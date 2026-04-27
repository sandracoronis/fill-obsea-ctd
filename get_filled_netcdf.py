"""
Author:  Sandra González-Villà (Coronis Computing S.L.)
Date:    29-01-2026
Version: 1.0.0

Description:
    Downloads OBSEA CTD data from ERDDAP, fills all gaps using a three-priority
    strategy (own sensor → other sensors → neural model / linear interpolation),
    and writes a gap-filled NetCDF file.

    When a model checkpoint is provided (--ckpt), gaps are filled using a trained
    neural imputation model. Otherwise, linear interpolation is used as fallback.

    If ERDDAP returns no data for the requested interval, the script runs the
    model on a synthetic all-NaN series (filling from climatology) and writes
    the result with QC=8.

    Optionally generates QC-colored diagnostic plots (-p).

Example usage:
    python get_filled_netcdf.py --start YYYYMMDD --end YYYYMMDD --out output_dir -p
    python get_filled_netcdf.py --time now --ckpt model/best_model.pt --out output_dir
"""

import json
import warnings
from argparse import ArgumentParser, ArgumentTypeError
from datetime import datetime, timedelta, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import requests
import torch
import xarray as xr

from models import build_model


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
VARIABLES  = ("CNDC", "PSAL", "TEMP", "SVEL", "PRES")
FILL_VALUE = -999990.0
VALID_QC   = frozenset({1, 7})

QC_DEFINITIONS = {
    0: ("unknown",                          "gray"),
    1: ("good_data",                        "forestgreen"),
    2: ("probably_good_data",               "steelblue"),
    3: ("potentially_correctable_bad_data", "orange"),
    4: ("bad_data",                         "indianred"),
    7: ("nominal_value",                    "purple"),
    8: ("interpolated/imputed_value",       "gold"),
    9: ("missing_value",                    "black"),
}

# Source flags tracking where each filled value came from
SOURCE_OWN   = 0   # value came from this sensor's own observations
SOURCE_OTHER = 1   # value borrowed from another sensor
SOURCE_MODEL = 2   # value from the neural model or linear interpolation
NO_SOURCE    = np.int8(127)   # sentinel: no value assigned yet


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def valid_date_yyyymmdd(s):
    """Parse and validate a date string in YYYYMMDD format."""
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        raise ArgumentTypeError(f"Invalid date '{s}'. Expected format YYYYMMDD.")

def valid_date_yyyymmddhhmm(s):
    """
    Parse a datetime string in YYYYMMDD-hh:mm format, or the literal 'now'.
    Minutes are snapped to the nearest previous half-hour (00 or 30).
    """
    if s == "now":
        now = datetime.now()
        minute = 30 if now.minute >= 30 else 0
        return datetime(now.year, now.month, now.day, now.hour, minute)

    try:
        ts = datetime.strptime(s, "%Y%m%d-%H:%M")
        minute = ts.minute
        if minute not in (0, 30):
            print(f"[WARNING] Minute {minute} is not 00 or 30 — snapping to previous half hour.")
            minute = 30 if minute > 30 else 0
        return datetime(ts.year, ts.month, ts.day, ts.hour, minute)
    except ValueError:
        raise ArgumentTypeError(f"Invalid date '{s}'. Expected format YYYYMMDD-hh:mm.")

def parse_args():
    """Define and parse command-line arguments."""
    parser = ArgumentParser(description="OBSEA_CTD_30min NetCDF gap filler")
    parser.add_argument(
        "--start", type=valid_date_yyyymmdd, default="20230101",
        help="Start date (YYYYMMDD). Used when --time is not given.",
    )
    parser.add_argument(
        "--end", type=valid_date_yyyymmdd, default="20240101",
        help="End date (YYYYMMDD). Used when --time is not given.",
    )
    parser.add_argument(
        "--time", type=valid_date_yyyymmddhhmm, default=None,
        help="Single timepoint to process (YYYYMMDD-hh:mm or 'now'). "
             "The filled NC file and plots span from 00:00 of that day to the given timestamp. "
             "A wider context window is downloaded automatically for the model but excluded from the output.",
    )
    parser.add_argument(
        "--ckpt", type=str,
        default=None,
        help="Path to the neural model checkpoint (.pt file). "
             "The climatology (cmems_climatology.json), configuration (config.json) and normalization (mu.npy and std.npy) "
             "files are expected in the same directory as the checkpoint. If not provided, linear interpolation is used.",
    )
    parser.add_argument(
        "--out", type=str,
        default="results/",
        help="Output directory for the filled NetCDF file.",
    )
    parser.add_argument(
        "--save-mode", choices=["one-timestamp", "all-sensors"], default="all-sensors",
        help=(
            "Output format for the filled NetCDF file. "
            "one-timestamp: one row per timestamp, sensor_id encodes the value source "
            "(original sensor id, 'model', or 'interp'). "
            "all-sensors: one row per sensor per timestamp covering the full range, "
            "one contiguous block per sensor."
        ),
    )
    parser.add_argument(
        "--valid-qc", nargs="+", type=int, default=[1, 7], metavar="QC",
        help=(
            "QC flag values considered 'valid' in the original file. "
            "Values with these flags are kept as-is with their original QC. "
            "Values with other QC flags are treated as missing and replaced by "
            "data from another sensor or by interpolation/imputation (QC=8). "
            "Default: 1 7"
        ),
    )
    parser.add_argument("-p", "--plot",    action="store_true", help="Generate QC plots.")
    parser.add_argument("-k", "--keep",    action="store_true", help="Keep raw download and per-sensor files.")
    return parser.parse_args()


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------
def _get_sequence_start(timestamp, seq_length):
    """Return the start of a sequence of `seq_length` 30-min steps ending at `timestamp`."""
    if isinstance(timestamp, str):
        timestamp = pd.to_datetime(timestamp)
    return timestamp - timedelta(minutes=30) * (seq_length - 1)

def resolve_time_range(args, seq_len):
    """
    Determine download and output time ranges from parsed arguments.

    Returns
    -------
    (download_start, download_end, output_start, output_end)
        download_* : range fetched from ERDDAP (may include context window)
        output_*   : range written to the filled NC file and plots

    When --time is given the download range is extended backwards by `seq_len`
    steps so the model has a full context window, but the output range is
    restricted to the requested day: 00:00 → args.time.
    """
    if args.time is not None:
        end_datetime   = args.time
        output_start   = datetime(end_datetime.year, end_datetime.month,
                                  end_datetime.day, tzinfo=end_datetime.tzinfo)
        output_end     = end_datetime
        download_start = _get_sequence_start(output_start, seq_len) if seq_len else output_start
        download_end   = end_datetime
    else:
        download_start = output_start = args.start
        download_end   = output_end   = args.end
    return download_start, download_end, output_start, output_end


# -----------------------------------------------------------------------------
# ERDDAP download
# -----------------------------------------------------------------------------
def build_erddap_url(dataset_name, start_datetime, end_datetime):
    """Build the ERDDAP tabledap URL for OBSEA_CTD_30min."""
    ts_start = (
        f"&time%3E={start_datetime.year}-{start_datetime.month}-"
        f"{start_datetime.day}T00%3A00%3A00Z"
    )
    ts_end = (
        f"&time%3C={end_datetime.year}-{end_datetime.month}-"
        f"{end_datetime.day}T23%3A59%3A59Z"
    )
    return (
        f"https://data.obsea.es/erddap/tabledap/{dataset_name}.nc"
        "?time%2Clatitude%2Clongitude%2Cdepth%2Csensor_id"
        "%2CCNDC%2CPSAL%2CTEMP%2CSVEL%2CPRES"
        "%2CCNDC_QC%2CPSAL_QC%2CTEMP_QC%2CSVEL_QC%2CPRES_QC"
        "%2Clatitude_QC%2Clongitude_QC%2Cdepth_QC"
        f"{ts_start}{ts_end}"
    )

def download_erddap_nc(url, filename, timeout=300):
    """
    Download a NetCDF file from ERDDAP and save it to disk.

    Returns
    -------
    int : 0 = success
          1 = HTTP error (server/network problem — do not proceed)
          2 = file write error
          3 = empty response
          4 = HTTP 404 / no data available for the requested interval
    """
    filename = Path(filename)
    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            if r.status_code == 404:
                print("[INFO] ERDDAP returned 404 — no data available for this interval.")
                return 4
            r.raise_for_status()
            bytes_written = 0
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

        if bytes_written == 0 or not filename.exists():
            return 3
        return 0

    except requests.RequestException as e:
        print(f"[ERROR] Download failed: {e}")
        return 1
    except OSError as e:
        print(f"[ERROR] File write failed: {e}")
        return 2


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_model(ckpt_path):
    """
    Load model architecture from config.json and weights from the checkpoint.

    Returns
    -------
    (model, mu, std, cfg) or (None, None, None, None) if loading fails.
    """
    config_path = ckpt_path.parent / "config.json"
    if not config_path.exists():
        print(f"[ERROR] config.json not found at {config_path}")
        return None, None, None, None

    with open(config_path) as f:
        cfg = json.load(f)

    arch = cfg.get("model", "gru").lower()
    if arch == "gru":
        kwargs = {"hidden_dim": cfg.get("hidden_dim", 64),
                  "n_layers":   cfg.get("n_layers",   2),
                  "dropout":    cfg.get("dropout",     0.1)}
    elif arch == "saits":
        hd     = cfg.get("hidden_dim", 128)
        kwargs = {"d_model":  hd,
                  "n_heads":  max(1, hd // 16),
                  "n_layers": cfg.get("n_layers", 2),
                  "d_ff":     hd * 2,
                  "dropout":  cfg.get("dropout", 0.1)}
    elif arch == "brits":
        kwargs = {"hidden_dim": cfg.get("hidden_dim", 64)}
    elif arch == "resgru":
        kwargs = {"hidden_dim": cfg.get("hidden_dim", 128),
                  "dropout":    cfg.get("dropout", 0.1)}
    else:
        kwargs = {"hidden_dim": cfg.get("hidden_dim", 64)}

    model = build_model(arch, **kwargs)
    ckpt  = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[INFO] Loaded {arch.upper()} model "
          f"(epoch {ckpt.get('epoch', '?')}, "
          f"val_loss {ckpt.get('val_loss', float('nan')):.5f})")

    mu_path  = ckpt_path.parent / "mu.npy"
    std_path = ckpt_path.parent / "std.npy"
    if mu_path.exists() and std_path.exists():
        mu  = np.load(mu_path)
        std = np.load(std_path)
        print(f"[INFO] Loaded normalisation stats from {mu_path.parent}")
    else:
        print("[WARNING] mu.npy / std.npy not found — "
              "computing from input data (suboptimal).")
        mu = std = None

    return model, mu, std, cfg

def load_climatology(path):
    """Load a climatology JSON file saved by save_climatology()."""
    with open(path) as f:
        raw = json.load(f)
    return {var: {int(k): float(v) for k, v in d.items()}
            for var, d in raw.items()}

def load_sensor_nc(path):
    """
    Load an OBSEA CTD NetCDF file into a tidy DataFrame.

    Returns
    -------
    DataFrame with columns: datetime, sensor, CNDC, PSAL, TEMP, SVEL, PRES
    and their corresponding _QC flags. Fill values are replaced with NaN;
    no QC filtering is applied here (that happens in _build_sensor_grid).
    """
    from scipy.io import netcdf_file

    f    = netcdf_file(str(path), "r", mmap=False)
    time = f.variables["time"][:].copy()
    sid  = f.variables["sensor_id"][:].copy()

    data, qc = {}, {}
    for var in VARIABLES:
        arr = f.variables[var][:].copy().astype(np.float32)
        arr[arr <= FILL_VALUE] = np.nan
        data[var] = arr
        qc[f"{var}_QC"] = f.variables[f"{var}_QC"][:].copy()
    f.close()

    sensors = np.array(
        ["".join(c.decode("utf-8", errors="replace") for c in row).strip("\x00").strip()
         for row in sid]
    )
    dt = pd.to_datetime(time, unit="s", origin="1970-01-01", utc=True)

    df = pd.DataFrame({"datetime": dt, "sensor": sensors})
    for var in VARIABLES:
        df[var]         = data[var]
        df[f"{var}_QC"] = qc[f"{var}_QC"]
    return df


def _build_sensor_grid(df, sensor, freq="30min", apply_qc=True, valid_qc=VALID_QC):
    """
    Reindex one sensor's observations onto a regular datetime grid.

    Missing timesteps have NaN for all variables. When apply_qc=True,
    values with QC flags not in `valid_qc` are set to NaN.
    """
    sub = (df[df["sensor"] == sensor]
           .drop_duplicates("datetime")
           .set_index("datetime")
           .sort_index())
    full = pd.date_range(sub.index.min(), sub.index.max(), freq=freq, tz="UTC")
    sub  = sub.reindex(full)

    if apply_qc:
        for var in VARIABLES:
            qccol = f"{var}_QC"
            if qccol in sub.columns:
                sub.loc[~sub[qccol].isin(valid_qc), var] = np.nan

    return sub

def build_merged_series(df, freq="30min", apply_qc=True, valid_qc=VALID_QC):
    """
    Merge all sensors into a single 30-min time series.

    When multiple sensors have simultaneous good observations, the first
    sensor (alphabetically) takes precedence.
    """
    sensors = sorted(df["sensor"].unique())
    full    = pd.date_range(df["datetime"].min(), df["datetime"].max(),
                            freq=freq, tz="UTC")
    grids   = {s: _build_sensor_grid(df, s, freq=freq, apply_qc=apply_qc,
                                    valid_qc=valid_qc)
               for s in sensors}

    merged = pd.DataFrame(index=full)
    for var in VARIABLES:
        col = pd.Series(np.nan, index=full, dtype=np.float32)
        for s in sensors:
            g = grids[s]
            if var in g.columns:
                obs_mask = g[var].notna()
                col[obs_mask & col.isna()] = g.loc[obs_mask & col.isna(), var]
        merged[var] = col

    for var in VARIABLES:
        merged[f"{var}_QC"] = np.where(merged[var].notna(), 1, 0).astype(np.int8)

    return merged


# -----------------------------------------------------------------------------
# NetCDF output
# -----------------------------------------------------------------------------
def _nc_compatible_string(series):
    """Convert a string Series to a fixed-width char array for NetCDF storage."""
    strlen = max(series.str.len()) if not series.empty else 2
    return np.array([list(s.ljust(strlen, "\0")) for s in series.values], "S1"), strlen

def _nc_process_data_column(series, ncfile):
    """
    Detect the appropriate NetCDF dtype and fill value for a pandas Series,
    and create any required string-length dimension.

    Returns
    -------
    (nc_dtype, nc_fill_value, zlib, values, dimensions)
    """
    dtype = series.dtype

    if str(series.name).endswith("_QC"):
        series = series.copy()
        series[series.isna()] = 9
        return "i1", 127, True, series.astype("i1").to_numpy(), ("row",)

    if pd.api.types.is_float_dtype(dtype):
        if dtype == "float64":
            return "double", None, True, series.to_numpy(), ("row",)
        return "float32", -999999.0, True, series.to_numpy(), ("row",)

    if pd.api.types.is_unsigned_integer_dtype(dtype):
        return "u4", 4294967295, True, series.to_numpy(), ("row",)

    if pd.api.types.is_integer_dtype(dtype):
        return "i4", -2147483648, True, series.to_numpy(), ("row",)

    if pd.api.types.is_string_dtype(dtype):
        values, strlen = _nc_compatible_string(series)
        dim_name = str(series.name) + "_strlen"
        if dim_name not in ncfile.dimensions:
            ncfile.createDimension(dim_name, strlen)
        return "S1", None, False, values, ("row", dim_name)

    if pd.api.types.is_datetime64_any_dtype(dtype):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            times = np.array(series.dt.to_pydatetime())
        times = nc.date2num(times, "seconds since 1970-01-01", calendar="standard")
        return "double", -9999.99, True, times, ("row",)

    raise ValueError(f"Cannot convert dtype '{dtype}' to NetCDF type!")

def _decode_sid_column(raw_sid_col):
    """
    Decode a sensor_id char array (n_raw, [1,] strlen) of S1 bytes/strings
    to a plain 1-D string array, one entry per raw row.
    """
    if raw_sid_col.ndim == 3:
        raw_sid_col = raw_sid_col[:, 0, :]
    return np.array([
        "".join(c.decode() if isinstance(c, bytes) else c for c in r
                ).rstrip("\x00").strip()
        for r in raw_sid_col
    ])

def _build_orig_qc(ds_nc, decoded_sids, sensors, time_sec, valid_qc=VALID_QC):
    """
    Pre-align original QC flags onto the full time axis for each sensor.

    The original QC flag is restored only where:
      - the sensor had a real (non-fill) value in the raw file, AND
      - that value's QC flag is in `valid_qc`.
    All other positions (missing, fill, or invalid-QC) default to QC=8
    (interpolated/imputed), since the output value came from another source.

    Returns
    -------
    dict[sensor_id_str -> dict[qc_var -> np.ndarray[int8, n_time]]]
    """
    n_time       = len(time_sec)
    raw_time_sec = ds_nc.variables["time"][:].astype("float64")
    qc_vars      = [v for v in ds_nc.variables if v.endswith("_QC")]

    orig_qc = {}
    for sensor in sensors:
        sid_str = sensor if isinstance(sensor, str) else sensor.decode()
        mask    = decoded_sids == sid_str
        s_times = raw_time_sec[mask]
        s_idx   = np.searchsorted(time_sec, s_times)
        # Clip before indexing to avoid out-of-bounds when raw rows fall outside
        # the (trimmed) output window; those positions will be excluded by `valid`.
        valid   = (s_idx < n_time) & (time_sec[np.minimum(s_idx, n_time - 1)] == s_times)

        orig_qc[sensor] = {}
        for qv in qc_vars:
            arr = np.full(n_time, 8, dtype="i1")
            if valid.any():
                raw_qc_vals = ds_nc.variables[qv][:][mask][valid].astype("i1")
                data_var    = qv.replace("_QC", "")
                if data_var in ds_nc.variables:
                    # Restore QC only where the raw value was present (not fill)
                    # AND its QC flag is in valid_qc. Rows that fail either check
                    # had their data replaced by another source → keep QC=8.
                    raw_vals  = ds_nc.variables[data_var][:][mask][valid]
                    has_value = np.asarray(raw_vals) > FILL_VALUE
                    is_valid  = np.isin(raw_qc_vals, list(valid_qc))
                    keep      = has_value & is_valid
                    arr[s_idx[valid][keep]] = raw_qc_vals[keep]
                else:
                    # No corresponding data variable (e.g. a pure metadata QC):
                    # restore unconditionally.
                    arr[s_idx[valid]] = raw_qc_vals
            orig_qc[sensor][qv] = arr

    return orig_qc

def _write_global_attrs(ncfile, ds_attrs, time_sec_array, modifier):
    """Copy global attributes from the raw file and update provenance fields."""
    ncfile.setncatts(ds_attrs.copy())
    min_date = pd.to_datetime(time_sec_array.min(), unit="s", origin="1970-01-01", utc=True)
    max_date = pd.to_datetime(time_sec_array.max(), unit="s", origin="1970-01-01", utc=True)
    ncfile.setncatts({
        "time_coverage_start": min_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "time_coverage_end":   max_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "date_modified":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modified_by":         modifier,
    })

def build_sensor_filled_series(df, filled_df, valid_qc=VALID_QC):
    """
    For every sensor in `df`, build a gapless series over `filled_df.index`
    using the three-priority fill strategy:
      1. sensor's own value, only where its QC flag is in `valid_qc`.
      2. value from another sensor (first available, alphabetical order),
         only where that sensor's QC flag is in `valid_qc`.
      3. model-imputed / interpolated value from `filled_df`, QC=8.

    Returns
    -------
    dict[sensor_id -> dict[var -> (values: pd.Series, source: pd.Series)]]
        `values` is float, `source` is int (SOURCE_OWN / SOURCE_OTHER / SOURCE_MODEL).
    """
    full_idx = filled_df.index
    sensors  = sorted(df["sensor"].unique())

    # One grid per sensor filtered by valid_qc — used for both own and borrowed values.
    grids = {s: _build_sensor_grid(df, s, freq="30min", apply_qc=True, valid_qc=valid_qc)
             for s in sensors}

    result = {}
    for sensor in sensors:
        sensor_data = {}

        for var in VARIABLES:
            if var not in filled_df.columns:
                continue

            grid    = grids[sensor]
            own     = grid[var].reindex(full_idx) if var in grid.columns \
                      else pd.Series(np.nan, index=full_idx, dtype="float32")
            imputed = filled_df[var]

            values = own.copy().astype("float32")
            source = pd.Series(np.where(own.notna(), SOURCE_OWN, -1),
                               index=full_idx, dtype="int8")

            # Priority 2: borrow valid-QC data from other sensors
            for other_s, other_grid in grids.items():
                if other_s == sensor or var not in other_grid.columns:
                    continue
                other_vals = other_grid[var].reindex(full_idx)
                mask = (source == -1) & other_vals.notna()
                values[mask] = other_vals[mask]
                source[mask] = SOURCE_OTHER

            # Priority 3: model imputation / interpolation
            mask = source == -1
            values[mask] = imputed[mask].astype("float32")
            source[mask] = SOURCE_MODEL

            sensor_data[var] = (values, source)

        result[sensor] = sensor_data

    return result


def write_imputed_nc_mode_a(nc_file, sensor_series, filled_path, valid_qc=VALID_QC):
    """
    Save-mode one-timestamp: one row per timestamp.

    For each timestep the best available value is chosen in priority order
    (own sensor → other sensor → model/interpolation). The `sensor_id` field
    records the source: the original sensor id, or 'model' for imputed values.

    Parameters
    ----------
    nc_file       : str — raw downloaded NetCDF (used as structural template)
    sensor_series : output of build_sensor_filled_series()
    filled_path   : Path — output file path
    """
    ds_xr   = xr.open_dataset(nc_file, decode_times=False)
    ds_nc   = nc.Dataset(nc_file, "r")
    sensors = list(sensor_series.keys())

    full_idx = next(iter(next(iter(sensor_series.values())).values()))[0].index
    time_sec = (full_idx.astype("int64") // 1_000_000_000).to_numpy()
    n_rows   = len(time_sec)

    # --- Build best-value and provenance arrays ---
    # Initialise best_source to NO_SOURCE=127 so that every real value
    # (including model-imputed, SOURCE_MODEL=2) is considered "better".
    best_vals   = {var: np.full(n_rows, np.nan, dtype="float32") for var in VARIABLES}
    best_source = {var: np.full(n_rows, NO_SOURCE, dtype="int8") for var in VARIABLES}
    best_sid    = np.full(n_rows, b"model", dtype="S32")

    for sensor, var_data in sensor_series.items():
        sid_bytes = sensor.encode("utf-8") if isinstance(sensor, str) else sensor
        for var, (values, source) in var_data.items():
            better = source.values < best_source[var]
            best_vals[var][better]   = values.values[better]
            best_source[var][better] = source.values[better]
            best_sid[better]         = sid_bytes

    # Decode sensor_id column once (reused for scalar lookup and QC alignment)
    decoded_sids = _decode_sid_column(ds_nc.variables["sensor_id"][:])

    # Build per-sensor lookup of scalar field values (latitude, longitude, depth, …)
    _SKIP_VARS = set(VARIABLES) | {f"{v}_QC" for v in VARIABLES} | {"time", "sensor_id"}
    sensor_scalars: dict[str, dict[str, float]] = {}
    for sensor in sensors:
        sid_str = sensor if isinstance(sensor, str) else sensor.decode()
        mask    = decoded_sids == sid_str
        sensor_scalars[sid_str] = {}
        if not mask.any():
            continue
        for v in ds_nc.variables:
            if v in _SKIP_VARS:
                continue
            raw_v = ds_nc.variables[v][:]
            if raw_v.dtype.kind in ("S", "U", "O"):
                continue
            try:
                sensor_scalars[sid_str][v] = float(raw_v[mask][0])
            except (IndexError, ValueError, TypeError):
                pass

    # Pre-align original QC flags for each sensor onto the full time axis
    orig_qc_a = _build_orig_qc(ds_nc, decoded_sids, sensors, time_sec, valid_qc)

    # String version of best_sid (for vectorized sensor lookup below)
    sid_strs = np.array([
        s.decode() if isinstance(s, bytes) else s for s in best_sid
    ])

    with nc.Dataset(str(filled_path), "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("row", n_rows)

        for var in ds_nc.variables:
            var_type = ds_nc.variables[var].getncattr("variable_type") \
                       if "variable_type" in ds_nc.variables[var].ncattrs() else None

            nc_dtype, nc_fill_value, zlib, _, dimensions = \
                _nc_process_data_column(ds_xr[var].to_series(), ncfile)

            myvar = ncfile.createVariable(var, nc_dtype, dimensions,
                                          zlib=zlib, fill_value=nc_fill_value)
            myvar.setncatts(ds_xr[var].attrs)

            if var == "time":
                myvar[:] = time_sec

            elif nc_dtype == "S1":
                # sensor_id: encode the source sensor (or 'model') per timestep
                strlen_dim = next(
                    (d for d in ds_nc.variables[var].dimensions if d != "row"), None
                )
                strlen = ds_nc.dimensions[strlen_dim].size if strlen_dim else 1
                rows = np.array([
                    list((s.decode() if isinstance(s, bytes) else s
                          ).ljust(strlen, "\x00")[:strlen])
                    for s in best_sid
                ], dtype="S1")
                myvar._Encoding = "ISO-8859-1"
                myvar[:] = rows
                myvar.setncattr("comment",
                                "Source sensor id, or 'model' for imputed/interpolated values.")

            elif var in VARIABLES:
                flat = best_vals[var]
                myvar[:] = flat
                myvar.setncattr("actual_range",
                                np.array([np.nanmin(flat), np.nanmax(flat)],
                                         dtype="float32"))

            elif var.endswith("_QC"):
                # Preserve original QC where the value came from the sensor's own data
                # (including e.g. QC=7 nominal_value); elsewhere QC=8 (imputed).
                data_var  = var.replace("_QC", "")
                proxy_var = data_var if data_var in VARIABLES else next(
                    (v for v in VARIABLES if v in best_vals), None
                )
                src      = best_source.get(proxy_var, np.full(n_rows, NO_SOURCE))
                qc       = np.full(n_rows, 8, dtype="i1")
                own_mask = src == SOURCE_OWN
                for sid_str, qc_dict in orig_qc_a.items():
                    if var not in qc_dict:
                        continue
                    mask = own_mask & (sid_strs == sid_str)
                    qc[mask] = qc_dict[var][mask]
                myvar[:] = qc

            else:
                # All remaining variables (sensor/platform constants and other numeric):
                # use the value from whichever sensor provided the best value at each
                # timestep, falling back to the first sensor that has the variable.
                fallback = next(
                    (sensor_scalars[s][var] for s in sensor_scalars
                     if var in sensor_scalars[s]),
                    nc_fill_value if nc_fill_value is not None else 0.0,
                )
                data = np.full(n_rows, fallback, dtype=nc_dtype)
                for sid_str, sc_dict in sensor_scalars.items():
                    if var in sc_dict:
                        data[sid_strs == sid_str] = sc_dict[var]
                myvar[:] = data

        _write_global_attrs(ncfile, ds_xr.attrs, time_sec,
                            "Coronis Computing neural gap filler (mode A)")

    ds_nc.close()
    ds_xr.close()
    print(f"[INFO] Mode one-timestamp — filled NetCDF written to {filled_path}")

def write_imputed_nc_mode_b(nc_file, sensor_series, filled_path, valid_qc=VALID_QC):
    """
    Save-mode all-sensors: one row per sensor per timestamp, covering the full time range.

    The output has exactly the same variables as the raw NetCDF file.
    Each sensor gets a contiguous block of rows spanning the full period.
    For each sensor block:
      - `time` and `sensor_id` are tiled/repeated appropriately.
      - Scalar variables (latitude, longitude, depth, …) are filled with the
        sensor's own constant value.
      - Data variables (CNDC, PSAL, …) are filled using the 3-priority strategy
        via `sensor_series`.
      - QC flags: original value where the sensor had its own observation (e.g.
        QC=7 nominal_value is preserved), QC=8 elsewhere.

    Parameters
    ----------
    nc_file       : str — raw downloaded NetCDF (structural template)
    sensor_series : output of build_sensor_filled_series()
    filled_path   : Path — output file path
    """
    ds_nc   = nc.Dataset(nc_file, "r")
    ds_xr   = xr.open_dataset(nc_file, decode_times=False)
    sensors = list(sensor_series.keys())

    full_idx = next(iter(next(iter(sensor_series.values())).values()))[0].index
    time_sec = (full_idx.astype("int64") // 1_000_000_000).to_numpy()
    n_time   = len(time_sec)
    n_rows   = n_time * len(sensors)

    # Decode sensor_id once and build per-sensor masks over raw rows
    decoded_sids  = _decode_sid_column(ds_nc.variables["sensor_id"][:])
    sensor_masks  = {
        sensor: decoded_sids == (sensor if isinstance(sensor, str) else sensor.decode())
        for sensor in sensors
    }

    # Pre-align original QC flags onto the full time axis for each sensor
    orig_qc = _build_orig_qc(ds_nc, decoded_sids, sensors, time_sec, valid_qc)

    with nc.Dataset(str(filled_path), "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("row", n_rows)

        for var in ds_xr.variables:
            raw_series = ds_xr[var].to_series()
            nc_dtype, nc_fill_value, zlib, _, dimensions = \
                _nc_process_data_column(raw_series, ncfile)

            myvar = ncfile.createVariable(var, nc_dtype, dimensions,
                                          zlib=zlib, fill_value=nc_fill_value)
            myvar.setncatts(ds_xr[var].attrs)

            if var == "time":
                # Tile the full time axis once per sensor block
                myvar[:] = np.tile(time_sec, len(sensors))

            elif nc_dtype == "S1":
                # Build char array directly from the known sensor id string.
                # (xarray decodes S1 to Python strings, so we cannot use ds_xr here)
                myvar._Encoding = "ISO-8859-1"
                strlen_dim = next(
                    (d for d in ds_nc.variables[var].dimensions if d != "row"), None
                )
                strlen = ds_nc.dimensions[strlen_dim].size if strlen_dim else 1
                chunks = []
                for sensor in sensors:
                    sid_str = sensor if isinstance(sensor, str) else sensor.decode()
                    row = np.array(
                        list(sid_str.ljust(strlen, "\x00")[:strlen]), dtype="S1"
                    )
                    chunks.append(np.repeat(row[np.newaxis], n_time, axis=0))
                myvar[:] = np.concatenate(chunks, axis=0)

            elif var in VARIABLES:
                # Data variable: use priority-filled values from sensor_series
                chunks = []
                for sensor in sensors:
                    if var in sensor_series[sensor]:
                        values, _ = sensor_series[sensor][var]
                        chunks.append(values.values.astype("float32"))
                    else:
                        chunks.append(np.full(n_time, nc_fill_value, dtype="float32"))
                flat = np.concatenate(chunks)
                myvar[:] = flat
                myvar.setncattr("actual_range",
                                np.array([np.nanmin(flat), np.nanmax(flat)],
                                         dtype="float32"))

            elif var.endswith("_QC"):
                # orig_qc already has QC=8 at non-own positions, preserving
                # the original flag (e.g. 7=nominal_value) where the sensor
                # had its own observation.
                chunks = []
                for sensor in sensors:
                    if var in orig_qc[sensor]:
                        chunks.append(orig_qc[sensor][var])
                    else:
                        chunks.append(np.full(n_time, 8, dtype="i1"))
                myvar[:] = np.concatenate(chunks)

            else:
                # All remaining variables (scalar per sensor: latitude, longitude,
                # depth, and any other numeric constants): repeat each sensor's
                # first observed value across its time block.
                raw_var = ds_nc.variables[var][:]
                chunks  = []
                for sensor in sensors:
                    mask = sensor_masks[sensor]
                    val  = raw_var[mask][0] if mask.any() else nc_fill_value
                    chunks.append(np.full(n_time, val, dtype=nc_dtype))
                myvar[:] = np.concatenate(chunks)

        _write_global_attrs(ncfile, ds_xr.attrs, time_sec,
                            "Coronis Computing neural gap filler (mode B)")

    ds_nc.close()
    ds_xr.close()
    print(f"[INFO] Mode all-sensors — filled NetCDF written to {filled_path}")

def write_empty_nc(filled_path, out_start, out_end, filled_df=None, freq="30min"):
    """
    Write a NetCDF file covering [out_start, out_end] for the case where no
    sensor data was available from ERDDAP.

    When `filled_df` is provided (model/interpolation ran on a synthetic all-NaN
    series), its values are written with QC=8 (interpolated/imputed).
    When `filled_df` is None, all values are NaN with QC=9 (missing_value).

    Parameters
    ----------
    filled_path : Path     — output file path
    out_start   : datetime — start of the output window (inclusive)
    out_end     : datetime — end of the output window (inclusive)
    filled_df   : pd.DataFrame or None — imputed values indexed by DatetimeIndex
    freq        : str      — time step (default 30min)
    """
    time_idx = pd.date_range(
        pd.Timestamp(out_start, tz="UTC"),
        pd.Timestamp(out_end,   tz="UTC"),
        freq=freq,
    )
    time_sec = (time_idx.astype("int64") // 1_000_000_000).to_numpy()
    n_rows   = len(time_sec)

    sid_str = "climatology"
    strlen  = len(sid_str)

    with nc.Dataset(str(filled_path), "w", format="NETCDF4") as ncfile:
        ncfile.createDimension("row", n_rows)
        ncfile.createDimension("sensor_id_strlen", strlen)

        # time
        tv = ncfile.createVariable("time", "f8", ("row",), zlib=True)
        tv.units    = "seconds since 1970-01-01T00:00:00Z"
        tv.calendar = "standard"
        tv[:]       = time_sec

        # sensor_id — "climatology" repeated for every row
        sv = ncfile.createVariable("sensor_id", "S1",
                                   ("row", "sensor_id_strlen"))
        sv._Encoding = "ISO-8859-1"
        row = np.array(list(sid_str), dtype="S1")
        sv[:] = np.repeat(row[np.newaxis], n_rows, axis=0)

        # scalar spatial fields — fixed OBSEA location, QC=7 (nominal_value)
        for field, val, qc_field in [
            ("latitude",  41.18212128, "latitude_QC"),
            ("longitude",  1.75257003, "longitude_QC"),
            ("depth",     20.0,        "depth_QC"),
        ]:
            fv = ncfile.createVariable(field, "f4", ("row",), zlib=True,
                                       fill_value=-999999.0)
            fv[:] = np.full(n_rows, val, dtype="float32")
            qv = ncfile.createVariable(qc_field, "i1", ("row",), zlib=True,
                                       fill_value=np.int8(127))
            qv[:] = np.full(n_rows, 7, dtype="i1")

        # data variables
        for var in VARIABLES:
            dv = ncfile.createVariable(var, "f4", ("row",), zlib=True,
                                       fill_value=-999999.0)
            qv = ncfile.createVariable(f"{var}_QC", "i1", ("row",), zlib=True,
                                       fill_value=np.int8(127))

            if filled_df is not None and var in filled_df.columns:
                vals = filled_df[var].reindex(time_idx).values.astype("float32")
                dv[:] = vals
                # QC=8 where imputed, QC=9 where still NaN (model couldn't fill)
                qv[:] = np.where(np.isfinite(vals), np.int8(8), np.int8(9))
            else:
                dv[:] = np.full(n_rows, np.nan, dtype="float32")
                qv[:] = np.full(n_rows, 9, dtype="i1")

        min_date = pd.to_datetime(time_sec.min(), unit="s", origin="1970-01-01", utc=True)
        max_date = pd.to_datetime(time_sec.max(), unit="s", origin="1970-01-01", utc=True)
        comment  = ("No sensor data available. Values imputed by model from climatology."
                    if filled_df is not None else
                    "No sensor data available for this interval.")
        ncfile.setncatts({
            "Conventions":         "CF-1.6, ACDD-1.3",
            "cdm_data_type":       "Point",
            "featureType":         "point",
            "time_coverage_start": min_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time_coverage_end":   max_date.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "date_modified":       datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "modified_by":         "Coronis Computing gap filler",
            "comment":             comment,
        })

    print(f"[INFO] NC written to {filled_path} ({n_rows} timesteps, "
          f"{'model-imputed' if filled_df is not None else 'all QC=9/missing'})")


# -----------------------------------------------------------------------------
# Gap filling
# -----------------------------------------------------------------------------
def _clim_predict(clim, datetimes, variables=VARIABLES):
    """
    Vectorised climatology lookup.

    Parameters
    ----------
    clim      : {var: {doy_int: value}}
    datetimes : DatetimeIndex on the 30-min grid
    variables : ordered tuple of variable names

    Returns
    -------
    [T, V] float32 array, NaN where the climatology has no entry
    """
    doys = datetimes.dayofyear.to_numpy()
    out  = np.full((len(datetimes), len(variables)), np.nan, dtype=np.float32)
    for j, var in enumerate(variables):
        if var in clim:
            vmap   = clim[var]
            out[:, j] = np.array([vmap.get(int(d), np.nan) for d in doys],
                                  dtype=np.float32)
    return out

def _model_fill_gaps(series, obs_mask, clim_norm, time_feat, labels,
                     model, seq_len, device):
    """
    Fill positions labelled "model" by sliding the GRU over the series.

    A window of `seq_len` is centred around each model position. Overlapping
    windows are averaged. The obs_mask is set to 1 at model positions to match
    the training convention (artificially masked positions always had obs_mask=1).

    Parameters
    ----------
    series    : [T, V] float32, normalised, NaN where missing
    obs_mask  : [T, V] bool
    clim_norm : [T, V] float32, normalised climatology
    time_feat : [T, 4] float32
    labels    : [T, V] object array with values "model" / "observed" / etc.
    model     : trained torch.nn.Module
    seq_len   : int
    device    : torch.device

    Returns
    -------
    [T, V] float32 with model positions filled
    """
    T, _     = series.shape
    filled   = series.copy()
    pred_sum = np.zeros_like(series)
    pred_cnt = np.zeros_like(series)

    model.eval()
    model_positions = np.where((labels == "model").any(axis=1))[0]
    if len(model_positions) == 0:
        return filled

    # Collect unique window start positions centred on each model timestep
    window_starts = set()
    for t in model_positions:
        ws = max(0, t - seq_len // 2)
        we = min(T, ws + seq_len)
        ws = max(0, we - seq_len)
        window_starts.add(ws)

    with torch.no_grad():
        for ws in sorted(window_starts):
            we  = min(T, ws + seq_len)
            sl  = slice(ws, we)
            W   = we - ws

            x_w    = filled[sl].copy()
            obs_w  = obs_mask[sl].copy().astype(np.float32)
            lbl_w  = labels[sl]
            clim_w = clim_norm[sl].copy()
            tf_w   = time_feat[sl].copy()

            # Match training convention: obs_mask=1 at masked (model) positions
            obs_w = np.where(lbl_w == "model", 1.0, obs_w)

            # Model input: anomaly w.r.t. climatology (NaN → clim)
            x_in   = np.where(np.isnan(x_w), clim_w, x_w)
            x_anom = x_in - clim_w

            # Pad short windows at the end of the series
            pad = seq_len - W
            if pad > 0:
                x_anom = np.pad(x_anom, ((0, pad), (0, 0)))
                obs_w  = np.pad(obs_w,  ((0, pad), (0, 0)))
                clim_w = np.pad(clim_w, ((0, pad), (0, 0)))
                tf_w   = np.pad(tf_w,   ((0, pad), (0, 0)))

            xt  = torch.tensor(x_anom[None], dtype=torch.float32, device=device)
            om  = torch.tensor(obs_w[None],  dtype=torch.float32, device=device)
            tm  = torch.zeros_like(om)
            cl  = torch.tensor(clim_w[None], dtype=torch.float32, device=device)
            tf  = torch.tensor(tf_w[None],   dtype=torch.float32, device=device)

            out = model(xt, om, tm, cl, tf).cpu().numpy()[0, :W]  # [W, V]

            pred_sum[sl] += out
            pred_cnt[sl] += 1

    valid_cnt  = pred_cnt > 0
    averaged   = np.where(valid_cnt, pred_sum / np.maximum(pred_cnt, 1), np.nan)
    model_mask = labels == "model"
    filled     = np.where(model_mask & valid_cnt, averaged, filled)

    return filled

def _build_time_features(index):
    """
    Return [T, 4] array of cyclic time encodings:
        sin/cos of day-of-year (seasonal)
        sin/cos of hour-of-day (diurnal)
    """
    doy  = index.dayofyear.to_numpy().astype(np.float32)
    hod  = (index.hour + index.minute / 60).to_numpy().astype(np.float32)
    return np.stack([
        np.sin(2 * np.pi * doy / 366),
        np.cos(2 * np.pi * doy / 366),
        np.sin(2 * np.pi * hod / 24),
        np.cos(2 * np.pi * hod / 24),
    ], axis=-1).astype(np.float32)

def run_neural_imputation(merged, model, mu, std, clim, seq_len, device):
    """
    Run the full neural imputation pipeline on a merged sensor DataFrame.

    Steps
    -----
    1. Normalise the data and build the climatology baseline.
    2. Label all missing positions as "model" targets.
    3. Slide the neural model over the series and collect predictions.
    4. Denormalise and write results back into the DataFrame.

    Returns
    -------
    filled_df  : DataFrame with same shape as `merged`, gaps imputed
    obs_mask   : [T, V] bool, True where originally observed
    """
    variables = [v for v in VARIABLES if v in merged.columns]
    raw       = np.stack([merged[v].values for v in variables], axis=-1)
    miss_mask = np.isnan(raw)

    # All missing positions are targets for the model
    labels = np.where(miss_mask, "model", "observed").astype(object)

    clim_raw  = _clim_predict(clim, merged.index, variables=tuple(variables))

    eps     = 1e-6
    mu_v    = np.array([mu[list(VARIABLES).index(v)]  for v in variables], dtype=np.float32)
    std_v   = np.array([std[list(VARIABLES).index(v)] for v in variables], dtype=np.float32)
    obs_mask = ~miss_mask

    raw_norm = np.where(
        miss_mask,
        np.nan,
        np.clip((raw - mu_v) / (std_v + eps), -10, 10),
    ).astype(np.float32)
    clim_norm = np.where(
        np.isnan(clim_raw), 0.0,
        (clim_raw - mu_v) / (std_v + eps),
    ).astype(np.float32)

    time_feat = _build_time_features(merged.index)

    raw_norm_filled = _model_fill_gaps(
        raw_norm, obs_mask, clim_norm, time_feat,
        labels, model, seq_len, device,
    )

    # Denormalise predictions back to original units
    raw_filled = np.where(
        labels == "model",
        raw_norm_filled * std_v + mu_v,
        raw,
    )

    filled_df = merged.copy()
    for j, var in enumerate(variables):
        filled_df[var] = raw_filled[:, j]

    return filled_df, obs_mask

def run_linear_imputation(merged):
    """
    Fill all gaps in the merged sensor DataFrame by linear interpolation.

    Interior gaps are filled by linear interpolation between the two nearest
    observed values. Leading / trailing gaps are filled by propagating the
    nearest observed value outward (equivalent to np.interp clamping).

    Returns
    -------
    filled_df : DataFrame with same shape as `merged`, all gaps filled
    """
    filled_df = merged.copy()
    for var in VARIABLES:
        if var in filled_df.columns:
            filled_df[var] = (filled_df[var]
                              .interpolate(method="linear")
                              .ffill()
                              .bfill())
    return filled_df


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------
def _extract_numeric_vars_and_qc(ds):
    """
    Return (data_vars, qc_vars) lists for all numeric variables in `ds`,
    excluding the QC variables themselves.
    """
    data_vars, qc_vars = [], []
    for var, da in ds.data_vars.items():
        if var.endswith("_QC"):
            continue
        if not np.issubdtype(da.dtype, np.number):
            continue
        data_vars.append(var)
        qc_name = f"{var}_QC"
        if qc_name in ds.data_vars:
            qc_vars.append(qc_name)
    return data_vars, qc_vars

def plot_whole_range(nc_file, outdir, suffix="", merge_sensors=False, xlim=None,
                     all_vars=False):
    """
    Generate QC-colored time-series scatter plots for numeric variables.

    Parameters
    ----------
    merge_sensors : bool
        If False (default), produce one figure per sensor per variable.
        If True, produce one figure per variable with all sensors overlaid —
        useful for one-timestamp output where all data is in a single series.
    xlim : (datetime-like, datetime-like) or None
        If given, sets the x-axis limits on every figure. Useful when the NC
        file contains a wider context window than the intended output range.
    all_vars : bool
        If False (default), only plot VARIABLES (CNDC, PSAL, TEMP, SVEL, PRES).
        If True, plot all numeric variables found in the file (including
        latitude, longitude, depth, etc.).
    """
    print(f"[INFO] Generating plots: {suffix}")

    with xr.open_dataset(nc_file, decode_times=False) as ds_raw:
        sensors = list(np.unique(ds_raw["sensor_id"].values))
        data_vars, qc_vars = _extract_numeric_vars_and_qc(ds_raw)
        if not all_vars:
            all_qc_vars = set(qc_vars)
            data_vars   = [v for v in data_vars if v in VARIABLES]
            qc_vars     = [f"{v}_QC" for v in data_vars if f"{v}_QC" in all_qc_vars]
        ds = ds_raw[data_vars + qc_vars + ["time", "sensor_id"]].load()

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Found {len(data_vars)} variables with potential QC flags")

    if merge_sensors:
        # One plot per variable with all rows combined
        time     = ds["time"].values.astype("float64")
        datetime = pd.to_datetime(time, unit="s", utc=True)

        for var in data_vars:
            qc_var = f"{var}_QC"
            if qc_var not in qc_vars:
                print(f"[INFO] Skipping {var}: no QC column")
                continue

            print(f"[INFO] Plotting {var} (all sensors merged)")

            plt.figure(figsize=(14, 4))
            for qc_value, (meaning, color) in QC_DEFINITIONS.items():
                mask = ds[qc_var].values == qc_value
                if mask.any():
                    plt.scatter(datetime[mask], ds[var].values[mask],
                                s=1, c=color, label=meaning, alpha=0.8)

            plt.xlabel("Time (UTC)")
            plt.ylabel(var)
            plt.title(f"{var} vs Time (QC-colored){suffix}")
            plt.legend(title="QC flag", markerscale=2, fontsize="small", ncol=3)
            if xlim is not None:
                plt.xlim(xlim)
            plt.tight_layout()
            plt.savefig(output_dir / f"{var}_vs_time_QC{suffix}.png", dpi=300)
            plt.close()

    else:
        # One plot per sensor per variable
        for sensor in sensors:
            sensor_str  = sensor.decode("utf-8") if isinstance(sensor, bytes) else str(sensor)
            sensor_mask = ds["sensor_id"].values == sensor
            if not sensor_mask.any():
                raise ValueError(f"Sensor '{sensor}' not found")

            try:
                ds_s = ds.isel(obs=sensor_mask)
            except Exception:
                ds_s = ds.isel(row=sensor_mask)

            data_vars_s, qc_vars_s = _extract_numeric_vars_and_qc(ds_s)
            time     = ds_s["time"].values.astype("float64")
            datetime = pd.to_datetime(time, unit="s", utc=True)

            for var in data_vars_s:
                qc_var = f"{var}_QC"
                if qc_var not in qc_vars_s:
                    print(f"[INFO] Skipping {var}: no QC column")
                    continue

                print(f"[INFO] Plotting {var} for sensor {sensor_str}")

                plt.figure(figsize=(10, 4))
                for qc_value, (meaning, color) in QC_DEFINITIONS.items():
                    mask = ds_s[qc_var].values == qc_value
                    if mask.any():
                        plt.scatter(datetime[mask], ds_s[var].values[mask],
                                    s=1, c=color, label=meaning, alpha=0.8)

                plt.xlabel("Time (UTC)")
                plt.ylabel(var)
                plt.title(f"{sensor_str}: {var} vs Time (QC-colored)")
                plt.legend(title="QC flag", markerscale=2, fontsize="small", ncol=3)
                if xlim is not None:
                    plt.xlim(xlim)
                plt.tight_layout()
                plt.savefig(output_dir / f"{sensor_str}_{var}_vs_time_QC{suffix}.png", dpi=300)
                plt.close()

    print(f"[INFO] QC plots saved to {output_dir}")

def plot_imputed_vs_observed(df, filled_df, outdir, valid_qc=VALID_QC):
    """
    For each variable, produce a figure with one panel per sensor.

    Each panel fills values in priority order:
      1. (blue)   sensor's own observed values
      2. (green)  values from other sensors where this sensor has NaN
      3. (orange) model-imputed values where no sensor had data

    When `df` is None (no sensor data available), a single panel per variable
    is produced showing all values as model-imputed.

    Parameters
    ----------
    df        : pd.DataFrame or None — raw per-row sensor data from load_sensor_nc()
    filled_df : pd.DataFrame — merged imputed series indexed by DatetimeIndex
    outdir    : str or Path
    """
    print("\n[INFO] Generating imputed-vs-observed plots")

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- No-sensor fallback: single panel per variable, all values are model-imputed ---
    if df is None or df.empty:
        full_idx = filled_df.index
        for var_name in VARIABLES:
            if var_name not in filled_df.columns:
                continue
            fig, ax = plt.subplots(figsize=(14, 3), dpi=150, constrained_layout=True)
            ax.scatter(full_idx, filled_df[var_name], s=1, color="orange",
                       label="Model imputed", alpha=0.9)
            ax.set_ylabel(var_name)
            ax.set_title(f"{var_name} — model imputed (no sensor data)")
            ax.set_xlabel("Time (UTC)")
            ax.legend(loc="upper right", fontsize="small")
            ax.set_xlim(full_idx.min(), full_idx.max())
            fig.savefig(output_dir / f"{var_name}_imputed.png", dpi=150)
            plt.close(fig)
        print(f"[INFO] Plots saved to {output_dir}")
        return

    sensors = sorted(df["sensor"].unique())

    # Build per-sensor grids once, reused across variables
    grids = {s: _build_sensor_grid(df, s, freq="30min", apply_qc=True, valid_qc=valid_qc)
             for s in sensors}

    for var_name in VARIABLES:
        if var_name not in filled_df.columns:
            continue

        fig, axes = plt.subplots(
            nrows=len(sensors), ncols=1,
            figsize=(14, 3 * len(sensors)), dpi=150,
            sharex=True, constrained_layout=True,
        )
        if len(sensors) == 1:
            axes = [axes]

        for ax, sensor in zip(axes, sensors):
            grid = grids[sensor]

            if var_name not in grid.columns:
                ax.set_title(f"{sensor} — {var_name} (no data)")
                continue

            # Use the full filled_df time axis so that periods where this sensor
            # has no data (grid ends early) are still covered by other sensors or model.
            full_idx = filled_df.index
            own      = grid[var_name].reindex(full_idx)
            imputed  = filled_df[var_name]

            # Priority 2: fill own NaNs with values from other sensors
            from_other = pd.Series(np.nan, index=full_idx, dtype="float32")
            for other_sensor, other_grid in grids.items():
                if other_sensor == sensor or var_name not in other_grid.columns:
                    continue
                other_vals = other_grid[var_name].reindex(full_idx)
                still_nan  = own.isna() & from_other.isna()
                from_other[still_nan] = other_vals[still_nan]

            # Priority 3: remaining NaNs get model values
            from_model = pd.Series(np.nan, index=full_idx, dtype="float32")
            still_nan  = own.isna() & from_other.isna()
            from_model[still_nan] = imputed[still_nan]

            ax.scatter(full_idx, own,        s=1, lw=0.7, color="steelblue", label="This sensor")
            ax.scatter(full_idx, from_other, s=1, lw=0.7, color="green",     label="Other sensor",  alpha=0.85)
            ax.scatter(full_idx, from_model, s=1, lw=0.8, color="orange",    label="Model imputed", alpha=0.9)

            ax.set_ylabel(var_name)
            ax.set_title(f"{sensor} — {var_name}")
            ax.legend(loc="upper right", fontsize="small")
            ax.set_xlim(full_idx.min(), full_idx.max())

        axes[-1].set_xlabel("Time (UTC)")
        fig.savefig(output_dir / f"{var_name}_imputed.png", dpi=150)
        plt.close(fig)

    print(f"[INFO] Plots saved to {output_dir}")



# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    args      = parse_args()
    out_dir   = Path(args.out)
    valid_qc  = frozenset(args.valid_qc)
    out_dir.mkdir(parents=True, exist_ok=True)
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load model (if checkpoint exists) ---
    ckpt      = args.ckpt
    use_model = ckpt is not None and Path(ckpt).exists()

    if use_model:
        model, mu, std, cfg = load_model(Path(ckpt))
        model.to(device)
        seq_len = cfg.get("seq_len", 48)
    else:
        seq_len = None

    # --- Resolve time range ---
    dl_start, dl_end, out_start, out_end = resolve_time_range(args, seq_len)
    print(f"[INFO] Download range : {dl_start}  →  {dl_end}")
    if args.time is not None:
        print(f"[INFO] Output range   : {out_start}  →  {out_end}")

    # --- Build file paths ---
    # Raw file uses the full download range; filled file uses the output range.
    dataset_name = "OBSEA_CTD_30min"
    dl_start_str  = f"{dl_start.year}{dl_start.month:02d}{dl_start.day:02d}"
    dl_end_str    = f"{dl_end.year}{dl_end.month:02d}{dl_end.day:02d}"
    out_start_str = f"{out_start.year}{out_start.month:02d}{out_start.day:02d}"
    out_end_str   = f"{out_end.year}{out_end.month:02d}{out_end.day:02d}"
    nc_file     = str(out_dir / f"{dataset_name}_{dl_start_str}_{dl_end_str}_raw.nc")
    filled_path = out_dir / f"{dataset_name}_{out_start_str}_{out_end_str}.nc"

    # --- Download raw data from ERDDAP ---
    url = build_erddap_url(dataset_name, dl_start, dl_end)
    print("[INFO] Downloading NetCDF file")
    status = download_erddap_nc(url=url, filename=nc_file)

    if status == 0:
        print("[INFO] Download successful\n")
    elif status == 4:
        # No sensor data from ERDDAP — run model imputation on a synthetic all-NaN
        # series so the model can fill from climatology, then write the result.
        print("[WARNING] No sensor data. Running model-only imputation from climatology.\n")
        out_idx = pd.date_range(pd.Timestamp(out_start, tz="UTC"),
                                pd.Timestamp(out_end,   tz="UTC"), freq="30min")
        synthetic = pd.DataFrame(
            {var: pd.Series(np.nan, index=out_idx, dtype="float32") for var in VARIABLES}
        )
        if use_model:
            clim_path = Path(ckpt).parent / "cmems_climatology.json"
            if not clim_path.exists():
                raise FileNotFoundError(f"Climatology not found: {clim_path}")
            clim = load_climatology(clim_path)
            filled_df, _ = run_neural_imputation(synthetic, model, mu, std, clim,
                                                  seq_len, device)
        else:
            filled_df = run_linear_imputation(synthetic)  # stays NaN — no data to interpolate
        write_empty_nc(filled_path, out_start, out_end, filled_df=filled_df)
        
        if args.plot:
            plots_dir = out_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            plot_imputed_vs_observed(None, filled_df, plots_dir)

            plot_xlim = (pd.Timestamp(out_start, tz="UTC"),
                         pd.Timestamp(out_end,   tz="UTC")) if args.time is not None else None
            try:
                plot_whole_range(str(filled_path), plots_dir, suffix="_filled",
                                 merge_sensors=(args.save_mode == "one-timestamp"),
                                 xlim=plot_xlim)
            except ValueError as e:
                print(f"[WARNING] Plotting error: {e}")
        return
    else:
        print(f"[ERROR] Download failed with status {status}")
        return

    # --- Gap filling ---
    if use_model:
        print(f"[INFO] Loading {Path(nc_file).name} …")
        df     = load_sensor_nc(nc_file)
        merged = build_merged_series(df)  # always uses VALID_QC={1,7} to match training

        n_missing = merged[list(VARIABLES)].isna().any(axis=1).sum()
        print(f"[INFO] Merged series: {len(merged)} timesteps, "
              f"{n_missing} with at least one gap ({100*n_missing/len(merged):.1f}%)")

        clim_path = Path(ckpt).parent / "cmems_climatology.json"
        if clim_path.exists():
            print(f"[INFO] Loading CMEMS climatology from {clim_path} …")
            clim = load_climatology(clim_path)
        else:
            raise FileNotFoundError(f"Climatology not found: {clim_path}")

        filled_df, _ = run_neural_imputation(
            merged, model, mu, std, clim, seq_len, device,
        )

        # Trim to output window (day 00:00 → args.time) when --time is used
        if args.time is not None:
            filled_df = filled_df.loc[
                pd.Timestamp(out_start, tz="UTC") : pd.Timestamp(out_end, tz="UTC")
            ]

        remaining = filled_df[list(VARIABLES)].isna().any(axis=1).sum()
        if remaining:
            print(f"[WARNING] {remaining} timesteps still have NaN after imputation.")
        else:
            print("[INFO] All gaps filled successfully.")

    else:
        # Linear interpolation fallback: gaps in the merged series are filled by
        # linear interpolation. The 3-priority per-sensor strategy (own → other → model)
        # is applied afterwards in build_sensor_filled_series, same as the model path.
        print("[INFO] No model checkpoint — using linear interpolation")
        df     = load_sensor_nc(nc_file)
        merged = build_merged_series(df)  # always uses VALID_QC={1,7} to match training

        n_missing = merged[list(VARIABLES)].isna().any(axis=1).sum()
        print(f"[INFO] Merged series: {len(merged)} timesteps, "
              f"{n_missing} with at least one gap ({100*n_missing/len(merged):.1f}%)")

        filled_df = run_linear_imputation(merged)

        # Trim to output window (day 00:00 → args.time) when --time is used
        if args.time is not None:
            filled_df = filled_df.loc[
                pd.Timestamp(out_start, tz="UTC") : pd.Timestamp(out_end, tz="UTC")
            ]

    sensor_series = build_sensor_filled_series(df, filled_df, valid_qc=valid_qc)

    if args.save_mode == "one-timestamp":
        write_imputed_nc_mode_a(nc_file, sensor_series, filled_path, valid_qc=valid_qc)
    else:
        write_imputed_nc_mode_b(nc_file, sensor_series, filled_path, valid_qc=valid_qc)


    # --- Optional QC plots ---
    if args.plot:
        plots_dir = out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_imputed_vs_observed(df, filled_df, plots_dir, valid_qc=valid_qc)

        merge = (args.save_mode == "one-timestamp")
        # When --time is used, fix x-axis to the output window so the context
        # window downloaded for model inference doesn't inflate the plot range.
        plot_xlim = (pd.Timestamp(out_start, tz="UTC"),
                     pd.Timestamp(out_end,   tz="UTC")) if args.time is not None else None
        try:
            plot_whole_range(str(filled_path), plots_dir, suffix="_filled", merge_sensors=merge, xlim=plot_xlim)
        except ValueError as e:
            print(f"[WARNING] Plotting error: {e}")

    # --- Cleanup temporary files ---
    if not args.keep:
        print("\n[INFO] Removing temporary files")
        for pattern in ("*_raw.nc", "*_filled.nc"):
            for path in out_dir.glob(pattern):
                if path.is_file():
                    path.unlink()
                    print(f"[INFO] Removed: {path.name}")


if __name__ == "__main__":
    main()
