"""
Author:  Sandra González-Villà (Coronis Computing S.L.)
Date:    29-01-2026
Version: 1.0.0

Description:
    This script processes an interval of ERDDAP data from OBSEA_CTD_30min
    dataset, interpolates missing TIME samples, and generates a new NetCDF
    file with filled missing values.

    Optionally, it also generates plots for the original and interpolated data.
    
Example usage:
    python get_filled_netcdf.py --start YYYYMMDD --end YYYYMMDD -o output_dir -p 
"""


from pathlib import Path
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentTypeError
import requests
from datetime import datetime, timezone


QC_DEFINITIONS = {
    0: ("unknown", "gray"),
    1: ("good_data", "green"),
    2: ("probably_good_data", "blue"),
    3: ("potentially_correctable_bad_data", "orange"),
    4: ("bad_data", "red"),
    7: ("nominal_value", "purple"),
    8: ("interpolated_value", "yellow"),
    9: ("missing_value", "black"),
}


# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------
def valid_date_yyyymmdd(s):
    """
    Validate date in YYYYMMDD format.
    """
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        raise ArgumentTypeError(
            f"Invalid date '{s}'. Expected format YYYYMMDD."
        )

def extract_numeric_vars_and_qc(ds):
    """
    Extract numeric data variables and their corresponding QC variables
    from an xarray.Dataset.

    Returns
    -------
    data_vars : list[str]
        Numeric variables excluding QC variables
    qc_vars : list[str]
        Variables that have QC
    """

    data_vars = []
    qc_vars = []

    for var, da in ds.data_vars.items():

        # Skip QC variables themselves
        if var.endswith("_QC"):
            continue

        # Keep only numeric dtypes
        if not np.issubdtype(da.dtype, np.number):
            continue

        data_vars.append(var)

        qc_name = f"{var}_QC"
        if qc_name in ds.data_vars:
            qc_vars.append(qc_name)

    return data_vars, qc_vars

def extend_time_to_full_range(time_orig, start_date, end_date, dt):
    """
    Extend a numeric TIME array (seconds since Unix epoch) so that it spans
    a user-defined continuous time range with fixed sampling interval.

    Missing timestamps are prepended and/or appended using the provided
    sampling step, while preserving the original sampling phase.

    Parameters
    ----------
    time_orig : np.ndarray
        Original TIME array (seconds since 1970-01-01T00:00:00Z),
        assumed to be regularly sampled.
    start_date : datetime.date or datetime.datetime
        Start date of the desired time range. The generated TIME
        sequence will begin at 00:00:00 UTC of this date.
    end_date : datetime.date or datetime.datetime
        End date of the desired time range. The generated TIME
        sequence will end at 23:59:59 UTC of this date.
    dt : int or float
        Sampling interval in seconds (e.g. 1800 for 30-minute data).

    Returns
    -------
    np.ndarray
        Extended TIME array covering the full requested temporal range
        with uniform sampling.
    """

    # Target temporal boundaries
    date_start = pd.Timestamp(f"{start_date.year}-{start_date.month}-{start_date.day} 00:00:00", tz="UTC")
    date_end = pd.Timestamp(f"{end_date.year}-{end_date.month}-{end_date.day} 23:59:59", tz="UTC")

    # Convert original TIME endpoints to datetime for boundary checks
    t0_dt = pd.to_datetime(time_orig[0], unit="s", origin="1970-01-01", utc=True)
    t1_dt = pd.to_datetime(time_orig[-1], unit="s", origin="1970-01-01", utc=True)


    # Prepend missing timestamps at the beginning
    prepend = []
    if t0_dt > date_start:
        t = time_orig[0]
        while True:
            t -= dt
            if pd.to_datetime(t, unit="s", origin="1970-01-01", utc=True) < date_start:
                break
            prepend.append(t)
        prepend = prepend[::-1]  # chronological order

    # Append missing timestamps at the end
    append = []
    if t1_dt < date_end:
        t = time_orig[-1]
        while True:
            t += dt
            if pd.to_datetime(t, unit="s", origin="1970-01-01", utc=True) > date_end:
                break
            append.append(t)

    # Assemble TIME vector
    time_full = np.concatenate([
        np.asarray(prepend, dtype="float64"),
        time_orig,
        np.asarray(append, dtype="float64"),
    ])

    return time_full

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
def download_erddap_nc(url, filename, timeout=300):
    """
    Download a NetCDF file from ERDDAP and save it to disk.

    Parameters
    ----------
    url : str
        ERDDAP download URL.
    filename : str or Path
        Full path of the output NetCDF file.
    timeout : int, optional
        Request timeout in seconds (default: 300).

    Returns
    -------
    int
        Status code:
        0 = success
        1 = request / HTTP error
        2 = file write error
        3 = empty or invalid file
    """

    filename = Path(filename)

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()

            bytes_written = 0
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        bytes_written += len(chunk)

        # Sanity check
        if bytes_written == 0 or not filename.exists():
            return 3

        return 0

    except requests.RequestException as e:
        print(f"Download failed: {e}")
        return 1

    except OSError as e:
        print(f"File write failed: {e}")
        return 2

def interpolate_nc_with_gaps_sensor(nc_file, start_date, end_date, sensorid, dt=1800):
    """
    Interpolate NetCDF variables along the TIME dimension to fill gaps
    and extend the dataset to a user-defined temporal range,
    for a single sensor only.

    Parameters
    ----------
    nc_file : str or Path
        Path to the input NetCDF file
    start_date : datetime.date or datetime.datetime
        Start date of desired time range (00:00:00 UTC)
    end_date : datetime.date or datetime.datetime
        End date of desired time range (23:59:59 UTC)
    sensorid : str
        Sensor ID to extract and interpolate
    dt : int
        Sampling interval in seconds (default: 1800)

    Returns
    -------
    xr.Dataset
        Gap-filled dataset with continuous TIME coordinate
        for the selected sensor
    """

    # Load and select sensor
    ds = xr.open_dataset(nc_file, decode_times=False)

    if "sensor_id" not in ds.data_vars:
        raise ValueError("NetCDF file has no 'sensor_id' variable")

    sensor_mask = ds["sensor_id"].values == sensorid
    if not sensor_mask.any():
        raise ValueError(f"Sensor '{sensorid}' not found")

    ds = ds.isel(row=sensor_mask)  # filter rows


    time_orig = ds["time"].values.astype("float64")
    ds.close()

    # Create full continuous time axis
    time_full = np.arange(time_orig[0], time_orig[-1]+dt, dt)
    time_full = extend_time_to_full_range(time_full, start_date, end_date, dt=dt)

    ds_filled = xr.Dataset()

    for var in ds.data_vars:
        data = ds[var].values
        dims = ds[var].dims
        
        if var == "time":
            new_data = time_full.copy()
        elif var == "sensor_id":
            new_data = np.array([sensorid] * len(time_full))
        elif var.endswith("_QC"):
            # Initialize all as interpolated (8)
            new_data = np.full(
                data.shape[:-1] + (len(time_full),),
                8,
                dtype=data.dtype
            )

            # Original time indices
            orig_time = ds["time"].values.astype(np.float64)
            orig_idx = np.searchsorted(time_full, orig_time)

            # Corresponding data variable
            data_var = var.replace("_QC", "")
            data_values = ds[data_var].values

            # Only keep QC for valid data
            for idx in np.ndindex(data_values.shape[:-1]):
                y = data_values[idx + (slice(None),)]
                qc_row = data[idx + (slice(None),)]

                valid = np.isfinite(y)
                new_data[idx + (orig_idx[valid],)] = qc_row[valid]

        else:
            # Interpolate numeric variable
            new_data = np.empty(data.shape[:-1] + (len(time_full),), dtype=data.dtype)
            it = np.ndindex(data.shape[:-1])
            for idx in it:

                x = ds["time"].values.astype(np.float64)
                y = data[idx + (slice(None),)]

                valid = np.isfinite(y)

                if valid.sum() == 0:
                    # no data at all: set fake value
                    new_data[idx + (slice(None),)] = -999999.0

                elif valid.sum() == 1:
                    # only one value: repeat it everywhere
                    new_data[idx + (slice(None),)] = y[valid][0]

                else:
                    x_valid = x[valid]
                    y_valid = y[valid]

                    new_data[idx + (slice(None),)] = np.interp(
                        time_full,
                        x_valid,
                        y_valid,
                        left=y_valid[0],    # replicate first value
                        right=y_valid[-1],  # replicate last value
                    )

        ds_filled[var] = (dims, new_data)
        ds_filled[var].attrs = ds[var].attrs.copy() # copy variable attributes
    
    # Copy global attributes
    ds_filled.attrs = ds.attrs.copy()

    # Update metadata
    min_date = pd.to_datetime(ds_filled["time"].min(), unit="s", origin="1970-01-01", utc=True)
    max_date = pd.to_datetime(ds_filled["time"].max(), unit="s", origin="1970-01-01", utc=True)

    ds_filled = ds_filled.assign_attrs({"time_coverage_start": min_date.strftime("%Y-%m-%dT%H:%M:%SZ")})
    ds_filled = ds_filled.assign_attrs({"time_coverage_end": max_date.strftime("%Y-%m-%dT%H:%M:%SZ")})
    ds_filled = ds_filled.assign_attrs({"date_modified": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    ds_filled = ds_filled.assign_attrs({"modified_by": "Coronis Computing gap filler"})
    
    return ds_filled

def interpolate_nc_with_gaps(nc_file, start_date, end_date, dt=1800):
    """
    Interpolate multi-sensor NetCDF along TIME, allowing multiple sensors.

    Parameters
    ----------
    nc_file : str or Path
        Path to input NetCDF file
    start_date, end_date : datetime.date or datetime.datetime
        Desired temporal range
    dt : int
        Sampling interval in seconds

    Returns
    -------
    xr.Dataset
        Dataset with gap-filled variables
    """

    ds = xr.open_dataset(nc_file, decode_times=False)
    sensors = list(np.unique(ds["sensor_id"].values))
    ds.close()

    all_ds = []
    for sensor in sensors:

        print(f'Processing sensor: {sensor}')
        ds_filled = interpolate_nc_with_gaps_sensor(nc_file, start_date, end_date, sensorid=sensor, dt=dt)
        all_ds.append(ds_filled)

    # Combine all sensors
    combined_ds = xr.concat(all_ds, dim="row")
    return combined_ds

def plot_whole_range(nc_file, outdir, suffix='', verbose=False):
    """
    Generate QC-colored time-series plots for all numeric variables
    and sensors in the NetCDF file.

    Parameters
    ----------
    nc_file : str or Path
        Path to the input NetCDF file. The dataset must include a 
        numeric TIME variable expressed as seconds since the Unix 
        epoch (1970-01-01T00:00:00Z).
    outdir : str or Path
        Output directory where the generated PNG plots will be saved.
        The directory is created if it does not already exist.
    suffix : str, optional
        Optional string appended to each output filename.
    verbose : bool, optional
        If True, print progress and diagnostic messages during plotting
        (default: False).

    Returns
    -------
    None
        One PNG file per plotted variable is written to `outdir`.
    """

    print(f'\nGenerating plots: {suffix}...')

    ds = xr.open_dataset(nc_file, decode_times=False)

    if "sensor_id" not in ds.data_vars:
        raise ValueError("NetCDF file has no 'sensor_id' variable")

    sensors = list(np.unique(ds["sensor_id"].values))
    data_vars, qc_vars = extract_numeric_vars_and_qc(ds)
    vars_to_keep = data_vars + qc_vars + ["time", "sensor_id"]
    ds = ds[vars_to_keep]
    ds.close()

    ds_original = ds.copy()
    
    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose: print(f"Found {len(data_vars)} variables with potential QC flags")

    for sensor in sensors:

        sensor_mask = ds_original["sensor_id"].values == sensor
        if not sensor_mask.any():
            raise ValueError(f"Sensor '{sensor}' not found")

        ds = ds_original.isel(row=sensor_mask)  # filter rows
        data_vars, qc_vars = extract_numeric_vars_and_qc(ds)

        time = ds["time"].values.astype("float64")
        datetime = pd.to_datetime(time, unit="s", utc=True)

        for var in data_vars:
            qc_var = f"{var}_QC"

            if qc_var not in qc_vars:
                if verbose: print(f"Skipping {var}: no QC column")
                continue

            if verbose: print(f"Plotting {var} with QC coloring")

            plt.figure(figsize=(10, 4))

            # Plot per-QC category
            for qc_value, (meaning, color) in QC_DEFINITIONS.items():
                mask = ds[qc_var].values == qc_value
                variable = ds[var].values
                if mask.any(): 
                    plt.scatter(
                        datetime[mask],
                        variable[mask],
                        s=1,
                        c=color,
                        label=meaning,
                        alpha=0.8
                    )

            plt.xlabel("Time (UTC)")
            plt.ylabel(var)
            plt.title(f"{sensor}: {var} vs Time (QC-colored)")
            plt.legend(
                title="QC flag",
                markerscale=2,
                fontsize="small",
                ncol=3
            )
            plt.tight_layout()

            output_file = output_dir / f"{sensor}_{var}_vs_time_QC{suffix}.png"
            plt.savefig(output_file, dpi=300)
            plt.close()

    if verbose: print(f"QC plots saved in: {output_dir}")


# -----------------------------------------------------------------------------
# Main function
# -----------------------------------------------------------------------------
if __name__ == "__main__":

    argparser = ArgumentParser(description='OBSEA_CTD_30min NetCDF file interpolator')
    argparser.add_argument("--start", type=valid_date_yyyymmdd, required=True, help="Start date to be processed (YYYYMMDD)") 
    argparser.add_argument("--end", type=valid_date_yyyymmdd, required=True, help="End date to be processed (YYYYMMDD)")
    argparser.add_argument("-o", "--output", type=str, required=True, help="Name of the output folder where the interpolated NetCDF file will be stored")
    argparser.add_argument("-p", "--plot", action="store_true", help="Plot results")
    argparser.add_argument("-k", "--keep", action="store_true", help="Keep raw data used for gap filling")
    argparser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = argparser.parse_args()

    start_date = f'{args.start.year}{args.start.month:02d}{args.start.day:02d}'
    end_date = f'{args.end.year}{args.end.month:02d}{args.end.day:02d}'
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)


    dataset_name = 'OBSEA_CTD_30min'
    nc_file = f'{out_dir}/{dataset_name}_{start_date}_{end_date}_raw.nc'

    url = (
        f"https://data.obsea.es/erddap/tabledap/{dataset_name}.nc"
        "?time%2Clatitude%2Clongitude%2Cdepth%2Csensor_id"
        "%2CCNDC%2CPSAL%2CTEMP%2CSVEL%2CPRES"
        "%2CCNDC_QC%2CPSAL_QC%2CTEMP_QC%2CSVEL_QC%2CPRES_QC"
        "%2Clatitude_QC%2Clongitude_QC%2Cdepth_QC"
        f"&time%3E={args.start.year}-{args.start.month}-{args.start.day}T00%3A00%3A00Z"
        f"&time%3C={args.end.year}-{args.end.month}-{args.end.day}T23%3A59%3A59Z"
    )

    # Download file
    print(f'\nDownloading NetCDF file...')
    status = download_erddap_nc(
        url=url,
        filename=nc_file,
    )

    if status == 0:
        print("Download successful")
    else:
        print(f"Download failed with status {status}")
        exit()


    # Interpolate values
    print(f'\nFilling gaps...')
    ds_filled = interpolate_nc_with_gaps(nc_file, args.start, args.end)
    filled_filename = out_dir / f'{dataset_name}_{start_date}_{end_date}.nc'
    ds_filled.to_netcdf(filled_filename)


    # Plot variables
    if args.plot:
        plots_dir = out_dir / ' plots'
        plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_whole_range(filled_filename, plots_dir, suffix='_filled', verbose=args.verbose)
            plot_whole_range(nc_file, plots_dir, suffix='_original', verbose=args.verbose)
        except ValueError as e:
            print(f"Plotting error: {e}")


    # Remove temporal data
    if not args.keep:
        print(f'\nRemoving temporal data...')
        path = Path(nc_file)
        if path.is_file():
            path.unlink()