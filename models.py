"""
src/models.py — Imputation model architectures.

All models share the same interface:
    forward(x, obs_mask, train_mask, clim, time_feat) -> pred

where:
    x         : [B, T, V]  normalised anomaly (residual w.r.t. clim)
    obs_mask  : [B, T, V]  1 = truly observed
    train_mask: [B, T, V]  1 = artificially masked (training only)
    clim      : [B, T, V]  normalised climatology baseline
    time_feat : [B, T, 4]  cyclic time encodings

Output is the full [B, T, V] reconstruction in normalised units.
The loss is computed only at train_mask positions against the target.

Architecture summary
--------------------
GRU       : Bidirectional GRU + linear head.  Best overall for short-to-
            medium gaps in smooth oceanographic signals.
SAITS     : Self-Attention Imputation (Transformer encoder).  Stronger
            for longer sequences with complex periodicity.
BRITS     : Bidirectional RNN Imputation with temporal smoothness.
            Adds a forward/backward consistency loss.
ResGRU    : GRU + residual MLP blocks.  Extra depth without extra recurrence.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_VARS    = 5   # CNDC, PSAL, TEMP, SVEL, PRES
T_FEAT    = 4   # sin/cos DOY + sin/cos HOD


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _build_input(x: "torch.Tensor",
                 clim: "torch.Tensor") -> "torch.Tensor":
    """
    Concatenate the anomaly signal with the climatology baseline.

    x    : [B, T, V] — anomaly (x_input - clim), already prepared in dataset
    clim : [B, T, V]

    We pass both the anomaly AND the raw clim so the model can learn the
    absolute scale without re-deriving it.
    → [B, T, 2V]
    """
    return torch.cat([x, clim], dim=-1)


if HAS_TORCH:

    # ------------------------------------------------------------------ #
    # GRU Imputer
    # ------------------------------------------------------------------ #

    class GRUImputer(nn.Module):
        """
        Bidirectional GRU for time-series imputation.

        The model sees: [anomaly | clim | obs_mask | time_feat]
        and outputs normalised reconstructions for every timestep.

        The output head adds the prediction back to the clim baseline so
        the model learns pure residuals (easier for smooth oceanographic data).
        """

        def __init__(
            self,
            n_vars:     int = N_VARS,
            hidden_dim: int = 64,
            n_layers:   int = 2,
            dropout:    float = 0.1,
        ) -> None:
            super().__init__()
            self.n_vars     = n_vars
            self.hidden_dim = hidden_dim
            in_dim          = 2 * n_vars + n_vars + T_FEAT  # x|clim|obs_mask|t

            self.gru = nn.GRU(
                input_size   = in_dim,
                hidden_size  = hidden_dim,
                num_layers   = n_layers,
                batch_first  = True,
                bidirectional = True,
                dropout      = dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, n_vars),
            )
            self._init_weights()

        def _init_weights(self) -> None:
            for name, p in self.gru.named_parameters():
                if "weight_hh" in name:
                    nn.init.orthogonal_(p)
                elif "weight_ih" in name:
                    nn.init.xavier_uniform_(p)
                elif "bias" in name:
                    nn.init.zeros_(p)
            for m in self.head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.zeros_(m.bias)

        def forward(
            self,
            x:          "torch.Tensor",
            obs_mask:   "torch.Tensor",
            train_mask: "torch.Tensor",
            clim:       "torch.Tensor",
            time_feat:  "torch.Tensor",
        ) -> "torch.Tensor":
            inp  = torch.cat([x, clim, obs_mask, time_feat], dim=-1)
            out, _ = self.gru(inp)
            pred = self.head(out)
            return pred + clim   # residual output

    # ------------------------------------------------------------------ #
    # SAITS Imputer (Self-Attention)
    # ------------------------------------------------------------------ #

    class SAITSImputer(nn.Module):
        """
        Self-Attention Imputation Transformer (SAITS-style).

        Two stacked TransformerEncoder layers with learned positional
        encoding.  Better than GRU for longer sequences but needs more data.
        """

        def __init__(
            self,
            n_vars:     int = N_VARS,
            d_model:    int = 64,
            n_heads:    int = 4,
            n_layers:   int = 2,
            d_ff:       int = 128,
            dropout:    float = 0.1,
            max_len:    int = 512,
        ) -> None:
            super().__init__()
            self.n_vars = n_vars
            in_dim      = 2 * n_vars + n_vars + T_FEAT

            self.input_proj = nn.Linear(in_dim, d_model)
            self.pos_emb    = nn.Embedding(max_len, d_model)

            layer = nn.TransformerEncoderLayer(
                d_model       = d_model,
                nhead         = n_heads,
                dim_feedforward = d_ff,
                dropout       = dropout,
                batch_first   = True,
                norm_first    = True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
            self.head    = nn.Linear(d_model, n_vars)

        def forward(
            self,
            x:          "torch.Tensor",
            obs_mask:   "torch.Tensor",
            train_mask: "torch.Tensor",
            clim:       "torch.Tensor",
            time_feat:  "torch.Tensor",
        ) -> "torch.Tensor":
            B, T, _ = x.shape
            assert T <= self.pos_emb.num_embeddings, \
                f"seq_len {T} > max_len {self.pos_emb.num_embeddings}"

            inp  = torch.cat([x, clim, obs_mask, time_feat], dim=-1)
            h    = self.input_proj(inp)
            pos  = self.pos_emb(torch.arange(T, device=x.device)).unsqueeze(0)
            h    = self.encoder(h + pos)
            pred = self.head(h)
            return pred + clim

    # ------------------------------------------------------------------ #
    # BRITS Imputer
    # ------------------------------------------------------------------ #

    class _BRITSCell(nn.Module):
        """Single direction of BRITS: GRU with temporal decay."""

        def __init__(self, n_vars: int, hidden_dim: int) -> None:
            super().__init__()
            self.n_vars = n_vars
            self.gru    = nn.GRUCell(n_vars * 2 + T_FEAT, hidden_dim)
            self.head   = nn.Linear(hidden_dim, n_vars)
            self.decay  = nn.Linear(n_vars, n_vars)

        def forward(
            self,
            x_seq:    "torch.Tensor",    # [B, T, V]
            obs_seq:  "torch.Tensor",    # [B, T, V]
            tfeat:    "torch.Tensor",    # [B, T, 4]
            clim_seq: "torch.Tensor",    # [B, T, V]
        ) -> "torch.Tensor":
            B, T, V = x_seq.shape
            h       = torch.zeros(B, self.gru.hidden_size, device=x_seq.device)
            preds   = []
            for t in range(T):
                xt  = x_seq[:, t, :]
                om  = obs_seq[:, t, :]
                tf  = tfeat[:, t, :]
                cl  = clim_seq[:, t, :]
                inp = torch.cat([xt * om + cl * (1 - om), om, tf], dim=-1)
                h   = self.gru(inp, h)
                preds.append(self.head(h))
            return torch.stack(preds, dim=1)   # [B, T, V]


    class BRITSImputer(nn.Module):
        """
        Bidirectional RNN Imputation (BRITS-style).
        Combines forward and backward GRU predictions.
        Side effect: sets self.consistency_loss after each forward pass
        for use in the training loop.
        """

        def __init__(
            self,
            n_vars:     int = N_VARS,
            hidden_dim: int = 64,
        ) -> None:
            super().__init__()
            self.fwd = _BRITSCell(n_vars, hidden_dim)
            self.bwd = _BRITSCell(n_vars, hidden_dim)
            self.consistency_loss = torch.tensor(0.0)

        def forward(
            self,
            x:          "torch.Tensor",
            obs_mask:   "torch.Tensor",
            train_mask: "torch.Tensor",
            clim:       "torch.Tensor",
            time_feat:  "torch.Tensor",
        ) -> "torch.Tensor":
            pred_fwd = self.fwd(x, obs_mask, time_feat, clim)
            pred_bwd = self.bwd(
                x.flip(1), obs_mask.flip(1),
                time_feat.flip(1), clim.flip(1),
            ).flip(1)
            self.consistency_loss = F.mse_loss(pred_fwd, pred_bwd) * 0.1
            return (pred_fwd + pred_bwd) / 2 + clim

    # ------------------------------------------------------------------ #
    # ResGRU Imputer
    # ------------------------------------------------------------------ #

    class ResGRUImputer(nn.Module):
        """GRU backbone + residual MLP refinement blocks."""

        def __init__(
            self,
            n_vars:     int = N_VARS,
            hidden_dim: int = 64,
            n_res:      int = 2,
            dropout:    float = 0.1,
        ) -> None:
            super().__init__()
            in_dim   = 2 * n_vars + n_vars + T_FEAT
            self.gru = nn.GRU(in_dim, hidden_dim, batch_first=True,
                              bidirectional=True, num_layers=1)
            blocks   = []
            for _ in range(n_res):
                blocks += [
                    nn.Linear(2 * hidden_dim, 2 * hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            self.res_blocks = nn.Sequential(*blocks)
            self.head        = nn.Linear(2 * hidden_dim, n_vars)

        def forward(
            self,
            x:          "torch.Tensor",
            obs_mask:   "torch.Tensor",
            train_mask: "torch.Tensor",
            clim:       "torch.Tensor",
            time_feat:  "torch.Tensor",
        ) -> "torch.Tensor":
            inp      = torch.cat([x, clim, obs_mask, time_feat], dim=-1)
            h, _     = self.gru(inp)
            h        = h + self.res_blocks(h)
            pred     = self.head(h)
            return pred + clim


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {}
if HAS_TORCH:
    MODEL_REGISTRY = {
        "gru":    GRUImputer,
        "saits":  SAITSImputer,
        "brits":  BRITSImputer,
        "resgru": ResGRUImputer,
    }


def build_model(
    name:       str,
    n_vars:     int = N_VARS,
    hidden_dim: int = 64,
    **kwargs,
) -> "nn.Module":
    """
    Instantiate a model by name.

    Parameters
    ----------
    name       : one of "gru", "saits", "brits", "resgru"
    n_vars     : number of sensor variables (default 5)
    hidden_dim : hidden dimension (applies to all models)
    **kwargs   : additional model-specific arguments
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch required: pip install torch")
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. "
                         f"Choose from {list(MODEL_REGISTRY)}")
    # return MODEL_REGISTRY[name](n_vars=n_vars, hidden_dim=hidden_dim, **kwargs)
    return MODEL_REGISTRY[name](n_vars=n_vars, **kwargs)


def count_parameters(model: "nn.Module") -> int:
    """Return total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
