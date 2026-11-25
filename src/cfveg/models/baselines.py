"""Non-learned baselines: persistence and climatology."""
from __future__ import annotations

import warnings

import numpy as np
import torch

from ..data.dataset import ndvi_to_scaled


def persistence(batch: dict, H: int) -> torch.Tensor:
    """Last observed (forward-filled) context frame repeated over the horizon (scaled units)."""
    return ndvi_to_scaled(batch["last_ndvi"])[:, None].repeat(1, H, 1, 1)


def fit_pixel_climatology(cube, cfg: dict, phys_fit: dict) -> np.ndarray:
    """Per-pixel, per-bin mean NDVI over TRAINING YEARS at each location -> [L, B, P, P].

    Uses only years in splits.train_years (never val/test years). Pixels/bins never observed fall
    back to the land-cover-group phenology mean fitted on the train split.
    """
    yrs = np.array(cube.years)
    ysel = np.isin(yrs, cfg["splits"]["train_years"])
    L, _, B, P, _ = cube.ndvi.shape
    mu = np.array(phys_fit["mu"], dtype=np.float32)          # [G, B]
    out = np.zeros((L, B, P, P), dtype=np.float32)
    for l in range(L):
        v = np.asarray(cube.ndvi[l][ysel], dtype=np.float32)  # [y, B, P, P]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            c = np.nanmean(v, axis=0)
        fb = mu[cube.lcg[l]].transpose(2, 0, 1)               # [B, P, P]
        out[l] = np.where(np.isfinite(c), c, fb)
    return out


def climatology(batch: dict, clim: np.ndarray, K: int, H: int) -> torch.Tensor:
    idx = batch["idx"].numpy()
    P = batch["tgt"].shape[-1]
    out = np.stack([clim[l, t0 + K:t0 + K + H, oy:oy + P, ox:ox + P] for l, y, t0, oy, ox in idx])
    return ndvi_to_scaled(torch.from_numpy(out))


def climatology_anomaly_persistence(batch: dict, clim: np.ndarray, K: int, H: int) -> torch.Tensor:
    """Climatology shifted by the last observed anomaly (a stronger simple baseline)."""
    idx = batch["idx"].numpy()
    P = batch["tgt"].shape[-1]
    last = batch["last_ndvi"].numpy()
    out = []
    for j, (l, y, t0, oy, ox) in enumerate(idx):
        c = clim[l, :, oy:oy + P, ox:ox + P]
        anom = last[j] - c[t0 + K - 1]
        out.append(c[t0 + K:t0 + K + H] + anom[None])
    return ndvi_to_scaled(torch.from_numpy(np.stack(out)))
