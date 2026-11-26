"""Splits, train-only normalisation statistics, sample windows and the PyTorch dataset.

Split rules (spatial AND temporal):
  train = train locations x train years
  val   = val locations   x val years
  test  = test locations  x test years
Fitted objects (normalisation, rate bounds, phenology envelopes, matching pools) use `train` only.
Per-location weather climatology / analog pools use training/climatology years only (see stress.py,
scenarios.py). `test_hist` (test locations x non-test years) is used ONLY as observed reference
data in the cross-year proxy validation, never for fitting.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

NDVI_CENTER = 0.4
NDVI_HALF_RANGE = 0.5  # NDVI in [-0.1, 0.9] maps to [-1, 1]
N_LC = 4


def ndvi_to_scaled(ndvi):
    return (ndvi - NDVI_CENTER) / NDVI_HALF_RANGE


def scaled_to_ndvi(x):
    return x * NDVI_HALF_RANGE + NDVI_CENTER


@dataclass
class Cube:
    ndvi: np.ndarray     # [L, Y, B, P, P] float16 (NaN invalid)
    static: np.ndarray   # [L, 2, P, P]
    lcg: np.ndarray      # [L, P, P]
    wx: np.ndarray       # [L, Y, B, F]
    s_daily: np.ndarray  # [L, Y, B*bin_days]
    meta: dict

    @classmethod
    def load(cls, path: str | Path, mmap: bool = True) -> "Cube":
        p = Path(path)
        mode = "r" if mmap else None
        return cls(np.load(p / "ndvi.npy", mmap_mode=mode), np.load(p / "static.npy"), np.load(p / "lcg.npy"),
                   np.load(p / "wx.npy"), np.load(p / "s_daily.npy"), json.loads((p / "meta.json").read_text()))

    @property
    def years(self) -> list[int]:
        return list(self.meta["years"])

    @property
    def split(self) -> np.ndarray:
        return np.array(self.meta["split"])

    @property
    def bin_doy(self) -> np.ndarray:
        return np.array(self.meta["bin_doy"], dtype=np.float32)

    def loc_year_mask(self, name: str, cfg: dict) -> np.ndarray:
        """Boolean [L, Y] mask of (location, year) pairs in a split."""
        s = cfg["splits"]
        yrs = np.array(self.years)
        if name == "train":
            lm, ym = self.split == "train", np.isin(yrs, s["train_years"])
        elif name == "val":
            lm, ym = self.split == "val", np.isin(yrs, s["val_years"])
        elif name == "test":
            lm, ym = self.split == "test", np.isin(yrs, s["test_years"])
        elif name == "test_hist":
            lm, ym = self.split == "test", ~np.isin(yrs, s["test_years"])
        elif name == "val_hist":
            lm, ym = self.split == "val", ~np.isin(yrs, s["val_years"] + s["test_years"])
        else:
            raise ValueError(name)
        return lm[:, None] & ym[None, :]


def fit_norm_stats(cube: Cube, cfg: dict) -> dict:
    """Weather and static normalisation statistics from the TRAIN split only."""
    m = cube.loc_year_mask("train", cfg)
    wx = cube.wx[m]                             # [n, B, F]
    wx = wx.reshape(-1, wx.shape[-1]).astype(np.float64)
    lm = m.any(1)
    st = cube.static[lm].astype(np.float64)     # [n, 2, P, P]
    return {
        "wx_mean": np.nanmean(wx, 0).tolist(), "wx_std": (np.nanstd(wx, 0) + 1e-6).tolist(),
        "static_mean": np.nanmean(st, axis=(0, 2, 3)).tolist(),
        "static_std": (np.nanstd(st, axis=(0, 2, 3)) + 1e-6).tolist(),
        "fitted_on": "train locations x train years",
    }


def make_windows(cube: Cube, cfg: dict, split: str, stride: int | None = None,
                 quadrants: bool = True, filter_valid: bool = True) -> np.ndarray:
    """Return int array [N, 5] of (loc, year, t0, oy, ox) windows for a split."""
    w = cfg["window"]
    K, H, P = w["context"], w["horizon"], w["patch"]
    stride = stride or w["stride"]
    full = cube.ndvi.shape[-1]
    offs = [(0, 0)] if P >= full or not quadrants else [(oy, ox) for oy in range(0, full - P + 1, P)
                                                        for ox in range(0, full - P + 1, P)]
    nb = cube.ndvi.shape[2]
    m = cube.loc_year_mask(split, cfg)
    cov = np.isfinite(cube.ndvi).mean(axis=(-2, -1)) if filter_valid else None  # [L, Y, B] (whole patch)
    out = []
    for l, y in zip(*np.nonzero(m)):
        for t0 in range(0, nb - K - H + 1, stride):
            if filter_valid:
                if cov[l, y, t0:t0 + K].mean() < w["min_context_valid"]:
                    continue
                if cov[l, y, t0 + K:t0 + K + H].mean() < w["min_target_valid"]:
                    continue
                if not np.isfinite(cube.ndvi[l, y, t0 + K - 3:t0 + K]).any():
                    continue
            for oy, ox in offs:
                out.append((l, y, t0, oy, ox))
    return np.array(out, dtype=np.int64).reshape(-1, 5)


def fill_context(ctx: np.ndarray) -> np.ndarray:
    """Forward-fill invalid context pixels in time, then backward-fill; remaining -> NaN."""
    out = ctx.copy()
    for t in range(1, len(out)):
        bad = ~np.isfinite(out[t])
        out[t][bad] = out[t - 1][bad]
    for t in range(len(out) - 2, -1, -1):
        bad = ~np.isfinite(out[t])
        out[t][bad] = out[t + 1][bad]
    return out


NEVER_OBSERVED = 99.0


def last_gap(ctx_mask: np.ndarray) -> np.ndarray:
    """Per pixel: number of 5-day steps from the last valid context frame to the first target frame."""
    K = ctx_mask.shape[0]
    gap = np.full(ctx_mask.shape[1:], NEVER_OBSERVED, dtype=np.float32)
    for t in range(K):
        gap[ctx_mask[t]] = K - t
    return gap


def horizon_stress(cube: Cube, l: int, y: int, t0: int, K: int, H: int) -> np.ndarray:
    bd = cube.meta["bin_days"]
    return cube.s_daily[l, y, (t0 + K) * bd:(t0 + K + H) * bd]


class WindowDataset(torch.utils.data.Dataset):
    """Yields dict samples. `wx_override` lets callers replace the weather sequence (scenarios)."""

    def __init__(self, cube: Cube, cfg: dict, windows: np.ndarray, stats: dict):
        self.cube, self.cfg, self.win = cube, cfg, windows
        w = cfg["window"]
        self.K, self.H, self.P = w["context"], w["horizon"], w["patch"]
        self.wx_mean = np.array(stats["wx_mean"], dtype=np.float32)
        self.wx_std = np.array(stats["wx_std"], dtype=np.float32)
        self.st_mean = np.array(stats["static_mean"], dtype=np.float32)[:, None, None]
        self.st_std = np.array(stats["static_std"], dtype=np.float32)[:, None, None]
        self.thr = cfg["drought"]["threshold"]
        self.min_days = cfg["drought"]["min_stressed_days"]

    def __len__(self) -> int:
        return len(self.win)

    def norm_wx(self, wx: np.ndarray) -> np.ndarray:
        return (wx - self.wx_mean) / self.wx_std

    def raw_weather(self, i: int) -> np.ndarray:
        l, y, t0, _, _ = self.win[i]
        return self.cube.wx[l, y, t0:t0 + self.K + self.H].astype(np.float32)

    def __getitem__(self, i: int) -> dict:
        l, y, t0, oy, ox = (int(v) for v in self.win[i])
        K, H, P = self.K, self.H, self.P
        sl = (slice(oy, oy + P), slice(ox, ox + P))
        seq = np.asarray(self.cube.ndvi[l, y, t0:t0 + K + H, sl[0], sl[1]], dtype=np.float32)
        ctx_raw, tgt_raw = seq[:K], seq[K:]
        ctx_mask = np.isfinite(ctx_raw)
        ctx_f = fill_context(ctx_raw)
        filled_mask = np.isfinite(ctx_f)
        ctx_f = np.where(filled_mask, ctx_f, NDVI_CENTER)
        tgt_mask = np.isfinite(tgt_raw)
        tgt = np.where(tgt_mask, tgt_raw, NDVI_CENTER)
        st = (self.cube.static[l][:, sl[0], sl[1]] - self.st_mean) / self.st_std
        lcg = self.cube.lcg[l][sl].astype(np.int64)
        onehot = np.eye(N_LC, dtype=np.float32)[lcg].transpose(2, 0, 1)
        wx = self.raw_weather(i)
        sh = horizon_stress(self.cube, l, y, t0, K, H)
        doy = self.cube.bin_doy[t0:t0 + K + H]
        return {
            "ctx": torch.from_numpy(ndvi_to_scaled(ctx_f)).float(),
            "ctx_mask": torch.from_numpy(ctx_mask.astype(np.float32)),
            "tgt": torch.from_numpy(ndvi_to_scaled(tgt)).float(),
            "tgt_mask": torch.from_numpy(tgt_mask.astype(np.float32)),
            "static": torch.from_numpy(np.concatenate([st, onehot]).astype(np.float32)),
            "lcg": torch.from_numpy(lcg),
            "wx": torch.from_numpy(self.norm_wx(wx)).float(),
            "wx_raw": torch.from_numpy(wx),
            "doy": torch.from_numpy(doy.astype(np.float32)),
            # scalar stress = mean bin-level s over the horizon (recomputable for swapped scenarios)
            "stress": torch.tensor(float(wx[K:, -1].mean()), dtype=torch.float32),
            "drought": torch.tensor(float((sh >= self.thr).sum() >= self.min_days)),
            "last_ndvi": torch.from_numpy(ctx_f[-1].astype(np.float32)),
            # bins between the last VALID context observation and the first target frame (1 = last bin valid);
            # never-observed pixels get NEVER_OBSERVED so rate constraints on the first step do not bind
            "last_gap": torch.from_numpy(last_gap(ctx_mask)),
            "idx": torch.tensor([l, y, t0, oy, ox]),
        }
