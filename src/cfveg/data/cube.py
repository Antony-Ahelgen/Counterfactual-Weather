"""Assemble per-location raw downloads into one analysis-ready cube (NumPy arrays on disk).

Layout of data/processed/cube/:
  ndvi.npy    float16 [L, Y, B, P, P]  NaN = invalid (cloud/shadow/snow/no-data)
  static.npy  float32 [L, 2, P, P]     elevation (m), slope (deg)
  lcg.npy     int8    [L, P, P]        land-cover group (0 forest, 1 grass/shrub, 2 cropland, 3 other)
  wx.npy      float32 [L, Y, B, 11]    per-bin weather (6 raw aggregates, 4 anomalies, s)
  s_daily.npy float32 [L, Y, B*5]      daily stress index
  meta.json   loc ids, lon, lat, split, years, bin doys, feature names
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .sentinel2 import bin_doys
from .static import lc_group
from .stress import ANOM_NAMES, RAW_BIN_VARS, bin_weather, daily_anomalies
from .weather import load_weather

WX_NAMES = RAW_BIN_VARS + ANOM_NAMES + ["s"]


class CoverageError(RuntimeError):
    pass


def coverage(ndvi: np.ndarray) -> np.ndarray:
    """Fraction of valid pixels per time step; ndvi [..., T, H, W]."""
    return np.isfinite(ndvi).mean(axis=(-2, -1))


def check_coverage(ndvi_loc: np.ndarray, min_frac: float = 0.15, loc_id: str = "") -> float:
    """Raise CoverageError if a location is mostly cloud over the whole record."""
    c = float(np.isfinite(ndvi_loc).mean())
    if c < min_frac:
        raise CoverageError(f"location {loc_id}: only {c:.1%} valid pixels (< {min_frac:.0%})")
    return c


def climatology_years(cfg: dict) -> list[int]:
    y0, y1 = cfg["data"]["weather_years"]
    held = set(cfg["splits"]["val_years"]) | set(cfg["splits"]["test_years"])
    return [y for y in range(y0, y1 + 1) if y not in held]


def build_cube(cfg: dict, root: Path, min_loc_valid: float = 0.15) -> dict:
    d = cfg["data"]
    raw = root / d["raw_dir"]
    out = root / d["processed_dir"] / "cube"
    out.mkdir(parents=True, exist_ok=True)
    locs = pd.read_csv(root / d.get("locations_file", "manifests/locations.csv"))
    years = d["years"]
    clim_years = climatology_years(cfg)
    ndvi_l, static_l, lcg_l, wx_l, sd_l, keep, dropped = [], [], [], [], [], [], []
    for r in locs.itertuples():
        f = raw / "s2" / f"{r.loc_id}.npz"
        if not f.exists():
            dropped.append((r.loc_id, "missing s2"))
            continue
        z = np.load(f)
        nd = z["ndvi"].astype(np.float32)
        try:
            check_coverage(nd, min_loc_valid, r.loc_id)
        except CoverageError as e:
            dropped.append((r.loc_id, str(e)))
            continue
        st = np.load(raw / "static" / f"{r.loc_id}.npz")
        wdf = load_weather(raw / "weather" / f"{r.loc_id}.csv")
        anom = daily_anomalies(wdf, clim_years, cfg["drought"]["weights"], cfg["drought"]["stress_window_days"])
        wx, sd = bin_weather(wdf, anom, years, d["doy_start"], d["bin_days"], d["n_bins"])
        ndvi_l.append(nd.astype(np.float16))
        static_l.append(np.stack([st["elev"], st["slope"]]).astype(np.float32))
        lcg_l.append(lc_group(st["lc"]))
        wx_l.append(wx)
        sd_l.append(sd)
        keep.append(r)
    kept = pd.DataFrame(keep).drop(columns=["Index"], errors="ignore")
    np.save(out / "ndvi.npy", np.stack(ndvi_l))
    np.save(out / "static.npy", np.stack(static_l))
    np.save(out / "lcg.npy", np.stack(lcg_l))
    np.save(out / "wx.npy", np.stack(wx_l))
    np.save(out / "s_daily.npy", np.stack(sd_l))
    meta = {
        "loc_id": kept.loc_id.tolist(), "lon": kept.lon.tolist(), "lat": kept.lat.tolist(),
        "split": kept.split.tolist(), "years": years, "climatology_years": clim_years,
        "bin_doy": bin_doys(d["doy_start"], d["bin_days"], d["n_bins"]).tolist(),
        "bin_days": d["bin_days"], "wx_names": WX_NAMES, "dropped": dropped,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta
