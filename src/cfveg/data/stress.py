"""Weather → 5-day bins, standardised anomalies and a moisture-stress index s.

All climatologies are per location and per day of year, fitted on *climatology years only*
(all downloaded weather years except validation and test years). Never on val/test years.

Stress index (daily):
  a_wb  = -z(30-day sum of P - ET0)   (water-balance deficit; SPEI-like but without log-logistic fit)
  a_t   =  z(30-day mean T2m)
  a_vpd =  z(30-day mean VPD)
  a_sm  = -z(30-day mean soil moisture 0-7 cm)
  s_raw = sum_i w_i a_i / sum_i w_i, then re-standardised per location/doy on climatology years
so s ≈ N(0,1) in the climatology period and s >= 1 is roughly a 1-sigma dry/hot anomaly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ANOM_NAMES = ["a_wb", "a_t", "a_vpd", "a_sm"]
RAW_BIN_VARS = ["precip", "t2m", "vpd", "srad", "sm", "et0"]   # aggregated per bin
SUM_VARS = {"precip", "srad", "et0"}


def _doy(idx: pd.DatetimeIndex) -> np.ndarray:
    d = idx.day_of_year.values
    return np.minimum(d, 365)  # fold Dec 31 of leap years


def doy_climatology(x: pd.Series, clim_mask: np.ndarray, half_window: int = 15) -> tuple[np.ndarray, np.ndarray]:
    """Per-doy mean/std of x using only rows where clim_mask is True, smoothed ±half_window days."""
    doy = _doy(x.index)
    v = x.values.astype(np.float64)
    mu = np.zeros(366)
    sd = np.zeros(366)
    for d in range(1, 366):
        dist = np.abs(((doy - d + 182) % 365) - 182)
        sel = clim_mask & (dist <= half_window) & np.isfinite(v)
        mu[d] = v[sel].mean()
        sd[d] = v[sel].std() + 1e-6
    mu[0], sd[0] = mu[1], sd[1]
    return mu, sd


def daily_anomalies(df: pd.DataFrame, clim_years: list[int], weights: dict[str, float],
                    window: int = 30) -> pd.DataFrame:
    """Return daily DataFrame with anomaly columns and stress index s."""
    clim_mask = np.isin(df.index.year, clim_years)
    roll = pd.DataFrame(index=df.index)
    roll["wb"] = (df["precip"] - df["et0"]).rolling(window, min_periods=window).sum()
    roll["t"] = df["t2m"].rolling(window, min_periods=window).mean()
    roll["vpd"] = df["vpd"].rolling(window, min_periods=window).mean()
    roll["sm"] = df["sm"].rolling(window, min_periods=window).mean()
    doy = _doy(df.index)
    out = pd.DataFrame(index=df.index)
    sign = {"wb": -1.0, "t": 1.0, "vpd": 1.0, "sm": -1.0}
    for k in ["wb", "t", "vpd", "sm"]:
        mu, sd = doy_climatology(roll[k], clim_mask)
        out[f"a_{k}"] = sign[k] * (roll[k].values - mu[doy]) / sd[doy]
    w = np.array([weights["precip_deficit"], weights["temp"], weights["vpd"], weights["soil_moisture"]])
    s_raw = (out[ANOM_NAMES].values * w).sum(1) / w.sum()
    s_raw = pd.Series(s_raw, index=df.index)
    mu, sd = doy_climatology(s_raw, clim_mask)
    out["s"] = (s_raw.values - mu[doy]) / sd[doy]
    return out


def bin_weather(df: pd.DataFrame, anom: pd.DataFrame, years: list[int], doy_start: int, bin_days: int,
                n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Aggregate to bins. Returns wx [Y, B, 6 raw + 4 anom + 1 s] and s_daily [Y, B*bin_days]."""
    feats = np.zeros((len(years), n_bins, len(RAW_BIN_VARS) + len(ANOM_NAMES) + 1), dtype=np.float32)
    s_daily = np.zeros((len(years), n_bins * bin_days), dtype=np.float32)
    for yi, y in enumerate(years):
        start = pd.Timestamp(y, 1, 1) + pd.Timedelta(days=doy_start - 1)
        idx = pd.date_range(start, periods=n_bins * bin_days, freq="D")
        d = df.reindex(idx)
        a = anom.reindex(idx)
        s_daily[yi] = a["s"].values
        for j, v in enumerate(RAW_BIN_VARS):
            arr = d[v].values.reshape(n_bins, bin_days)
            feats[yi, :, j] = arr.sum(1) if v in SUM_VARS else arr.mean(1)
        for j, v in enumerate(ANOM_NAMES + ["s"]):
            feats[yi, :, len(RAW_BIN_VARS) + j] = a[v].values.reshape(n_bins, bin_days).mean(1)
    return feats, s_daily


def stressed_days(s_daily_window: np.ndarray, threshold: float) -> int:
    return int((s_daily_window >= threshold).sum())


def is_drought(s_daily_window: np.ndarray, threshold: float, min_days: int) -> bool:
    """Drought label for a forecast horizon: >= min_days days with s >= threshold."""
    return stressed_days(s_daily_window, threshold) >= min_days
