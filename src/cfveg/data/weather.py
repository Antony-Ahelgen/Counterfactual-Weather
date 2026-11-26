"""Daily weather from the Open-Meteo historical archive (key-free), model `era5_seamless`.

era5_seamless = ERA5-Land (0.1°) where available (2 m temperature, dewpoint, VPD, soil moisture),
ERA5 (0.25°) otherwise (precipitation, shortwave radiation, ET0 are null in Open-Meteo's
era5_land daily product, so they come from ERA5). Data licence: CC BY 4.0 (Open-Meteo),
underlying data © ECMWF / Copernicus Climate Change Service.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARS = [
    "precipitation_sum",
    "temperature_2m_mean",
    "dew_point_2m_mean",
    "vapour_pressure_deficit_max",
    "shortwave_radiation_sum",
    "soil_moisture_0_to_7cm_mean",
    "et0_fao_evapotranspiration",
]
SHORT = {"precipitation_sum": "precip", "temperature_2m_mean": "t2m", "dew_point_2m_mean": "d2m",
         "vapour_pressure_deficit_max": "vpd", "shortwave_radiation_sum": "srad",
         "soil_moisture_0_to_7cm_mean": "sm", "et0_fao_evapotranspiration": "et0"}
WEATHER_VARS = [SHORT[v] for v in DAILY_VARS]


def fetch_batch(lons: list[float], lats: list[float], start: str, end: str, model: str,
                tries: int = 8) -> list[pd.DataFrame]:
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "start_date": start, "end_date": end, "daily": ",".join(DAILY_VARS),
        "models": model, "timezone": "GMT",
    }
    for k in range(tries):
        try:
            r = requests.get(URL, params=params, timeout=120)
            if r.status_code == 429 or r.status_code >= 500:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            r.raise_for_status()
            js = r.json()
            js = js if isinstance(js, list) else [js]
            out = []
            for j in js:
                df = pd.DataFrame(j["daily"]).rename(columns=SHORT)
                df["time"] = pd.to_datetime(df["time"])
                df.attrs["grid_lat"], df.attrs["grid_lon"] = j["latitude"], j["longitude"]
                out.append(df.set_index("time"))
            return out
        except Exception as e:  # noqa: BLE001
            wait = min(30 * 2**k, 900)
            print(f"  weather retry {k + 1}/{tries}: {str(e)[:160]} (sleep {wait}s)", flush=True)
            time.sleep(wait)
    raise RuntimeError("weather download failed after retries")


def download_weather(locs: pd.DataFrame, out_dir: Path, start: str, end: str, model: str,
                     batch: int = 5, pause: float = 5.0) -> None:
    """Idempotent: writes out_dir/<loc_id>.csv; skips existing files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in locs.itertuples() if not (out_dir / f"{r.loc_id}.csv").exists()]
    print(f"weather: {len(todo)} locations to fetch", flush=True)
    for i in range(0, len(todo), batch):
        chunk = todo[i:i + batch]
        dfs = fetch_batch([r.lon for r in chunk], [r.lat for r in chunk], start, end, model)
        for r, df in zip(chunk, dfs):
            df.to_csv(out_dir / f"{r.loc_id}.csv")
            meta = {"loc_id": r.loc_id, "grid_lat": df.attrs["grid_lat"], "grid_lon": df.attrs["grid_lon"],
                    "model": model, "start": start, "end": end, "vars": DAILY_VARS}
            (out_dir / f"{r.loc_id}.json").write_text(json.dumps(meta))
        print(f"  weather {min(i + batch, len(todo))}/{len(todo)}", flush=True)
        time.sleep(pause)


def load_weather(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["time"], index_col="time")
    # small gaps -> linear interpolation (documented); precipitation gaps -> 0 is NOT used
    return df[WEATHER_VARS].astype(np.float32).interpolate(limit=3).ffill().bfill()
