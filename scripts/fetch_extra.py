"""Download EXTRA locations for later rounds (owner request 2026-11-26). Reserved data: not used by the v2 round.

Waits until the v2 fresh test-set download has finished, then samples new candidate points in the same region
(same rules: >= 0.3 deg from every existing location and each other, >= 60 % vegetated by WorldCover), keeps up to
80, and downloads static layers, weather and Sentinel-2 composites. IDs X0000...; manifest
manifests/locations_extra.csv (split = "extra"). Resumable: finished files are skipped.

  python scripts/fetch_extra.py
(The process name deliberately does not contain "download_data", so the v2 pipeline does not wait for it.)
"""
from __future__ import annotations

import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from download_data import _s2_job, _static_job  # noqa: E402

from cfveg.data.weather import download_weather  # noqa: E402
from cfveg.utils.provenance import load_preset  # noqa: E402

MAN = ROOT / "manifests" / "locations_extra.csv"
N_KEEP = 80


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def v2_test_done() -> bool:
    t = pd.read_csv(ROOT / "manifests" / "locations_v2.csv")
    return all((ROOT / "data" / "raw" / "s2" / f"{i}.npz").exists() for i in t[t.split == "test"].loc_id)


def sample_extra(cfg: dict) -> pd.DataFrame:
    d = cfg["data"]
    existing = pd.read_csv(ROOT / "manifests" / "location_candidates.csv")[["lon", "lat"]].values
    rng = np.random.default_rng(20261126)
    lo0, la0, lo1, la1 = d["region_bbox"]
    pts: list[tuple[float, float]] = []
    allp = [tuple(p) for p in existing]
    tries = 0
    while len(pts) < 260 and tries < 300_000:
        tries += 1
        lon, lat = rng.uniform(lo0, lo1), rng.uniform(la0, la1)
        ok = all(((lon - x) * np.cos(np.radians(lat))) ** 2 + (lat - y) ** 2 >= d["min_spacing_deg"] ** 2
                 for x, y in allp)
        if ok:
            pts.append((round(lon, 5), round(lat, 5)))
            allp.append((lon, lat))
    df = pd.DataFrame(pts, columns=["lon", "lat"])
    df.insert(0, "loc_id", [f"X{i:04d}" for i in range(len(df))])
    return df


def main() -> None:
    cfg = load_preset("cpu")
    log("waiting for the v2 fresh test-set download to finish")
    while not v2_test_done():
        time.sleep(120)
    raw = ROOT / cfg["data"]["raw_dir"]
    (raw / "static").mkdir(parents=True, exist_ok=True)
    if not MAN.exists():
        cand = sample_extra(cfg)
        log(f"{len(cand)} extra candidates; static layers")
        res = {}
        with ProcessPoolExecutor(4) as ex:
            futs = {ex.submit(_static_job, r.loc_id, r.lon, r.lat, cfg): r.loc_id for r in cand.itertuples()}
            for f in as_completed(futs):
                try:
                    res[futs[f]] = f.result()
                except Exception as e:  # noqa: BLE001
                    res[futs[f]] = {"loc_id": futs[f], "ok": False, "veg_frac": 0.0, "elev": np.nan}
                    log(f"static fail {futs[f]}: {str(e)[:80]}")
        cand = cand.merge(pd.DataFrame(res.values()), on="loc_id")
        keep = cand[cand.ok & (cand.veg_frac >= cfg["data"]["min_veg_fraction"])].head(N_KEEP).copy()
        keep["split"] = "extra"
        keep[["loc_id", "lon", "lat", "veg_frac", "elev", "split"]].to_csv(MAN, index=False)
        log(f"kept {len(keep)} extra locations")
    locs = pd.read_csv(MAN)
    y0, y1 = cfg["data"]["weather_years"]
    download_weather(locs, raw / "weather", f"{y0}-01-01", f"{y1}-12-31", cfg["data"]["weather_model"])
    (raw / "s2").mkdir(parents=True, exist_ok=True)
    todo = [r for r in locs.itertuples() if not (raw / "s2" / f"{r.loc_id}.npz").exists()]
    log(f"Sentinel-2 for {len(todo)} extra locations")
    with ProcessPoolExecutor(4) as ex:
        futs = {ex.submit(_s2_job, r.loc_id, r.lon, r.lat, cfg, 6): r.loc_id for r in todo}
        for k, f in enumerate(as_completed(futs)):
            try:
                log(f"extra s2 {k + 1}/{len(todo)} {f.result()}")
            except Exception as e:  # noqa: BLE001
                log(f"extra s2 FAIL {futs[f]}: {str(e)[:120]}")
    log("EXTRA DOWNLOAD DONE")


if __name__ == "__main__":
    main()
