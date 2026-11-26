"""Resumable, idempotent downloads. Stages: locations (incl. static), weather, s2.

  python scripts/download_data.py --stage locations --workers 6
  python scripts/download_data.py --stage weather
  python scripts/download_data.py --stage s2 --workers 5
Outputs (git-ignored): data/raw/{static,s2,weather}/<loc_id>.*
Manifests (committed): manifests/locations.csv, manifests/s2_scenes.csv, manifests/weather.csv
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfveg.utils.provenance import load_preset  # noqa: E402


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _static_job(loc_id: str, lon: float, lat: float, cfg: dict) -> dict:
    from cfveg.data.stac_common import configure_gdal
    from cfveg.data.static import load_static, veg_fraction

    out = ROOT / cfg["data"]["raw_dir"] / "static" / f"{loc_id}.npz"
    if out.exists():
        z = np.load(out)
        return {"loc_id": loc_id, "veg_frac": veg_fraction(z["lc"]), "elev": float(np.nanmean(z["elev"])),
                "ok": True}
    configure_gdal()
    d = cfg["data"]
    st = load_static(lon, lat, d["patch_px"], d["resolution_m"], d["crs"])
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **st)
    tmp.replace(out)
    return {"loc_id": loc_id, "veg_frac": veg_fraction(st["lc"]), "elev": float(np.nanmean(st["elev"])), "ok": True}


def _s2_job(loc_id: str, lon: float, lat: float, cfg: dict, threads: int) -> dict:
    from cfveg.data.sentinel2 import download_location
    from cfveg.data.stac_common import configure_gdal

    out = ROOT / cfg["data"]["raw_dir"] / "s2" / f"{loc_id}.npz"
    if out.exists():
        return {"loc_id": loc_id, "skipped": True}
    configure_gdal()
    t = time.time()
    r = download_location(lon, lat, cfg, n_workers=threads)
    tmp = out.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, ndvi=r["ndvi"].astype(np.float16), nobs=r["nobs"],
                        years=np.array(cfg["data"]["years"]), lenient_years=np.array(r["lenient_years"]),
                        failed_days=np.array(r["failed_days"]))
    tmp.replace(out)
    man = pd.DataFrame(r["manifest"])
    man.insert(0, "loc_id", loc_id)
    man.to_csv(out.with_suffix(".scenes.csv"), index=False)
    return {"loc_id": loc_id, "secs": round(time.time() - t), "n_scenes": len(man),
            "valid_frac": float(np.isfinite(r["ndvi"]).mean()), "lenient_years": r["lenient_years"],
            "n_failed_days": len(r["failed_days"])}


def stage_locations(cfg: dict, workers: int) -> None:
    from cfveg.data.locations import assign_splits, sample_candidates

    d, s = cfg["data"], cfg["splits"]
    (ROOT / d["raw_dir"] / "static").mkdir(parents=True, exist_ok=True)
    cand = sample_candidates(d["region_bbox"], d["n_candidate_locations"], d["min_spacing_deg"], d["location_seed"])
    print(f"{len(cand)} candidates", flush=True)
    res = {}
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_static_job, r.loc_id, r.lon, r.lat, cfg): r.loc_id for r in cand.itertuples()}
        for k, f in enumerate(as_completed(futs)):
            lid = futs[f]
            try:
                res[lid] = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  static FAIL {lid}: {e}", flush=True)
                res[lid] = {"loc_id": lid, "ok": False, "veg_frac": 0.0, "elev": np.nan}
            if k % 20 == 0:
                print(f"  static {k + 1}/{len(cand)}", flush=True)
    meta = pd.DataFrame([res[i] for i in cand.loc_id])
    cand = cand.merge(meta, on="loc_id")
    keep = cand[(cand.ok) & (cand.veg_frac >= d["min_veg_fraction"])].head(d["n_locations"]).copy()
    print(f"kept {len(keep)} / {len(cand)} with veg_frac >= {d['min_veg_fraction']}", flush=True)
    keep = assign_splits(keep[["loc_id", "lon", "lat", "veg_frac", "elev"]], s["frac_val_locations"],
                         s["frac_test_locations"], s["split_seed"])
    (ROOT / "manifests").mkdir(exist_ok=True)
    keep.to_csv(ROOT / "manifests" / "locations.csv", index=False)
    cand.to_csv(ROOT / "manifests" / "location_candidates.csv", index=False)
    print(keep.split.value_counts().to_dict(), flush=True)


def stage_weather(cfg: dict) -> None:
    from cfveg.data.weather import download_weather

    d = cfg["data"]
    locs = pd.read_csv(ROOT / "manifests" / "locations.csv")
    y0, y1 = d["weather_years"]
    out = ROOT / d["raw_dir"] / "weather"
    download_weather(locs, out, f"{y0}-01-01", f"{y1}-12-31", d["weather_model"])
    rows = []
    for r in locs.itertuples():
        m = json.loads((out / f"{r.loc_id}.json").read_text())
        m["sha256_16"] = _sha(out / f"{r.loc_id}.csv")
        m.pop("vars", None)
        rows.append(m)
    pd.DataFrame(rows).to_csv(ROOT / "manifests" / "weather.csv", index=False)


def stage_s2(cfg: dict, workers: int, threads: int) -> None:
    d = cfg["data"]
    (ROOT / d["raw_dir"] / "s2").mkdir(parents=True, exist_ok=True)
    locs = pd.read_csv(ROOT / "manifests" / "locations.csv")
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(workers) as ex:
        futs = {ex.submit(_s2_job, r.loc_id, r.lon, r.lat, cfg, threads): r.loc_id for r in locs.itertuples()}
        for f in as_completed(futs):
            lid = futs[f]
            done += 1
            try:
                print(f"  s2 {done}/{len(locs)} {f.result()} elapsed={time.time() - t0:.0f}s", flush=True)
            except Exception:  # noqa: BLE001
                print(f"  s2 FAIL {lid}:\n{traceback.format_exc()[-800:]}", flush=True)
    scenes = sorted((ROOT / d["raw_dir"] / "s2").glob("*.scenes.csv"))
    if scenes:
        pd.concat([pd.read_csv(p) for p in scenes]).to_csv(ROOT / "manifests" / "s2_scenes.csv", index=False)
        sums = [{"loc_id": p.name.split(".")[0], "sha256_16": _sha(p.with_name(p.name.split(".")[0] + ".npz"))}
                for p in scenes]
        pd.DataFrame(sums).to_csv(ROOT / "manifests" / "s2_checksums.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["locations", "weather", "s2", "all"])
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    cfg = load_preset(a.preset)
    if a.stage in ("locations", "all"):
        stage_locations(cfg, a.workers)
    if a.stage in ("weather", "all"):
        stage_weather(cfg)
    if a.stage in ("s2", "all"):
        stage_s2(cfg, a.workers, a.threads)


if __name__ == "__main__":
    main()
