"""At the cut-off time: stop the S2 download, re-split the downloaded locations (same rule and seed, before any
training) into manifests/locations_medium.csv, then run the reduced-scale pipeline with preset `medium` into
results/medium (the current main results stay untouched until promoted).

  python scripts/medium_cutoff.py --cutoff 18:15
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PY = sys.executable


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="18:15")
    a = ap.parse_args()
    hh, mm = map(int, a.cutoff.split(":"))
    now = dt.datetime.now()
    cut = now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    log(f"waiting until {cut:%H:%M} to stop the download")
    while dt.datetime.now() < cut:
        time.sleep(20)
    ps = ("Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like "
          "'*download_data*' -or $_.CommandLine -like '*multiprocessing*' } | ForEach-Object { Stop-Process -Id "
          "$_.ProcessId -Force }")
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)
    time.sleep(5)
    for f in (ROOT / "data" / "raw" / "s2").glob("*.tmp.npz"):
        f.unlink()
    from cfveg.data.locations import assign_splits
    from cfveg.utils.provenance import load_preset

    cfg = load_preset("cpu")
    locs = pd.read_csv(ROOT / "manifests" / "locations.csv")
    have = locs[[(ROOT / "data" / "raw" / "s2" / f"{i}.npz").exists() for i in locs.loc_id]].drop(columns=["split"])
    s = cfg["splits"]
    med = assign_splits(have.reset_index(drop=True), s["frac_val_locations"], s["frac_test_locations"], s["split_seed"])
    med.to_csv(ROOT / "manifests" / "locations_medium.csv", index=False)
    log(f"{len(med)} locations: {med.split.value_counts().to_dict()}; starting medium pipeline")
    env = dict(os.environ, CFVEG_RESULTS=str(ROOT / "results" / "medium"))
    subprocess.run([PY, "scripts/run_fast.py", "--preset", "medium", "--test_windows", "120",
                    "--effect_windows", "100"], cwd=ROOT, env=env, check=True)
    log("medium pipeline finished")


if __name__ == "__main__":
    main()
