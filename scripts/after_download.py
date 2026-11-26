"""Unattended driver: resumable Sentinel-2 download passes until all locations are present (or a pass makes
no progress), then the full pipeline. Safe to re-run at any time (every stage skips finished work).

  python scripts/after_download.py
Logs: runs/download/s2_pass<k>.log, runs/pipeline.log
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def n_done() -> int:
    d = ROOT / "data" / "raw" / "s2"
    return len([f for f in d.glob("L*.npz") if not f.name.endswith(".tmp.npz")])


def main() -> None:
    total = len(pd.read_csv(ROOT / "manifests" / "locations.csv"))
    (ROOT / "runs" / "download").mkdir(parents=True, exist_ok=True)
    for k in range(1, 7):
        before = n_done()
        if before >= total:
            break
        print(f"[{time.strftime('%H:%M:%S')}] download pass {k}: {before}/{total} present", flush=True)
        for f in (ROOT / "data" / "raw" / "s2").glob("*.tmp.npz"):
            f.unlink()
        with open(ROOT / "runs" / "download" / f"s2_pass{k}.log", "a") as lf:
            subprocess.run([PY, "scripts/download_data.py", "--stage", "s2", "--workers", "4", "--threads", "6"],
                           cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, check=False)
        if n_done() == before:
            print(f"[{time.strftime('%H:%M:%S')}] no progress in pass {k}; continuing with {before} locations",
                  flush=True)
            break
    print(f"[{time.strftime('%H:%M:%S')}] {n_done()}/{total} locations; starting pipeline", flush=True)
    subprocess.run([PY, "scripts/run_pipeline.py", "--preset", "cpu", "--parallel", "2"], cwd=ROOT, check=True)
    print(f"[{time.strftime('%H:%M:%S')}] pipeline finished", flush=True)


if __name__ == "__main__":
    main()
