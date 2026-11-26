"""Wait for the running S2 download to finish (it writes manifests/s2_scenes.csv at the end), fill any
failed locations with one more resumable pass, then run the full pipeline. Resumable and idempotent."""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
start = time.time()
man = ROOT / "manifests" / "s2_scenes.csv"
print(f"[{time.strftime('%H:%M:%S')}] waiting for download to finish", flush=True)
while not (man.exists() and man.stat().st_mtime > start):
    time.sleep(60)
print(f"[{time.strftime('%H:%M:%S')}] download finished; filling gaps", flush=True)
subprocess.run([PY, "scripts/download_data.py", "--stage", "s2", "--workers", "5", "--threads", "6"], cwd=ROOT,
               check=False, stdout=open(ROOT / "runs" / "download" / "s2_fill.log", "a"), stderr=subprocess.STDOUT)
n = len(list((ROOT / "data" / "raw" / "s2").glob("*.npz")))
print(f"[{time.strftime('%H:%M:%S')}] {n} locations downloaded; starting pipeline", flush=True)
subprocess.run([PY, "scripts/run_pipeline.py", "--preset", "cpu", "--parallel", "2"], cwd=ROOT, check=True)
print(f"[{time.strftime('%H:%M:%S')}] pipeline finished", flush=True)
