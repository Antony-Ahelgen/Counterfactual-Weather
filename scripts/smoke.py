"""End-to-end smoke run of every stage on the REAL-data test fixture (CPU, a few minutes).
Outputs go to results/smoke/ and runs/smoke_smoke/ (git-ignored). Never reported as findings."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable


def run(*a: str) -> None:
    print("+", " ".join(a), flush=True)
    subprocess.run([PY, *a], cwd=ROOT, check=True)


def main() -> None:
    for d in [ROOT / "runs" / "smoke_smoke", ROOT / "results" / "smoke", ROOT / "data" / "smoke"]:
        shutil.rmtree(d, ignore_errors=True)
    run("scripts/build_data.py", "--preset", "smoke", "--smoke")
    run("scripts/train_all.py", "--preset", "smoke", "--smoke", "--variants", "det", "diff_plain", "diff_phys")
    run("scripts/eval_all.py", "--preset", "smoke", "--smoke", "--split", "test", "--tag", "test", "--variants",
        "persistence", "climatology", "clim_anomaly", "det", "diff_plain", "diff_phys+G+P")
    run("scripts/effects.py", "--preset", "smoke", "--smoke", "--members", "4", "--max_windows", "10",
        "--variants", "diff_phys+G+P", "diff_phys+G+P+C", "det")
    run("scripts/validate.py", "--preset", "smoke", "--smoke")
    run("scripts/counterfactual_demo.py", "--preset", "smoke", "--smoke", "--members", "4")
    print("SMOKE OK")


if __name__ == "__main__":
    main()
