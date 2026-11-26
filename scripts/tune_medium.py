"""Validation-only tuning on the medium data (owner request 2026-11-26), then a single test evaluation.

1. wait for runs/medium/diff_phys_s0/final.pt (lambda scale 1)
2. train diff_phys_l0.3 and diff_phys_l3.0 on the medium data (same steps)
3. lambda selected on VALIDATION with the pre-registered pipeline rule (scripts/run_pipeline.choose_lambda)
4. runs/medium_tuned = det, diff_plain (unchanged) + diff_phys with the chosen lambda
5. wait for the untuned medium pipeline to finish, then run_fast --preset medium_tuned with gamma and eta
   re-selected on VALIDATION, one test evaluation, effects, validation, assets -> results/medium_tuned
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pipeline as rp  # noqa: E402

RUNS = ROOT / "runs"
OUT = ROOT / "results" / "medium_tuned"


def wait_for(path: Path, what: str) -> None:
    rp.log(f"tune: waiting for {what}")
    while not path.exists():
        time.sleep(30)


def main() -> None:
    os.environ["CFVEG_RESULTS"] = str(OUT)
    wait_for(RUNS / "medium" / "diff_phys_s0" / "final.pt", "medium diff_phys (lambda 1)")
    rp.log("tune: training lambda 0.3 and 3.0")
    rp.run_parallel(rp.train_jobs("medium", ["diff_phys_l0.3", "diff_phys_l3.0"], None, 4), 2)
    l1 = RUNS / "medium" / "diff_phys_l1.0_s0"
    if not l1.exists():
        shutil.copytree(RUNS / "medium" / "diff_phys_s0", l1)
    lam_f = OUT / "val" / "lambda_choice.json"
    if not lam_f.exists():
        rp.run(["scripts/eval_all.py", "--preset", "medium", "--split", "val", "--tag", "val_lambda", "--members", "8",
                "--n_windows", "40", "--reference", "diff_phys_l1.0", "--variants", "diff_phys_l0.3",
                "diff_phys_l1.0", "diff_phys_l3.0"], RUNS / "eval_val_lambda_medium.log")
        lam = rp.choose_lambda(pd.read_csv(OUT / "val_lambda" / "metrics.csv"))
        lam_f.parent.mkdir(parents=True, exist_ok=True)
        lam_f.write_text(json.dumps({"lambda_scale": lam, "rule": "lowest violations within 5% of best val CRPS",
                                     "candidates": rp.LAMBDAS, "preset": "medium"}, indent=1))
        rp.log(f"tune: chosen lambda scale = {lam}")
    lam = json.loads(lam_f.read_text())["lambda_scale"]
    tuned = RUNS / "medium_tuned"
    tuned.mkdir(exist_ok=True)
    for v in ["det", "diff_plain"]:
        if not (tuned / f"{v}_s0").exists():
            shutil.copytree(RUNS / "medium" / f"{v}_s0", tuned / f"{v}_s0")
    if not (tuned / "diff_phys_s0").exists():
        shutil.copytree(RUNS / "medium" / f"diff_phys_l{lam}_s0", tuned / "diff_phys_s0")
        (tuned / "diff_phys_s0" / "SELECTED_LAMBDA.txt").write_text(f"lambda scale {lam} (validation)\n")
    rp.log("tune: waiting for the untuned medium pipeline to finish")
    while "FAST PIPELINE DONE (medium)" not in (RUNS / "pipeline.log").read_text():
        time.sleep(30)
    rp.run(["scripts/run_fast.py", "--preset", "medium_tuned", "--test_windows", "120", "--effect_windows", "100",
            "--eta_grid", "0.5,1.0"], RUNS / "run_medium_tuned.log")
    rp.log("TUNED PIPELINE DONE")


if __name__ == "__main__":
    main()
