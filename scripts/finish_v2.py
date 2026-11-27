"""Finish the v2 round on the fresh test locations downloaded so far (owner decision 2026-11-27 15:50 IST, before any
v2 test result was seen: deadline 19:00 IST, so the 26 test locations still downloading are left out; the cube build
drops locations without imagery and records them in meta.json).

Seeds 1 and 2 are evaluated in parallel with seed 0 into their own tag folders (test_s1, test_s2) to avoid
concurrent writes, then merged into results/v2/test. Resumable: finished steps are skipped.

  python scripts/finish_v2.py
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pipeline as rp  # noqa: E402
from run_v2 import PROC, RES, TEST_VARIANTS  # noqa: E402

P = "v2"


def merge_seed_folders() -> None:
    test = RES / "test"
    for seed in [1, 2]:
        src = RES / f"test_s{seed}"
        for f in src.glob("*"):
            if f.name in ("metrics.csv", "metrics_ci.csv", "paired_tests.csv"):
                main = test / f.name
                new = pd.read_csv(f)
                if main.exists():
                    old = pd.read_csv(main)
                    new = pd.concat([old[old.seed != seed], new], ignore_index=True)
                new.to_csv(main, index=False)
            elif f.is_file() and f"_s{seed}" in f.name:
                shutil.copy2(f, test / f.name)


def main() -> None:
    os.environ["CFVEG_RESULTS"] = str(RES)
    marker = PROC / "FULL_BUILT"
    if not marker.exists():
        rp.log("v2-finish: rebuilding cube with the downloaded fresh test locations")
        rp.run(["scripts/build_data.py", "--preset", P], ROOT / "runs" / "build_data_v2_full.log")
        marker.write_text(time.strftime("%Y-%m-%d %H:%M") + " (test locations downloaded by 2026-11-27 15:45)")
    jobs = []
    for seed, tag in [(0, "test"), (1, "test_s1"), (2, "test_s2")]:
        if not (RES / tag / "metrics.csv").exists():
            jobs.append((["scripts/eval_all.py", "--preset", P, "--split", "test", "--tag", tag, "--n_windows", "200",
                          "--members", "8", "--seed", str(seed), "--variants"] + TEST_VARIANTS,
                         ROOT / "runs" / f"v2_eval_test_s{seed}.log"))
    if not (RES / "effects" / "effects_det_s0.csv").exists():
        jobs.append((["scripts/effects.py", "--preset", P, "--members", "8", "--max_windows", "200"],
                     ROOT / "runs" / "v2_effects.log"))
    rp.log(f"v2-finish: {len(jobs)} parallel test jobs")
    rp.run_parallel(jobs, 4)
    merge_seed_folders()
    rp.run(["scripts/validate.py", "--preset", P], ROOT / "runs" / "v2_validate.log")
    rp.run(["scripts/counterfactual_demo.py", "--preset", P, "--members", "8"], ROOT / "runs" / "v2_demo.log")
    rp.run(["scripts/paper_assets.py"], ROOT / "runs" / "v2_assets.log")
    for seed in [1, 2, 0]:
        rp.run(["scripts/render_summary.py", "--seed", str(seed)], ROOT / "runs" / "v2_summary.log")
    rp.log("V2 PIPELINE DONE")


if __name__ == "__main__":
    main()
