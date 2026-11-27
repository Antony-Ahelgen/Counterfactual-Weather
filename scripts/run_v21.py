"""v2.1 confirmation run on the RESERVED fresh test locations (design fixed 2026-11-27 ~18:20 IST, before any result).

Test set  = the v2 test locations that were NOT in the v2 cube (left out at the v2 deadline) and whose Sentinel-2
            download has completed when this script starts. No model has seen them; the 55 v2 test locations are
            excluded. Train/val locations and all fitted objects are the same as v2 (same models: runs/v2).
Variants  = (a) replication of v2 with 8 members: observed, persistence, climatology, clim_anomaly, det, diff_plain,
            diff_phys+G+P; (b) calibration with 32 members: diff_phys+G+P+Cal (bias + spread fitted on validation,
            scripts/fit_calibration.py), and diff_phys+G+P with 32 members (seed 0) to separate ensemble size from
            the correction.
Verdict   = calibration rule unchanged (90 % coverage in [0.8, 0.95] and spread-skill in [0.7, 1.3]), per seed;
            supported only if supported in every seed. Other claims: replication via render_summary.
Results in results/v21 (main results untouched).

  python scripts/run_v21.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pipeline as rp  # noqa: E402

P = "v21"
RES = ROOT / "results" / "v21"
MAN = ROOT / "manifests" / "locations_v21.csv"
REP = ["observed", "persistence", "climatology", "clim_anomaly", "det", "diff_plain", "diff_phys+G+P"]


def make_manifest() -> pd.DataFrame:
    v2 = pd.read_csv(ROOT / "manifests" / "locations_v2.csv")
    used = set(json.loads((ROOT / "data" / "processed_v2" / "cube" / "meta.json").read_text())["loc_id"])
    test = v2[(v2.split == "test") & ~v2.loc_id.isin(used)]
    test = test[[(ROOT / "data" / "raw" / "s2" / f"{i}.npz").exists() for i in test.loc_id]]
    dev = v2[v2.split.isin(["train", "val"])]
    man = pd.concat([dev, test], ignore_index=True)
    man.to_csv(MAN, index=False)
    return test


def main() -> None:
    os.environ["CFVEG_RESULTS"] = str(RES)
    if not MAN.exists():
        test = make_manifest()
        rp.log(f"v21: reserved test locations available: {len(test)} {test.loc_id.tolist()}")
    (RES / "val").mkdir(parents=True, exist_ok=True)
    for f in ["guidance_choice.json", "spread_choice.json", "calib_choice.json"]:
        shutil.copy2(ROOT / "results" / "val" / f, RES / "val" / f)
    if not (ROOT / "data" / "processed_v21" / "clim.npy").exists():
        rp.run(["scripts/build_data.py", "--preset", P], ROOT / "runs" / "build_data_v21.log")
    jobs = []
    for seed, tag in [(0, "test"), (1, "test_s1"), (2, "test_s2")]:
        if not (RES / tag / "metrics.csv").exists():
            jobs.append((["scripts/eval_all.py", "--preset", P, "--split", "test", "--tag", tag, "--n_windows", "100",
                          "--members", "8", "--seed", str(seed), "--variants"] + REP,
                         ROOT / "runs" / f"v21_eval_s{seed}.log"))
        cv = ["diff_phys+G+P+Cal"] + (["diff_phys+G+P"] if seed == 0 else [])
        if not (RES / f"m32_s{seed}" / "metrics.csv").exists():
            jobs.append((["scripts/eval_all.py", "--preset", P, "--split", "test", "--tag", f"m32_s{seed}",
                          "--n_windows", "100", "--members", "32", "--seed", str(seed), "--reference",
                          "diff_phys+G+P+Cal", "--variants"] + cv, ROOT / "runs" / f"v21_m32_s{seed}.log"))
    rp.log(f"v21: {len(jobs)} parallel evaluation jobs")
    rp.run_parallel(jobs, 6)
    rp.log("V21 EVAL DONE")


if __name__ == "__main__":
    main()
