"""Reduced-scale REAL-data pipeline (preset `fast`), chosen by the project owner for time reasons.

  python scripts/run_fast.py
Differences from scripts/run_pipeline.py (all documented in docs/LIMITATIONS.md):
  - 69 locations (downloads completed at the time), re-split 70/15/15 before training (manifests/locations_fast.csv)
  - 3 trained models (det, diff_plain, diff_phys) x 2500 steps; lambda NOT tuned (config values)
  - physics ablation through sampling variants (guidance / projection on/off), not retrained models
  - guidance scale chosen on VALIDATION with the same pre-registered rule, smaller grid {0, 0.01, 0.1}
  - test: 100 windows, 8 members for every probabilistic variant; effects: 80 windows, 8 members
Resumable: every stage is skipped when its output exists.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_pipeline as rp  # noqa: E402

GAMMAS = [0.01, 0.1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="fast")
    ap.add_argument("--test_windows", default="100")
    ap.add_argument("--effect_windows", default="80")
    ap.add_argument("--eta_grid", default="", help="e.g. 0.5,1.0: select DDIM eta on validation (0 always included)")
    a = ap.parse_args()
    P = a.preset
    sys.path.insert(0, str(ROOT / "src"))
    from cfveg.utils.provenance import load_preset

    rp.GAMMAS = GAMMAS
    (ROOT / "runs").mkdir(exist_ok=True)
    proc = ROOT / load_preset(P)["data"]["processed_dir"]
    res = Path(os.environ.get("CFVEG_RESULTS", str(ROOT / "results")))
    res = res if res.is_absolute() else ROOT / res
    val = res / "val"
    val.mkdir(parents=True, exist_ok=True)
    if not (proc / "clim.npy").exists():
        rp.run(["scripts/build_data.py", "--preset", P], ROOT / "runs" / f"build_data_{P}.log")
    rp.log("fast: train det, diff_plain, diff_phys (parallel)")
    rp.run_parallel(rp.train_jobs(P, ["det", "diff_plain", "diff_phys"], None, 5), 3)
    g_f = val / "guidance_choice.json"
    if not g_f.exists():
        rp.log("fast: guidance sweep on validation")
        rp.run(["scripts/eval_all.py", "--preset", P, "--split", "val", "--tag", "val_guidance", "--members", "8",
                "--n_windows", "40", "--reference", "diff_phys+P", "--variants", "diff_phys+P"] +
               [f"diff_phys+Gx{g}+P" for g in GAMMAS], ROOT / "runs" / "eval_val_guidance.log")
        g = rp.choose_gamma(pd.read_csv(res / "val_guidance" / "metrics.csv"))
        g_f.write_text(json.dumps({"guidance_scale": g, "rule": "largest gamma with val CRPS within 2% of gamma=0 "
                                   "(with projection)", "candidates": [0.0] + GAMMAS, "preset": P}, indent=1))
        rp.log(f"chosen guidance scale = {g}")
    e_f = val / "eta_choice.json"
    if a.eta_grid and not e_f.exists():
        etas = [float(x) for x in a.eta_grid.split(",")]
        rp.log(f"eta sweep on validation: {[0.0] + etas}")
        rp.run(["scripts/eval_all.py", "--preset", P, "--split", "val", "--tag", "val_eta", "--members", "8",
                "--n_windows", "40", "--reference", "diff_phys+G+P", "--variants", "diff_phys+G+P"] +
               [f"diff_phys+G+P+Ex{e}" for e in etas], ROOT / "runs" / "eval_val_eta.log")
        vm = pd.read_csv(res / "val_eta" / "metrics.csv")
        base = vm[vm.variant == "diff_phys+G+P"].iloc[-1]
        cands = [(0.0, base.crps, base.cov90)] + [
            (e, r.crps, r.cov90) for e in etas for _, r in vm[vm.variant == f"diff_phys+G+P+Ex{e}"].tail(1).iterrows()]
        ok = [(abs(c90 - 0.9), e) for e, c, c90 in cands if c <= 1.02 * base.crps]
        eta = sorted(ok)[0][1]
        e_f.write_text(json.dumps({"eta": eta, "rule": "cov90 closest to 0.9 among eta with val CRPS within 2% of "
                                   "eta=0; ties -> smaller eta", "candidates": [0.0] + etas, "preset": P}, indent=1))
        rp.log(f"chosen eta = {eta}")
    if not (res / "test" / "metrics.csv").exists():
        rp.run(["scripts/eval_all.py", "--preset", P, "--split", "test", "--tag", "test", "--n_windows", a.test_windows,
                "--members", "8", "--variants", "observed", "persistence", "climatology", "clim_anomaly", "det",
                "diff_plain", "diff_phys+G+P", "diff_phys", "diff_phys+G", "diff_phys+P", "diff_plain+G+P"],
               ROOT / "runs" / "eval_test.log")
    if not (res / "effects" / "effects_det_s0.csv").exists():
        rp.run(["scripts/effects.py", "--preset", P, "--members", "8", "--max_windows", a.effect_windows],
               ROOT / "runs" / "effects.log")
    rp.run(["scripts/validate.py", "--preset", P], ROOT / "runs" / "validate.log")
    rp.run(["scripts/counterfactual_demo.py", "--preset", P, "--members", "8"], ROOT / "runs" / "demo.log")
    rp.run(["scripts/paper_assets.py"], ROOT / "runs" / "paper_assets.log")
    rp.run(["scripts/render_summary.py"], ROOT / "runs" / "render_summary.log")
    rp.log(f"FAST PIPELINE DONE ({P})")


if __name__ == "__main__":
    main()
