"""v2 improvement round: resumable end to end (safe to stop at any time and re-run the same command).

  python scripts/run_v2.py            # results go to results/v2 (main results untouched until promoted)

Stages (each skipped when done):
  1 download the fresh test locations (background, resumable) while development work runs
  2 build the development cube; train det / diff_plain / diff_phys for seeds 0,1,2 (resume from checkpoints)
  3 VALIDATION selections (pre-registered, written before any v2 result):
      gamma  : among {0, 0.003, 0.01, 0.03, 0.1, 0.3} with val CRPS within 2 % of gamma=0, the one whose envelope
               violation fraction is closest to that of the OBSERVATIONS; ties -> smaller gamma (seed 0)
      spread : per seed and variant, factor = skill/spread (1 / spread-skill ratio) on validation
  4 wait for the download; rebuild the cube with the fresh test locations
  5 test evaluation once per seed (0,1,2); effects/validation/demo on seed 0; assets; per-seed summaries
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_pipeline as rp  # noqa: E402

P = "v2"
SEEDS = [0, 1, 2]
GAMMAS = [0.003, 0.01, 0.03, 0.1, 0.3]
RES = ROOT / "results" / "v2"
PROC = ROOT / "data" / "processed_v2"
PROB = ["diff_plain", "diff_phys+G+P", "diff_phys", "diff_phys+G", "diff_phys+P", "diff_plain+G+P"]
TEST_VARIANTS = ["observed", "persistence", "climatology", "clim_anomaly", "det"] + PROB


def test_todo() -> list[str]:
    t = pd.read_csv(ROOT / "manifests" / "locations_v2.csv")
    return [i for i in t[t.split == "test"].loc_id if not (ROOT / "data" / "raw" / "s2" / f"{i}.npz").exists()]


def download_running() -> bool:
    ps = ("(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -like "
          "'*download_data*' }).Count")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


def main() -> None:
    os.environ["CFVEG_RESULTS"] = str(RES)
    (RES / "val").mkdir(parents=True, exist_ok=True)
    dl = None
    if test_todo() and not download_running():
        rp.log(f"v2: starting download of {len(test_todo())} fresh test locations")
        dl = subprocess.Popen([sys.executable, "scripts/download_data.py", "--stage", "s2", "--workers", "5",
                               "--threads", "6"], cwd=ROOT,
                              stdout=open(ROOT / "runs" / "download" / "s2_v2test.log", "a"), stderr=subprocess.STDOUT)
    if not (PROC / "clim.npy").exists():
        rp.run(["scripts/build_data.py", "--preset", P], ROOT / "runs" / "build_data_v2.log")
    for seed in SEEDS:
        jobs = []
        for v in ["det", "diff_plain", "diff_phys"]:
            if not (ROOT / "runs" / P / f"{v}_s{seed}" / "final.pt").exists():
                jobs.append((["scripts/train_all.py", "--preset", P, "--variants", v, "--seeds", str(seed),
                              "--threads", "5"], ROOT / "runs" / P / f"{v}_s{seed}" / "train.log"))
        rp.log(f"v2: training seed {seed} ({len(jobs)} jobs)")
        rp.run_parallel(jobs, 3)
    g_f = RES / "val" / "guidance_choice.json"
    if not g_f.exists():
        rp.run(["scripts/eval_all.py", "--preset", P, "--split", "val", "--tag", "val_guidance", "--members", "8",
                "--n_windows", "60", "--no_spread", "--reference", "diff_phys+P", "--variants", "observed",
                "diff_phys+P"] + [f"diff_phys+Gx{g}+P" for g in GAMMAS], ROOT / "runs" / "v2_val_guidance.log")
        vm = pd.read_csv(RES / "val_guidance" / "metrics.csv")
        obs = vm[vm.variant == "observed"].iloc[-1].phen_frac
        base = vm[vm.variant == "diff_phys+P"].iloc[-1]
        cands = [(0.0, base.crps, base.phen_frac)] + [
            (g, r.crps, r.phen_frac) for g in GAMMAS for _, r in vm[vm.variant == f"diff_phys+Gx{g}+P"].tail(1).iterrows()]
        ok = sorted((abs(pf - obs), g) for g, c, pf in cands if c <= 1.02 * base.crps)
        gamma = ok[0][1]
        g_f.write_text(json.dumps({"guidance_scale": gamma, "observed_envelope_violation": obs,
                                   "rule": "envelope violation closest to observed among gamma with val CRPS within "
                                           "2% of gamma=0", "candidates": [0.0] + GAMMAS, "preset": P}, indent=1))
        rp.log(f"v2: chosen guidance scale = {gamma}")
    s_f = RES / "val" / "spread_choice.json"
    spread = json.loads(s_f.read_text()) if s_f.exists() else {}
    for seed in SEEDS:
        if f"s{seed}" in spread:
            continue
        rp.run(["scripts/eval_all.py", "--preset", P, "--split", "val", "--tag", "val_spread", "--members", "8",
                "--n_windows", "60", "--seed", str(seed), "--no_spread", "--reference", "diff_phys+G+P",
                "--variants"] + PROB, ROOT / "runs" / "v2_val_spread.log")
        vm = pd.read_csv(RES / "val_spread" / "metrics.csv")
        vm = vm[vm.seed == seed]
        spread[f"s{seed}"] = {r.variant.replace("+", "_"): float(1.0 / r.spread_skill) for r in vm.itertuples()
                              if r.spread_skill and r.spread_skill == r.spread_skill}
        s_f.write_text(json.dumps(spread, indent=1))
        rp.log(f"v2: spread factors seed {seed}: {spread[f's{seed}']}")
    if dl is not None:
        rp.log("v2: waiting for the fresh test-set download")
        dl.wait()
    while download_running():
        time.sleep(60)
    marker = PROC / "FULL_BUILT"
    if not marker.exists():
        rp.log(f"v2: rebuilding cube with fresh test locations ({len(test_todo())} still missing)")
        rp.run(["scripts/build_data.py", "--preset", P], ROOT / "runs" / "build_data_v2_full.log")
        marker.write_text(time.strftime("%Y-%m-%d %H:%M"))
    for seed in SEEDS:
        done = (RES / "test" / "metrics.csv").exists() and (pd.read_csv(RES / "test" / "metrics.csv").seed == seed).any()
        if not done:
            rp.run(["scripts/eval_all.py", "--preset", P, "--split", "test", "--tag", "test", "--n_windows", "200",
                    "--members", "8", "--seed", str(seed), "--variants"] + TEST_VARIANTS,
                   ROOT / "runs" / f"v2_eval_test_s{seed}.log")
    if not (RES / "effects" / "effects_det_s0.csv").exists():
        rp.run(["scripts/effects.py", "--preset", P, "--members", "8", "--max_windows", "200"],
               ROOT / "runs" / "v2_effects.log")
    rp.run(["scripts/validate.py", "--preset", P], ROOT / "runs" / "v2_validate.log")
    rp.run(["scripts/counterfactual_demo.py", "--preset", P, "--members", "8"], ROOT / "runs" / "v2_demo.log")
    rp.run(["scripts/paper_assets.py"], ROOT / "runs" / "v2_assets.log")
    for seed in [1, 2, 0]:
        rp.run(["scripts/render_summary.py", "--seed", str(seed)], ROOT / "runs" / "v2_summary.log")
    rp.log("V2 PIPELINE DONE")


if __name__ == "__main__":
    main()
