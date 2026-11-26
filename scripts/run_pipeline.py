"""Unattended main pipeline (resumable; each stage is skipped when its output exists).

  python scripts/run_pipeline.py --preset cpu [--parallel 2]

Stages
  1 build_data                         -> data/processed/*, results/data/*
  2 train det + diff_plain             (parallel)
  3 lambda sweep on VALIDATION         diff_phys_l{0.3,1,3} at sweep_steps -> results/val/lambda_choice.json
  4 train diff_phys + ablations        (no_rate, no_phen, no_mono) with the chosen lambda scale
  5 guidance-scale sweep on VALIDATION -> results/val/guidance_choice.json
  6 test evaluation (once per final variant), effects, validation, demo

Pre-registered selection rules (fixed before any result was seen; validation split only):
  lambda: among sweep scales, the one with the lowest summed violation rate (rate+phen+mono) whose
          validation CRPS is within 5 % of the best validation CRPS; ties -> smaller scale.
  gamma : among {0, 0.003, 0.01, 0.03, 0.1}, the largest gamma whose validation CRPS is within 2 % of
          gamma=0's CRPS (guidance should fix physics without costing accuracy); swept with projection (+P)
          as in the reported variant. Amended after the code review, before any real-data result existed.
  All test variants use the same ensemble size (12 members).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
LAMBDAS = [0.3, 1.0, 3.0]
SMOKE: list[str] = []          # ["--smoke"] when testing the orchestrator on the real-data fixture
RUNS_SUFFIX = ""
RES = ROOT / "results"
GAMMAS = [0.003, 0.01, 0.03, 0.1]


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(ROOT / "runs" / "pipeline.log", "a") as f:
        f.write(line + "\n")


def run(args: list[str], logfile: Path | None = None) -> None:
    log("+ " + " ".join(args))
    if logfile:
        logfile.parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "a") as f:
            subprocess.run([PY, *args], cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT)
    else:
        subprocess.run([PY, *args], cwd=ROOT, check=True)


def run_parallel(jobs: list[tuple[list[str], Path]], n: int) -> None:
    procs: list[subprocess.Popen] = []
    queue = list(jobs)
    while queue or procs:
        while queue and len(procs) < n:
            args, lf = queue.pop(0)
            lf.parent.mkdir(parents=True, exist_ok=True)
            log("+ (bg) " + " ".join(args))
            procs.append(subprocess.Popen([PY, *args], cwd=ROOT, stdout=open(lf, "a"), stderr=subprocess.STDOUT))
        time.sleep(20)
        for p in list(procs):
            if p.poll() is not None:
                procs.remove(p)
                if p.returncode != 0:
                    raise RuntimeError(f"job failed: {p.args}")


def train_jobs(preset: str, variants: list[str], steps: int | None, threads: int) -> list:
    jobs = []
    for v in variants:
        if (ROOT / "runs" / (preset + RUNS_SUFFIX) / f"{v}_s0" / "final.pt").exists():
            continue
        a = ["scripts/train_all.py", "--preset", preset, *SMOKE, "--variants", v, "--threads", str(threads)]
        if steps:
            a += ["--steps", str(steps)]
        jobs.append((a, ROOT / "runs" / (preset + RUNS_SUFFIX) / f"{v}_s0" / "train.log"))
    return jobs


def choose_lambda(val_metrics: pd.DataFrame) -> float:
    rows = []
    for a in LAMBDAS:
        r = val_metrics[val_metrics.variant == f"diff_phys_l{a}"]
        if len(r):
            r = r.iloc[-1]
            rows.append((a, r.crps, r.rate_frac + r.phen_frac + r.get("mono_frac", 0.0)))
    best = min(c for _, c, _ in rows)
    ok = [(v, a) for a, c, v in rows if c <= 1.05 * best]
    return sorted(ok)[0][1]


def choose_gamma(val_metrics: pd.DataFrame) -> float:
    """Swept WITH projection, because the reported variant is +G+P (red-team minor)."""
    base = val_metrics[val_metrics.variant == "diff_phys+P"].iloc[-1].crps
    ok = [0.0]
    for g in GAMMAS:
        r = val_metrics[val_metrics.variant == f"diff_phys+Gx{g}+P"]
        if len(r) and r.iloc[-1].crps <= 1.02 * base:
            ok.append(g)
    return max(ok)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--sweep_steps", type=int, default=3000)
    ap.add_argument("--val_windows", type=int, default=150)
    ap.add_argument("--smoke", action="store_true", help="test the orchestrator on the real-data fixture")
    a = ap.parse_args()
    global SMOKE, RUNS_SUFFIX, RES
    if a.smoke:
        SMOKE, RUNS_SUFFIX, RES = ["--smoke"], "_smoke", ROOT / "results" / "smoke"
    p = a.preset
    (ROOT / "runs").mkdir(exist_ok=True)
    thr = max(2, 14 // a.parallel)
    proc = ROOT / "data" / ("smoke" if a.smoke else "processed")
    val = RES / "val"
    val.mkdir(parents=True, exist_ok=True)

    if not (proc / "clim.npy").exists():
        run(["scripts/build_data.py", "--preset", p, *SMOKE], ROOT / "runs" / "build_data.log")
    log("stage 2: det + diff_plain")
    run_parallel(train_jobs(p, ["det", "diff_plain"], None, thr), a.parallel)

    lam_f = val / "lambda_choice.json"
    if not lam_f.exists():
        log("stage 3: lambda sweep (validation)")
        run_parallel(train_jobs(p, [f"diff_phys_l{x}" for x in LAMBDAS], a.sweep_steps, max(2, 14 // 3)), 3)
        run(["scripts/eval_all.py", "--preset", p, *SMOKE, "--split", "val", "--tag", "val_lambda", "--members", "8",
             "--n_windows", str(a.val_windows), "--reference", "diff_phys_l1.0", "--variants"] +
            [f"diff_phys_l{x}" for x in LAMBDAS], ROOT / "runs" / "eval_val_lambda.log")
        m = pd.read_csv(RES / "val_lambda" / "metrics.csv")
        lam = choose_lambda(m)
        lam_f.write_text(json.dumps({"lambda_scale": lam, "rule": "lowest violations within 5% of best val CRPS",
                                     "candidates": LAMBDAS}, indent=1))
        log(f"chosen lambda scale = {lam}")

    log("stage 4: diff_phys + ablations")
    run_parallel(train_jobs(p, ["diff_phys", "diff_phys_no_rate", "diff_phys_no_phen", "diff_phys_no_mono"], None, thr),
                 a.parallel)

    g_f = val / "guidance_choice.json"
    if not g_f.exists():
        log("stage 5: guidance sweep (validation)")
        run(["scripts/eval_all.py", "--preset", p, *SMOKE, "--split", "val", "--tag", "val_guidance", "--members", "8",
             "--n_windows", str(a.val_windows), "--reference", "diff_phys+P", "--variants", "diff_phys+P"] +
            [f"diff_phys+Gx{g}+P" for g in GAMMAS], ROOT / "runs" / "eval_val_guidance.log")
        m = pd.read_csv(RES / "val_guidance" / "metrics.csv")
        g = choose_gamma(m)
        g_f.write_text(json.dumps({"guidance_scale": g, "rule": "largest gamma with val CRPS within 2% of gamma=0",
                                   "candidates": [0.0] + GAMMAS}, indent=1))
        log(f"chosen guidance scale = {g}")

    log("stage 6: test evaluation, effects, validation, demo")
    if not (RES / "test" / "metrics.csv").exists():
        run(["scripts/eval_all.py", "--preset", p, *SMOKE, "--split", "test", "--tag", "test", "--n_windows", "200",
             "--members", "12", "--variants", "observed", "persistence", "climatology", "clim_anomaly", "det",
             "diff_plain", "diff_phys+G+P", "diff_plain+G", "diff_plain+G+P", "diff_phys", "diff_phys+G",
             "diff_phys+P", "diff_phys_no_rate+G+P", "diff_phys_no_phen+G+P", "diff_phys_no_mono+G+P"],
            ROOT / "runs" / "eval_test.log")
    if not (RES / "effects" / "effects_det_s0.csv").exists():
        run(["scripts/effects.py", "--preset", p, *SMOKE, "--members", "10", "--max_windows", "160"],
            ROOT / "runs" / "effects.log")
    run(["scripts/validate.py", "--preset", p, *SMOKE], ROOT / "runs" / "validate.log")
    run(["scripts/counterfactual_demo.py", "--preset", p, *SMOKE], ROOT / "runs" / "demo.log")
    log("PIPELINE DONE")


if __name__ == "__main__":
    main()
