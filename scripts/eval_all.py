"""Evaluate forecast variants on validation or test windows (factual forecasts, observed weather).

Variant spec strings:
  persistence | climatology | clim_anomaly | det
  <run>[+G[x<scale>]][+P]   diffusion run name (e.g. diff_plain, diff_phys, diff_phys_no_rate) with
                            +G physics guidance (optional scale override) and +P hard projection.
Examples:
  python scripts/eval_all.py --preset cpu --split val --variants diff_phys+Gx0.003 diff_phys+Gx0.01 diff_phys+Gx0.03
  python scripts/eval_all.py --preset cpu --split test --tag main
Outputs results/<tag>/{metrics.csv, per_window_<variant>.csv, metrics_ci.csv, paired_tests.csv} + provenance.
The test split must only be run once per final variant (no tuning on test).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
# results root: env CFVEG_RESULTS (e.g. results/full for the full-scale run), default results/
RESULTS = Path(__import__("os").environ.get("CFVEG_RESULTS", str(ROOT / "results")))
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS
sys.path.insert(0, str(ROOT / "src"))

from cfveg.counterfactual.paired import Sampler, SamplerOpts  # noqa: E402
from cfveg.data.dataset import Cube, WindowDataset, make_windows  # noqa: E402
from cfveg.models.diffusion import Diffusion  # noqa: E402
from cfveg.models.train import collate, load_model  # noqa: E402
from cfveg.physics.constraints import Physics  # noqa: E402
from cfveg.utils.provenance import load_preset, write_provenance  # noqa: E402
from cfveg.validation.evaluate import (baseline_variants, bootstrap_ci, deterministic_variant,  # noqa: E402
                                       diffusion_variant, evaluate, evaluate_observed, paired_bootstrap)

MAIN_TEST = ["observed", "persistence", "climatology", "clim_anomaly", "det", "diff_plain", "diff_phys+G+P"]
ABLATION = ["diff_plain+G", "diff_plain+G+P", "diff_phys", "diff_phys+G", "diff_phys_no_rate+G+P",
            "diff_phys_no_phen+G+P", "diff_phys_no_mono+G+P"]


class Ctx:
    def __init__(self, preset: str, smoke: bool, seed: int):
        self.cfg = load_preset(preset)
        self.preset, self.smoke, self.seed = preset, smoke, seed
        self.pdir = ROOT / ("data/smoke" if smoke else self.cfg["data"]["processed_dir"])
        self.cube = Cube.load(self.pdir / "cube")
        self.stats = json.loads((self.pdir / "stats.json").read_text())
        self.phys_fit = json.loads((self.pdir / "physics.json").read_text())
        self.clim = np.load(self.pdir / "clim.npy")
        self.phys = Physics(self.phys_fit, self.cfg)
        self.diff = Diffusion(self.cfg["diffusion"]["timesteps"], self.cfg["diffusion"]["prediction"])
        self.K, self.H = self.cfg["window"]["context"], self.cfg["window"]["horizon"]
        self.wx_mean = torch.tensor(self.stats["wx_mean"])
        self.wx_std = torch.tensor(self.stats["wx_std"])
        self.runs = ROOT / "runs" / (preset + ("_smoke" if smoke else ""))
        g_f = RESULTS / ("smoke/" if smoke else "") / "val" / "guidance_choice.json"
        if g_f.exists():   # guidance scale selected on VALIDATION (run_pipeline stage 5)
            self.cfg["physics"]["guidance_scale"] = json.loads(g_f.read_text())["guidance_scale"]
        s_f = g_f.with_name("spread_choice.json")   # post-hoc spread factors fitted on VALIDATION
        self.spread = json.loads(s_f.read_text()) if s_f.exists() else {}
        e_f = g_f.with_name("eta_choice.json")
        self.cfg["diffusion"]["eta"] = json.loads(e_f.read_text())["eta"] if e_f.exists() else 0.0

    def model(self, run: str, kind: str = "diffusion"):
        d = self.runs / f"{run}_s{self.seed}"
        return load_model(self.cfg, d / "final.pt", kind)

    def sampler(self, run: str, guidance: bool, projection: bool, scale: float | None,
                eta: float | None = None) -> Sampler:
        p = self.cfg["physics"]
        cons = tuple(p["constraints"])
        m = re.match(r"diff_phys_no_(\w+)", run)
        if m:
            cons = tuple(c for c in cons if c != m.group(1))
        o = SamplerOpts(guidance=guidance, guidance_scale=scale if scale is not None else p["guidance_scale"],
                        projection=projection, ddim_steps=self.cfg["diffusion"]["ddim_steps"],
                        lambda_rate=p["lambda_rate"], lambda_phen=p["lambda_phen"], lambda_range=p["lambda_range"],
                        constraints=cons, eta=self.cfg["diffusion"].get("eta", 0.0) if eta is None else eta,
                        projection_rate=p.get("projection_rate", True))
        return Sampler(self.model(run), self.diff, self.phys, o)

    def variant(self, spec: str):
        base = {v.name: v for v in baseline_variants(self.clim, self.K, self.H)}
        if spec in base:
            return base[spec]
        if spec == "det":
            return deterministic_variant(self.model("det", "deterministic"))
        parts = spec.split("+")
        run, guidance, projection, scale, eta = parts[0], False, False, None, None
        for p in parts[1:]:
            if p.startswith("Ex"):
                eta = float(p[2:])
            elif p.startswith("G"):
                guidance = True
                if p.startswith("Gx"):
                    scale = float(p[2:])
            elif p == "P":
                projection = True
        key = spec.split("+Ex")[0].replace("+", "_")
        sp = self.spread.get(f"s{self.seed}", {}).get(key, 1.0) if not getattr(self, "no_spread", False) else 1.0
        v = diffusion_variant(self.sampler(run, guidance, projection, scale, eta), spec, spread=sp)
        return v


def eval_batches(ctx: Ctx, split: str, n_windows: int, batch_size: int = 16, seed: int = 0) -> list[dict]:
    win = make_windows(ctx.cube, ctx.cfg, split)
    rng = np.random.default_rng(seed)
    if len(win) > n_windows:
        win = win[np.sort(rng.choice(len(win), n_windows, replace=False))]
    ds = WindowDataset(ctx.cube, ctx.cfg, win, ctx.stats)
    return [collate([ds[i] for i in range(j, min(j + batch_size, len(ds)))]) for j in range(0, len(ds), batch_size)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--n_windows", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="real-data test fixture")
    ap.add_argument("--reference", default="diff_phys+G+P", help="variant for paired comparisons")
    ap.add_argument("--no_spread", action="store_true", help="ignore spread_choice.json (for fitting it)")
    a = ap.parse_args()
    ctx = Ctx(a.preset, a.smoke, a.seed)
    ctx.no_spread = a.no_spread
    torch.set_num_threads(ctx.cfg["train"].get("threads", 8))
    variants = a.variants or (MAIN_TEST + ABLATION)
    tag = a.tag or f"{a.split}"
    out = RESULTS / (("smoke/" if a.smoke else "") + tag)
    out.mkdir(parents=True, exist_ok=True)
    nw = a.n_windows or ctx.cfg["eval"]["n_eval_windows"]
    M = a.members or ctx.cfg["eval"]["n_members"]
    batches = eval_batches(ctx, a.split, nw)
    print(f"{a.split}: {sum(b['tgt'].shape[0] for b in batches)} windows, {M} members", flush=True)
    rows, per = [], {}
    mpath = out / "metrics.csv"
    old = pd.read_csv(mpath) if mpath.exists() else None
    for spec in variants:
        t = time.time()
        if spec == "observed":
            res = evaluate_observed(batches, ctx.phys)
            res.update({"seed": a.seed, "members": 0, "secs": round(time.time() - t, 1)})
            rows.append(res)
            print(f"observed: rate={res['rate_frac']:.4f} phen={res['phen_frac']:.4f}", flush=True)
            continue
        try:
            v = ctx.variant(spec)
        except FileNotFoundError as e:
            print(f"skip {spec}: {e}", flush=True)
            continue
        res, pw = evaluate(v, batches, ctx.phys, M, ctx.wx_mean, ctx.wx_std, ctx.K, seed=a.seed)
        res["seed"] = a.seed
        res["members"] = M if v.probabilistic else 1
        res["secs"] = round(time.time() - t, 1)
        res.pop("rank_hist", None) if False else None
        safe = spec.replace("+", "_")
        pw.to_csv(out / f"per_window_{safe}_s{a.seed}.csv", index=False)
        if "rank_hist" in res:
            (out / f"rank_hist_{safe}_s{a.seed}.json").write_text(json.dumps(res.pop("rank_hist")))
        per[spec] = pw
        rows.append(res)
        print(f"{spec}: mae={res.get('mae', float('nan')):.4f} crps={res.get('crps', float('nan')):.4f} "
              f"rate={res.get('rate_frac', float('nan')):.4f} phen={res.get('phen_frac', float('nan')):.4f} "
              f"mono={res.get('mono_frac', float('nan'))} grad_ratio={res.get('grad_ratio', float('nan')):.3f} "
              f"({res['secs']}s)", flush=True)
    df = pd.DataFrame(rows)
    if old is not None:
        df = pd.concat([old[~((old.variant.isin(df.variant)) & (old.seed == a.seed))], df], ignore_index=True)
    df.to_csv(mpath, index=False)
    write_provenance(mpath, config=ctx.cfg, seed=a.seed, extra={"split": a.split, "n_windows": nw, "members": M,
                                                                 "smoke_fixture": a.smoke})
    # bootstrap CIs over locations and paired tests vs reference
    ci_rows, pt_rows = [], []
    for f in sorted(out.glob(f"per_window_*_s{a.seed}.csv")):
        name = f.stem.removeprefix("per_window_").removesuffix(f"_s{a.seed}")
        pw = pd.read_csv(f)
        for metric in ["mae", "rmse", "crps"]:
            pt, lo, hi = bootstrap_ci(pw, metric)
            ci_rows.append({"variant": name, "metric": metric, "value": pt, "lo95": lo, "hi95": hi, "seed": a.seed})
    ref = out / f"per_window_{a.reference.replace('+', '_')}_s{a.seed}.csv"
    if ref.exists():
        rdf = pd.read_csv(ref)
        for f in sorted(out.glob(f"per_window_*_s{a.seed}.csv")):
            if f == ref:
                continue
            name = f.stem.removeprefix("per_window_").removesuffix(f"_s{a.seed}")
            for metric in ["mae", "crps"]:
                r = paired_bootstrap(rdf, pd.read_csv(f), metric)
                pt_rows.append({"reference": a.reference, "other": name, "metric": metric, **r, "seed": a.seed})
    for name, rr in [("metrics_ci.csv", ci_rows), ("paired_tests.csv", pt_rows)]:
        if rr:
            new = pd.DataFrame(rr)
            if (out / name).exists():      # keep other seeds' rows
                old_df = pd.read_csv(out / name)
                new = pd.concat([old_df[old_df.seed != a.seed], new], ignore_index=True)
            new.to_csv(out / name, index=False)
            write_provenance(out / name, seed=a.seed, extra={"split": a.split, "bootstrap": "1000 x locations"})


if __name__ == "__main__":
    main()
