"""Fit the post-hoc calibration (bias correction + spread factor) on VALIDATION windows only.

Rule (fixed 2026-11-27 ~18:15 IST, before any result on the reserved test locations):
  bias[h] = mean over valid validation pixels of (ensemble mean - observation) at lead h (NDVI)
  k       = 1 / spread-skill ratio of the bias-corrected ensemble on validation (so validation spread-skill = 1)
Fitted per training seed and variant from the raw ensembles (no spread_choice factor). Written per seed to
<results>/val/calib_choice_s<seed>.json, then merged into calib_choice.json.
Why not a coverage-targeted width: with M members the empirical 5-95 % quantile interval of a perfectly calibrated
ensemble covers clearly less than 90 % (even [min, max] covers only (M-1)/(M+1), 78 % for M = 8), so forcing 90 %
coverage over-disperses the ensemble (smoke test: spread-skill 2.4). Coverage is instead evaluated with 32 members.

  python scripts/fit_calibration.py --preset v2 --seed 0 [--n_windows 150]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from eval_all import RESULTS, Ctx, eval_batches  # noqa: E402

from cfveg.data.dataset import scaled_to_ndvi  # noqa: E402
from cfveg.utils.provenance import write_provenance  # noqa: E402
from cfveg.validation import metrics as M  # noqa: E402
from cfveg.validation.evaluate import calibrate  # noqa: E402

VARIANTS = ["diff_phys+G+P", "diff_plain"]
GRID = np.round(np.arange(0.80, 4.0001, 0.02), 2)


def cov90(ens_l, obs_l, mask_l) -> float:
    s = n = 0.0
    for e, o, m in zip(ens_l, obs_l, mask_l):
        c = M.coverage(e, o, m, levels=(0.9,))
        s += c["cov90_sum"]
        n += c["n"]
    return s / max(n, 1)


def spread_skill(ens_l, obs_l, mask_l) -> float:
    v = q = 0.0
    for e, o, m in zip(ens_l, obs_l, mask_l):
        r = M.spread_skill(e, o, m)
        v += r["var_sum"]
        q += r["sqerr_sum"]
    return float(np.sqrt(v / max(q, 1e-12)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_windows", type=int, default=150)
    ap.add_argument("--members", type=int, default=16)
    a = ap.parse_args()
    ctx = Ctx(a.preset, False, a.seed)
    ctx.no_spread = True
    torch.set_num_threads(ctx.cfg["train"].get("threads", 5))
    batches = eval_batches(ctx, "val", a.n_windows)
    out = {}
    for spec in VARIANTS:
        v = ctx.variant(spec)
        ens_l, obs_l, mask_l = [], [], []
        for bi, b in enumerate(batches):
            ens_l.append(v.forecast(b, a.members, a.seed + 1000 * bi))
            obs_l.append(scaled_to_ndvi(b["tgt"]))
            mask_l.append(b["tgt_mask"])
        H = obs_l[0].shape[1]
        num, den = np.zeros(H), np.zeros(H)
        for e, o, m in zip(ens_l, obs_l, mask_l):
            d = (e.mean(0) - o) * m
            num += d.sum(dim=(0, 2, 3)).numpy()
            den += m.sum(dim=(0, 2, 3)).numpy()
        bias = (num / np.maximum(den, 1)).tolist()
        ss_b = spread_skill([calibrate(e, 1.0, bias) for e in ens_l], obs_l, mask_l)
        k_sel, c_sel = float(1.0 / ss_b), None
        cal = [calibrate(e, k_sel, bias) for e in ens_l]
        rec = {"k": k_sel, "bias": bias, "val_cov90_raw": cov90(ens_l, obs_l, mask_l),
               "val_cov90_cal": c_sel if c_sel is not None else cov90(cal, obs_l, mask_l),
               "val_spread_skill_raw": spread_skill(ens_l, obs_l, mask_l),
               "val_spread_skill_cal": spread_skill(cal, obs_l, mask_l),
               "n_windows": sum(o.shape[0] for o in obs_l)}
        out[spec.replace("+", "_")] = rec
        print(f"seed {a.seed} {spec}: k={k_sel} bias={np.round(bias, 4).tolist()} "
              f"cov90 {rec['val_cov90_raw']:.3f}->{rec['val_cov90_cal']:.3f} "
              f"spread-skill {rec['val_spread_skill_raw']:.2f}->{rec['val_spread_skill_cal']:.2f}", flush=True)
    f = RESULTS / "val" / f"calib_choice_s{a.seed}.json"   # per seed (parallel runs); merged afterwards
    f.parent.mkdir(parents=True, exist_ok=True)
    allc = {}
    allc[f"s{a.seed}"] = out
    allc["rule"] = ("bias per lead = mean(ensemble mean - obs) on validation; k = 1 / spread-skill of the "
                    "bias-corrected validation ensemble")
    f.write_text(json.dumps(allc, indent=1))
    write_provenance(f, config=ctx.cfg, seed=a.seed, extra={"split": "val", "n_windows": a.n_windows})


if __name__ == "__main__":
    main()
