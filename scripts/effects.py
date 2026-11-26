"""Compute per-window counterfactual effects on test windows for one or more forecasters.

  python scripts/effects.py --preset cpu --variants diff_phys+G+P diff_phys+G+P+C diff_plain det
  (+C = monotone coupling of the drought branch to the normal branch, delta = mono_eps)
Outputs results/effects/effects_<variant>_s<seed>.csv (+ provenance) and a few showcase windows
(pixel-level arrays) in results/effects/showcase_<variant>.npz (git-ignored if large).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from eval_all import Ctx  # noqa: E402

from cfveg.counterfactual.effects import (det_forecast_fn, effects_table, sampler_forecast_fn,  # noqa: E402
                                          window_effects)
from cfveg.data.dataset import WindowDataset, make_windows  # noqa: E402
from cfveg.data.scenarios import ScenarioBuilder  # noqa: E402
from cfveg.utils.provenance import write_provenance  # noqa: E402
from cfveg.validation.framework import cross_year_proxy, did_pixels  # noqa: E402


def effect_windows(ctx: Ctx, split: str, t0_stride: int, max_windows: int, seed: int = 0) -> np.ndarray:
    """One random quadrant per (location, year, t0) with t0 on a coarse stride."""
    w = make_windows(ctx.cube, ctx.cfg, split, stride=t0_stride)
    rng = np.random.default_rng(seed)
    keys = {}
    for row in w:
        keys.setdefault(tuple(row[:3]), []).append(row)
    sel = np.array([opts[rng.integers(len(opts))] for _, opts in sorted(keys.items())])
    if len(sel) > max_windows:
        sel = sel[np.sort(rng.choice(len(sel), max_windows, replace=False))]
    return sel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--split", default="test")
    ap.add_argument("--variants", nargs="*", default=["diff_phys+G+P", "diff_phys+G+P+C", "diff_phys", "diff_plain", "det"])
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--t0_stride", type=int, default=4)
    ap.add_argument("--max_windows", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="real-data test fixture")
    a = ap.parse_args()
    ctx = Ctx(a.preset, a.smoke, a.seed)
    torch.set_num_threads(ctx.cfg["train"].get("threads", 8))
    M = a.members or ctx.cfg["eval"]["n_members"]
    win = effect_windows(ctx, a.split, a.t0_stride, a.max_windows, a.seed)
    ds = WindowDataset(ctx.cube, ctx.cfg, win, ctx.stats)
    sb = ScenarioBuilder(ctx.cube, ctx.cfg)
    out = ROOT / "results" / ("smoke/" if a.smoke else "") / ("effects" if a.split == "test" else f"effects_{a.split}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(win)} windows, {M} members", flush=True)
    for spec in a.variants:
        t = time.time()
        coupled = spec.endswith("+C")
        base = spec[:-2] if coupled else spec
        if base == "det":
            fn = det_forecast_fn(ctx.model("det", "deterministic"))
        else:
            parts = base.split("+")
            fn = sampler_forecast_fn(ctx.sampler(parts[0], "G" in parts, "P" in parts, None))
        delta = ctx.cfg["physics"]["mono_eps"] if coupled else None
        recs, show = [], {}
        for i in range(len(ds)):
            s = ds[i]
            l, y, t0, oy, ox = (int(v) for v in win[i])
            r = window_effects(fn, s, sb, l, y, t0, M, ctx.wx_mean, ctx.wx_std, ctx.K, seed=a.seed * 100000 + i,
                               coupling_delta=delta)
            r.update({"loc": l, "year": ctx.cube.years[y], "t0": t0, "oy": oy, "ox": ox,
                      "drought": float(s["drought"])})
            if len(show) < 6 and r["drought"] == 1.0:
                show[f"w{i}"] = {"xN": r["xN"], "xD": r["xD"], "xO": r["xO"], "obs_ctx": s["ctx"].numpy(),
                                 "tgt": s["tgt"].numpy(), "tgt_mask": s["tgt_mask"].numpy(), "idx": win[i]}
            last = s["last_ndvi"].numpy()[None]
            K_, H_, P_ = ctx.K, ctx.H, s["tgt"].shape[-1]
            clim = ctx.clim[l, :, oy:oy + P_, ox:ox + P_]
            fc = {"normal": r["xN"].mean(0) - last, "factual": r["xO"].mean(0) - last,
                  "persistence": np.zeros_like(r["xN"][0]),
                  "climatology": clim[t0 + K_:t0 + K_ + H_] - clim[t0 + K_ - 1][None]}
            anom = r["xO"].mean(0) - r["xN"].mean(0)
            r.update(cross_year_proxy(ctx.cube, ctx.cfg, l, y, t0, oy, ox, fc, model_anom=anom))
            # model effect on the natural-experiment (DiD) pixel set: horizon mean over pixels observed
            # both before and after the forecast start
            seq = np.asarray(ctx.cube.ndvi[l, y, t0:t0 + K_ + H_, oy:oy + P_, ox:ox + P_], dtype=np.float32)
            _, _, msk = did_pixels(seq, K_)
            if msk.any():
                r["dON_did"] = (r["xO"] - r["xN"]).mean(1)[:, msk].mean(1)
            for k in ["xN", "xD", "xO"]:
                r.pop(k)
            recs.append(r)
            if i % 25 == 0:
                print(f"  {spec} {i}/{len(ds)} ({time.time() - t:.0f}s)", flush=True)
        safe = spec.replace("+", "_")
        df = effects_table(recs)
        f = out / f"effects_{safe}_s{a.seed}.csv"
        df.to_csv(f, index=False)
        write_provenance(f, config=ctx.cfg, seed=a.seed, extra={"variant": spec, "members": M, "split": a.split,
                                                                 "coupled": coupled})
        np.savez_compressed(out / f"showcase_{safe}_s{a.seed}.npz",
                            **{f"{w}_{k}": v for w, d in show.items() for k, v in d.items()})
        print(f"{spec}: done in {time.time() - t:.0f}s; mean dDN={df.dDN_mean.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
