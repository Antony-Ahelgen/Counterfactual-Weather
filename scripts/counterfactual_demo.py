"""Phase 6 demo: counterfactual maps with uncertainty for one test scene (one command).

  python scripts/counterfactual_demo.py --preset cpu [--variant diff_phys+G+P]
Picks the test window with the strongest observed drought stress, runs paired sampling
(normal analog ensemble vs observed drought weather) with common random numbers, and also with
independent noise to quantify the CRN variance reduction. Saves
results/counterfactual/{demo.json, demo_maps.pdf} (+ provenance).
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

from eval_all import Ctx  # noqa: E402

from cfveg.counterfactual.paired import estimate_effect, summarize_delta  # noqa: E402
from cfveg.data.dataset import WindowDataset, make_windows, scaled_to_ndvi  # noqa: E402
from cfveg.data.scenarios import ScenarioBuilder  # noqa: E402
from cfveg.utils import plotting  # noqa: E402
from cfveg.utils.provenance import write_provenance  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--variant", default="diff_phys+G+P")
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ctx = Ctx(a.preset, a.smoke, a.seed)
    torch.set_num_threads(ctx.cfg["train"].get("threads", 8))
    M = a.members or ctx.cfg["eval"]["n_members"]
    sb = ScenarioBuilder(ctx.cube, ctx.cfg)
    win = make_windows(ctx.cube, ctx.cfg, "test")
    stress = np.array([ctx.cube.wx[l, y, t0 + ctx.K:t0 + ctx.K + ctx.H, -1].mean() for l, y, t0, _, _ in win])
    cov = np.array([np.isfinite(ctx.cube.ndvi[l, y, t0 + ctx.K:t0 + ctx.K + ctx.H, oy:oy + ctx.cfg["window"]["patch"], ox:ox + ctx.cfg["window"]["patch"]]).mean()
                    for l, y, t0, oy, ox in win])
    score = np.where(cov > 0.5, stress, -np.inf)
    i = int(np.argmax(score))
    l, yi, t0, oy, ox = (int(v) for v in win[i])
    ds = WindowDataset(ctx.cube, ctx.cfg, win[i:i + 1], ctx.stats)
    s = ds[0]
    parts = a.variant.split("+")
    sampler = ctx.sampler(parts[0], "G" in parts, "P" in parts, None)
    normal, n_method = sb.normal(l, yi, t0)
    drought, d_method = sb.drought(l, yi, t0)
    r = estimate_effect(sampler, s, normal, drought, M, ctx.wx_mean, ctx.wx_std, ctx.K, crn=True, seed=a.seed)
    r_ind = estimate_effect(sampler, s, normal, drought, M, ctx.wx_mean, ctx.wx_std, ctx.K, crn=False, seed=a.seed)
    sm = summarize_delta(r["delta"])
    sm_ind = summarize_delta(r_ind["delta"])
    out = ROOT / "results" / ("smoke/" if a.smoke else "") / "counterfactual"
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "variant": a.variant, "members": M, "loc_id": ctx.cube.meta["loc_id"][l], "year": ctx.cube.years[yi],
        "t0_bin": t0, "quadrant": [oy, ox], "normal_method": n_method, "n_normal_analogs": len(normal),
        "drought_method": d_method, "horizon_stress_drought": sb.horizon_stress(drought[0]),
        "horizon_stress_normal_mean": float(np.mean([sb.horizon_stress(w) for w in normal])),
        "area_mean_delta_crn": sm["area_mean"], "frac_sig_loss_crn": sm["frac_sig_loss"],
        "member_area_delta_sd_crn": float(np.std(sm["area_mean_members"], ddof=1)),
        "member_area_delta_sd_independent": float(np.std(sm_ind["area_mean_members"], ddof=1)),
        "pixel_delta_var_crn": float(np.var(r["delta"], axis=0, ddof=1).mean()),
        "pixel_delta_var_independent": float(np.var(r_ind["delta"], axis=0, ddof=1).mean()),
    }
    res["crn_variance_ratio_pixel"] = res["pixel_delta_var_crn"] / res["pixel_delta_var_independent"]
    f = out / "demo.json"
    f.write_text(json.dumps(res, indent=1))
    write_provenance(f, config=ctx.cfg, seed=a.seed)
    # ---- figure: rows = horizon frames 2, 5, 8; cols = observed, normal, drought, delta mean, delta sd
    plotting.setup()
    import matplotlib.pyplot as plt

    obs = scaled_to_ndvi(s["tgt"]).numpy()
    obs = np.where(s["tgt_mask"].numpy() > 0, obs, np.nan)
    xa, xb = r["xa"].mean(0), r["xb"].mean(0)
    H = obs.shape[0]
    frames = sorted({1, H // 2, H - 1})
    fig, ax = plt.subplots(len(frames), 5, figsize=(plotting.COL_W, 0.95 * len(frames) + 0.9), layout="constrained")
    vmax_d = max(0.02, float(np.nanpercentile(np.abs(sm["mean"]), 98)))
    titles = ["Observed", "Normal fc.", "Drought fc.", "Δ mean", "Δ s.d."]
    ims = []
    for rr, h in enumerate(frames):
        for cc, (img, cmap, vmin, vmax) in enumerate([(obs[h], plotting.SEQ_CMAP, 0, 0.9),
                                                      (xa[h], plotting.SEQ_CMAP, 0, 0.9),
                                                      (xb[h], plotting.SEQ_CMAP, 0, 0.9),
                                                      (sm["mean"][h], plotting.DIV_CMAP, -vmax_d, vmax_d),
                                                      (sm["sd"][h], "Greys", 0, vmax_d)]):
            im = ax[rr, cc].imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
            ax[rr, cc].set_facecolor("0.85")   # grey = no cloud-free observation
            ax[rr, cc].set_xticks([])
            ax[rr, cc].set_yticks([])
            if rr == 0:
                ax[rr, cc].set_title(titles[cc])
                ims.append(im)
        ax[rr, 0].set_ylabel(f"+{5 * (h + 1)} d")
    for cc, lab in [(2, "NDVI"), (3, "Δ NDVI"), (4, "s.d.")]:
        cb = fig.colorbar(ims[cc], ax=ax[-1, cc] if cc > 2 else ax[-1, :3], location="bottom", shrink=0.9, label=lab)
        if cc > 2:
            from matplotlib.ticker import MaxNLocator
            cb.locator = MaxNLocator(3)
            cb.update_ticks()
    fig.suptitle(f"{res['loc_id']} {res['year']}: drought ({d_method}) vs normal ({n_method}, "
                 f"{len(normal)} analogs); {M} members, CRN", fontsize=6)
    plotting.save(fig, out / "demo_maps")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
