"""Build paper tables (LaTeX) and figures (vector PDF, grayscale-safe, >= 6 pt) from results/ only.

  python scripts/paper_assets.py [--smoke]
Outputs results/tables/*.tex and results/figures/*.pdf (+png previews).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
# results root: env CFVEG_RESULTS (e.g. results/full for the full-scale run), default results/
RESULTS = Path(__import__("os").environ.get("CFVEG_RESULTS", str(ROOT / "results")))
if not RESULTS.is_absolute():
    RESULTS = ROOT / RESULTS
sys.path.insert(0, str(ROOT / "src"))

from cfveg.utils import plotting  # noqa: E402
from cfveg.utils.provenance import write_provenance  # noqa: E402

NAMES = {
    "persistence": "Persistence", "climatology": "Climatology", "clim_anomaly": "Clim.+anomaly",
    "det": "Deterministic U-Net", "diff_plain": "Diffusion (no physics)", "diff_plain_G": "Diff. + guidance",
    "diff_plain_G_P": "Diff. + guid. + proj.", "diff_phys": "Diff. + phys. loss",
    "diff_phys_G": "Diff. + loss + guid.", "diff_phys_G_P": "Physics-constrained (proposed)",
    "diff_phys_no_rate_G_P": "  w/o rate", "diff_phys_no_phen_G_P": "  w/o phenology",
    "diff_phys_no_mono_G_P": "  w/o monotonicity", "diff_phys_G_P_C": "Proposed + mono. coupling",
}
MAIN = ["persistence", "climatology", "clim_anomaly", "det", "diff_plain", "diff_phys_G_P"]
ABL = ["diff_plain", "diff_plain_G", "diff_plain_G_P", "diff_phys", "diff_phys_G", "diff_phys_G_P",
       "diff_phys_no_rate_G_P", "diff_phys_no_phen_G_P", "diff_phys_no_mono_G_P"]


def safe(v: str) -> str:
    return v.replace("+", "_")


def fmt(x, d=3):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{d}f}"


def tex_table(df: pd.DataFrame, cols: list[tuple[str, str, int]], caption: str, label: str) -> str:
    head = " & ".join(["Model"] + [c[1] for c in cols]) + r" \\"
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{" + caption + "}", r"\label{" + label + "}",
             r"\scriptsize", r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\hline", head, r"\hline"]
    for _, r in df.iterrows():
        lines.append(" & ".join([NAMES.get(r["key"], r["key"])] + [r.get(f"_{c[0]}", fmt(r.get(c[0]), c[2]))
                                                                      for c in cols]) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    base = RESULTS / ("smoke" if a.smoke else "")
    tdir, fdir = base / "tables", base / "figures"
    tdir.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    plotting.setup()
    import matplotlib.pyplot as plt

    m = pd.read_csv(base / "test" / "metrics.csv")
    m = m[m.seed == a.seed].copy()
    m["key"] = m.variant.map(safe)
    ci = pd.read_csv(base / "test" / "metrics_ci.csv") if (base / "test" / "metrics_ci.csv").exists() else None
    if ci is not None and "seed" in ci:
        ci = ci[ci.seed == a.seed]
    if ci is not None:
        for metric in ["mae", "crps"]:
            c = ci[ci.metric == metric].set_index("variant")
            m[f"_{metric}"] = [f"{c.loc[k, 'value']:.3f} [{c.loc[k, 'lo95']:.3f}, {c.loc[k, 'hi95']:.3f}]"
                               if k in c.index else fmt(r) for k, r in zip(m.key, m[metric])]
    # ---------- Table 1: main comparison
    t1 = m[m.key.isin(MAIN)].set_index("key").loc[[k for k in MAIN if k in set(m.key)]].reset_index()
    cols = [("mae", "MAE [95\\% CI]", 3), ("crps", "CRPS [95\\% CI]", 3), ("ssim", "SSIM", 3),
            ("grad_ratio", "Grad. ratio", 2), ("hf_ratio", "HF ratio", 2), ("rate_frac", "Rate viol.", 3),
            ("phen_frac", "Env. viol.", 3), ("mono_frac", "Mono. viol.", 3)]
    (tdir / "main.tex").write_text(tex_table(t1, cols, "Held-out test locations in test years (2021, 2022). MAE/CRPS "
                                             "in NDVI; ratios are prediction/observation (1 = as sharp as observed); "
                                             "violation rates are fractions of pixel-timesteps.", "tab:main"))
    # ---------- Table 2: ablation
    t2 = m[m.key.isin(ABL)].set_index("key").loc[[k for k in ABL if k in set(m.key)]].reset_index()
    cols2 = [("mae", "MAE", 3), ("crps", "CRPS", 3), ("grad_ratio", "Grad. ratio", 2), ("rate_frac", "Rate viol.", 3),
             ("phen_frac", "Env. viol.", 3), ("mono_frac", "Mono. viol.", 3), ("members", "Members", 0)]
    (tdir / "ablation.tex").write_text(tex_table(t2, cols2, "Physics ablation on the test split.", "tab:ablation"))
    # ---------- Fig: skill with CIs
    if ci is not None:
        fig, axs = plt.subplots(1, 2, figsize=(plotting.COL_W, 1.9), layout="constrained")
        keys = [k for k in MAIN if k in set(ci.variant)]
        for ax, metric in zip(axs, ["mae", "crps"]):
            c = ci[ci.metric == metric].set_index("variant").loc[keys]
            y = np.arange(len(keys))
            ax.errorbar(c.value, y, xerr=[c.value - c.lo95, c.hi95 - c.value], fmt="o", color="0", ms=3, capsize=2)
            ax.set_yticks(y)
            ax.set_yticklabels([NAMES[k] for k in keys] if metric == "mae" else [])
            ax.set_xlabel(f"{metric.upper()} (NDVI)")
            ax.invert_yaxis()
            ax.grid(axis="x", color="0.85", lw=0.5)
        plotting.save(fig, fdir / "fig_skill")
    # ---------- Fig: violations + sharpness (ablation variants)
    t2v = t2.copy()
    if len(t2v):
        fig, axs = plt.subplots(1, 2, figsize=(plotting.COL_W, 2.2), layout="constrained")
        y = np.arange(len(t2v))
        hatches = ["", "////", "...."]
        for j, (col, lab) in enumerate([("rate_frac", "rate"), ("phen_frac", "envelope"), ("mono_frac", "monotone")]):
            if col in t2v:
                axs[0].barh(y + (j - 1) * 0.27, t2v[col].fillna(0), height=0.27, color=["0.15", "0.55", "0.85"][j],
                            hatch=hatches[j], edgecolor="0", lw=0.3, label=lab)
        axs[0].set_yticks(y)
        axs[0].set_yticklabels([NAMES.get(k, k) for k in t2v.key])
        axs[0].invert_yaxis()
        axs[0].set_xlabel("Violation fraction")
        axs[0].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
        if "grad_ratio" in t2v:
            axs[1].barh(y - 0.2, t2v.grad_ratio, height=0.4, color="0.3", label="gradient")
            axs[1].barh(y + 0.2, t2v.hf_ratio, height=0.4, color="0.75", hatch="////", edgecolor="0", lw=0.3,
                        label="high-freq.")
            axs[1].axvline(1, color="0", ls="--", lw=0.8)
            axs[1].set_yticks(y)
            axs[1].set_yticklabels([])
            axs[1].invert_yaxis()
            axs[1].set_xlabel("Sharpness (pred./obs.)")
            axs[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2)
        plotting.save(fig, fdir / "fig_physics_ablation")
    # ---------- Fig: rank histograms
    hs = {k: json.loads((base / "test" / f"rank_hist_{k}_s{a.seed}.json").read_text())
          for k in ["diff_plain", "diff_phys_G_P"] if (base / "test" / f"rank_hist_{k}_s{a.seed}.json").exists()}
    if hs:
        fig, axs = plt.subplots(1, len(hs), figsize=(plotting.COL_W * 0.75, 1.5), squeeze=False, layout="constrained")
        for ax, (k, h) in zip(axs[0], hs.items()):
            ax.bar(np.arange(len(h)), h, color="0.5", width=0.9)
            ax.axhline(1 / len(h), color="0", ls="--", lw=0.8)
            ax.set_title(NAMES[k])
            ax.set_xlabel("Rank of observation")
        axs[0, 0].set_ylabel("Frequency")
        plotting.save(fig, fdir / "fig_rank_hist")
    # ---------- Fig: guidance sweep (validation)
    g = base / "val_guidance" / "metrics.csv"
    if g.exists():
        gm = pd.read_csv(g)
        gm["gamma"] = [float(v.split("Gx")[1].split("+")[0]) if "Gx" in v else 0.0 for v in gm.variant]
        gm = gm.sort_values("gamma")
        fig, ax1 = plt.subplots(figsize=(plotting.COL_W * 0.6, 1.8), layout="constrained")
        xs = np.arange(len(gm))
        ax1.plot(xs, gm.crps, "o-", color="0", ms=3, label="CRPS")
        ax2 = ax1.twinx()
        ax2.plot(xs, gm.rate_frac + gm.phen_frac, "s--", color="0.5", ms=3, label="rate+env. viol.")
        ax1.set_xticks(xs)
        ax1.set_xticklabels([f"{v:g}" for v in gm.gamma])
        ax1.set_xlabel("Guidance scale γ (validation)")
        ax1.set_ylabel("CRPS")
        ax2.set_ylabel("Violation fraction")
        fig.legend(frameon=False, loc="upper center", ncol=2)
        plotting.save(fig, fdir / "fig_guidance_sweep")
    # ---------- Fig: placebo / effects distributions and natural experiment
    vdir = base / "validation"
    eff = base / "effects" / f"effects_diff_phys_G_P_s{a.seed}.csv"
    if eff.exists():
        e = pd.read_csv(eff)
        fig, axs = plt.subplots(1, 2, figsize=(plotting.COL_W, 1.9), layout="constrained")
        bins = np.linspace(min(e[["dON_mean", "dN2_mean", "dNi_mean"]].min().min(), -0.01),
                           max(e[["dON_mean", "dN2_mean", "dNi_mean"]].max().max(), 0.01), 30)
        axs[0].hist(e.loc[e.drought == 1, "dON_mean"], bins=bins, histtype="stepfilled", color="0.3", alpha=0.8,
                    label="drought windows")
        axs[0].hist(e.loc[e.drought == 0, "dON_mean"], bins=bins, histtype="step", color="0", ls="--",
                    label="non-drought windows")
        axs[0].hist(e.loc[e.n_analogs >= 2, "dN2_mean"], bins=bins, histtype="step", color="0.5", ls=":", lw=1.2,
                    label="scenario placebo")
        axs[0].axvline(0, color="0", lw=0.6)
        axs[0].set_xlabel("Estimated effect Δ (NDVI)")
        axs[0].set_ylabel("Windows")
        axs[0].legend(frameon=False, loc="upper left", fontsize=5.5)
        pairs = vdir / "natexp_pairs.csv"
        if pairs.exists():
            pr = pd.read_csv(pairs)
            axs[1].errorbar(pr.did, pr.model_mean, yerr=[pr.model_mean - pr.model_q05, pr.model_q95 - pr.model_mean],
                            fmt="o", ms=2.5, color="0", ecolor="0.6", elinewidth=0.6)
            lim = [min(pr.did.min(), pr.model_q05.min()), max(pr.did.max(), pr.model_q95.max())]
            axs[1].plot(lim, lim, "--", color="0.4", lw=0.8)
            axs[1].set_xlabel("Observed DiD effect (NDVI)")
            axs[1].set_ylabel("Model effect (NDVI)")
            axs[1].set_title("Natural experiment (matched)")
        plotting.save(fig, fdir / "fig_validation")
    # ---------- Table 3: validation summary for proposed vs unconstrained
    vs = vdir / "validation_summary.csv"
    if vs.exists():
        s = pd.read_csv(vs)
        keep = ["2_natural_experiment", "3_placebo"]
        s = s[s.part.isin(keep) & s.variant.isin(["diff_phys_G_P", "diff_plain", "det", "diff_phys_G_P_C"])]
        pv = s.pivot_table(index="metric", columns="variant", values="value", aggfunc="first")
        lines = [r"\begin{table}[t]", r"\centering", r"\caption{Validation without ground truth (test split).}",
                 r"\label{tab:validation}", r"\scriptsize", r"\begin{tabular}{l" + "r" * len(pv.columns) + "}",
                 r"\hline", " & ".join(["Metric"] + [NAMES.get(c, c) for c in pv.columns]) + r" \\", r"\hline"]
        for metric, row in pv.iterrows():
            lines.append(" & ".join([metric.replace("_", r"\_")] + [fmt(v, 3) for v in row.values]) + r" \\")
        lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
        (tdir / "validation.tex").write_text("\n".join(lines) + "\n")
    # copy data and demo figures
    for src, dst in [(base / "data" / "sanity_timeseries.pdf", fdir / "fig_data_timeseries.pdf"),
                     (base / "counterfactual" / "demo_maps.pdf", fdir / "fig_counterfactual_maps.pdf")]:
        if src.exists():
            shutil.copy(src, dst)
            if src.with_suffix(".png").exists():
                shutil.copy(src.with_suffix(".png"), dst.with_suffix(".png"))
    write_provenance(tdir / "main.tex", extra={"inputs": "results/test, results/validation, results/val_guidance"})
    print("tables:", sorted(p.name for p in tdir.glob("*.tex")))
    print("figures:", sorted(p.name for p in fdir.glob("*.pdf")))


if __name__ == "__main__":
    main()
