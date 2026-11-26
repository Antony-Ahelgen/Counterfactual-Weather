"""Five-part validation: builds tables from results/test (eval_all) and results/effects (effects.py).

  python scripts/validate.py --preset cpu
Outputs results/validation/{proxy_backtest,proxy_crossyear,natural_experiment,natexp_pairs,placebo,
calibration,violations,validation_summary}.csv (+ provenance).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfveg.data.dataset import Cube  # noqa: E402
from cfveg.utils.provenance import load_preset, write_provenance  # noqa: E402
from cfveg.validation.evaluate import bootstrap_ci  # noqa: E402
from cfveg.validation.framework import natural_experiment, placebos  # noqa: E402


def save(df: pd.DataFrame, path: Path, **extra) -> None:
    df.to_csv(path, index=False)
    write_provenance(path, extra=extra)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--main", default="diff_phys+G+P")
    a = ap.parse_args()
    cfg = load_preset(a.preset)
    base = ROOT / "results" / ("smoke" if a.smoke else "")
    test_dir, eff_dir, out = base / "test", base / "effects", base / "validation"
    out.mkdir(parents=True, exist_ok=True)
    cube = Cube.load(ROOT / ("data/smoke" if a.smoke else cfg["data"]["processed_dir"]) / "cube")
    s = a.seed
    summary = []

    # ---- 1a back-test on non-drought windows (factual forecasts vs observations)
    rows = []
    for f in sorted(test_dir.glob(f"per_window_*_s{s}.csv")):
        name = f.stem.removeprefix("per_window_").removesuffix(f"_s{s}")
        pw = pd.read_csv(f)
        nd = pw[pw.drought == 0]
        for metric in ["mae", "rmse", "crps"]:
            pt, lo, hi = bootstrap_ci(nd, metric)
            rows.append({"variant": name, "metric": metric, "value": pt, "lo95": lo, "hi95": hi,
                         "n_windows": len(nd), "subset": "non-drought"})
    bt = pd.DataFrame(rows)
    save(bt, out / "proxy_backtest.csv", part="1a")

    # ---- 1b cross-year proxy (normal-scenario change vs observed change in non-drought years)
    main_safe = a.main.replace("+", "_")
    eff_files = {f.stem.removeprefix("effects_").removesuffix(f"_s{s}"): f for f in eff_dir.glob(f"effects_*_s{s}.csv")}
    rows = []
    for name, f in sorted(eff_files.items()):
        e = pd.read_csv(f)
        for subset, sel in [("drought windows", e.drought == 1), ("all windows", e.drought >= 0)]:
            d = e[sel & (e.xp_n_years > 0)] if "xp_n_years" in e else e.iloc[:0]
            if not len(d):
                continue
            w = d.xp_n_pix
            r = {"variant": name, "subset": subset, "n_windows": len(d)}
            for k in ["normal", "factual", "persistence", "climatology"]:
                col = f"xp_mae_{k}"
                if col in d:
                    r[f"mae_{k}"] = float(np.average(d[col], weights=w))
            rows.append(r)
    xp = pd.DataFrame(rows)
    save(xp, out / "proxy_crossyear.csv", part="1b")

    # ---- 2 natural experiment + 3 placebos for every effects file
    ne_rows, pl_rows = [], []
    for name, f in sorted(eff_files.items()):
        e = pd.read_csv(f)
        ne = natural_experiment(cube, cfg, e)
        r = {"variant": name, **{k: v for k, v in ne["summary"].items() if k != "smd"}}
        if "smd" in ne["summary"]:
            r.update({f"smd_{k}": v for k, v in ne["summary"]["smd"].items()})
        ne_rows.append(r)
        if name == main_safe and len(ne["pairs"]):
            save(ne["pairs"], out / "natexp_pairs.csv", part="2", variant=name)
        pl_rows.append({"variant": name, **placebos(e, seed=s),
                        "frac_drought_above_normal": float(e.frac_drought_above_normal.mean()),
                        "mean_dDN": float(e.dDN_mean.mean()),
                        "mean_dDN_drought_windows": float(e.loc[e.drought == 1, "dDN_mean"].mean())})
    save(pd.DataFrame(ne_rows), out / "natural_experiment.csv", part="2")
    save(pd.DataFrame(pl_rows), out / "placebo.csv", part="3")

    # ---- 4 calibration and 5 violations from eval metrics
    m = pd.read_csv(test_dir / "metrics.csv")
    m = m[m.seed == s]
    cal_cols = [c for c in ["variant", "crps", "cov50", "cov80", "cov90", "spread_skill"] if c in m]
    cal = m[cal_cols].copy()
    hists = {}
    for f in test_dir.glob(f"rank_hist_*_s{s}.json"):
        hists[f.stem.removeprefix("rank_hist_").removesuffix(f"_s{s}")] = json.loads(f.read_text())
    cal["rank_hist_chi2_flatness"] = [float(np.sum((np.array(hists[v.replace("+", "_")]) - 1 / len(hists[v.replace("+", "_")])) ** 2) * len(hists[v.replace("+", "_")]))
                                      if v.replace("+", "_") in hists else np.nan for v in cal.variant]
    ne_df = pd.DataFrame(ne_rows)
    if "did_in_model_90pi" in ne_df:
        cal["did_in_effect_90pi"] = [ne_df.set_index("variant").did_in_model_90pi.get(v.replace("+", "_"), np.nan)
                                     for v in cal.variant]
    save(cal, out / "calibration.csv", part="4")
    vio_cols = [c for c in ["variant", "rate_frac", "rate_mag", "phen_frac", "phen_mag", "range_frac", "mono_frac",
                            "mono_mag"] if c in m]
    save(m[vio_cols], out / "violations.csv", part="5")

    # ---- summary (long format: part, variant, metric, value)
    def add(part, variant, metric, value):
        summary.append({"part": part, "variant": variant, "metric": metric, "value": value})

    for r in bt[bt.metric == "mae"].itertuples():
        add("1a_backtest", r.variant, "mae_nondrought", r.value)
    for r in xp.to_dict("records"):
        for k in ["mae_normal", "mae_persistence", "mae_climatology", "mae_factual"]:
            if k in r:
                add("1b_crossyear", r["variant"] + f" [{r['subset']}]", k, r[k])
    for r in ne_rows:
        for k in ["n_matched", "max_abs_smd", "did_mean", "model_mean", "bias_model_minus_did", "corr",
                  "did_in_model_90pi"]:
            if k in r:
                add("2_natural_experiment", r["variant"], k, r[k])
    for r in pl_rows:
        for k in ["scenario_placebo_mean", "noise_placebo_mean", "scenario_placebo_frac_below_tol", "temporal_auc",
                  "perm_real_contrast", "perm_null_mean", "perm_p_one_sided", "crn_var_ratio",
                  "frac_drought_above_normal", "mean_dDN_drought_windows"]:
            if k in r:
                add("3_placebo", r["variant"], k, r[k])
    for r in cal.to_dict("records"):
        for k, v in r.items():
            if k != "variant":
                add("4_calibration", r["variant"], k, v)
    for r in m[vio_cols].to_dict("records"):
        for k, v in r.items():
            if k != "variant":
                add("5_violations", r["variant"], k, v)
    save(pd.DataFrame(summary), out / "validation_summary.csv", parts="1-5")
    print(pd.DataFrame(summary).groupby("part").size())


if __name__ == "__main__":
    main()
