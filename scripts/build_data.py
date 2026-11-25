"""Build analysis-ready cube + train-only fitted objects + data summary + sanity figure.

  python scripts/build_data.py --preset cpu              # real data (after download_data.py)
  python scripts/build_data.py --preset smoke --synthetic # SYNTHETIC cube for smoke tests
  python scripts/build_data.py --preset cpu --download    # run downloads first (resumable)
Outputs: <processed>/cube/*, <processed>/stats.json, physics.json, clim.npy;
results/data/{data_summary.json, drought_sensitivity.csv, sanity_timeseries.pdf} (+ provenance)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfveg.data.cube import build_cube  # noqa: E402
from cfveg.data.dataset import Cube, fit_norm_stats, make_windows  # noqa: E402
from cfveg.data.stress import is_drought  # noqa: E402
from cfveg.data.synthetic import make_synthetic_cube  # noqa: E402
from cfveg.models.baselines import fit_pixel_climatology  # noqa: E402
from cfveg.physics.constraints import fit_physics  # noqa: E402
from cfveg.utils import plotting  # noqa: E402
from cfveg.utils.provenance import load_preset, write_provenance  # noqa: E402


def processed_dir(cfg: dict, synthetic: bool) -> Path:
    return ROOT / ("data/smoke" if synthetic else cfg["data"]["processed_dir"])


def drought_stats(cube: Cube, cfg: dict, split: str, thr: float, min_days: int) -> dict:
    w = make_windows(cube, cfg, split, quadrants=False)
    K, H = cfg["window"]["context"], cfg["window"]["horizon"]
    bd = cube.meta["bin_days"]
    lab = [is_drought(cube.s_daily[l, y, (t0 + K) * bd:(t0 + K + H) * bd], thr, min_days) for l, y, t0, _, _ in w]
    return {"n_windows": len(w), "drought_frac": float(np.mean(lab)) if lab else float("nan")}


def sanity_figure(cube: Cube, cfg: dict, out: Path, synthetic: bool) -> None:
    plotting.setup()
    import matplotlib.pyplot as plt

    yrs = cube.years
    # pick the train location with the best coverage
    cov = np.isfinite(cube.ndvi).mean(axis=(1, 2, 3, 4))
    l = int(np.argmax(np.where(cube.split == "train", cov, -1)))
    doy = cube.bin_doy
    fig, axes = plt.subplots(3, 1, figsize=(plotting.COL_W, 3.6), sharex=True)
    t_all, nd_all, s_all, p_all, flag = [], [], [], [], []
    K, H = cfg["window"]["context"], cfg["window"]["horizon"]
    bd = cube.meta["bin_days"]
    for yi, y in enumerate(yrs):
        nd = np.asarray(cube.ndvi[l, yi], dtype=np.float32)
        med = np.array([np.nanmedian(f) if np.isfinite(f).mean() > 0.3 else np.nan for f in nd])
        t = y + (doy - 1) / 365.0
        t_all.append(t)
        nd_all.append(med)
        s_all.append(cube.wx[l, yi, :, -1])
        p_all.append(cube.wx[l, yi, :, 0])
        fl = np.zeros(len(doy))
        for t0 in range(len(doy) - K - H + 1):
            if is_drought(cube.s_daily[l, yi, (t0 + K) * bd:(t0 + K + H) * bd], cfg["drought"]["threshold"],
                          cfg["drought"]["min_stressed_days"]):
                fl[t0 + K:t0 + K + H] = 1
        flag.append(fl)
    for t, nd, s, p, fl in zip(t_all, nd_all, s_all, p_all, flag):
        axes[0].plot(t, nd, "-", color="0", marker=".", ms=2)
        axes[1].bar(t, p, width=4 / 365, color="0.5")
        axes[2].plot(t, s, "-", color="0")
        axes[2].fill_between(t, -3, 3, where=fl > 0, color="0.8", step="mid", lw=0)
    axes[2].axhline(cfg["drought"]["threshold"], ls="--", color="0.3", lw=0.8)
    axes[0].set_ylabel("NDVI (patch median)")
    axes[1].set_ylabel("Precip. (mm/5 d)")
    axes[2].set_ylabel("Stress index s")
    axes[2].set_ylim(-3, 3)
    axes[2].set_xlabel("Year (Mar–Oct only)")
    title = f"{'SYNTHETIC ' if synthetic else ''}location {cube.meta['loc_id'][l]} " \
            f"({cube.meta['lat'][l]:.2f}N, {cube.meta['lon'][l]:.2f}E); grey = drought-labelled horizon bins"
    axes[0].set_title(title)
    plotting.save(fig, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    cfg = load_preset(a.preset)
    pdir = processed_dir(cfg, a.synthetic)
    if a.synthetic:
        make_synthetic_cube(pdir / "cube", n_loc=12, P=cfg["window"]["patch"], seed=0)
        res_dir = ROOT / "results" / "smoke"
    else:
        if a.download:
            for st in ["locations", "weather", "s2"]:
                subprocess.run([sys.executable, str(ROOT / "scripts" / "download_data.py"), "--stage", st,
                                "--preset", a.preset], check=True)
        meta = build_cube(cfg, ROOT)
        print(f"cube: {len(meta['loc_id'])} locations; dropped {len(meta['dropped'])}", flush=True)
        res_dir = ROOT / "results" / "data"
    cube = Cube.load(pdir / "cube")
    stats = fit_norm_stats(cube, cfg)
    (pdir / "stats.json").write_text(json.dumps(stats, indent=1))
    phys = fit_physics(cube, cfg)
    (pdir / "physics.json").write_text(json.dumps(phys))
    np.save(pdir / "clim.npy", fit_pixel_climatology(cube, cfg, phys))
    res_dir.mkdir(parents=True, exist_ok=True)
    summary = {"synthetic": bool(a.synthetic), "n_locations": len(cube.meta["loc_id"]),
               "split_counts": pd.Series(cube.meta["split"]).value_counts().to_dict(),
               "dropped_locations": cube.meta["dropped"],
               "valid_pixel_frac": float(np.isfinite(cube.ndvi).mean()),
               "valid_frac_by_year": {int(y): float(np.isfinite(cube.ndvi[:, i]).mean()) for i, y in enumerate(cube.years)},
               "rate_bounds_inc_per_day": phys["r_inc"], "rate_bounds_dec_per_day": phys["r_dec"]}
    for sp in ["train", "val", "test"]:
        summary[f"windows_{sp}"] = drought_stats(cube, cfg, sp, cfg["drought"]["threshold"],
                                                 cfg["drought"]["min_stressed_days"])
    f = res_dir / "data_summary.json"
    f.write_text(json.dumps(summary, indent=1))
    write_provenance(f, config=cfg, extra={"synthetic": bool(a.synthetic)})
    rows = []
    for thr in [0.5, 0.75, 1.0, 1.25, 1.5]:
        for md in [20, 30, 40]:
            if md > cfg["window"]["horizon"] * cube.meta["bin_days"]:
                continue
            for sp in ["train", "test"]:
                rows.append({"threshold": thr, "min_days": md, "split": sp, **drought_stats(cube, cfg, sp, thr, md)})
    f2 = res_dir / "drought_sensitivity.csv"
    pd.DataFrame(rows).to_csv(f2, index=False)
    write_provenance(f2, config=cfg["drought"])
    sanity_figure(cube, cfg, res_dir / "sanity_timeseries", a.synthetic)
    print(json.dumps({k: v for k, v in summary.items() if k.startswith("windows")}, indent=1))


if __name__ == "__main__":
    main()
