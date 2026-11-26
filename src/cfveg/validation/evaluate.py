"""Evaluate forecast variants on a set of windows: accuracy, sharpness, probabilistic, physics violations.

A variant is a callable `forecast(batch, n_members, seed) -> NDVI ensemble [M, B, H, P, P]` plus a
name. Monotone-response violations are measured by forecasting the same window under the observed
weather w and a stressed w' = stress_weather(w, +1 sigma) with common random numbers, counting
pixel-timesteps where the stressed forecast is higher by more than tolerance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
import torch

from ..data.dataset import scaled_to_ndvi
from ..models.baselines import climatology, climatology_anomaly_persistence, persistence
from ..physics.monotonicity import stress_weather, with_weather
from . import metrics as M

MONO_TOL = 0.01


@dataclass
class Variant:
    name: str
    forecast: Callable          # (batch, n_members, seed) -> NDVI tensor [M, B, H, P, P]
    probabilistic: bool
    learned: bool               # responds to weather (monotonicity test applies)


def baseline_variants(clim: np.ndarray, K: int, H: int) -> list[Variant]:
    return [
        Variant("persistence", lambda b, m, s: scaled_to_ndvi(persistence(b, H))[None], False, False),
        Variant("climatology", lambda b, m, s: scaled_to_ndvi(climatology(b, clim, K, H))[None], False, False),
        Variant("clim_anomaly", lambda b, m, s: scaled_to_ndvi(climatology_anomaly_persistence(b, clim, K, H))[None],
                False, False),
    ]


def deterministic_variant(model, name: str = "det") -> Variant:
    @torch.no_grad()
    def f(b, m, s):
        return scaled_to_ndvi(model(None, torch.zeros(b["tgt"].shape[0], dtype=torch.long), b))[None]
    return Variant(name, f, False, True)


def diffusion_variant(sampler, name: str) -> Variant:
    def f(b, m, s):
        B = b["tgt"].shape[0]
        g = torch.Generator().manual_seed(s)
        noise = torch.randn((m, *b["tgt"].shape), generator=g)
        big = {k: v.repeat(m, *([1] * (v.ndim - 1))) for k, v in b.items()}
        x = sampler.sample(big, noise.view(m * B, *b["tgt"].shape[1:]))
        return scaled_to_ndvi(x).view(m, B, *b["tgt"].shape[1:])
    return Variant(name, f, True, True)


def evaluate_observed(batches: list[dict], phys) -> dict:
    """Violation rates of the OBSERVED targets (valid pixels only; rate steps need consecutive valid frames).
    Reference for over-constraint: a forecast with far fewer violations than reality is not 'more physical'."""
    viol = {"rate_frac": [], "phen_frac": [], "range_frac": [], "rate_mag": [], "phen_mag": []}
    w = []
    for b in batches:
        v = phys.violations(b["tgt"], b, 1.0 - b["drought"], mask=b["tgt_mask"])
        for k in viol:
            viol[k].append(v[k])
        w.append(b["tgt_mask"].sum().item())
    out = {k: float(np.average(v, weights=w)) for k, v in viol.items()}
    out["variant"] = "observed"
    return out


def evaluate(variant: Variant, batches: list[dict], phys, n_members: int, wx_mean, wx_std, K: int,
             seed: int = 0, mono: bool = True) -> tuple[dict, pd.DataFrame]:
    acc = M.Accumulator()
    rng = np.random.default_rng(seed)
    rows = []
    viol = {"rate_frac": [], "phen_frac": [], "range_frac": [], "rate_mag": [], "phen_mag": [], "mono_frac": [],
            "mono_mag": []}
    for bi, b in enumerate(batches):
        obs = scaled_to_ndvi(b["tgt"])
        mask = b["tgt_mask"]
        ens = variant.forecast(b, n_members, seed + 1000 * bi)          # [M, B, H, P, P]
        mean = ens.mean(0)
        e = M.masked_err(mean, obs, mask)
        acc.add(e)
        acc.add({"ssim": M.ssim_frames(ens[0], obs, mask)})              # single member: no averaging blur
        acc.add(M.sharpness(ens[0], obs, mask))
        if variant.probabilistic and ens.shape[0] > 1:
            acc.add(M.crps_ensemble(ens, obs, mask), "p_")
            acc.add(M.coverage(ens, obs, mask), "c_")
            acc.add(M.spread_skill(ens, obs, mask), "ss_")
            acc.add_hist(M.rank_hist(ens, obs, mask, rng))
        else:
            acc.add({"crps_sum": e["abs_sum"], "n": e["n"]}, "p_")
        normal = 1.0 - b["drought"]
        from ..data.dataset import ndvi_to_scaled

        for mi in range(min(ens.shape[0], 4)):
            v = phys.violations(ndvi_to_scaled(ens[mi]), b, normal)
            for k in v:
                viol[k].append(v[k])
        if mono and variant.learned:
            b2 = with_weather(b, stress_weather(b["wx_raw"], K, 1.0), wx_mean, wx_std, K)
            m_mono = min(n_members, 4)
            es = variant.forecast(b2, m_mono, seed + 1000 * bi)           # same seed => CRN
            eb = ens[:m_mono]
            d = es - eb
            viol["mono_frac"].append((d > MONO_TOL).float().mean().item())
            viol["mono_mag"].append(d[d > MONO_TOL].mean().item() if (d > MONO_TOL).any() else 0.0)
        # per-window records (for bootstrap over locations)
        d = (mean - obs) * mask
        crps_w = None
        if variant.probabilistic and ens.shape[0] > 1:
            Mm = ens.shape[0]
            t1 = (ens - obs[None]).abs().mean(0)
            s_, _ = torch.sort(ens, dim=0)
            i = torch.arange(1, Mm + 1, dtype=ens.dtype).view(-1, 1, 1, 1, 1)
            crps_w = ((t1 - (2 * i - Mm - 1).mul(s_).sum(0) / (Mm * (Mm - 1))) * mask).sum((1, 2, 3))
        for j in range(obs.shape[0]):
            l, y, t0, oy, ox = b["idx"][j].tolist()
            n = mask[j].sum().item()
            rows.append({"loc": l, "year": y, "t0": t0, "oy": oy, "ox": ox, "drought": b["drought"][j].item(),
                         "n": n, "abs_sum": d[j].abs().sum().item(), "sq_sum": (d[j] ** 2).sum().item(),
                         "crps_sum": crps_w[j].item() if crps_w is not None else d[j].abs().sum().item()})
    out = acc.finalize()
    for k, v in viol.items():
        if v:
            out[k] = float(np.mean(v))
    out["variant"] = variant.name
    return out, pd.DataFrame(rows)


def bootstrap_ci(per_window: pd.DataFrame, metric: str = "mae", n_boot: int = 1000, seed: int = 0,
                 unit: str = "loc") -> tuple[float, float, float]:
    """Point estimate and 95 % CI, resampling locations (clusters) with replacement."""
    g = per_window.groupby(unit)[["abs_sum", "sq_sum", "crps_sum", "n"]].sum()
    rng = np.random.default_rng(seed)

    def stat(df):
        if metric == "mae":
            return df.abs_sum.sum() / df.n.sum()
        if metric == "rmse":
            return np.sqrt(df.sq_sum.sum() / df.n.sum())
        return df.crps_sum.sum() / df.n.sum()

    pt = stat(g)
    bs = [stat(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(n_boot)]
    return float(pt), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, metric: str = "mae", n_boot: int = 1000,
                     seed: int = 0) -> dict:
    """Difference a - b in metric with a 95 % CI over location clusters (same windows required)."""
    key = ["loc", "year", "t0", "oy", "ox"]
    m = a.merge(b, on=key, suffixes=("_a", "_b"))
    col = {"mae": "abs_sum", "rmse": "sq_sum", "crps": "crps_sum"}[metric]
    g = m.groupby("loc")[[f"{col}_a", f"{col}_b", "n_a"]].sum()
    rng = np.random.default_rng(seed)

    def stat(df):
        va, vb = df[f"{col}_a"].sum() / df.n_a.sum(), df[f"{col}_b"].sum() / df.n_a.sum()
        if metric == "rmse":
            va, vb = np.sqrt(va), np.sqrt(vb)
        return va - vb

    pt = stat(g)
    bs = np.array([stat(g.iloc[rng.integers(0, len(g), len(g))]) for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff": float(pt), "lo": float(lo), "hi": float(hi), "p_two_sided": float(2 * min((bs <= 0).mean(),
                                                                                           (bs >= 0).mean()))}
