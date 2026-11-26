"""Per-window effect estimation shared by the validation framework.

For each window (same initial scene) and a forecaster we run, with common random numbers:
  N  : normal scenario (analog-year ensemble, member i uses analog i mod n)
  O  : observed (factual) weather
  D  : drought scenario (ScenarioBuilder.drought: observed if drought window, else analog)
  N2 : a *different* normal analog assignment (members shifted by one analog) -> scenario placebo
  Ni : normal scenario with INDEPENDENT noise                                   -> noise placebo
Effects (NDVI units, area-and-horizon means per member):
  delta_DN = D - N  (drought effect, the paper's counterfactual)
  delta_ON = O - N  (effect of the observed weather; temporal & permutation placebo)
  delta_N2 = N2 - N, delta_Ni = Ni - N (placebos; should be ~0)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from ..data.dataset import scaled_to_ndvi
from ..physics.monotonicity import with_weather
from .paired import repeat_batch


def _wx_members(wx_list: list[np.ndarray], n: int, shift: int = 0) -> torch.Tensor:
    return torch.from_numpy(np.stack([wx_list[(i + shift) % len(wx_list)] for i in range(n)])).float()


def window_effects(forecast_fn, sample: dict, sb, l: int, yi: int, t0: int, n: int, wx_mean, wx_std, K: int,
                   seed: int, coupling_delta: float | None = None) -> dict:
    """forecast_fn(batch, noise, hook) -> scaled x0 [B, H, P, P]. Returns per-member area-mean effects
    and pixel summaries for the drought effect."""
    normal, n_method = sb.normal(l, yi, t0)
    drought, d_method = sb.drought(l, yi, t0)
    obs = [sb.factual(l, yi, t0)]
    base = repeat_batch(sample, n)
    g = torch.Generator().manual_seed(seed)
    noise = torch.randn(base["tgt"].shape, generator=g)
    noise_ind = torch.randn(base["tgt"].shape, generator=g)
    branches = {
        "N": (_wx_members(normal, n), 0.0, noise),
        "O": (_wx_members(obs, n), float(sample["drought"]), noise),
        "D": (_wx_members(drought, n), 1.0, noise),
        "N2": (_wx_members(normal, n, shift=1), 0.0, noise),
        "Ni": (_wx_members(normal, n), 0.0, noise_ind),
    }
    names = list(branches)
    batches = [with_weather(base, w, wx_mean, wx_std, K, drought=torch.full((n,), f)) for w, f, _ in branches.values()]
    big = {k: torch.cat([b[k] for b in batches]) for k in batches[0]}
    nz = torch.cat([v[2] for v in branches.values()])
    hook = None
    if coupling_delta is not None:
        iN, iD = names.index("N"), names.index("D")

        def hook(x0, i):
            nd = scaled_to_ndvi(x0)
            xn = nd[iN * n:(iN + 1) * n]
            xd = torch.minimum(nd[iD * n:(iD + 1) * n], xn + coupling_delta)
            nd = nd.clone()
            nd[iD * n:(iD + 1) * n] = xd
            from ..data.dataset import ndvi_to_scaled
            return ndvi_to_scaled(nd)
    x = scaled_to_ndvi(forecast_fn(big, nz, hook))
    xs = {k: x[i * n:(i + 1) * n] for i, k in enumerate(names)}
    am = lambda t: t.mean((1, 2, 3)).numpy()  # noqa: E731
    dDN = xs["D"] - xs["N"]
    return {
        "n_method": n_method, "d_method": d_method, "n_analogs": len(normal),
        "s_N": float(np.mean([sb.horizon_stress(w) for w in normal])),
        "s_D": float(np.mean([sb.horizon_stress(w) for w in drought])),
        "s_O": sb.horizon_stress(obs[0]),
        "dDN": am(dDN), "dON": am(xs["O"] - xs["N"]), "dN2": am(xs["N2"] - xs["N"]), "dNi": am(xs["Ni"] - xs["N"]),
        "xN": xs["N"].numpy(), "xD": xs["D"].numpy(), "xO": xs["O"].numpy(),
        "frac_drought_above_normal": float((dDN > 0.01).float().mean()),
        # per-pixel variance of Delta with CRN vs. expected variance with independent noise
        "var_crn": float(dDN.var(0, unbiased=True).mean()) if n > 1 else 0.0,
        "var_ind": float((xs["D"].var(0, unbiased=True) + xs["N"].var(0, unbiased=True)).mean()) if n > 1 else 0.0,
    }


def sampler_forecast_fn(sampler):
    def f(batch, noise, hook):
        return sampler.sample(batch, noise, x0_hook=hook)
    return f


def det_forecast_fn(model):
    @torch.no_grad()
    def f(batch, noise, hook):
        out = model(None, torch.zeros(batch["tgt"].shape[0], dtype=torch.long), batch)
        if hook is not None:
            out = hook(out, 0)
        return out
    return f


def effects_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {k: r[k] for k in ["loc", "year", "t0", "oy", "ox", "drought", "n_method", "d_method", "n_analogs",
                                 "s_N", "s_D", "s_O", "frac_drought_above_normal", "var_crn", "var_ind"]}
        row.update({k: v for k, v in r.items() if k.startswith("xp_")})
        for k in ["dDN", "dON", "dN2", "dNi"]:
            v = r[k]
            row[f"{k}_mean"] = float(np.mean(v))
            row[f"{k}_q05"], row[f"{k}_q95"] = (float(np.percentile(v, 5)), float(np.percentile(v, 95)))
            row[f"{k}_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)
