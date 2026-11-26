"""Physics-guided sampling and paired counterfactual estimation with common random numbers (CRN).

estimate_effect(): play the SAME initial scene forward under scenario a (normal) and b (drought),
member i of both branches starting from the IDENTICAL initial noise (CRN). Delta = x_b - x_a.
Optional monotone coupling projects x0_b <- min(x0_b, x0_a + delta) at every reverse step; this
forces Delta <= delta and can bias placebo tests, so coupled and uncoupled results are kept apart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..data.dataset import ndvi_to_scaled, scaled_to_ndvi
from ..physics.monotonicity import with_weather


@dataclass
class SamplerOpts:
    guidance: bool = False
    guidance_scale: float = 1.0
    projection: bool = False
    projection_rate: bool = True
    ddim_steps: int = 25
    lambda_rate: float = 1.0
    lambda_phen: float = 0.5
    lambda_range: float = 1.0
    constraints: tuple = ("rate", "phen", "mono")
    eta: float = 0.0


class Sampler:
    def __init__(self, model, diff, phys, opts: SamplerOpts):
        self.model, self.diff, self.phys, self.o = model, diff, phys, opts

    def energy(self, x0: torch.Tensor, batch: dict) -> torch.Tensor:
        """Per-sample guidance energy as a SUM over pixels (so gradients are O(local violation))."""
        n = x0[0].numel()
        e = self.o.lambda_range * self.phys.range_energy(x0)
        if "rate" in self.o.constraints:
            e = e + self.o.lambda_rate * self.phys.rate_energy(x0, batch)
        if "phen" in self.o.constraints:
            e = e + self.o.lambda_phen * self.phys.phen_energy(x0, batch, 1.0 - batch["drought"])
        # dimensionless: violation relative to ~0.1 NDVI (phen/range) or to the rate bound (rate), squared
        return e * n

    def project(self, x0: torch.Tensor, batch: dict) -> torch.Tensor:
        return self.phys.project(x0, batch, rate=self.o.projection_rate and "rate" in self.o.constraints)

    def sample(self, batch: dict, noise: torch.Tensor, x0_hook=None, z_keys=None, z_seed: int = 0) -> torch.Tensor:
        g = self.energy if (self.o.guidance and self.phys is not None) else None
        p = self.project if (self.o.projection and self.phys is not None) else None
        return self.diff.sample(self.model, batch, noise, self.o.ddim_steps, guidance=g,
                                guidance_scale=self.o.guidance_scale, projection=p, x0_hook=x0_hook,
                                eta=self.o.eta, z_keys=z_keys, z_seed=z_seed)


def repeat_batch(sample: dict, n: int) -> dict:
    """Single-window sample dict (unbatched tensors) -> batch of n identical copies."""
    return {k: v.unsqueeze(0).repeat(n, *([1] * v.ndim)) for k, v in sample.items()}


def batch_with_scenarios(base: dict, wx_list: list[np.ndarray], n: int, wx_mean, wx_std, K: int,
                         drought_flag: float) -> dict:
    """Member i uses scenario weather wx_list[i % len(wx_list)] (analog ensemble)."""
    w = torch.from_numpy(np.stack([wx_list[i % len(wx_list)] for i in range(n)])).float()
    return with_weather(base, w, wx_mean, wx_std, K, drought=torch.full((n,), drought_flag))


def estimate_effect(sampler: Sampler, sample: dict, wx_a: list[np.ndarray], wx_b: list[np.ndarray],
                    n_members: int, wx_mean, wx_std, K: int, crn: bool = True, coupling_delta: float | None = None,
                    seed: int = 0, flag_a: float = 0.0, flag_b: float = 1.0) -> dict:
    """Returns NDVI-unit arrays: xa, xb [M, H, P, P], delta [M, H, P, P]."""
    base = repeat_batch(sample, n_members)
    ba = batch_with_scenarios(base, wx_a, n_members, wx_mean, wx_std, K, flag_a)
    bb = batch_with_scenarios(base, wx_b, n_members, wx_mean, wx_std, K, flag_b)
    g = torch.Generator().manual_seed(seed)
    shape = base["tgt"].shape
    na = torch.randn(shape, generator=g)
    nb = na.clone() if crn else torch.randn(shape, generator=g)
    both = {k: torch.cat([ba[k], bb[k]]) for k in ba}
    noise = torch.cat([na, nb])
    hook = None
    if coupling_delta is not None:
        M = n_members

        def hook(x0, i):
            nd = scaled_to_ndvi(x0)
            xb = torch.minimum(nd[M:], nd[:M] + coupling_delta)
            return ndvi_to_scaled(torch.cat([nd[:M], xb]))
    keys = torch.cat([torch.arange(n_members), torch.arange(n_members) + (0 if crn else 10_000)])
    x = scaled_to_ndvi(sampler.sample(both, noise, x0_hook=hook, z_keys=keys, z_seed=seed))
    xa, xb = x[:n_members], x[n_members:]
    return {"xa": xa.numpy(), "xb": xb.numpy(), "delta": (xb - xa).numpy()}


def summarize_delta(delta: np.ndarray) -> dict:
    """Ensemble summaries per pixel/time and aggregates. delta [M, H, P, P] NDVI units."""
    mean = delta.mean(0)
    sd = delta.std(0, ddof=1)
    q05, q95 = np.percentile(delta, [5, 95], axis=0)
    sig_loss = q95 < 0
    return {"mean": mean, "sd": sd, "q05": q05, "q95": q95,
            "area_mean": float(mean.mean()), "area_mean_members": delta.mean((1, 2, 3)),
            "frac_sig_loss": float(sig_loss.mean())}
