"""Gaussian diffusion: cosine schedule, v-prediction, masked loss, DDIM sampling with hooks.

Hooks used by the physics module:
  guidance(x0_hat, t_index, batch) -> energy (scalar per sample); its gradient w.r.t. x_t steers sampling
  projection(x0_hat, batch) -> x0_hat projected onto the valid set (hard constraints)
"""
from __future__ import annotations

import math
from typing import Callable

import torch


def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    t = torch.arange(T + 1, dtype=torch.float64) / T
    f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    ab = f / f[0]
    betas = (1 - ab[1:] / ab[:-1]).clamp(max=0.999)
    return torch.cumprod(1 - betas, 0).float()   # alpha_bar for t = 0..T-1


class Diffusion:
    def __init__(self, T: int = 1000, prediction: str = "v"):
        self.T = T
        self.prediction = prediction
        self.ab = cosine_alpha_bar(T)

    def _a(self, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ab = self.ab.to(t.device)[t].view(-1, 1, 1, 1)
        return ab.sqrt(), (1 - ab).sqrt()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        a, s = self._a(t)
        return a * x0 + s * noise

    def target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        if self.prediction == "eps":
            return noise
        a, s = self._a(t)
        return a * noise - s * x0

    def x0_from_out(self, x_t: torch.Tensor, t: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        a, s = self._a(t)
        if self.prediction == "eps":
            return (x_t - s * out) / a
        return a * x_t - s * out

    def eps_from_x0(self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor) -> torch.Tensor:
        a, s = self._a(t)
        return (x_t - a * x0) / s

    def snr_weight(self, t: torch.Tensor) -> torch.Tensor:
        """alpha_bar(t) in [0,1]: ~1 at low noise, ~0 at high noise (for auxiliary physics losses)."""
        return self.ab.to(t.device)[t]

    def loss(self, model, batch: dict, t: torch.Tensor | None = None, noise: torch.Tensor | None = None):
        """Masked diffusion loss. Returns (loss, x_t, t, x0_hat)."""
        x0 = batch["tgt"]
        m = batch["tgt_mask"]
        B = x0.shape[0]
        t = torch.randint(0, self.T, (B,), device=x0.device) if t is None else t
        noise = torch.randn_like(x0) if noise is None else noise
        x_t = self.q_sample(x0, t, noise)
        out = model(x_t, t, batch)
        tgt = self.target(x0, t, noise)
        per = ((out - tgt) ** 2 * m).sum((1, 2, 3)) / m.sum((1, 2, 3)).clamp(min=1)
        return per.mean(), x_t, t, self.x0_from_out(x_t, t, out)

    def ddim_timesteps(self, steps: int) -> list[int]:
        return torch.linspace(self.T - 1, 0, steps).round().long().tolist()

    @torch.no_grad()
    def sample(self, model, batch: dict, noise: torch.Tensor, steps: int = 25,
               guidance: Callable | None = None, guidance_scale: float = 0.0,
               projection: Callable | None = None, x0_hook: Callable | None = None) -> torch.Tensor:
        """Deterministic DDIM (eta=0) from the given initial noise (common random numbers)."""
        x = noise
        ts = self.ddim_timesteps(steps)
        for i, ti in enumerate(ts):
            t = torch.full((x.shape[0],), ti, device=x.device, dtype=torch.long)
            if guidance is not None and guidance_scale > 0:
                with torch.enable_grad():
                    xg = x.detach().requires_grad_(True)
                    x0 = self.x0_from_out(xg, t, model(xg, t, batch))
                    energy = guidance(x0, batch).sum()
                    grad = torch.autograd.grad(energy, xg)[0]
                x0 = x0.detach()
                # Energy is a per-pixel SUM, so grad is O(local violation). Step on the clean estimate;
                # the eps used for the DDIM update is recomputed from the guided x0 below, which is
                # equivalent to adjusting the predicted noise. Scale grows as noise falls (alpha_bar).
                w = guidance_scale * self.ab[ti].item()
                x0 = x0 - w * grad
            else:
                x0 = self.x0_from_out(x, t, model(x, t, batch))
            if projection is not None:
                x0 = projection(x0, batch)
            if x0_hook is not None:
                x0 = x0_hook(x0, i)
            eps = self.eps_from_x0(x, t, x0)
            if i == len(ts) - 1:
                x = x0
            else:
                ab_next = self.ab[ts[i + 1]]
                x = ab_next.sqrt() * x0 + (1 - ab_next).sqrt() * eps
        return x


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        import copy

        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, step: int) -> None:
        d = min(self.decay, (1 + step) / (10 + step))
        for pe, p in zip(self.model.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.model.buffers(), model.buffers()):
            be.copy_(b)
