"""Compact conditional U-Net with FiLM conditioning.

Spatial input channels: [noisy target (H, diffusion only)] + context NDVI (K) + context mask (K)
+ static layers (6). Conditioning vector (FiLM in every residual block): diffusion timestep,
the full weather sequence (K+H frames x F features), day-of-year (sin/cos per frame), the scalar
stress index and the drought flag.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(a), torch.cos(a)], dim=1)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, emb_dim: int, dropout: float = 0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(min(8, cin), cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.film = nn.Linear(emb_dim, 2 * cout)
        self.n2 = nn.GroupNorm(min(8, cout), cout)
        self.drop = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        nn.init.zeros_(self.c2.weight)
        nn.init.zeros_(self.c2.bias)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, emb: torch.Tensor) -> torch.Tensor:
        h = self.c1(F.silu(self.n1(x)))
        scale, shift = self.film(emb)[:, :, None, None].chunk(2, dim=1)
        h = self.n2(h) * (1 + scale) + shift
        h = self.c2(self.drop(F.silu(h)))
        return h + self.skip(x)


class CondUNet(nn.Module):
    def __init__(self, K: int, H: int, n_wx: int, n_static: int = 6, base_ch: int = 32,
                 ch_mult: tuple = (1, 2, 2), emb_dim: int = 128, dropout: float = 0.0,
                 diffusion: bool = True):
        super().__init__()
        self.K, self.H, self.diffusion = K, H, diffusion
        cin = (H if diffusion else 0) + 2 * K + n_static
        self.emb_dim = emb_dim
        self.t_mlp = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))
        wx_in = (K + H) * (n_wx + 2) + 2
        self.c_mlp = nn.Sequential(nn.Linear(wx_in, 2 * emb_dim), nn.SiLU(), nn.Linear(2 * emb_dim, emb_dim))
        self.inp = nn.Conv2d(cin, base_ch, 3, padding=1)
        chs = [base_ch * m for m in ch_mult]
        self.down = nn.ModuleList()
        self.pool = nn.ModuleList()
        c = base_ch
        skips = []
        for i, co in enumerate(chs):
            self.down.append(nn.ModuleList([ResBlock(c, co, emb_dim, dropout), ResBlock(co, co, emb_dim, dropout)]))
            skips.append(co)
            c = co
            self.pool.append(nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chs) - 1 else nn.Identity())
        self.mid = nn.ModuleList([ResBlock(c, c, emb_dim, dropout), ResBlock(c, c, emb_dim, dropout)])
        self.up = nn.ModuleList()
        for i, co in reversed(list(enumerate(chs))):
            self.up.append(nn.ModuleList([ResBlock(c + skips[i], co, emb_dim, dropout),
                                          ResBlock(co, co, emb_dim, dropout)]))
            c = co
        self.out = nn.Sequential(nn.GroupNorm(min(8, c), c), nn.SiLU(), nn.Conv2d(c, H, 3, padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def cond_vector(self, wx: torch.Tensor, doy: torch.Tensor, stress: torch.Tensor,
                    drought: torch.Tensor) -> torch.Tensor:
        ang = 2 * math.pi * doy / 365.0
        seq = torch.cat([wx, torch.sin(ang)[..., None], torch.cos(ang)[..., None]], dim=-1)
        v = torch.cat([seq.flatten(1), stress[:, None], drought[:, None]], dim=1)
        return self.c_mlp(v)

    def forward(self, x_t: torch.Tensor | None, t: torch.Tensor, batch: dict) -> torch.Tensor:
        parts = ([x_t] if self.diffusion else []) + [batch["ctx"], batch["ctx_mask"], batch["static"]]
        h = self.inp(torch.cat(parts, dim=1))
        emb = self.t_mlp(timestep_embedding(t, self.emb_dim)) + self.cond_vector(
            batch["wx"], batch["doy"], batch["stress"], batch["drought"])
        emb = F.silu(emb)
        hs = []
        for blocks, pool in zip(self.down, self.pool):
            for b in blocks:
                h = b(h, emb)
            hs.append(h)
            h = pool(h)
        for b in self.mid:
            h = b(h, emb)
        for blocks in self.up:
            skip = hs.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = torch.cat([h, skip], dim=1)
            for b in blocks:
                h = b(h, emb)
        return self.out(h)


def build_unet(cfg: dict, diffusion: bool = True, n_wx: int = 11) -> CondUNet:
    w, m = cfg["window"], cfg["model"]
    return CondUNet(w["context"], w["horizon"], n_wx, 6, m["base_ch"], tuple(m["ch_mult"]), m["emb_dim"],
                    m.get("dropout", 0.0), diffusion)
