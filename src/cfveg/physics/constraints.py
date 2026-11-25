"""Physics constraints as differentiable violation energies on the predicted clean sample.

ALL constraint maths happens in *unscaled NDVI units* (NDVI per day for rates). Inputs in the
diffusion's scaled space are converted with `scaled_to_ndvi` first (tested in test_physics.py).

Fitted quantities (from the TRAIN split only, see fit_physics):
  r_inc[g], r_dec[g]   99.5th percentile of clean consecutive NDVI increases / decreases per day,
                       per land-cover group g
  mu[g, b], sigma[g, b] phenology envelope per land-cover group and 5-day bin, from non-drought
                       training windows (bin stress s < threshold)
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from ..data.dataset import N_LC, scaled_to_ndvi

NDVI_MIN, NDVI_MAX = -1.0, 1.0


def fit_physics(cube, cfg: dict) -> dict:
    """Rate bounds and phenology envelopes from train locations x train years only."""
    m = cube.loc_year_mask("train", cfg)
    pct = cfg["physics"]["rate_percentile"]
    thr = cfg["drought"]["threshold"]
    bd = cube.meta["bin_days"]
    nb = cube.ndvi.shape[2]
    r_inc = np.zeros(N_LC)
    r_dec = np.zeros(N_LC)
    sums = np.zeros((N_LC, nb))
    sq = np.zeros((N_LC, nb))
    cnt = np.zeros((N_LC, nb))
    incs = [[] for _ in range(N_LC)]
    decs = [[] for _ in range(N_LC)]
    rng = np.random.default_rng(0)
    for l, y in zip(*np.nonzero(m)):
        nd = np.asarray(cube.ndvi[l, y], dtype=np.float32)   # [B, P, P]
        g = cube.lcg[l]
        d = (nd[1:] - nd[:-1]) / bd                          # per day, adjacent bins both valid
        s_bin = cube.wx[l, y, :, -1]
        for k in range(N_LC):
            sel = g == k
            if not sel.any():
                continue
            dk = d[:, sel]
            dk = dk[np.isfinite(dk)]
            if dk.size > 20000:
                dk = rng.choice(dk, 20000, replace=False)
            incs[k].append(dk[dk > 0])
            decs[k].append(-dk[dk < 0])
            v = nd[:, sel]                                    # [B, n]
            ok = np.isfinite(v) & (s_bin < thr)[:, None]
            vv = np.where(ok, v, 0.0)
            sums[k] += vv.sum(1)
            sq[k] += (vv ** 2).sum(1)
            cnt[k] += ok.sum(1)
    for k in range(N_LC):
        ii = np.concatenate(incs[k]) if incs[k] else np.array([])
        dd = np.concatenate(decs[k]) if decs[k] else np.array([])
        r_inc[k] = np.percentile(ii, pct) if ii.size > 100 else np.nan
        r_dec[k] = np.percentile(dd, pct) if dd.size > 100 else np.nan
    # groups with too little data fall back to the pooled value
    r_inc = np.where(np.isfinite(r_inc), r_inc, np.nanmax(r_inc))
    r_dec = np.where(np.isfinite(r_dec), r_dec, np.nanmax(r_dec))
    with np.errstate(invalid="ignore", divide="ignore"):
        mu = sums / cnt
        sd = np.sqrt(np.maximum(sq / cnt - mu ** 2, 0))
    pooled_mu = np.nansum(sums, 0) / np.maximum(cnt.sum(0), 1)
    pooled_sd = np.sqrt(np.maximum(np.nansum(sq, 0) / np.maximum(cnt.sum(0), 1) - pooled_mu ** 2, 0))
    low = cnt < 200
    mu[low] = np.broadcast_to(pooled_mu, mu.shape)[low]
    sd[low] = np.broadcast_to(pooled_sd, sd.shape)[low]
    # light smoothing over bins
    ker = np.array([0.25, 0.5, 0.25])
    mu = np.stack([np.convolve(np.pad(r, 1, mode="edge"), ker, "valid") for r in mu])
    sd = np.stack([np.convolve(np.pad(r, 1, mode="edge"), ker, "valid") for r in sd])
    return {"r_inc": r_inc.tolist(), "r_dec": r_dec.tolist(), "mu": mu.tolist(), "sigma": sd.tolist(),
            "count": cnt.tolist(), "percentile": pct, "fitted_on": "train locations x train years"}


class Physics:
    """Holds fitted bounds as tensors and computes violation energies."""

    def __init__(self, phys: dict, cfg: dict, device: str = "cpu"):
        self.r_inc = torch.tensor(phys["r_inc"], dtype=torch.float32, device=device)
        self.r_dec = torch.tensor(phys["r_dec"], dtype=torch.float32, device=device)
        self.mu = torch.tensor(phys["mu"], dtype=torch.float32, device=device)       # [G, B]
        self.sigma = torch.tensor(phys["sigma"], dtype=torch.float32, device=device)
        self.k = cfg["physics"]["phen_k"]
        self.bin_days = cfg["data"]["bin_days"]
        self.K = cfg["window"]["context"]
        self.H = cfg["window"]["horizon"]

    # ---------- helpers ----------
    def _bin_idx(self, batch: dict) -> torch.Tensor:
        """Global bin index of each horizon frame, [B, H]."""
        t0 = batch["idx"][:, 2]
        return t0[:, None] + self.K + torch.arange(self.H, device=t0.device)[None]

    def envelope(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-pixel phenology bounds (lo, hi) in NDVI, [B, H, P, P]."""
        g = batch["lcg"]                                     # [B, P, P]
        b = self._bin_idx(batch).clamp(max=self.mu.shape[1] - 1)   # [B, H]
        mu = self.mu[g[:, None], b[:, :, None, None]]        # [B, H, P, P]
        sd = self.sigma[g[:, None], b[:, :, None, None]]
        return mu - self.k * sd, mu + self.k * sd

    def rates(self, ndvi: torch.Tensor, batch: dict) -> torch.Tensor:
        """Daily rate between consecutive frames incl. step from last context frame. [B, H, P, P]."""
        prev = torch.cat([batch["last_ndvi"][:, None], ndvi[:, :-1]], dim=1)
        return (ndvi - prev) / self.bin_days

    # ---------- energies (per-sample mean; multiply by #pixels for sums) ----------
    def rate_energy(self, x0_scaled: torch.Tensor, batch: dict) -> torch.Tensor:
        ndvi = scaled_to_ndvi(x0_scaled)
        g = self.rates(ndvi, batch)
        ri = self.r_inc[batch["lcg"]][:, None]
        rd = self.r_dec[batch["lcg"]][:, None]
        e = F.relu(g - ri) ** 2 + F.relu(-g - rd) ** 2
        return e.mean((1, 2, 3)) / (ri.mean() ** 2)          # normalised by typical bound

    def phen_energy(self, x0_scaled: torch.Tensor, batch: dict, normal: torch.Tensor) -> torch.Tensor:
        """normal: [B] 1 for normal-scenario samples (lower bound active), 0 for drought."""
        ndvi = scaled_to_ndvi(x0_scaled)
        lo, hi = self.envelope(batch)
        e = F.relu(ndvi - hi) ** 2 + normal.view(-1, 1, 1, 1) * F.relu(lo - ndvi) ** 2
        return e.mean((1, 2, 3)) / 0.01                      # in units of (0.1 NDVI)^2

    def range_energy(self, x0_scaled: torch.Tensor) -> torch.Tensor:
        ndvi = scaled_to_ndvi(x0_scaled)
        return (F.relu(ndvi - NDVI_MAX) ** 2 + F.relu(NDVI_MIN - ndvi) ** 2).mean((1, 2, 3)) / 0.01

    @staticmethod
    def mono_energy(x0_stressed: torch.Tensor, x0_base: torch.Tensor, eps: float) -> torch.Tensor:
        """Penalise stressed forecast exceeding base forecast by more than eps (NDVI units)."""
        d = scaled_to_ndvi(x0_stressed) - scaled_to_ndvi(x0_base)
        return (F.relu(d - eps) ** 2).mean((1, 2, 3)) / 0.01

    # ---------- hard projection ----------
    def project(self, x0_scaled: torch.Tensor, batch: dict, rate: bool = True) -> torch.Tensor:
        ndvi = scaled_to_ndvi(x0_scaled).clamp(NDVI_MIN, NDVI_MAX)
        if rate:
            ri = self.r_inc[batch["lcg"]] * self.bin_days
            rd = self.r_dec[batch["lcg"]] * self.bin_days
            prev = batch["last_ndvi"]
            frames = []
            for h in range(ndvi.shape[1]):
                cur = torch.minimum(torch.maximum(ndvi[:, h], prev - rd), prev + ri)
                frames.append(cur)
                prev = cur
            ndvi = torch.stack(frames, 1)
        from ..data.dataset import ndvi_to_scaled

        return ndvi_to_scaled(ndvi)

    # ---------- violation metrics (fractions, NDVI units) ----------
    @torch.no_grad()
    def violations(self, x0_scaled: torch.Tensor, batch: dict, normal: torch.Tensor, tol: float = 1e-4) -> dict:
        ndvi = scaled_to_ndvi(x0_scaled)
        g = self.rates(ndvi, batch)
        ri = self.r_inc[batch["lcg"]][:, None]
        rd = self.r_dec[batch["lcg"]][:, None]
        rate_ex = F.relu(g - ri) + F.relu(-g - rd)
        lo, hi = self.envelope(batch)
        phen_ex = F.relu(ndvi - hi) + normal.view(-1, 1, 1, 1) * F.relu(lo - ndvi)
        rng_ex = F.relu(ndvi - NDVI_MAX) + F.relu(NDVI_MIN - ndvi)
        return {
            "rate_frac": (rate_ex > tol).float().mean().item(),
            "rate_mag": rate_ex[rate_ex > tol].mean().item() if (rate_ex > tol).any() else 0.0,
            "phen_frac": (phen_ex > tol).float().mean().item(),
            "phen_mag": phen_ex[phen_ex > tol].mean().item() if (phen_ex > tol).any() else 0.0,
            "range_frac": (rng_ex > tol).float().mean().item(),
        }
