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
    min_fv = cfg["physics"].get("rate_min_frame_valid", 0.95)
    env_fv = cfg["physics"].get("env_min_frame_valid", 0.8)
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
        # "clean" differences: both frames nearly cloud-free (limits residual cloud/haze contamination)
        fv = np.isfinite(nd).mean((1, 2))
        d[~((fv[1:] >= min_fv) & (fv[:-1] >= min_fv))] = np.nan
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
            ok = np.isfinite(v) & (s_bin < thr)[:, None] & (fv >= env_fv)[:, None]
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
            "count": cnt.tolist(), "percentile": pct, "min_frame_valid": min_fv, "env_min_frame_valid": env_fv, "fitted_on": "train locations x train years"}


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

    def _dt(self, batch: dict, H: int) -> torch.Tensor:
        """Days spanned by each step [B, H, P, P]: first step uses the true gap since the last valid
        context observation (never-observed pixels -> very long gap, i.e. effectively unconstrained)."""
        gap = batch.get("last_gap")
        B, P = batch["last_ndvi"].shape[0], batch["last_ndvi"].shape[-1]
        dt = torch.full((B, H, P, P), float(self.bin_days), device=batch["last_ndvi"].device)
        if gap is not None:
            dt[:, 0] = gap * self.bin_days
        return dt

    def rates(self, ndvi: torch.Tensor, batch: dict) -> torch.Tensor:
        """Daily rate between consecutive frames incl. the step from the last valid context value. [B,H,P,P]."""
        prev = torch.cat([batch["last_ndvi"][:, None], ndvi[:, :-1]], dim=1)
        return (ndvi - prev) / self._dt(batch, ndvi.shape[1])

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
            dt = self._dt(batch, ndvi.shape[1])
            ri_d = self.r_inc[batch["lcg"]]
            rd_d = self.r_dec[batch["lcg"]]
            prev = batch["last_ndvi"]
            frames = []
            for h in range(ndvi.shape[1]):
                cur = torch.minimum(torch.maximum(ndvi[:, h], prev - rd_d * dt[:, h]), prev + ri_d * dt[:, h])
                frames.append(cur)
                prev = cur
            ndvi = torch.stack(frames, 1)
        from ..data.dataset import ndvi_to_scaled

        return ndvi_to_scaled(ndvi)

    # ---------- violation metrics (fractions, NDVI units) ----------
    @torch.no_grad()
    def violations(self, x0_scaled: torch.Tensor, batch: dict, normal: torch.Tensor, tol: float = 1e-4,
                   mask: torch.Tensor | None = None) -> dict:
        """Violation fractions over pixel-timesteps. Never-observed first steps are excluded from the rate
        check. With `mask` (e.g. observed targets), only valid pixels count; rate steps need both ends valid."""
        ndvi = scaled_to_ndvi(x0_scaled)
        g = self.rates(ndvi, batch)
        ri = self.r_inc[batch["lcg"]][:, None]
        rd = self.r_dec[batch["lcg"]][:, None]
        rate_ex = F.relu(g - ri) + F.relu(-g - rd)
        lo, hi = self.envelope(batch)
        phen_ex = F.relu(ndvi - hi) + normal.view(-1, 1, 1, 1) * F.relu(lo - ndvi)
        rng_ex = F.relu(ndvi - NDVI_MAX) + F.relu(NDVI_MIN - ndvi)
        m = torch.ones_like(ndvi) if mask is None else mask.float()
        first_ok = (batch.get("last_gap", torch.ones_like(batch["last_ndvi"])) < 50).float()
        prev_m = torch.cat([first_ok[:, None], m[:, :-1]], dim=1)
        rm = m * prev_m
        if mask is not None:   # observed series: a valid step needs consecutive valid frames (no gaps)
            rm[:, 0] = rm[:, 0] * (batch.get("last_gap", torch.ones_like(batch["last_ndvi"])) <= 1).float()

        def frac(ex, w):
            return ((ex > tol).float() * w).sum().item() / max(w.sum().item(), 1.0)

        def mag(ex, w):
            sel = (ex > tol) & (w > 0)
            return ex[sel].mean().item() if sel.any() else 0.0
        return {"rate_frac": frac(rate_ex, rm), "rate_mag": mag(rate_ex, rm), "phen_frac": frac(phen_ex, m),
                "phen_mag": mag(phen_ex, m), "range_frac": frac(rng_ex, m)}
