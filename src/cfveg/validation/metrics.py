"""Metrics (all in NDVI units, masked so invalid/cloudy pixels never contribute).

Accuracy:      masked MAE, RMSE; SSIM on frames with >= 70 % valid pixels, averaged over 7x7 windows
               centred on valid pixels (invalid pixels inside a window are filled with the prediction).
Sharpness:     mean spatial gradient magnitude and high-frequency share of the 2-D power spectrum
               (power at radial frequency > 0.25 cycles/pixel, i.e. above half Nyquist), computed on
               the same near-cloud-free frames for prediction and observation. Ratio pred/obs < 1
               means the forecast is blurrier than reality.
Probabilistic: fair ensemble CRPS, central-interval coverage (50/80/90 %), spread-skill ratio,
               PIT/rank histogram.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

FULL_FRAME = 0.7


def masked_err(pred: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> dict:
    """Returns sums so results can be aggregated over batches: abs_sum, sq_sum, n."""
    d = (pred - obs) * mask
    return {"abs_sum": d.abs().sum().item(), "sq_sum": (d ** 2).sum().item(), "n": mask.sum().item()}


def _gauss(win: int = 7, sigma: float = 1.5) -> torch.Tensor:
    x = torch.arange(win, dtype=torch.float32) - win // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] * g[None]).view(1, 1, win, win)


def ssim_frames(pred: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, data_range: float = 1.0) -> list[float]:
    """SSIM per near-cloud-free frame. pred/obs/mask [B, H, P, P]."""
    vals = []
    k = _gauss()
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    frac = mask.mean((2, 3))
    for b, h in zip(*torch.nonzero(frac >= FULL_FRAME, as_tuple=True)):
        m = mask[b, h]
        x = pred[b, h][None, None]
        y = torch.where(m > 0, obs[b, h], pred[b, h])[None, None]
        mx, my = F.conv2d(x, k), F.conv2d(y, k)
        sxx = F.conv2d(x * x, k) - mx ** 2
        syy = F.conv2d(y * y, k) - my ** 2
        sxy = F.conv2d(x * y, k) - mx * my
        s = ((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))
        r = k.shape[-1] // 2
        mc = m[r:-r, r:-r]                       # only windows centred on a valid observed pixel count
        if mc.sum() > 0:
            vals.append(((s[0, 0] * mc).sum() / mc.sum()).item())
    return vals


def grad_mag(x: torch.Tensor) -> torch.Tensor:
    gx = x[..., :, 1:] - x[..., :, :-1]
    gy = x[..., 1:, :] - x[..., :-1, :]
    return 0.5 * (gx.abs().mean((-2, -1)) + gy.abs().mean((-2, -1)))


def hf_share(x: torch.Tensor, cutoff: float = 0.25) -> torch.Tensor:
    """Share of (mean-removed) spectral power above `cutoff` cycles/pixel. x [..., P, P]."""
    P = x.shape[-1]
    xm = x - x.mean((-2, -1), keepdim=True)
    ps = torch.fft.fft2(xm).abs() ** 2
    f = torch.fft.fftfreq(P)
    r = torch.sqrt(f[:, None] ** 2 + f[None] ** 2)
    hi = (r > cutoff).float()
    tot = ps.sum((-2, -1)).clamp(min=1e-12)
    return (ps * hi).sum((-2, -1)) / tot


def sharpness(pred: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> dict:
    """Sharpness on frames with >= 95 % valid pixels (gaps filled by the prediction)."""
    frac = mask.mean((2, 3))
    sel = frac >= 0.95
    if not sel.any():
        return {"grad_pred": [], "grad_obs": [], "hf_pred": [], "hf_obs": []}
    p = pred[sel]
    o = torch.where(mask[sel] > 0, obs[sel], pred[sel])
    return {"grad_pred": grad_mag(p).tolist(), "grad_obs": grad_mag(o).tolist(),
            "hf_pred": hf_share(p).tolist(), "hf_obs": hf_share(o).tolist()}


def crps_ensemble(ens: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> dict:
    """Fair CRPS. ens [M, B, H, P, P]. Returns sum and count."""
    M = ens.shape[0]
    t1 = (ens - obs[None]).abs().mean(0)
    # E|X - X'| with fair (M-1) normalisation, via sorted ensemble
    s, _ = torch.sort(ens, dim=0)
    i = torch.arange(1, M + 1, dtype=ens.dtype).view(-1, *([1] * (ens.ndim - 1)))
    emd = (2 * (2 * i - M - 1) * s).sum(0) / (M * (M - 1))
    crps = t1 - 0.5 * emd
    return {"crps_sum": (crps * mask).sum().item(), "n": mask.sum().item()}


def coverage(ens: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, levels=(0.5, 0.8, 0.9)) -> dict:
    out = {}
    for lv in levels:
        lo = torch.quantile(ens, (1 - lv) / 2, dim=0)
        hi = torch.quantile(ens, 1 - (1 - lv) / 2, dim=0)
        inside = ((obs >= lo) & (obs <= hi)).float()
        out[f"cov{int(lv * 100)}_sum"] = (inside * mask).sum().item()
    out["n"] = mask.sum().item()
    return out


def spread_skill(ens: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor) -> dict:
    M = ens.shape[0]
    mean = ens.mean(0)
    var = ens.var(0, unbiased=True)
    return {"var_sum": (var * mask).sum().item(),
            "sqerr_sum": (((mean - obs) ** 2) * mask).sum().item() * M / (M + 1),
            "n": mask.sum().item()}


def rank_hist(ens: torch.Tensor, obs: torch.Tensor, mask: torch.Tensor, rng: np.random.Generator) -> np.ndarray:
    """Counts of the observation's rank among members (0..M), random tie-breaking."""
    M = ens.shape[0]
    below = (ens < obs[None]).sum(0)
    ties = (ens == obs[None]).sum(0)
    r = below + torch.from_numpy(rng.integers(0, 1 << 30, size=tuple(below.shape))) % (ties + 1)
    r = r[mask > 0].numpy()
    return np.bincount(r, minlength=M + 1)


class Accumulator:
    """Accumulates summed metric pieces across batches and finalises them."""

    def __init__(self):
        self.s: dict[str, float] = {}
        self.lists: dict[str, list] = {}
        self.hist = None

    def add(self, d: dict, prefix: str = "") -> None:
        for k, v in d.items():
            if isinstance(v, list):
                self.lists.setdefault(prefix + k, []).extend(v)
            else:
                self.s[prefix + k] = self.s.get(prefix + k, 0.0) + v

    def add_hist(self, h: np.ndarray) -> None:
        self.hist = h if self.hist is None else self.hist + h

    def finalize(self) -> dict:
        s, L = self.s, self.lists
        out = {}
        if "n" in s:
            out["mae"] = s["abs_sum"] / s["n"]
            out["rmse"] = np.sqrt(s["sq_sum"] / s["n"])
        if "ssim" in L and L["ssim"]:
            out["ssim"] = float(np.mean(L["ssim"]))
        for k in ["grad", "hf"]:
            if L.get(f"{k}_pred"):
                out[f"{k}_pred"] = float(np.mean(L[f"{k}_pred"]))
                out[f"{k}_obs"] = float(np.mean(L[f"{k}_obs"]))
                out[f"{k}_ratio"] = out[f"{k}_pred"] / out[f"{k}_obs"]
        if "p_n" in s:
            out["crps"] = s["p_crps_sum"] / s["p_n"]
        if "c_n" in s:
            for lv in (50, 80, 90):
                out[f"cov{lv}"] = s[f"c_cov{lv}_sum"] / s["c_n"]
        if "ss_n" in s:
            out["spread_skill"] = float(np.sqrt(s["ss_var_sum"] / s["ss_n"]) / np.sqrt(s["ss_sqerr_sum"] / s["ss_n"]))
        if self.hist is not None:
            h = self.hist / self.hist.sum()
            out["rank_hist"] = h.tolist()
        return out
