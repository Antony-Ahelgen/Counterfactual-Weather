"""Construct a strictly more stressed weather sequence w' from w (horizon frames only).

Raw features (see cube.WX_NAMES): precip, t2m, vpd, srad, sm, et0, a_wb, a_t, a_vpd, a_sm, s.
A stress increment `amount` (in units of the standardised stress index) is applied as:
  precip *= (1 - 0.25*amount) (floored at 0), t2m += 1.0*amount, vpd *= (1 + 0.15*amount),
  sm *= (1 - 0.08*amount), et0 *= (1 + 0.08*amount), all anomalies a_* += amount, s += amount.
so s' >= s by construction (tested).
"""
from __future__ import annotations

import torch

IDX = {"precip": 0, "t2m": 1, "vpd": 2, "srad": 3, "sm": 4, "et0": 5, "a_wb": 6, "a_t": 7, "a_vpd": 8,
       "a_sm": 9, "s": 10}


def stress_weather(wx_raw: torch.Tensor, K: int, amount: torch.Tensor | float) -> torch.Tensor:
    """wx_raw [B, K+H, F] raw units; amount scalar or [B] (>= 0). Returns perturbed copy."""
    w = wx_raw.clone()
    a = torch.as_tensor(amount, dtype=w.dtype, device=w.device)
    a = a.view(-1, 1) if a.ndim == 1 else a
    h = slice(K, None)
    w[:, h, IDX["precip"]] = (w[:, h, IDX["precip"]] * (1 - 0.25 * a)).clamp(min=0)
    w[:, h, IDX["t2m"]] = w[:, h, IDX["t2m"]] + 1.0 * a
    w[:, h, IDX["vpd"]] = w[:, h, IDX["vpd"]] * (1 + 0.15 * a)
    w[:, h, IDX["sm"]] = w[:, h, IDX["sm"]] * (1 - 0.08 * a)
    w[:, h, IDX["et0"]] = w[:, h, IDX["et0"]] * (1 + 0.08 * a)
    for k in ["a_wb", "a_t", "a_vpd", "a_sm", "s"]:
        w[:, h, IDX[k]] = w[:, h, IDX[k]] + a
    return w


def with_weather(batch: dict, wx_raw_new: torch.Tensor, wx_mean: torch.Tensor, wx_std: torch.Tensor,
                 K: int, drought: torch.Tensor | None = None) -> dict:
    """Return a shallow copy of batch conditioned on a new raw weather sequence."""
    b = dict(batch)
    b["wx_raw"] = wx_raw_new
    b["wx"] = (wx_raw_new - wx_mean) / wx_std
    b["stress"] = wx_raw_new[:, K:, IDX["s"]].mean(1)
    if drought is not None:
        b["drought"] = drought
    return b
