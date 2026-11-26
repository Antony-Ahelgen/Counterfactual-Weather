"""ScenarioBuilder: weather sequences for the 'normal' and 'drought' scenarios.

The model is trained on OBSERVED weather. Counterfactuals swap the HORIZON part of the weather
sequence at inference (the context part is the observed past and stays factual in both branches).

normal  = analog-year ensemble: the same horizon bins at the SAME location in TRAINING years
          (splits.train_years only) whose horizon is genuinely near-typical: |mean s| < normal_max_s (neither dry nor unusually wet) and
          at most normal_max_stressed_days days with s >= threshold. Fallback: per-location median
          over training years ("climatology").
drought = observed weather if the factual horizon is a drought; otherwise a drought analog from
          drought-labelled TRAINING-year windows at the same location, then at the nearest train
          locations within max_neighbour_deg; final fallback: the normal scenario perturbed until its
          mean horizon stress exceeds threshold + 0.25 (physics.monotonicity.stress_weather).
Test/val years never enter any analog pool.
"""
from __future__ import annotations

import numpy as np

from .stress import is_drought


class ScenarioBuilder:
    def __init__(self, cube, cfg: dict, n_neighbours: int = 5, max_neighbour_deg: float = 1.5):
        self.cube, self.cfg = cube, cfg
        w = cfg["window"]
        self.K, self.H = w["context"], w["horizon"]
        self.thr = cfg["drought"]["threshold"]
        self.min_days = cfg["drought"]["min_stressed_days"]
        self.bd = cube.meta["bin_days"]
        yrs = np.array(cube.years)
        self.pool_years = np.nonzero(np.isin(yrs, cfg["splits"]["train_years"]))[0]   # year indices
        lon, lat = np.array(cube.meta["lon"]), np.array(cube.meta["lat"])
        self.train_locs = np.nonzero(cube.split == "train")[0]
        self.lon, self.lat = lon, lat
        self.n_nb = n_neighbours
        self.max_nb_deg = max_neighbour_deg
        dcfg = cfg["drought"]
        self.normal_max_s = dcfg.get("normal_max_s", 0.5)
        self.normal_max_days = dcfg.get("normal_max_stressed_days", 10)

    def _drought_label(self, l: int, yi: int, t0: int) -> bool:
        s = self.cube.s_daily[l, yi, (t0 + self.K) * self.bd:(t0 + self.K + self.H) * self.bd]
        return is_drought(s, self.thr, self.min_days)

    def factual(self, l: int, yi: int, t0: int) -> np.ndarray:
        return self.cube.wx[l, yi, t0:t0 + self.K + self.H].astype(np.float32)

    def _swap(self, l: int, yi: int, t0: int, src_l: int, src_y: int) -> np.ndarray:
        w = self.factual(l, yi, t0).copy()
        w[self.K:] = self.cube.wx[src_l, src_y, t0 + self.K:t0 + self.K + self.H]
        return w

    def _horizon_s(self, l: int, yi: int, t0: int) -> np.ndarray:
        return self.cube.s_daily[l, yi, (t0 + self.K) * self.bd:(t0 + self.K + self.H) * self.bd]

    def is_normal(self, l: int, yi: int, t0: int) -> bool:
        """Near-typical horizon: neither dry nor unusually wet (stricter than 'not drought')."""
        s = self._horizon_s(l, yi, t0)
        return bool(abs(np.nanmean(s)) < self.normal_max_s and (s >= self.thr).sum() <= self.normal_max_days)

    def normal_pool(self, l: int, t0: int) -> list[tuple[int, int]]:
        return [(l, y) for y in self.pool_years if self.is_normal(l, y, t0)]

    def normal(self, l: int, yi: int, t0: int) -> tuple[list[np.ndarray], str]:
        """Return list of candidate normal weather sequences (analog ensemble) and the method used."""
        pool = self.normal_pool(l, t0)
        if pool:
            return [self._swap(l, yi, t0, sl, sy) for sl, sy in pool], "analog"
        w = self.factual(l, yi, t0).copy()
        w[self.K:] = np.median(self.cube.wx[l, self.pool_years, t0 + self.K:t0 + self.K + self.H], axis=0)
        return [w], "climatology"

    def neighbours(self, l: int) -> np.ndarray:
        d = (self.lon[self.train_locs] - self.lon[l]) ** 2 * np.cos(np.radians(self.lat[l])) ** 2 + \
            (self.lat[self.train_locs] - self.lat[l]) ** 2
        order = np.argsort(d)
        nb, dd = self.train_locs[order], np.sqrt(d[order])
        keep = (nb != l) & (dd <= self.max_nb_deg)
        return nb[keep][: self.n_nb]

    def drought(self, l: int, yi: int, t0: int) -> tuple[list[np.ndarray], str]:
        if self._drought_label(l, yi, t0):
            return [self.factual(l, yi, t0)], "observed"
        own = [(l, y) for y in self.pool_years if self._drought_label(l, y, t0)]
        if own:
            return [self._swap(l, yi, t0, sl, sy) for sl, sy in own], "analog_same_location"
        nb = [(n, y) for n in self.neighbours(l) for y in self.pool_years if self._drought_label(n, y, t0)]
        if nb:
            return [self._swap(l, yi, t0, sl, sy) for sl, sy in nb], "analog_neighbour"
        import torch

        from ..physics.monotonicity import stress_weather

        base = self.normal(l, yi, t0)[0][0]
        amount = max(1.5, self.thr + 0.25 - self.horizon_stress(base))
        w = stress_weather(torch.from_numpy(base)[None], self.K, amount)[0].numpy()
        return [w], "perturbed_climatology"

    def horizon_stress(self, w: np.ndarray) -> float:
        return float(w[self.K:, -1].mean())
