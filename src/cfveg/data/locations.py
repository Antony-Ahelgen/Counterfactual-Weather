"""Sample candidate study locations and assign spatial splits."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sample_candidates(bbox: list[float], n: int, min_spacing: float, seed: int) -> pd.DataFrame:
    """Random points in bbox with pairwise (lon-scaled) spacing >= min_spacing degrees."""
    rng = np.random.default_rng(seed)
    lo0, la0, lo1, la1 = bbox
    pts: list[tuple[float, float]] = []
    tries = 0
    while len(pts) < n and tries < 200_000:
        tries += 1
        lon, lat = rng.uniform(lo0, lo1), rng.uniform(la0, la1)
        ok = True
        for (x, y) in pts:
            dx = (lon - x) * np.cos(np.radians(lat))
            if dx * dx + (lat - y) ** 2 < min_spacing ** 2:
                ok = False
                break
        if ok:
            pts.append((round(lon, 5), round(lat, 5)))
    df = pd.DataFrame(pts, columns=["lon", "lat"])
    df.insert(0, "loc_id", [f"L{i:04d}" for i in range(len(df))])
    return df


def assign_splits(locs: pd.DataFrame, frac_val: float, frac_test: float, seed: int) -> pd.DataFrame:
    """Spatial split by location. Locations are already >= min_spacing apart (buffer)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(locs))
    n_te = int(round(frac_test * len(locs)))
    n_va = int(round(frac_val * len(locs)))
    split = np.array(["train"] * len(locs), dtype=object)
    split[idx[:n_te]] = "test"
    split[idx[n_te:n_te + n_va]] = "val"
    out = locs.copy()
    out["split"] = split
    return out
