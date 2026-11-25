"""Leakage tests: anything fitted must be invariant to val/test locations and val/test years.

Strategy: fit on a cube, then corrupt every value belonging to val/test locations OR val/test years
(NDVI, weather, stress, static of held-out locations) and refit. Outputs must be identical.
DO NOT weaken these tests.
"""
import copy

import numpy as np
import pytest

from cfveg.data.dataset import Cube, fit_norm_stats, make_windows
from cfveg.data.scenarios import ScenarioBuilder
from cfveg.data.synthetic import make_synthetic_cube
from cfveg.models.baselines import fit_pixel_climatology
from cfveg.physics.constraints import fit_physics
from cfveg.utils.provenance import load_preset


@pytest.fixture(scope="module")
def cfg():
    return load_preset("smoke")


@pytest.fixture(scope="module")
def cubes(tmp_path_factory, cfg):
    p = tmp_path_factory.mktemp("cube")
    make_synthetic_cube(p, n_loc=10, P=32, seed=3)
    clean = Cube.load(p, mmap=False)
    bad = copy.deepcopy(clean)
    bad.ndvi = np.array(bad.ndvi, dtype=np.float32)
    held_loc = bad.split != "train"
    yrs = np.array(bad.years)
    held_yr = np.isin(yrs, cfg["splits"]["val_years"] + cfg["splits"]["test_years"])
    rng = np.random.default_rng(0)
    for arr in [bad.ndvi, bad.wx, bad.s_daily]:
        arr[held_loc] = rng.normal(5, 3, arr[held_loc].shape)
        arr[:, held_yr] = rng.normal(-5, 3, arr[:, held_yr].shape)
    bad.static[held_loc] = 1e6
    return clean, bad


def _eq(a, b):
    if isinstance(a, dict):
        return all(_eq(a[k], b[k]) for k in a if k not in ("count",))
    if isinstance(a, str):
        return a == b
    return np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)


def test_norm_stats_train_only(cubes, cfg):
    c, b = cubes
    assert _eq(fit_norm_stats(c, cfg), fit_norm_stats(b, cfg))


def test_physics_fit_train_only(cubes, cfg):
    c, b = cubes
    assert _eq(fit_physics(c, cfg), fit_physics(b, cfg))


def test_pixel_climatology_no_heldout_years(cubes, cfg):
    c, b = cubes
    phys = fit_physics(c, cfg)
    yrs = np.array(c.years)
    held_yr = np.isin(yrs, cfg["splits"]["val_years"] + cfg["splits"]["test_years"])
    b2 = copy.deepcopy(c)
    b2.ndvi = np.array(c.ndvi, dtype=np.float32)
    b2.ndvi[:, held_yr] = 9.0
    assert np.allclose(fit_pixel_climatology(c, cfg, phys), fit_pixel_climatology(b2, cfg, phys))


def test_windows_respect_splits(cubes, cfg):
    c, _ = cubes
    s = cfg["splits"]
    yrs = np.array(c.years)
    for name, locs, years in [("train", "train", s["train_years"]), ("val", "val", s["val_years"]),
                              ("test", "test", s["test_years"])]:
        w = make_windows(c, cfg, name)
        assert len(w) > 0
        assert (c.split[w[:, 0]] == locs).all()
        assert np.isin(yrs[w[:, 1]], years).all()


def test_scenario_pools_exclude_heldout_years(cubes, cfg):
    c, _ = cubes
    sb = ScenarioBuilder(c, cfg)
    yrs = np.array(c.years)
    assert np.isin(yrs[sb.pool_years], cfg["splits"]["train_years"]).all()
    # analog weather for a test window must be invariant to corrupting all held-out years
    b2 = copy.deepcopy(c)
    held_yr = np.isin(yrs, cfg["splits"]["val_years"] + cfg["splits"]["test_years"])
    w = make_windows(c, cfg, "test")[0]
    l, yi, t0 = int(w[0]), int(w[1]), int(w[2])
    ref_n = sb.normal(l, yi, t0)[0]
    b2.wx = c.wx.copy()
    b2.s_daily = c.s_daily.copy()
    other = held_yr.copy()
    other[yi] = False          # the factual year itself provides the observed context only
    b2.wx[:, other] = 99.0
    b2.s_daily[:, other] = 99.0
    sb2 = ScenarioBuilder(b2, cfg)
    new_n = sb2.normal(l, yi, t0)[0]
    assert len(ref_n) == len(new_n) and all(np.allclose(a, b) for a, b in zip(ref_n, new_n))
    # neighbour pools use train locations only
    assert (c.split[sb.neighbours(l)] == "train").all()
