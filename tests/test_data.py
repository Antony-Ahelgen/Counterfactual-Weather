import datetime as dt

import numpy as np
import pandas as pd
import pytest

from cfveg.data.cube import CoverageError, check_coverage
from cfveg.data.dataset import (Cube, WindowDataset, fill_context, fit_norm_stats, make_windows,
                                ndvi_to_scaled, scaled_to_ndvi)
from cfveg.data.sentinel2 import bin_index, boa_offset, composite, ndvi_from_dn
from cfveg.data.stress import daily_anomalies, is_drought
from cfveg.data.synthetic import make_synthetic_cube
from cfveg.utils.provenance import load_preset


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    p = tmp_path_factory.mktemp("cube")
    make_synthetic_cube(p, n_loc=10, P=32, seed=1)
    return Cube.load(p)


@pytest.fixture(scope="module")
def cfg():
    return load_preset("smoke")


def test_boa_offset_and_ndvi():
    assert boa_offset("04.00") == -1000 and boa_offset("05.10") == -1000
    assert boa_offset("02.12") == 0 and boa_offset("03.00") == 0 and boa_offset(None) == 0
    # same surface reflectance in old and new baselines must give the same NDVI
    r4, r8 = 0.05, 0.35
    old = ndvi_from_dn(np.array([r4 * 1e4]), np.array([r8 * 1e4]), 0)
    new = ndvi_from_dn(np.array([r4 * 1e4 + 1000]), np.array([r8 * 1e4 + 1000]), -1000)
    assert np.allclose(old, new) and np.allclose(old, (r8 - r4) / (r8 + r4))
    assert np.isnan(ndvi_from_dn(np.array([0.0]), np.array([3000.0]), 0)[0])


def test_bin_index_and_composite():
    assert bin_index(dt.date(2019, 3, 2), 61, 5, 49) == 0
    assert bin_index(dt.date(2019, 1, 10), 61, 5, 49) is None
    st = np.array([[[0.5]], [[np.nan]], [[0.7]]], dtype=np.float32)
    comp, n = composite(st, np.array([0, 0, 1]), 3)
    assert comp[0, 0, 0] == pytest.approx(0.5) and n[0, 0, 0] == 1
    assert comp[1, 0, 0] == pytest.approx(0.7) and np.isnan(comp[2, 0, 0])


def test_coverage_check_fails_on_cloudy():
    cloudy = np.full((5, 10, 8, 8), np.nan)
    cloudy[0, 0] = 0.5
    with pytest.raises(CoverageError):
        check_coverage(cloudy, 0.15)
    ok = np.full((5, 10, 8, 8), 0.5)
    assert check_coverage(ok, 0.15) == 1.0


def test_scaling_roundtrip():
    x = np.linspace(-0.1, 0.9, 11)
    assert np.allclose(scaled_to_ndvi(ndvi_to_scaled(x)), x)
    assert ndvi_to_scaled(0.9) == pytest.approx(1.0) and ndvi_to_scaled(-0.1) == pytest.approx(-1.0)


def test_fill_context_never_zero():
    c = np.array([[[np.nan]], [[0.6]], [[np.nan]]], dtype=np.float32)
    f = fill_context(c)
    assert np.allclose(f[:, 0, 0], [0.6, 0.6, 0.6])


def test_dataset_shapes_and_masks(syn, cfg):
    stats = fit_norm_stats(syn, cfg)
    win = make_windows(syn, cfg, "train")
    assert len(win) > 0
    ds = WindowDataset(syn, cfg, win, stats)
    s = ds[0]
    K, H, P = cfg["window"]["context"], cfg["window"]["horizon"], cfg["window"]["patch"]
    assert s["ctx"].shape == (K, P, P) and s["tgt"].shape == (H, P, P)
    assert s["wx"].shape == (K + H, 11) and s["static"].shape == (6, P, P)
    assert s["tgt_mask"].min() >= 0 and s["tgt_mask"].max() <= 1
    assert np.isfinite(s["tgt"].numpy()).all()


def test_stress_index_and_drought():
    idx = pd.date_range("2010-01-01", "2019-12-31", freq="D")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"precip": rng.gamma(0.5, 4, len(idx)), "et0": 2 + np.sin(idx.day_of_year / 58),
                       "t2m": 10 + 10 * np.sin(idx.day_of_year / 58) + rng.normal(0, 2, len(idx)),
                       "vpd": 0.8 + rng.normal(0, 0.1, len(idx)), "sm": 0.3 + rng.normal(0, 0.02, len(idx))},
                      index=idx)
    dry = df.copy()
    sel = (idx >= "2018-05-01") & (idx <= "2018-08-01")
    dry.loc[sel, "precip"] = 0.0
    dry.loc[sel, "t2m"] += 4
    dry.loc[sel, "vpd"] += 0.5
    dry.loc[sel, "sm"] -= 0.08
    w = {"precip_deficit": 1.0, "temp": 0.5, "vpd": 0.5, "soil_moisture": 1.0}
    a = daily_anomalies(dry, list(range(2010, 2018)), w)
    s = a["s"]
    clim = s[(s.index.year < 2018) & s.notna()]
    assert abs(clim.mean()) < 0.1 and 0.8 < clim.std() < 1.2
    assert s["2018-07-01":"2018-07-31"].mean() > 1.5
    assert is_drought(s["2018-07-01":"2018-08-09"].values, 1.0, 30)
    # an undisturbed copy of the same weather must not be labelled drought in July 2018
    s_norm = daily_anomalies(df, list(range(2010, 2018)), w)["s"]
    assert s_norm["2018-07-01":"2018-07-31"].mean() < s["2018-07-01":"2018-07-31"].mean() - 1.0
