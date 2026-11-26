import numpy as np
import pytest
import torch

from cfveg.data.dataset import Cube, WindowDataset, fit_norm_stats, make_windows, ndvi_to_scaled
from cfveg.data.scenarios import ScenarioBuilder
from cfveg.models.train import collate
from cfveg.physics.constraints import Physics, fit_physics
from cfveg.physics.monotonicity import stress_weather
from cfveg.utils.provenance import load_preset


def _phys(cfg, r=0.01, mu=0.5, sd=0.05):
    G, B = 4, 49
    return Physics({"r_inc": [r] * G, "r_dec": [r] * G, "mu": np.full((G, B), mu).tolist(),
                    "sigma": np.full((G, B), sd).tolist()}, cfg)


def _batch(H, P=4, last=0.5, t0=0):
    return {"last_ndvi": torch.full((1, P, P), last), "lcg": torch.zeros((1, P, P), dtype=torch.long),
            "idx": torch.tensor([[0, 0, t0, 0, 0]])}


@pytest.fixture(scope="module")
def cfg():
    return load_preset("smoke")


def test_rate_units_are_unscaled_ndvi_per_day(cfg):
    ph = _phys(cfg, r=0.01)          # 0.01 NDVI/day -> 0.05 per 5-day bin
    H = cfg["window"]["horizon"]
    b = _batch(H)
    ok = ndvi_to_scaled(torch.tensor([0.5 + 0.04 * (h + 1) for h in range(H)]).view(1, H, 1, 1).repeat(1, 1, 4, 4))
    bad = ndvi_to_scaled(torch.tensor([0.5 + 0.08 * (h + 1) for h in range(H)]).view(1, H, 1, 1).repeat(1, 1, 4, 4))
    assert ph.rate_energy(ok, b).item() == 0.0
    assert ph.rate_energy(bad, b).item() > 0
    v = ph.violations(bad, b, torch.ones(1))
    assert v["rate_frac"] == 1.0
    # a 0.04/bin step is 0.08 in scaled units: if bounds were (wrongly) applied in scaled space with
    # the same numeric bound it would be flagged; in NDVI units it must not be.
    assert ph.violations(ok, b, torch.ones(1))["rate_frac"] == 0.0


def test_rate_includes_first_step_from_context(cfg):
    ph = _phys(cfg, r=0.01)
    H = cfg["window"]["horizon"]
    b = _batch(H, last=0.2)
    x = ndvi_to_scaled(torch.full((1, H, 4, 4), 0.6))     # jump of 0.4 in the first step only
    assert ph.rate_energy(x, b).item() > 0


def test_phenology_envelope(cfg):
    ph = _phys(cfg, mu=0.5, sd=0.05)   # envelope [0.4, 0.6]
    H = cfg["window"]["horizon"]
    b = _batch(H)
    inside = ndvi_to_scaled(torch.full((1, H, 4, 4), 0.55))
    above = ndvi_to_scaled(torch.full((1, H, 4, 4), 0.75))
    below = ndvi_to_scaled(torch.full((1, H, 4, 4), 0.25))
    one, zero = torch.ones(1), torch.zeros(1)
    assert ph.phen_energy(inside, b, one).item() == 0
    assert ph.phen_energy(above, b, one).item() > 0 and ph.phen_energy(above, b, zero).item() > 0
    assert ph.phen_energy(below, b, one).item() > 0       # normal: lower bound active
    assert ph.phen_energy(below, b, zero).item() == 0     # drought: may fall below


def test_monotonicity_energy():
    base = ndvi_to_scaled(torch.full((1, 2, 3, 3), 0.5))
    lower = ndvi_to_scaled(torch.full((1, 2, 3, 3), 0.45))
    higher = ndvi_to_scaled(torch.full((1, 2, 3, 3), 0.6))
    from cfveg.physics.constraints import Physics as P

    assert P.mono_energy(lower, base, 0.01).item() == 0
    assert P.mono_energy(higher, base, 0.01).item() > 0
    tiny = ndvi_to_scaled(torch.full((1, 2, 3, 3), 0.505))
    assert P.mono_energy(tiny, base, 0.01).item() == 0     # within tolerance


def test_stress_weather_increases_s():
    w = torch.randn(3, 8, 11)
    w[..., 0] = w[..., 0].abs() * 5
    w[..., 4] = 0.3
    w2 = stress_weather(w, 4, torch.tensor([0.5, 1.0, 1.5]))
    assert torch.equal(w2[:, :4], w[:, :4])                # context untouched
    assert (w2[:, 4:, 10] > w[:, 4:, 10]).all()
    assert (w2[:, 4:, 0] <= w[:, 4:, 0]).all() and (w2[:, 4:, 1] > w[:, 4:, 1]).all()


def test_projection_enforces_bounds(cfg):
    ph = _phys(cfg, r=0.01)
    H = cfg["window"]["horizon"]
    b = _batch(H, last=0.5)
    x = ndvi_to_scaled(torch.tensor([0.9, 0.1, 1.5, -2.0][:H] + [0.5] * max(0, H - 4)).view(1, H, 1, 1)
                       .repeat(1, 1, 4, 4))
    p = ph.project(x, b)
    assert ph.violations(p, b, torch.zeros(1))["rate_frac"] == 0.0
    assert ph.violations(p, b, torch.zeros(1))["range_frac"] == 0.0


@pytest.fixture(scope="module")
def syn(real_cube):
    return real_cube


def test_fit_physics_reasonable(syn, cfg):
    f = fit_physics(syn, cfg)
    assert all(0 < r < 0.2 for r in f["r_inc"]) and all(0 < r < 0.2 for r in f["r_dec"])
    mu = np.array(f["mu"])
    assert np.isfinite(mu).all() and mu.shape[1] == syn.ndvi.shape[2]


def test_scenarios_stress_levels(syn, cfg):
    sb = ScenarioBuilder(syn, cfg)
    thr = cfg["drought"]["threshold"]
    wins = make_windows(syn, cfg, "test")
    ns, ds_ = [], []
    for l, y, t0, _, _ in wins[::4][:30]:
        n, _ = sb.normal(l, y, t0)
        d, _ = sb.drought(l, y, t0)
        ns.append(np.mean([sb.horizon_stress(w) for w in n]))
        ds_.append(np.mean([sb.horizon_stress(w) for w in d]))
    assert abs(np.mean(ns)) < 0.75            # normal scenario ~ no stress
    assert np.mean(ds_) > thr * 0.9           # drought scenario past (or at) the threshold
    assert all(d > n for n, d in zip(ns, ds_))
