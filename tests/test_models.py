import numpy as np
import pytest
import torch

from cfveg.data.dataset import Cube, WindowDataset, fit_norm_stats, make_windows
from cfveg.data.synthetic import make_synthetic_cube
from cfveg.models.diffusion import Diffusion, cosine_alpha_bar
from cfveg.models.train import collate
from cfveg.models.unet import build_unet
from cfveg.utils.provenance import load_preset, seed_everything


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    p = tmp_path_factory.mktemp("cube")
    make_synthetic_cube(p, n_loc=8, P=32, seed=2)
    cube = Cube.load(p)
    cfg = load_preset("smoke")
    stats = fit_norm_stats(cube, cfg)
    ds = WindowDataset(cube, cfg, make_windows(cube, cfg, "train"), stats)
    batch = collate([ds[i] for i in range(4)])
    return cfg, batch


def test_schedule_monotone():
    ab = cosine_alpha_bar(1000)
    assert ab.shape == (1000,)
    assert (ab[1:] <= ab[:-1]).all() and ab[0] > 0.99 and ab[-1] < 1e-3


def test_v_prediction_roundtrip():
    d = Diffusion(100, "v")
    x0 = torch.randn(2, 4, 8, 8)
    n = torch.randn_like(x0)
    t = torch.tensor([3, 70])
    xt = d.q_sample(x0, t, n)
    v = d.target(x0, t, n)
    assert torch.allclose(d.x0_from_out(xt, t, v), x0, atol=1e-4)
    assert torch.allclose(d.eps_from_x0(xt, t, x0), n, atol=1e-3)


def test_unet_shapes(setup):
    cfg, b = setup
    m = build_unet(cfg)
    out = m(torch.randn_like(b["tgt"]), torch.zeros(4, dtype=torch.long), b)
    assert out.shape == b["tgt"].shape
    md = build_unet(cfg, diffusion=False)
    assert md(None, torch.zeros(4, dtype=torch.long), b).shape == b["tgt"].shape


def test_masked_loss_ignores_invalid(setup):
    cfg, b = setup
    seed_everything(0)
    m = build_unet(cfg)
    d = Diffusion(100)
    t = torch.full((4,), 10)
    n = torch.randn_like(b["tgt"])
    l1, *_ = d.loss(m, b, t=t, noise=n)
    b2 = dict(b)
    b2["tgt"] = torch.where(b["tgt_mask"] > 0, b["tgt"], b["tgt"] + 100.0)   # corrupt invalid pixels
    # x_t changes too, so compare loss terms computed with the corrupted target only on masked pixels
    out = m(d.q_sample(b["tgt"], t, n), t, b)
    tgt1 = d.target(b["tgt"], t, n)
    tgt2 = d.target(b2["tgt"], t, n)
    m_ = b["tgt_mask"]
    assert torch.allclose(((out - tgt1) ** 2 * m_).sum(), ((out - tgt2) ** 2 * m_).sum())


def test_overfit_tiny_batch(setup):
    cfg, b = setup
    seed_everything(0)
    m = build_unet(cfg, diffusion=False)
    opt = torch.optim.Adam(m.parameters(), lr=3e-3)
    mask = b["tgt_mask"]
    first = None
    for _ in range(150):
        pred = m(None, torch.zeros(4, dtype=torch.long), b)
        loss = ((pred - b["tgt"]) ** 2 * mask).sum() / mask.sum()
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first


def test_diffusion_loss_decreases(setup):
    cfg, b = setup
    seed_everything(0)
    m = build_unet(cfg)
    d = Diffusion(100)
    opt = torch.optim.Adam(m.parameters(), lr=2e-3)
    g = torch.Generator().manual_seed(0)
    ts = torch.randint(0, 100, (8, 4), generator=g)
    ns = torch.randn((8, *b["tgt"].shape), generator=g)

    def fixed_loss():
        with torch.no_grad():
            return np.mean([d.loss(m, b, t=ts[i], noise=ns[i])[0].item() for i in range(8)])

    before = fixed_loss()
    for k in range(200):
        l, *_ = d.loss(m, b, t=ts[k % 8], noise=ns[k % 8])
        opt.zero_grad()
        l.backward()
        opt.step()
    assert fixed_loss() < 0.5 * before


def test_ddim_deterministic_given_noise(setup):
    cfg, b = setup
    seed_everything(0)
    m = build_unet(cfg).eval()
    d = Diffusion(100)
    n = torch.randn_like(b["tgt"])
    a = d.sample(m, b, n, steps=5)
    c = d.sample(m, b, n.clone(), steps=5)
    assert torch.equal(a, c)
