"""Training loop for the diffusion model (with optional physics losses) and the deterministic baseline."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..data.dataset import Cube, WindowDataset, make_windows
from ..physics.constraints import Physics
from ..physics.monotonicity import stress_weather, with_weather
from ..utils.provenance import seed_everything
from .diffusion import EMA, Diffusion
from .unet import build_unet


def collate(samples: list[dict]) -> dict:
    return {k: torch.stack([s[k] for s in samples]) for k in samples[0]}


def infinite_loader(ds, batch_size: int, seed: int, num_workers: int = 0):
    g = torch.Generator().manual_seed(seed)
    while True:
        dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True,
                                         collate_fn=collate, generator=g, num_workers=num_workers,
                                         persistent_workers=False)
        yield from dl


def physics_aux_losses(model, diff: Diffusion, phys: Physics, batch: dict, x_t, t, x0_hat, pcfg: dict,
                       wx_mean, wx_std, K: int) -> dict:
    """SNR-weighted auxiliary physics losses on x0_hat. Returns dict of scalar tensors."""
    w = diff.snr_weight(t)                                   # [B]
    cons = set(pcfg["constraints"])
    out = {}
    if "rate" in cons and pcfg["lambda_rate"] > 0:
        out["rate"] = pcfg["lambda_rate"] * (w * phys.rate_energy(x0_hat, batch)).mean()
    if "phen" in cons and pcfg["lambda_phen"] > 0:
        normal = 1.0 - batch["drought"]
        out["phen"] = pcfg["lambda_phen"] * (w * phys.phen_energy(x0_hat, batch, normal)).mean()
    if "mono" in cons and pcfg["lambda_mono"] > 0:
        amt = torch.empty(x_t.shape[0]).uniform_(0.5, 1.5)
        b2 = with_weather(batch, stress_weather(batch["wx_raw"], K, amt), wx_mean, wx_std, K)
        x0_s = diff.x0_from_out(x_t, t, model(x_t, t, b2))
        out["mono"] = pcfg["lambda_mono"] * (w * phys.mono_energy(x0_s, x0_hat, pcfg["mono_eps"])).mean()
    return out


@torch.no_grad()
def val_loss(model, diff: Diffusion | None, loader_batches: list[dict], kind: str) -> float:
    model.eval()
    tot, n = 0.0, 0
    gen = torch.Generator().manual_seed(123)
    for b in loader_batches:
        B = b["tgt"].shape[0]
        if kind == "diffusion":
            t = torch.randint(0, diff.T, (B,), generator=gen)
            noise = torch.randn(b["tgt"].shape, generator=gen)
            l, *_ = diff.loss(model, b, t=t, noise=noise)
        else:
            pred = model(None, torch.zeros(B, dtype=torch.long), b)
            m = b["tgt_mask"]
            l = (((pred - b["tgt"]) ** 2 * m).sum() / m.sum().clamp(min=1))
        tot += l.item() * B
        n += B
    model.train()
    return tot / max(n, 1)


def train(cfg: dict, cube: Cube, stats: dict, phys_fit: dict | None, out_dir: Path, kind: str = "diffusion",
          use_physics_loss: bool = False, seed: int = 0, resume: bool = True, max_steps: int | None = None) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tc, pcfg = cfg["train"], cfg["physics"]
    torch.set_num_threads(tc.get("threads", 8))
    seed_everything(seed)
    K = cfg["window"]["context"]
    train_win = make_windows(cube, cfg, "train")
    val_win = make_windows(cube, cfg, "val")
    ds = WindowDataset(cube, cfg, train_win, stats)
    vds = WindowDataset(cube, cfg, val_win, stats)
    rng = np.random.default_rng(0)
    vsel = rng.choice(len(vds), size=min(len(vds), 8 * tc["batch_size"]), replace=False) if len(vds) else []
    vbatches = [collate([vds[i] for i in vsel[j:j + tc["batch_size"]]]) for j in range(0, len(vsel), tc["batch_size"])]
    model = build_unet(cfg, diffusion=(kind == "diffusion"))
    diff = Diffusion(cfg["diffusion"]["timesteps"], cfg["diffusion"]["prediction"]) if kind == "diffusion" else None
    phys = Physics(phys_fit, cfg) if (phys_fit is not None and use_physics_loss) else None
    wx_mean = torch.tensor(stats["wx_mean"])
    wx_std = torch.tensor(stats["wx_std"])
    opt = torch.optim.AdamW(model.parameters(), lr=tc["lr"], weight_decay=1e-4)
    ema = EMA(model, cfg["diffusion"]["ema_decay"])
    steps = max_steps or tc["steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 500) *
                                              (0.5 * (1 + np.cos(np.pi * min(s, steps) / steps)) * 0.9 + 0.1))
    step = 0
    ck = out_dir / "last.pt"
    if resume and ck.exists():
        st = torch.load(ck, weights_only=False)
        model.load_state_dict(st["model"])
        ema.model.load_state_dict(st["ema"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        print(f"resumed from step {step}", flush=True)
    meta = {"kind": kind, "physics_loss": use_physics_loss, "seed": seed, "n_train_windows": len(train_win),
            "n_val_windows": len(val_win), "steps": steps, "cfg": cfg}
    (out_dir / "train_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    log_f = open(out_dir / "log.csv", "a", newline="")
    log = csv.writer(log_f)
    if step == 0:
        log.writerow(["step", "loss", "diff", "rate", "phen", "mono", "val", "sec_per_step"])
    it = infinite_loader(ds, tc["batch_size"], seed + step, tc.get("num_workers", 0))
    best = float("inf")
    t0 = time.time()
    model.train()
    print(f"[{kind}] train windows={len(train_win)} val={len(val_win)} steps={steps} params="
          f"{sum(p.numel() for p in model.parameters()) / 1e6:.2f}M physics={use_physics_loss}", flush=True)
    while step < steps:
        b = next(it)
        aux = {}
        if kind == "diffusion":
            l_diff, x_t, t, x0_hat = diff.loss(model, b)
            if phys is not None and pcfg.get("use_loss", True):
                aux = physics_aux_losses(model, diff, phys, b, x_t, t, x0_hat, pcfg, wx_mean, wx_std, K)
        else:
            pred = model(None, torch.zeros(b["tgt"].shape[0], dtype=torch.long), b)
            m = b["tgt_mask"]
            l_diff = ((pred - b["tgt"]) ** 2 * m).sum() / m.sum().clamp(min=1)
        loss = l_diff + sum(aux.values()) if aux else l_diff
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        sched.step()
        ema.update(model, step)
        step += 1
        vl = ""
        if step % tc["val_every"] == 0 or step == steps:
            vl = val_loss(ema.model, diff, vbatches, kind) if vbatches else float("nan")
            if vl < best:
                best = vl
                torch.save({"ema": ema.model.state_dict(), "step": step, "val": vl}, out_dir / "best.pt")
        if step % tc["log_every"] == 0 or vl != "":
            sps = (time.time() - t0) / tc["log_every"] if step % tc["log_every"] == 0 else ""
            t0 = time.time() if step % tc["log_every"] == 0 else t0
            row = [step, f"{loss.item():.5f}", f"{l_diff.item():.5f}"] + \
                  [f"{aux[k].item():.5f}" if k in aux else "" for k in ["rate", "phen", "mono"]] + \
                  [f"{vl:.5f}" if vl != "" else "", f"{sps:.3f}" if sps != "" else ""]
            log.writerow(row)
            log_f.flush()
            print(" ".join(map(str, row)), flush=True)
        if step % tc["ckpt_every"] == 0 or step == steps:
            torch.save({"model": model.state_dict(), "ema": ema.model.state_dict(), "opt": opt.state_dict(),
                        "sched": sched.state_dict(), "step": step}, ck)
    torch.save({"ema": ema.model.state_dict(), "step": step}, out_dir / "final.pt")
    log_f.close()
    return out_dir / "final.pt"


def load_model(cfg: dict, path: Path, kind: str = "diffusion"):
    m = build_unet(cfg, diffusion=(kind == "diffusion"))
    st = torch.load(path, weights_only=False)
    m.load_state_dict(st["ema"])
    m.eval()
    return m
