"""Train model variants.

  python scripts/train_all.py --preset cpu                        # main variants
  python scripts/train_all.py --preset cpu --variants diff_phys_no_rate diff_phys_no_phen diff_phys_no_mono
  python scripts/train_all.py --preset smoke --smoke           # smoke
Variants:
  det                deterministic conditional U-Net (masked L2)
  diff_plain         conditional diffusion, physics off
  diff_phys          conditional diffusion + rate/phen/mono auxiliary losses
  diff_phys_no_X     diff_phys with constraint X removed from the losses (ablation)
  diff_phys_l{a}     diff_phys with all lambdas scaled by a (validation sweep), e.g. diff_phys_l0.3
Checkpoints: runs/<preset>/<variant>_s<seed>/ (resumable).
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfveg.data.dataset import Cube  # noqa: E402
from cfveg.models.train import train  # noqa: E402
from cfveg.utils.provenance import load_preset  # noqa: E402

MAIN = ["det", "diff_plain", "diff_phys"]


def variant_cfg(cfg: dict, v: str) -> tuple[dict, str, bool]:
    c = copy.deepcopy(cfg)
    if v == "det":
        return c, "deterministic", False
    if v == "diff_plain":
        return c, "diffusion", False
    if v.startswith("diff_phys"):
        if v.startswith("diff_phys_no_"):
            drop = v.removeprefix("diff_phys_no_")
            c["physics"]["constraints"] = [k for k in c["physics"]["constraints"] if k != drop]
        elif v.startswith("diff_phys_l"):
            a = float(v.removeprefix("diff_phys_l"))
            for k in ["lambda_rate", "lambda_phen", "lambda_mono"]:
                c["physics"][k] *= a
        return c, "diffusion", True
    raise ValueError(v)


def run_dir(preset: str, v: str, seed: int, smoke: bool) -> Path:
    return ROOT / "runs" / (preset + ("_smoke" if smoke else "")) / f"{v}_s{seed}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="cpu")
    ap.add_argument("--variants", nargs="*", default=MAIN)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="real-data test fixture")
    ap.add_argument("--threads", type=int, default=None)
    a = ap.parse_args()
    cfg = load_preset(a.preset)
    if a.threads:
        cfg["train"]["threads"] = a.threads
    lam_f = ROOT / "results" / ("smoke/" if a.smoke else "") / "val" / "lambda_choice.json"
    lam_scale = json.loads(lam_f.read_text())["lambda_scale"] if lam_f.exists() else 1.0
    pdir = ROOT / ("data/smoke" if a.smoke else cfg["data"]["processed_dir"])
    cube = Cube.load(pdir / "cube")
    stats = json.loads((pdir / "stats.json").read_text())
    phys = json.loads((pdir / "physics.json").read_text())
    seeds = a.seeds if a.seeds is not None else cfg["eval"]["seeds"]
    for seed in seeds:
        for v in a.variants:
            c, kind, use_phys = variant_cfg(cfg, v)
            if use_phys and not v.startswith("diff_phys_l"):
                # lambda scale selected on VALIDATION (run_pipeline stage 3); recorded in train_meta.json
                for k in ["lambda_rate", "lambda_phen", "lambda_mono"]:
                    c["physics"][k] *= lam_scale
                c["physics"]["lambda_scale_selected"] = lam_scale
            out = run_dir(a.preset, v, seed, a.smoke)
            if (out / "final.pt").exists():
                print(f"skip {v} seed {seed}: final.pt exists", flush=True)
                continue
            print(f"=== train {v} seed {seed} -> {out}", flush=True)
            train(c, cube, stats, phys, out, kind=kind, use_physics_loss=use_phys, seed=seed, max_steps=a.steps)


if __name__ == "__main__":
    main()
