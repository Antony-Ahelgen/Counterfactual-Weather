"""Seeding, provenance and small IO helpers."""
from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover
        pass


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts", "configs"],
                               cwd=ROOT, capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:  # pragma: no cover
        return "unknown"


def write_provenance(result_path: str | Path, config: dict | None = None, seed: int | None = None,
                     extra: dict | None = None) -> Path:
    """Write <result_path stem>.provenance.json next to a result file."""
    result_path = Path(result_path)
    prov = {
        "result_file": result_path.name,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "git_commit": git_commit(),
        "seed": seed,
        "config": config,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    if extra:
        prov.update(extra)
    out = result_path.with_name(result_path.stem + ".provenance.json")
    out.write_text(json.dumps(prov, indent=2, default=str))
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    import yaml

    with open(path) as f:
        return yaml.safe_load(f)


def load_preset(name: str) -> dict[str, Any]:
    """Merge configs/base.yaml with configs/presets/<name>.yaml (shallow per top-level key)."""
    base = load_yaml(ROOT / "configs" / "base.yaml")
    over = load_yaml(ROOT / "configs" / "presets" / f"{name}.yaml") or {}
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = {**base[k], **v}
        else:
            base[k] = v
    base["preset"] = name
    return base
