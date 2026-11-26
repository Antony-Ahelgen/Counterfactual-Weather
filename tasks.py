"""Cross-platform task runner (Windows has no `make`). Mirrors the Makefile targets.

Usage:  python tasks.py <target> [--preset cpu]
Targets: test, smoke, data, train, eval, paper-assets, audit
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    r = subprocess.run(list(args), cwd=ROOT)
    if r.returncode != 0:
        sys.exit(r.returncode)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target")
    ap.add_argument("--preset", default="cpu")
    a = ap.parse_args()
    p = a.preset
    t = a.target
    if t == "test":
        run(PY, "-m", "pytest", "-m", "not slow")
    elif t == "smoke":
        run(PY, "scripts/smoke.py")
    elif t == "data":
        run(PY, "scripts/build_data.py", "--preset", p)
    elif t == "train":
        run(PY, "scripts/run_pipeline.py", "--preset", p)
    elif t == "eval":
        # evaluation, effects, validation and demo are the last (resumable) stage of the pipeline
        run(PY, "scripts/run_pipeline.py", "--preset", p)
    elif t == "paper-assets":
        run(PY, "scripts/paper_assets.py")
        run(PY, "scripts/render_summary.py")
        run(PY, "scripts/audit_numbers.py")
    elif t == "audit":
        run(PY, "scripts/audit_numbers.py")
    else:
        sys.exit(f"unknown target {t}")


if __name__ == "__main__":
    main()
