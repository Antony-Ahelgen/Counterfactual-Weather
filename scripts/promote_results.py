"""Promote a completed run (e.g. results/medium) to be the main reported results.

The current main results are moved to results/archive_<name>/ (kept, committed, auditable), the promoted run's
files are moved into results/, then paper assets, the generated summary and the paper number macros are rebuilt.

  python scripts/promote_results.py --src results/medium --archive-name fast69
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
KEEP = {"smoke"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/medium")
    ap.add_argument("--archive-name", required=True)
    a = ap.parse_args()
    res = ROOT / "results"
    src = ROOT / a.src
    assert (src / "test" / "metrics.csv").exists() and (src / "validation" / "validation_summary.csv").exists(), \
        "source run incomplete; not promoting"
    arch = res / f"archive_{a.archive_name}"
    arch.mkdir(parents=True, exist_ok=False)
    for item in list(res.iterdir()):
        # other complete runs (e.g. results/medium) stay where they are as references
        if item.name in KEEP or item == src or item.name.startswith(("archive_", "medium")):
            continue
        shutil.move(str(item), str(arch / item.name))
    for item in list(src.iterdir()):
        shutil.move(str(item), str(res / item.name))
    src.rmdir()
    for cmd in (["scripts/paper_assets.py"], ["scripts/render_summary.py"], ["paper/make_numbers.py"],
                ["paper/check_macros.py"], ["scripts/audit_numbers.py"]):
        subprocess.run([PY, *cmd], cwd=ROOT, check=True)
    print(f"promoted {a.src} -> results/; previous results in {arch.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
