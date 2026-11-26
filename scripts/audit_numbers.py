"""Fail if a number in docs/*.md or paper/*.tex cannot be traced to a results file or config.

Checked tokens: decimal numbers (e.g. 0.0412, -1.5) and percentages (e.g. 12.3%). Integers
(years, counts, section numbers) are not checked. A token passes if some value v found in
results/**/*.csv|json (excluding smoke results) or configs/**/*.yaml satisfies round(v, d) == token
(d = decimals shown), or round(100*v, d) == token for percentages. Fenced code blocks are skipped.
Docs that are pure prose about methods/limitations can mark literal constants with `<!--noaudit-->`
on the same line.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"(?<![\w./])(-?\d+\.\d+)(\s?%)?(?![\w./])")  # "/" excludes DOIs like 10.5281/zenodo


def collect_values() -> set[float]:
    vals: set[float] = set()

    def add(x):
        try:
            f = float(x)
            if f == f and abs(f) != float("inf"):
                vals.add(f)
        except (TypeError, ValueError):
            pass

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        else:
            add(o)

    for p in (ROOT / "results").rglob("*"):
        if "smoke" in p.parts or not p.is_file():
            continue
        if p.suffix == ".json":
            try:
                walk(json.loads(p.read_text()))
            except Exception:  # noqa: BLE001
                pass
        elif p.suffix == ".csv":
            for line in p.read_text().splitlines():
                for cell in line.split(","):
                    add(cell)
    import yaml

    for p in (ROOT / "configs").rglob("*.yaml"):
        walk(yaml.safe_load(p.read_text()))
    return vals


def traceable(tok: str, pct: bool, vals: set[float]) -> bool:
    d = len(tok.split(".")[1])
    t = float(tok)
    for v in vals:
        for cand in ([v * 100, v] if pct else [v, abs(v) if t >= 0 else v]):
            if abs(round(cand, d) - t) < 0.5 * 10 ** (-d) + 1e-12:
                return True
    return False


def main() -> int:
    vals = collect_values()
    files = list((ROOT / "docs").glob("*.md")) + list((ROOT / "paper").glob("**/*.tex"))
    bad = []
    for f in files:
        in_code = False
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if line.strip().startswith("```"):
                in_code = not in_code
                continue
            if in_code or "noaudit" in line:
                continue
            for m in NUM.finditer(line):
                if not traceable(m.group(1), bool(m.group(2)), vals):
                    bad.append(f"{f.relative_to(ROOT)}:{i}: {m.group(0).strip()}")
    if bad:
        print("UNTRACEABLE NUMBERS:\n" + "\n".join(bad))
        return 1
    print(f"audit_numbers: OK ({len(files)} files checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
