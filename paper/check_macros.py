"""Check that every number macro used in paper/main.tex is defined in paper/numbers.tex (no '??' in the PDF)."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
tex = (HERE / "main.tex").read_text(encoding="utf-8")
nums = (HERE / "numbers.tex").read_text(encoding="utf-8")
defined = set(re.findall(r"\\csname ([^\\]+)\\endcsname", nums))
missing = []
for kind, pre in [("R", "r"), ("V", "v"), ("CF", "c")]:
    for a, b in re.findall(r"\\" + kind + r"\{([^}]*)\}\{([^}]*)\}", tex):
        if f"{pre}@{a}@{b}" not in defined:
            missing.append(f"\\{kind}{{{a}}}{{{b}}}")
for a in re.findall(r"\\D\{([^}]*)\}", tex):
    if f"d@{a}" not in defined:
        missing.append(f"\\D{{{a}}}")
notes = len(re.findall(r"\\draftnote\{", tex))   # the \newcommand definition does not match this pattern
print("missing macros:", missing or "none")
print("open draft notes:", notes)
sys.exit(1 if missing else 0)
