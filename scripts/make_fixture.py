"""Cut a small REAL-data fixture (committed) for unit tests and smoke runs.

Takes 6 train, 2 val and 2 test locations that have been downloaded, runs the normal cube builder
on them (real Sentinel-2 NDVI composites, real Open-Meteo/ERA5 weather, real DEM/WorldCover),
and crops the central 16x16 pixels. Output: tests/fixtures/real_cube/*.npy + meta.json.
No synthetic data is used anywhere in this project.
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cfveg.data.cube import build_cube  # noqa: E402
from cfveg.utils.provenance import load_preset, write_provenance  # noqa: E402

N = {"train": 6, "val": 2, "test": 2}
CROP = 16


def main() -> None:
    cfg = load_preset("cpu")
    locs = pd.read_csv(ROOT / "manifests" / "locations.csv")
    have = locs[[os.path.exists(ROOT / cfg["data"]["raw_dir"] / "s2" / f"{i}.npz") for i in locs.loc_id]]
    pick = pd.concat([have[have.split == s].head(n) for s, n in N.items()])
    assert all((pick.split == s).sum() == n for s, n in N.items()), "not enough downloaded locations yet"
    tmp_root = ROOT / "data" / "fixture_build"
    (tmp_root / "manifests").mkdir(parents=True, exist_ok=True)
    pick.to_csv(tmp_root / "manifests" / "locations.csv", index=False)
    c = copy.deepcopy(cfg)
    c["data"]["raw_dir"] = str(ROOT / cfg["data"]["raw_dir"])
    c["data"]["processed_dir"] = "processed"
    meta = build_cube(c, tmp_root, min_loc_valid=0.0)
    src = tmp_root / "processed" / "cube"
    out = ROOT / "tests" / "fixtures" / "real_cube"
    out.mkdir(parents=True, exist_ok=True)
    P = np.load(src / "ndvi.npy", mmap_mode="r").shape[-1]
    o = (P - CROP) // 2
    sl = (slice(o, o + CROP), slice(o, o + CROP))
    np.save(out / "ndvi.npy", np.load(src / "ndvi.npy")[..., sl[0], sl[1]].astype(np.float16))
    np.save(out / "static.npy", np.load(src / "static.npy")[..., sl[0], sl[1]])
    np.save(out / "lcg.npy", np.load(src / "lcg.npy")[..., sl[0], sl[1]])
    np.save(out / "wx.npy", np.load(src / "wx.npy"))
    np.save(out / "s_daily.npy", np.load(src / "s_daily.npy"))
    meta["fixture"] = f"REAL data subset: {len(pick)} locations, central {CROP}x{CROP} px crop"
    (out / "meta.json").write_text(json.dumps(meta))
    write_provenance(out / "meta.json", extra={"locations": pick.loc_id.tolist()})
    print(meta["fixture"], {k: (out / f"{k}.npy").stat().st_size // 1024 for k in ["ndvi", "wx", "s_daily"]}, "KB")


if __name__ == "__main__":
    main()
