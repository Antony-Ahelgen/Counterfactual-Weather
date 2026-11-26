"""Static layers: Copernicus DEM GLO-30 (elevation, slope) and ESA WorldCover land cover.

Both from Microsoft Planetary Computer (collections `cop-dem-glo-30`, `esa-worldcover`).
"""
from __future__ import annotations

import numpy as np

from .stac_common import bbox_ll, catalog, fresh_sign, patch_geobox, retry

VEG_CLASSES = (10, 20, 30, 40)  # tree, shrub, grassland, cropland
# Coarse land-cover groups used for rate bounds and phenology envelopes
LC_GROUPS = {10: 0, 20: 1, 30: 1, 40: 2}  # 0 forest, 1 grass/shrub, 2 cropland, 3 other
LC_GROUP_NAMES = ["forest", "grass_shrub", "cropland", "other"]


def lc_group(lc: np.ndarray) -> np.ndarray:
    out = np.full(lc.shape, 3, dtype=np.int8)
    for k, g in LC_GROUPS.items():
        out[lc == k] = g
    return out


def load_static(lon: float, lat: float, size_px: int, res_m: float, crs: str) -> dict[str, np.ndarray]:
    import odc.stac

    gbox = patch_geobox(lon, lat, size_px, res_m, crs)
    bb = bbox_ll(lon, lat, size_px, res_m)
    cat = catalog()

    wc_items = retry(lambda: list(cat.search(collections=["esa-worldcover"], bbox=bb).items()), what="wc search")
    wc_items = sorted(wc_items, key=lambda i: str(i.datetime or i.properties.get("start_datetime")))[-1:]
    # nearest resampling for categorical data
    wc = retry(lambda: odc.stac.load(fresh_sign(wc_items), bands=["map"], geobox=gbox, resampling="nearest").compute(),
               what="wc load")
    lc = wc["map"].isel(time=0).values.astype(np.uint8)

    dem_items = retry(lambda: list(cat.search(collections=["cop-dem-glo-30"], bbox=bb).items()), what="dem search")
    dem = retry(lambda: odc.stac.load(fresh_sign(dem_items), bands=["data"], geobox=gbox, resampling="bilinear",
                                      groupby="solar_day").compute(), what="dem load")
    elev = dem["data"].max("time").values.astype(np.float32)
    gy, gx = np.gradient(elev, res_m)
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(np.float32)
    return {"lc": lc, "elev": elev, "slope": slope, "wc_item": np.array(wc_items[0].id),
            "dem_items": np.array([i.id for i in dem_items])}


def veg_fraction(lc: np.ndarray) -> float:
    return float(np.isin(lc, VEG_CLASSES).mean())
