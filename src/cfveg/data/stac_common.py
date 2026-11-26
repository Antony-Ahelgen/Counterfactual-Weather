"""Shared helpers for Planetary Computer STAC access (no account needed)."""
from __future__ import annotations

import functools
import time
from typing import Callable, TypeVar

from odc.geo.geobox import GeoBox
from pyproj import Transformer

T = TypeVar("T")
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@functools.lru_cache(maxsize=1)
def catalog():
    import planetary_computer as pc
    import pystac_client

    # Unsigned client: items are signed right before reading (see fresh_sign) so SAS tokens
    # never expire mid-load.
    return pystac_client.Client.open(STAC_URL)


def fresh_sign(items: list, min_ttl_s: float = 1800.0) -> list:
    """Return signed copies of items, evicting cached SAS tokens with < min_ttl_s remaining."""
    import planetary_computer as pc
    from planetary_computer import sas

    for k, tok in list(sas.TOKEN_CACHE.items()):
        if tok.ttl() < min_ttl_s:
            sas.TOKEN_CACHE.pop(k, None)
    return [pc.sign(it.clone()) for it in items]


def configure_gdal() -> None:
    import odc.stac

    odc.stac.configure_rio(cloud_defaults=True, GDAL_HTTP_MAX_RETRY="5", GDAL_HTTP_RETRY_DELAY="2")


def patch_geobox(lon: float, lat: float, size_px: int, res_m: float, crs: str = "EPSG:3035") -> GeoBox:
    """Square GeoBox of size_px x size_px at res_m centred on (lon, lat), snapped to the res grid."""
    tr = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    x, y = tr.transform(lon, lat)
    half = size_px * res_m / 2
    x0 = round((x - half) / res_m) * res_m
    y0 = round((y - half) / res_m) * res_m
    return GeoBox.from_bbox((x0, y0, x0 + size_px * res_m, y0 + size_px * res_m), crs=crs,
                            resolution=res_m)


def bbox_ll(lon: float, lat: float, size_px: int, res_m: float) -> list[float]:
    """Generous lon/lat bbox around a patch for STAC search."""
    d = size_px * res_m / 111_000 * 0.8 + 0.002
    return [lon - d * 1.6, lat - d, lon + d * 1.6, lat + d]


def retry(fn: Callable[[], T], tries: int = 5, base: float = 3.0, what: str = "", quiet: bool = False) -> T:
    """Call fn with exponential backoff."""
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            if k == tries - 1:
                raise
            wait = base * 2**k
            if not quiet:
                print(f"  retry {k + 1}/{tries} {what}: {type(e).__name__}: {str(e)[:120]} (sleep {wait:.0f}s)",
                      flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")
