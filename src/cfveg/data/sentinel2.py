"""Sentinel-2 L2A download (Planetary Computer) → SCL-masked NDVI → 5-day composites.

Processing-baseline offset: products with processing baseline >= 04.00 (from 25 Jan 2022) carry
BOA_ADD_OFFSET = -1000 DN. Reflectance = (DN + offset) / 10000. We read the baseline per item
from STAC metadata (`s2:processing_baseline`) and apply the offset per acquisition, so NDVI is
comparable across years. DN == 0 is no-data.
"""
from __future__ import annotations

import datetime as dt

import numpy as np

from .stac_common import bbox_ll, catalog, fresh_sign, patch_geobox, retry

BOA_OFFSET_DN = 1000
NEW_BASELINE = 4.0
MAX_EMPTY_FRAC = 0.2
MAX_FAILED_DAY_FRAC = 0.1
BLOB_HOST = "sentinel2l2a01.blob.core.windows.net"


def wait_for_network(host: str = BLOB_HOST, max_wait_s: float = 3600.0) -> None:
    """Block until DNS for the data host resolves (e.g. after laptop sleep / Wi-Fi drop)."""
    import socket
    import time

    t0 = time.time()
    k = 0
    while True:
        try:
            socket.getaddrinfo(host, 443)
            return
        except OSError:
            if time.time() - t0 > max_wait_s:
                raise RuntimeError(f"network unavailable for {max_wait_s:.0f}s")
            if k % 10 == 0:
                print(f"  waiting for network ({host} unresolved)", flush=True)
            k += 1
            time.sleep(30)


def boa_offset(processing_baseline: str | None) -> int:
    """DN offset to *add* before scaling (0 or -1000)."""
    try:
        pb = float(processing_baseline) if processing_baseline is not None else 0.0
    except ValueError:
        pb = 0.0
    return -BOA_OFFSET_DN if pb >= NEW_BASELINE else 0


def ndvi_from_dn(b4: np.ndarray, b8: np.ndarray, offset: int) -> np.ndarray:
    """NDVI from L2A digital numbers with the baseline offset applied; NaN where invalid."""
    b4 = b4.astype(np.float32)
    b8 = b8.astype(np.float32)
    nodata = (b4 == 0) | (b8 == 0)
    r4 = np.clip(b4 + offset, 0, None) / 10000.0
    r8 = np.clip(b8 + offset, 0, None) / 10000.0
    den = r8 + r4
    with np.errstate(invalid="ignore", divide="ignore"):
        nd = (r8 - r4) / den
    nd[nodata | (den <= 1e-4)] = np.nan
    return np.clip(nd, -1, 1)


def bin_index(date: dt.date, doy_start: int, bin_days: int, n_bins: int) -> int | None:
    doy = date.timetuple().tm_yday
    k = (doy - doy_start) // bin_days
    return int(k) if 0 <= k < n_bins else None


def bin_doys(doy_start: int, bin_days: int, n_bins: int) -> np.ndarray:
    """Centre day-of-year of each bin."""
    return doy_start + bin_days * np.arange(n_bins) + (bin_days - 1) / 2


def composite(ndvi_stack: np.ndarray, bins: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Median composite of valid NDVI per bin. ndvi_stack [T,H,W] (NaN invalid), bins [T] (-1 drop)."""
    H, W = ndvi_stack.shape[1:]
    out = np.full((n_bins, H, W), np.nan, dtype=np.float32)
    nobs = np.zeros((n_bins, H, W), dtype=np.uint8)
    for b in range(n_bins):
        sel = bins == b
        if not sel.any():
            continue
        s = ndvi_stack[sel]
        nobs[b] = np.isfinite(s).sum(0)
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[b] = np.nanmedian(s, axis=0)
    return out, nobs


def download_location(lon: float, lat: float, cfg: dict, n_workers: int = 16) -> dict:
    """Return per-year NDVI composites [Y, B, H, W] plus scene manifest for one location."""
    import odc.stac

    d = cfg["data"]
    gbox = patch_geobox(lon, lat, d["patch_px"], d["resolution_m"], d["crs"])
    bb = bbox_ll(lon, lat, d["patch_px"], d["resolution_m"])
    years = d["years"]
    nb, bd, ds0 = d["n_bins"], d["bin_days"], d["doy_start"]
    valid_scl = np.array(d["valid_scl"])
    H = W = d["patch_px"]
    ndvi_all = np.full((len(years), nb, H, W), np.nan, dtype=np.float32)
    nobs_all = np.zeros((len(years), nb, H, W), dtype=np.uint8)
    manifest = []
    lenient_years: list[int] = []
    failed_days: list[str] = []
    for yi, y in enumerate(years):
        t0 = dt.date(y, 1, 1) + dt.timedelta(days=ds0 - 1)
        t1 = t0 + dt.timedelta(days=nb * bd - 1)
        items = retry(lambda: list(catalog().search(
            collections=["sentinel-2-l2a"], bbox=bb, datetime=f"{t0}/{t1}",
            query={"eo:cloud_cover": {"lt": d["max_scene_cloud"]}}).items()), what=f"search {y}")
        if not items:
            continue
        # Load each acquisition day separately with its own retries (transient DNS/HTTP failures under
        # parallel load otherwise sink a whole year). Days that still fail are dropped and counted; a
        # location-year with > MAX_FAILED_DAY_FRAC failed days is refused (retried in a later pass).
        items = sorted(items, key=lambda i: i.datetime)
        by_day: dict[dt.date, list] = {}
        for it in items:
            by_day.setdefault(it.datetime.date(), []).append(it)

        def _load_day(day_items):
            def once():
                wait_for_network()
                ds_ = odc.stac.load(fresh_sign(day_items), bands=["B04", "B08", "SCL"], geobox=gbox,
                                    dtype="uint16", groupby="solar_day", resampling="nearest", fail_on_error=True)
                return ds_.isel(time=0)
            return retry(once, tries=5, base=3.0, what="day load", quiet=True)

        from concurrent.futures import ThreadPoolExecutor

        days = sorted(by_day)
        loaded: dict[dt.date, object] = {}
        failed = []
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_load_day, by_day[dd]): dd for dd in days}
            for fu, dd in futs.items():
                try:
                    loaded[dd] = fu.result()
                except Exception:  # noqa: BLE001
                    failed.append(dd)
        if len(failed) > MAX_FAILED_DAY_FRAC * len(days):
            raise RuntimeError(f"{lon:.3f},{lat:.3f} {y}: {len(failed)}/{len(days)} acquisition days failed "
                               f"after retries; not saving, retry later")
        if failed:
            lenient_years.append(int(y))
            failed_days.extend(str(dd) for dd in failed)
        day_pb: dict[dt.date, str] = {}
        for it in items:
            dd = it.datetime.date()
            day_pb.setdefault(dd, it.properties.get("s2:processing_baseline"))
            manifest.append({"id": it.id, "date": str(dd), "pb": it.properties.get("s2:processing_baseline"),
                             "cloud": it.properties.get("eo:cloud_cover")})
        times = sorted(loaded)
        stack = np.full((len(times), H, W), np.nan, dtype=np.float32)
        bins = np.full(len(times), -1)
        for ti, tday in enumerate(times):
            dsd = loaded[tday]
            pb = day_pb.get(tday)
            nd = ndvi_from_dn(dsd.B04.values, dsd.B08.values, boa_offset(pb))
            scl = dsd.SCL.values
            nd[~np.isin(scl, valid_scl)] = np.nan
            stack[ti] = nd
            b = bin_index(tday, ds0, bd, nb)
            bins[ti] = -1 if b is None else b
        ndvi_all[yi], nobs_all[yi] = composite(stack, bins, nb)
    return {"ndvi": ndvi_all, "nobs": nobs_all, "manifest": manifest, "lenient_years": lenient_years,
            "failed_days": failed_days}
