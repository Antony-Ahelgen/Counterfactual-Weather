# PLAN

Start: 2026-11-25 22:13 IST. Budget: 72 h wall clock → hard stop 2026-11-28 22:13 IST.
Re-plan checkpoints: 50 % = 2026-11-27 10:13 IST, 80 % = 2026-11-28 07:49 IST.

## Decisions (with reasons)

| Item | Choice | Why |
|---|---|---|
| Compute preset | `cpu` (new, between smoke and small): 32x32 patches, K=8, H=8 (40 d + 40 d), U-Net base 32 ch, 1 seed, 20 members | No CUDA GPU (Intel UHD only). User approved CPU scale on 2026-11-25. |
| Python | 3.14.2 (brief asks 3.11) | Only interpreter installed; installing another would write outside the project. All deps have 3.14 wheels. |
| Storage format | NetCDF / NPZ, not Zarr | Windows Application Control blocks the `numcodecs` DLL that zarr needs. |
| Imagery | Sentinel-2 L2A from Microsoft Planetary Computer STAC (route 2) | GreenEarthNet (route 1) rejected: whole-split download, ~1 TB recommended disk, E-OBS weather without soil moisture; far over the 20 GB limit. |
| Weather | Open-Meteo historical archive, `models=era5_seamless` (ERA5-Land 0.1° for T2m, dewpoint, VPD, soil moisture; ERA5 0.25° for precipitation, radiation, ET0) | No `~/.cdsapirc`; key-free. Open-Meteo `era5_land` returns null precipitation/radiation; ARCO-ERA5 needs Zarr (blocked). |
| Static | Copernicus DEM GLO-30 + ESA WorldCover 2021 (Planetary Computer) | Key-free, same STAC. |
| Region | Central/Western Europe bbox lon −2..24, lat 44..54.5; 160 locations ≥0.3° apart, ≥60 % vegetated | Covers 2018, 2019, 2020, 2022 droughts plus normal years; ≥100 distinct weather cells. |
| Years | 2017–2023, Mar–Oct, 49 five-day bins/year | S2 L2A on PC from 2017. Growing season only (winter = cloud/snow). |
| Splits | locations 70/15/15 (train/val/test); years train {2017,2018,2019,2023}, val {2020}, test {2021,2022} | Spatially and temporally disjoint. Test contains 2022 drought year. |
| Drought | local stress index from training-year climatology, s ≥ 1 for ≥30 d of horizon | Brief default; sensitivity reported. |

## Time estimate per phase (wall clock)

| Phase | Est. | Notes |
|---|---|---|
| 0 Scaffold | 1 h | |
| 1 Data | 6 h | ~2 h download (background), cube building, splits, tests |
| 2 Scenarios | 2 h | |
| 3 Baselines | 4 h | deterministic U-Net training ~1.5 h CPU |
| 4 Diffusion | 8 h | CPU training ~4–5 h |
| 5 Physics | 10 h | physics model training ~5 h + ablations (shorter runs) |
| 6 Counterfactual | 3 h | |
| 7 Validation | 8 h | |
| 8 Experiments / assets | 12 h | ablation grid at reduced steps |
| 9 Paper draft | 4 h | P1 |
| 10 Stretch | rest | oracle first |

## Risks
- CPU training is slow → reduced steps; ablations use shorter schedules (documented).
- Cloud cover in Central Europe → masked losses, coverage check, min-valid filters.
- Weather is coarse (9–25 km) vs 20 m imagery → many locations, documented in LIMITATIONS.
- `gh` not installed → local commits only, push instructions printed.
- Weather API rate limits → caching, backoff, small number of requests (multi-year per call).
