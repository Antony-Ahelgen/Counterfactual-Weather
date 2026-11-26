# Data sources, licences and required attribution

All data are real observations or reanalysis. No synthetic data are used anywhere in this project.
Licence/attribution wording below was checked against the providers' own pages on 2026-11-26.

| Dataset | Used for | Access | Licence |
|---|---|---|---|
| Sentinel-2 Level-2A (bands B04, B08, SCL) | NDVI target and context | Microsoft Planetary Computer STAC, collection `sentinel-2-l2a` (no account) | Copernicus Sentinel data terms (free, full and open) |
| ERA5-Land / ERA5 reanalysis via Open-Meteo historical archive (`models=era5_seamless`) | Daily precipitation, 2 m temperature, dewpoint, VPD, shortwave radiation, top-layer soil moisture, ET0 | `https://archive-api.open-meteo.com/v1/archive` (no key) | Open-Meteo API data: CC BY 4.0; underlying data: Copernicus Climate Change Service (C3S) licence |
| Copernicus DEM GLO-30 | Elevation, slope | Planetary Computer, collection `cop-dem-glo-30` | Licence for the use of the Copernicus WorldDEM-30 (free) |
| ESA WorldCover 10 m 2021 v200 | Land-cover groups, location filtering | Planetary Computer, collection `esa-worldcover` | CC BY 4.0 |

## Required attribution text

- **Sentinel-2:** "Contains modified Copernicus Sentinel data [2017–2023]." Terms: https://scihub.copernicus.eu/twiki/pub/SciHubWebPortal/TermsConditions/Sentinel_Data_Terms_and_Conditions.pdf (linked from the Planetary Computer collection metadata).
- **Open-Meteo:** "Weather data by Open-Meteo.com" with a link to https://open-meteo.com/. Citation: Zippenfenig, P. (2023). Open-Meteo.com Weather API [Computer software]. Zenodo. https://doi.org/10.5281/ZENODO.7970649 (from https://open-meteo.com/en/licence).
- **ERA5 / ERA5-Land (C3S):** Open-Meteo states its reanalysis data come from the Copernicus Climate Change Service. Acknowledge: "Contains modified Copernicus Climate Change Service information [2010–2023]." Dataset references for the paper: ERA5-Land and ERA5 dataset DOIs on the CDS **[VERIFY exact DOIs before citing]**.
- **Copernicus DEM (modified data, licence §(b)):** "produced using Copernicus WorldDEM-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA; all rights reserved". Disclaimer required when redistributing (§(c)): "The organisations in charge of the Copernicus programme by law or by delegation do not incur any liability for any use of the Copernicus WorldDEM-30". Source: https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/DEM/resources/license/License-COPDEM-30.pdf
- **ESA WorldCover:** "© ESA WorldCover project / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium." Cite: Zanaga, D. et al. (2022). ESA WorldCover 10 m 2021 v200. Zenodo. https://doi.org/10.5281/zenodo.7254221 (full author list on the Zenodo record).
- **Microsoft Planetary Computer** (hosting): acknowledge as data access platform.

## Considered and rejected

- **GreenEarthNet** (Benson et al., CVPR 2024; https://github.com/vitusbenson/greenearthnet): preferred first option in the brief, but the toolkit downloads whole splits and recommends ~1 TB of free disk (far above the 20 GB limit), and its weather (E-OBS) has no soil moisture. Rejected on 2026-11-25.
- **ERA5-Land through the CDS API**: no `~/.cdsapirc` on this machine; key-free Open-Meteo route used instead (upgrade path in PROGRESS.md).
- **ARCO-ERA5 (Google Cloud Zarr)**: Zarr unusable on this machine (Windows Application Control blocks the `numcodecs` DLL).
- **Open-Meteo `models=era5_land`**: returns null precipitation/radiation/ET0 in the daily archive; `era5_seamless` used (ERA5-Land where available, ERA5 0.25° otherwise).
- **LEVIR-CD**: building change-detection dataset of very-high-resolution RGB image pairs: no near-infrared band (no NDVI), no regular time series and no weather. Cannot support vegetation forecasting. Not used.

## Reproducing the download
`manifests/locations.csv` (study locations and split), `manifests/s2_scenes.csv` (every Sentinel-2 item id, date, processing baseline, cloud cover), `manifests/s2_checksums.csv` and `manifests/weather.csv` (grid cell, model, period, checksums) are committed. Run `python scripts/download_data.py --stage locations|weather|s2`.
