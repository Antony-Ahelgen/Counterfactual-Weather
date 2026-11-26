# PROGRESS

Start: 2026-11-25 22:13 IST · Budget end: 2026-11-28 22:13 IST · Preset: cpu

## Push status
`gh` is not installed, so there is no remote yet. All work is committed locally. To push, run:
```
winget install GitHub.cli      # then restart terminal
gh auth login
cd C:\Users\senth\cfveg-project\counterfactual-vegetation-diffusion
gh repo create counterfactual-vegetation-diffusion --private --source=. --remote=origin --push
git push --follow-tags
```

## Optional upgrade: CDS key (ERA5-Land via CDS)
1. Register at https://cds.climate.copernicus.eu, open the ERA5-Land dataset page, accept the licence.
2. Copy your key from your profile into `%USERPROFILE%\.cdsapirc` (url + key lines, per CDS "How to API" page).
Not needed: the pipeline uses Open-Meteo era5_seamless (key-free).

## Log
- 2026-11-25 22:13 Recon done (no CUDA GPU, Py3.14, zarr DLL blocked, gh absent). User chose project dir outside OneDrive and CPU preset.
- 2026-11-25 22:59 Phase 0 scaffold: configs, tasks.py, Makefile, provenance utils, first tests pass. Agent permission settings file not created (blocked); a proposed version is kept locally, untracked.

## Next
Phase 1: location sampling (WorldCover filter) → S2 download + 5-day NDVI composites → weather → static → cubes → splits → stress index/drought labels → dataset class → tests.
- 2026-11-26 00:40 USER INSTRUCTION: never use synthetic data (no oracle, tests/smoke must use a real-data fixture). Synthetic generator to be removed once >=20 real locations are downloaded. Ask user for any access needed.
