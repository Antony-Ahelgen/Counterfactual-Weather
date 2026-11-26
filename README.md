# Counterfactual Vegetation Forecasting under Drought with Physics-Constrained Diffusion

Research code for: *Counterfactual Vegetation Forecasting under Drought with Physics-Constrained Diffusion Models*
(Antony Ahelgen A, Ferjith J S, V Muthulakshmi).

A weather-conditioned diffusion model forecasts Sentinel-2 NDVI from past imagery, static land attributes and a
weather scenario. Playing the same initial scene forward under a drought and a normal scenario (with common random
numbers) gives the drought effect Δ. Physical plausibility (rate bounds, phenology envelope, monotone response to
moisture stress) is enforced through auxiliary losses, guidance during sampling and a final projection. Because the
true counterfactual is never observed, results are checked with a five-part validation framework (proxy back-tests,
natural experiment, placebos, calibration, constraint violations). All data are real (no synthetic data).

Status and results: `docs/RESULTS_SUMMARY.md` (generated), `PROGRESS.md`. Methods: `docs/METHODS_NOTES.md`.
Limitations: `docs/LIMITATIONS.md`. Data sources and attribution: `docs/DATA_LICENSES.md`.

## Install
```
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu   # or a CUDA wheel
.venv/Scripts/python -m pip install -r requirements.lock
.venv/Scripts/python -m pip install -e . --no-deps
```
(Linux/macOS: use `.venv/bin/python`.) No accounts or API keys are needed.

## Run (one command per stage)
| Stage | Windows (no make) | make |
|---|---|---|
| unit tests (real-data fixture) | `python tasks.py test` | `make test` |
| end-to-end smoke run (~1–2 min CPU) | `python tasks.py smoke` | `make smoke` |
| download + build data (~1–2 GB) | `python scripts/build_data.py --preset cpu --download` | `make data` (after download) |
| train all models + validation-based selection | `python tasks.py train --preset cpu` | `make train` |
| evaluation, effects, validation, demo | `python tasks.py eval --preset cpu` | `make eval` |
| tables, figures, summary, number audit | `python tasks.py paper-assets` | `make paper-assets` |

`scripts/run_pipeline.py` (used by train/eval) is resumable and runs: data build → deterministic + unconstrained
diffusion → λ sweep (validation) → physics model + ablations → guidance-scale sweep (validation) → test
evaluation → counterfactual effects → five-part validation → demo maps. Selection rules are pre-registered in its
docstring.

Presets: `smoke` (tests), `cpu` (main results on this project's CPU-only machine), `small` (single GPU).

## Layout
`configs/` YAML configs · `src/cfveg/` package (data, models, physics, counterfactual, validation, utils) ·
`scripts/` entry points · `tests/` pytest (leakage, shapes, masks, constraint maths, schedule, determinism) ·
`manifests/` download manifests (locations, scene ids, checksums) · `results/` metrics with provenance,
LaTeX tables, figures · `docs/` notes.

## Data attribution
Contains modified Copernicus Sentinel data (2017–2023) and modified Copernicus Climate Change Service information.
Weather data by [Open-Meteo.com](https://open-meteo.com/) (CC BY 4.0). Produced using Copernicus WorldDEM-30
© DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union
and ESA; all rights reserved. © ESA WorldCover project / Contains modified Copernicus Sentinel data (2021)
processed by ESA WorldCover consortium. Imagery accessed through Microsoft Planetary Computer.
Full details: `docs/DATA_LICENSES.md`.
