# Limitations

## Causal interpretation
- The drought effect Δ = forecast(drought weather) − forecast(normal weather) is a **model-based estimate**, not ground truth and not causal proof. It is credible only if (i) the model has learned how weather drives NDVI at these locations, and (ii) no important unobserved factor drives both the weather and the vegetation outcome (e.g. irrigation, management, pests, fire, harvest timing chosen in response to weather). Neither assumption can be fully tested because the no-drought world is never observed.
- The validation framework (proxy back-tests, natural experiment, placebos, calibration, constraint violations) can *falsify* the estimates but cannot prove them. Each check has its own weaknesses, listed in `docs/METHODS_NOTES.md`.
- The cross-year proxy compares against other years at the same place; those years differ in more than weather (crop rotation, management), so it is a proxy, not truth.
- The natural experiment relies on matching on observed covariates; unobserved differences between treated and control locations remain.

## Data
- **Weather is coarse.** ERA5-Land is ~9 km and ERA5 (used for precipitation, radiation, ET0 via Open-Meteo `era5_seamless`) is ~25–31 km, while imagery is 20 m. Every pixel in a 640 m patch shares one weather sequence; within-patch variation must come from context imagery and static layers. We use 160 locations ≥0.3° apart (distinct weather cells) to give the model real weather variation, but the weather–vegetation relationship is learned from at most 160 weather series per year.
- **Precipitation and radiation come from ERA5 (0.25°), not ERA5-Land**, because Open-Meteo's `era5_land` daily product returns null for them. The CDS route would give native ERA5-Land values (needs a user key). <!--noaudit-->
- **Stress index is SPEI-like, not SPEI.** It combines standardised 30-day anomalies of P−ET0, temperature, VPD and soil moisture against a per-location, per-day-of-year climatology from 2010–2019 + 2023 (val/test years excluded). Eleven reference years is a short climatology; extremes are less well estimated.
- **Cloud masking uses the Sentinel-2 SCL layer only** (classes 4, 5, 6 kept). SCL misses some thin cloud, haze and shadow; this shows up as implausibly fast NDVI rises between composites and inflates the data-derived rate bounds (see results/data/data_summary.json).
- **Growing season only** (≈1 Mar–30 Oct). Winter dynamics are not modelled.
- **2017 coverage is thinner** (Sentinel-2B only from mid-2017; older processing baselines).
- **Processing-baseline offset** (−1000 DN from baseline 04.00, Jan 2022) is applied from STAC metadata per acquisition; residual radiometric differences between baselines may remain.
- **Land cover** is the 2021 WorldCover map applied to all years; land-cover change 2017–2023 is ignored.

## Model and compute
- **CPU only** (no CUDA GPU): reduced preset (32×32 px patches = 640 m, 8 context + 8 target 5-day frames, ~0.9 M-parameter U-Net, 8000 training steps, 1 seed, 20 ensemble members). Results may change with larger models, longer training or more seeds. Single-seed results do not capture training variance; bootstrap intervals are over locations only.
- Ablation models are trained with the same budget as the main models; differences smaller than the bootstrap intervals are not interpreted.
- Monotonicity is trained on a hand-designed stress perturbation of the weather (fixed direction in feature space) applied to half of each batch.
- Rate bounds are the 99.5th percentile of clean consecutive differences (both frames ≥95 % valid) per land-cover group; they are loose because of residual cloud contamination and real abrupt events (harvest, mowing).

## Reduced-scale run actually reported (preset `fast`)
- Owing to time constraints, the project owner chose (2026-11-26) a reduced-scale run on real data instead of the
  planned full run: the **69 of 160 locations** whose Sentinel-2 download had completed, re-split 70/15/15 with
  the same rule and seed **before any training** (`manifests/locations_fast.csv`); **2500 training steps**; three
  trained models (deterministic, diffusion without physics, diffusion with physics losses). Physics ablations
  are obtained by switching guidance and projection on/off at sampling time; the per-constraint loss ablations
  (retraining without each constraint) and the λ sweep were **not run**; λ values are the config defaults.
- Evaluation uses 100 test windows and 8 ensemble members per probabilistic variant; effects use 80 windows. Ten
  test locations make the bootstrap intervals over locations wide.
- The full-scale pipeline (`scripts/run_pipeline.py`, preset `cpu`) is implemented and smoke-tested but was not
  run to completion.

## Scope changes (honest record)
- The semi-synthetic oracle (brief Phase 7, part 6, P2) was **not run**: the project owner instructed on 2026-11-26 that no synthetic data be used anywhere.
- The optional Indian monsoon study area (P2) was not attempted.
