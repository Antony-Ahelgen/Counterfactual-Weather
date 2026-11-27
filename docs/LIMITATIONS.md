# Limitations

## Causal interpretation
- The drought effect Δ = forecast(drought weather) − forecast(normal weather) is a **model-based estimate**, not ground truth and not causal proof. It is credible only if (i) the model has learned how weather drives NDVI at these locations, and (ii) no important unobserved factor drives both the weather and the vegetation outcome (e.g. irrigation, management, pests, fire, harvest timing chosen in response to weather). Neither assumption can be fully tested because the no-drought world is never observed.
- The validation framework (proxy back-tests, natural experiment, placebos, calibration, constraint violations) can *falsify* the estimates but cannot prove them. Each check has its own weaknesses, listed in `docs/METHODS_NOTES.md`.
- The cross-year proxy compares against other years at the same place; those years differ in more than weather (crop rotation, management), so it is a proxy, not truth.
- The natural experiment relies on matching on observed covariates; unobserved differences between treated and control locations remain.

## Data
- **Weather is coarse.** ERA5-Land is ~9 km and ERA5 (used for precipitation, radiation, ET0 via Open-Meteo `era5_seamless`) is ~25–31 km, while imagery is 20 m. Every pixel in a 640 m patch shares one weather sequence; within-patch variation must come from context imagery and static layers. The design uses 160 locations ≥0.3° apart (distinct weather cells; the main reported run uses 135 of them) to give the model real weather variation, but the weather–vegetation relationship is learned from few weather series (68 training locations in the main run).
- **Precipitation and radiation come from ERA5 (0.25°), not ERA5-Land**, because Open-Meteo's `era5_land` daily product returns null for them. The CDS route would give native ERA5-Land values (needs a user key). <!--noaudit-->
- **Stress index is SPEI-like, not SPEI.** It combines standardised 30-day anomalies of P−ET0, temperature, VPD and soil moisture against a per-location, per-day-of-year climatology from 2010–2019 + 2023 (val/test years excluded). Eleven reference years is a short climatology; extremes are less well estimated.
- **Cloud masking uses the Sentinel-2 SCL layer only** (classes 4, 5, 6 kept). SCL misses some thin cloud, haze and shadow; this shows up as implausibly fast NDVI rises between composites and inflates the data-derived rate bounds (see results/data/data_summary.json).
- **Growing season only** (≈1 Mar–30 Oct). Winter dynamics are not modelled.
- **2017 coverage is thinner** (Sentinel-2B only from mid-2017; older processing baselines).
- **Processing-baseline offset** (−1000 DN from baseline 04.00, Jan 2022) is applied from STAC metadata per acquisition; residual radiometric differences between baselines may remain.
- **Land cover** is the 2021 WorldCover map applied to all years; land-cover change 2017–2023 is ignored.

## Model and compute
- **CPU only** (no CUDA GPU): 32×32 px patches (640 m), 8 context + 8 target 5-day frames, ~0.9 M-parameter U-Net,
  5000 training steps (main run), **3 seeds**, 8 ensemble members. Three seeds give only a rough view of training
  variance; claims are counted as supported only if supported in every seed (seed-robustness table in
  docs/RESULTS_SUMMARY.md).
- Monotonicity is trained AND scored with the same hand-designed stress perturbation of the weather (fixed direction
  in feature space), so the monotone-violation metric is in-sample for the physics models.
- Rate bounds are the 99.5th percentile of clean consecutive differences (both frames ≥95 % valid) per land-cover
  group; they are loose (residual cloud/haze, real harvest/mowing). Observations exceed them more often than the
  nominal 0.5 % (see the `observed` row in results/test/metrics.csv) because observed series include less clean
  frames and bounds are pooled per land-cover group.
- Physics guidance enforces the same envelope and rate bounds that the violation metrics score, so zero violations
  under guidance/rate projection would hold by construction. In the main run guidance is off (γ = 0 selected on
  validation) and rate projection is disabled, so violations come from the learned losses only; they are below the
  observations' own envelope and rate violation rates, which may partly reflect smoother forecasts.

## Runs actually reported
- **Main (reported) run: preset `v2`** — development set = the 80 locations of the earlier runs (68 train / 12 val);
  fresh test set = locations never downloaded before. Of 80 planned fresh test locations, the **55** whose downloads
  had completed by the owner's deadline (2026-11-27 15:45 IST) were used; this was decided before any v2 test result
  was seen, and the remaining 25 are reserved for later checks (`manifests/locations_v2.csv`). 5000 training steps,
  3 seeds; physics-loss weights ×3 (selected on validation in the earlier λ sweep); rate projection off; guidance
  scale chosen on validation by a pre-registered rule (envelope-violation rate closest to the observations' among
  γ within 2 % of the unguided CRPS) → γ = 0; post-hoc spread factors (1.10–1.22) fitted on validation per seed and
  model; deterministic DDIM. The test split was evaluated once with these choices.
- **Confirmation run `v21`** (`results/v21/CONFIRMATION.md`): the v2 models evaluated once on the 12 reserved fresh
  test locations downloaded by 18:55 IST on 2026-11-27 (design and validation-fitted calibration committed before any
  result). Replicates physics consistency (supported, 3 seeds), sharper than det (supported, 3 seeds), CRPS partly.
  Pre-registered calibrated model (bias + spread, 32 members): mixed (90 % coverage 0.811 / 0.809 / 0.794); the bias
  correction fitted on 2020 did not transfer to 2021-22. Exploratory: uncorrected 32-member ensembles meet the
  calibration rule in all seeds (seeds 1-2 added after seeing seed 0). Only 12 locations: wide uncertainty.
- **Evaluation ensemble size.** The calibration rule scores the empirical 5-95 % interval of the members; with 8
  members this interval cannot reach 90 % coverage even for a calibrated ensemble ((M-1)/(M+1) = 78 % for [min, max]),
  so the main-run calibration verdict is partly an ensemble-size effect.
- **Reference runs (kept, not tuned on test):** `results/archive_medium_tuned/` (80 locations, 56/12/12, λ×3,
  γ = 0.1, 1 seed), `results/medium/` (same data, untuned) and `results/archive_fast69/` (69 locations). Their
  conclusions agree in direction with the main run; the per-run numbers are in each folder's `RESULTS_SUMMARY.md`.
- Per-constraint retraining ablations were not run; physics ablations switch guidance/projection at sampling time.
- Evaluation: 200 test windows per seed and 8 ensemble members per probabilistic variant; effects on 200 windows
  (seed 0), of which only 23 are drought-labelled at 16 locations. Bootstrap intervals over 55 location clusters may
  be too narrow, and **no conclusion about drought effects is solid** (counts in docs/RESULTS_SUMMARY.md,
  "Statistical caveats").
- The full-scale pipeline (`scripts/run_pipeline.py`, preset `cpu`, all 160 locations) is implemented and
  smoke-tested but was not run to completion.

## Drought-effect bias
- The drought-effect estimates are compared with the observed drought anomaly in docs/RESULTS_SUMMARY.md (table
  "Drought-effect estimates", generated per run). Models whose anomaly is far from the observed one are biased; the
  size differs between runs (physics models: 2.7× fast, about 1.3× medium/medium_tuned, 1.8× v2), which itself shows the estimates are not stable at this
  sample size.

## Scope changes (honest record)
- The semi-synthetic oracle (brief Phase 7, part 6, P2) was **not run**: the project owner instructed on 2026-11-26 that no synthetic data be used anywhere.
- The optional Indian monsoon study area (P2) was not attempted.
