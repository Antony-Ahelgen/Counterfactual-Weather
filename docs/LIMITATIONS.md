# Limitations

## Causal interpretation
- The drought effect Δ = forecast(drought weather) − forecast(normal weather) is a **model-based estimate**, not ground truth and not causal proof. It is credible only if (i) the model has learned how weather drives NDVI at these locations, and (ii) no important unobserved factor drives both the weather and the vegetation outcome (e.g. irrigation, management, pests, fire, harvest timing chosen in response to weather). Neither assumption can be fully tested because the no-drought world is never observed.
- The validation framework (proxy back-tests, natural experiment, placebos, calibration, constraint violations) can *falsify* the estimates but cannot prove them. Each check has its own weaknesses, listed in `docs/METHODS_NOTES.md`.
- The cross-year proxy compares against other years at the same place; those years differ in more than weather (crop rotation, management), so it is a proxy, not truth.
- The natural experiment relies on matching on observed covariates; unobserved differences between treated and control locations remain.

## Data
- **Weather is coarse.** ERA5-Land is ~9 km and ERA5 (used for precipitation, radiation, ET0 via Open-Meteo `era5_seamless`) is ~25–31 km, while imagery is 20 m. Every pixel in a 640 m patch shares one weather sequence; within-patch variation must come from context imagery and static layers. The design uses 160 locations ≥0.3° apart (distinct weather cells; the main reported run uses 80 of them) to give the model real weather variation, but the weather–vegetation relationship is learned from few weather series (56 training locations in the main run).
- **Precipitation and radiation come from ERA5 (0.25°), not ERA5-Land**, because Open-Meteo's `era5_land` daily product returns null for them. The CDS route would give native ERA5-Land values (needs a user key). <!--noaudit-->
- **Stress index is SPEI-like, not SPEI.** It combines standardised 30-day anomalies of P−ET0, temperature, VPD and soil moisture against a per-location, per-day-of-year climatology from 2010–2019 + 2023 (val/test years excluded). Eleven reference years is a short climatology; extremes are less well estimated.
- **Cloud masking uses the Sentinel-2 SCL layer only** (classes 4, 5, 6 kept). SCL misses some thin cloud, haze and shadow; this shows up as implausibly fast NDVI rises between composites and inflates the data-derived rate bounds (see results/data/data_summary.json).
- **Growing season only** (≈1 Mar–30 Oct). Winter dynamics are not modelled.
- **2017 coverage is thinner** (Sentinel-2B only from mid-2017; older processing baselines).
- **Processing-baseline offset** (−1000 DN from baseline 04.00, Jan 2022) is applied from STAC metadata per acquisition; residual radiometric differences between baselines may remain.
- **Land cover** is the 2021 WorldCover map applied to all years; land-cover change 2017–2023 is ignored.

## Model and compute
- **CPU only** (no CUDA GPU): 32×32 px patches (640 m), 8 context + 8 target 5-day frames, ~0.9 M-parameter U-Net,
  3000 training steps (main run), **1 seed**, 8 ensemble members. Single-seed results do not capture training
  variance; differences of a few thousandths between diffusion variants may be within seed-to-seed variance.
- Monotonicity is trained AND scored with the same hand-designed stress perturbation of the weather (fixed direction
  in feature space), so the monotone-violation metric is in-sample for the physics models.
- Rate bounds are the 99.5th percentile of clean consecutive differences (both frames ≥95 % valid) per land-cover
  group; they are loose (residual cloud/haze, real harvest/mowing). Observations exceed them more often than the
  nominal 0.5 % (see the `observed` row in results/test/metrics.csv) because observed series include less clean
  frames and bounds are pooled per land-cover group.
- Physics guidance enforces the same envelope and rate bounds that the violation metrics score, so zero violations
  under guidance/projection hold by construction; violation rates below the observations' own indicate
  over-constraint, not added realism.

## Runs actually reported
- **Main (reported) run: preset `medium_tuned`** — the 80 of 160 locations whose Sentinel-2 download had completed by a
  fixed cut-off (owner time limit), re-split 56/12/12 with the same rule and seed **before any training**
  (`manifests/locations_medium.csv`); 3000 training steps; physics-loss weight scale λ selected on validation from
  {0.3, 1, 3} (chose 3); guidance scale γ selected on validation from {0, 0.01, 0.1} (chose 0.1, the grid edge, so
  the optimum is not identified); DDIM stochasticity η selected on validation from {0, 0.5, 1} to improve calibration
  (chose 0: stochastic sampling made validation coverage worse). The test split was evaluated once with these choices.
- **Reference runs (kept, not tuned on test):** `results/medium/` (same data, λ = 1, untuned) and
  `results/archive_fast69/` (69 locations, 2500 steps). Their conclusions agree in direction with the main run; the
  per-run numbers are in each folder's `RESULTS_SUMMARY.md`.
- Per-constraint retraining ablations were not run; physics ablations switch guidance/projection at sampling time.
- Evaluation: 120 test windows and 8 ensemble members per probabilistic variant; effects on 100 windows. With 12 test
  locations and few drought-labelled windows, bootstrap intervals are wide and **no conclusion about drought effects
  is solid** (counts in docs/RESULTS_SUMMARY.md, "Statistical caveats").
- The full-scale pipeline (`scripts/run_pipeline.py`, preset `cpu`, all 160 locations) is implemented and
  smoke-tested but was not run to completion.

## Drought-effect bias
- The drought-effect estimates are compared with the observed drought anomaly in docs/RESULTS_SUMMARY.md (table
  "Drought-effect estimates", generated per run). Models whose anomaly is far from the observed one are biased; the
  direction and size differ between runs (fast vs. medium), which itself shows the estimates are not stable at this
  sample size.

## Scope changes (honest record)
- The semi-synthetic oracle (brief Phase 7, part 6, P2) was **not run**: the project owner instructed on 2026-11-26 that no synthetic data be used anywhere.
- The optional Indian monsoon study area (P2) was not attempted.
