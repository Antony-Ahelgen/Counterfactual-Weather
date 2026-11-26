# Methods notes (adaptable into the paper)

All numbers here are configuration values. They describe the **planned design** (configs/base.yaml,
configs/presets/cpu.yaml). The **main reported run** used preset `medium_tuned` (configs/presets/medium_tuned.yaml):
80 of the 160 locations (downloaded by a fixed cut-off), re-split 56/12/12 before training, 3000 training steps,
8 ensemble members; λ scale selected on validation from {0.3, 1, 3}, guidance scale from {0, 0.01, 0.1}, DDIM η
from {0, 0.5, 1} (stochastic DDIM: σ_t = η·sqrt((1−ᾱ_{t−1})/(1−ᾱ_t))·sqrt(1−ᾱ_t/ᾱ_{t−1}), with step noise shared
between paired branches to keep common random numbers). Results are in `docs/RESULTS_SUMMARY.md`, generated from `results/`.

## 1 Data

**Study design.** 160 locations in Central/Western Europe (lon −2…24° E, lat 44…54.5° N), sampled at random
with a minimum spacing of 0.3° (more than two ERA5-Land grid cells, which also acts as a spatial buffer between
splits) and kept if ≥ 60 % of the patch is vegetated (ESA WorldCover tree, shrub, grassland or cropland).
Each location is a 64 × 64 pixel patch at 20 m (1.28 km) in ETRS89-LAEA (EPSG:3035). Years 2017–2023 cover <!--noaudit: code constants-->
the 2018, 2019/2020 and 2022 European droughts as well as normal years. Drought is defined locally from weather
(below), not by calendar year.

**Imagery.** Sentinel-2 L2A (Planetary Computer) bands B04, B08 and the scene classification layer (SCL). Scenes
with eo:cloud_cover < 80 % between ≈1 March and 30 October are read on the patch grid (nearest resampling).
Digital numbers are converted to reflectance as (DN + offset)/10 000 with offset = −1000 for processing
baseline ≥ 04.00 (read per scene from `s2:processing_baseline`) and 0 otherwise, so NDVI is comparable across
years. NDVI = (B08 − B04)/(B08 + B04). Pixels with SCL ∉ {4 vegetation, 5 bare, 6 water} or DN = 0 are invalid.
Per-pixel median composites are formed in 49 fixed 5-day bins per year starting at day-of-year 61. Invalid
pixels stay missing (never imputed with zeros in targets or losses).

**Weather.** Daily ERA5-Land/ERA5 from the Open-Meteo archive (`era5_seamless`, nearest grid cell, 2010–2023):
precipitation, 2 m mean temperature, dewpoint, maximum VPD, shortwave radiation, mean soil moisture 0–7 cm and
FAO ET0. Aggregated per 5-day bin (sums for precipitation, radiation, ET0; means otherwise).

**Stress index.** For each location and day, 30-day rolling quantities are standardised against a per-location,
per-day-of-year climatology (±15-day window) fitted on climatology years only (2010–2019 and 2023; validation
and test years excluded): a_wb = −z(Σ₃₀(P − ET0)), a_T = z(T̄₃₀), a_VPD = z(VPD₃₀), a_SM = −z(SM₃₀).
s_raw = (1·a_wb + 0.5·a_T + 0.5·a_VPD + 1·a_SM)/3, re-standardised the same way to give s. A forecast window is
labelled **drought** if s ≥ 1 on at least 30 of the horizon days. Thresholds are in config; sensitivity is
reported in results/data/drought_sensitivity.csv.

**Static inputs.** Copernicus DEM GLO-30 (elevation, slope) and WorldCover 2021 grouped into forest,
grass/shrub, cropland and other.

**Samples and splits.** A sample is K = 8 context composites (40 days) and H = 8 target composites (40 days)
on a 32 × 32 pixel quadrant, with the weather of all K + H bins, day-of-year, static layers and validity masks.
Windows require ≥ 30 % valid context and ≥ 20 % valid target pixels. Locations are split 70/15/15 into
train/validation/test; years are split train {2017, 2018, 2019, 2023}, validation {2020}, test {2021, 2022}.
A sample belongs to a split only if both its location and its year do. Normalisation statistics, rate bounds,
phenology envelopes and matching-covariate scaling are fitted on train locations × train years only. Analog
pools for scenarios use training years only. Leakage tests (tests/test_leakage.py) corrupt every held-out
value and require all fitted objects to be unchanged.

## 2 Scenarios
The model is trained on observed weather; counterfactuals swap the horizon part of the weather sequence at
inference (context weather stays observed). **Normal:** analog-year ensemble of the same horizon bins at the
same location in non-drought training years (fallback: per-location median over training years). **Drought:**
the observed weather if the window is a drought; otherwise drought-labelled training-year windows at the same
location, then at the five nearest training locations; final fallback: the normal scenario shifted by +1.5σ of
stress (precipitation ×(1 − 0.25a), temperature +a K, VPD ×(1 + 0.15a), soil moisture ×(1 − 0.08a),
ET0 ×(1 + 0.08a), anomalies and s + a). The model receives the full weather sequence plus the scalar horizon
stress and the drought flag.

## 3 Models
**Conditional U-Net.** Input channels: noisy target (H), context NDVI (K, forward/backward filled), context
validity mask (K), standardised elevation and slope, one-hot land-cover groups. Conditioning vector: sinusoidal
diffusion-timestep embedding plus an MLP of the flattened weather sequence (11 features × 16 bins), sin/cos
day-of-year per bin, scalar stress and drought flag; injected by FiLM (scale and shift after GroupNorm) in every
residual block. Three resolution levels (32, 64, 64 channels), one residual block per level (CPU preset).

**Diffusion.** Cosine noise schedule, T = 1000, v-prediction, masked MSE loss (no gradient from invalid
pixels), AdamW (lr 2·10⁻⁴, warm-up 500 steps, cosine decay), batch 32, 8000 steps, gradient clipping at 1,
EMA 0.999. Sampling: deterministic DDIM with 25 steps. No classifier-free guidance (it would exaggerate the
weather effect and bias Δ).

**Baselines.** Persistence (last valid context value per pixel); climatology (per-pixel, per-bin mean over the
location's training years, falling back to the land-cover phenology mean); climatology + last anomaly; a
deterministic U-Net with identical inputs and conditioning trained with masked MSE; the same diffusion model
without physics (unconstrained).

## 4 Physics constraints
All constraint maths is in unscaled NDVI (the network works in x = (NDVI − 0.4)/0.5; conversion is unit-tested). <!--noaudit: code constants-->
1. **Rate bounds.** g_t = (x_{t+1} − x_t)/5 days, including the step from the last observed context value.
   Bounds r_inc(c), r_dec(c) per land-cover group c are the 99.5th percentiles of clean consecutive composite
   differences (both frames ≥ 95 % valid) on the training split. E_rate = mean[relu(g − r_inc)² + relu(−g − r_dec)²]/r̄².
2. **Phenology envelope.** μ(c, b), σ(c, b) per land-cover group and bin from non-drought training bins
   (s < 1), smoothed over bins. E_phen = mean[relu(x − (μ + 2σ))² + 1_normal · relu((μ − 2σ) − x)²]/0.1².
   Under the drought scenario the forecast may fall below the envelope but not exceed it.
3. **Monotone response to stress.** With the same noisy x_t and timestep, x̂₀ is predicted under the observed
   weather w and a more stressed w′ (perturbation above, a ~ U(0.5, 1.5)); E_mono = mean[relu(x̂₀(w′) − x̂₀(w) − 0.01)²]/0.1². <!--noaudit: code constants-->

**Auxiliary losses.** L = L_diff + Σ_k λ_k α̅(t) E_k(x̂₀) with λ_rate = 1, λ_phen = 0.5, λ_mono = 1; α̅(t) weights
the terms towards low noise. E_mono is computed on half of each batch (compute budget).

**Guidance.** At each DDIM step: x̂₀ from the network; energy E = Σ over pixels of (λ_rate E_rate + λ_phen E_phen +
λ_range E_range); x̂₀ ← x̂₀ − γ α̅(t) ∇_{x_t} E (γ = guidance scale, chosen on validation); the noise estimate is
recomputed from the guided x̂₀. **Projection:** NDVI clipped to [−1, 1] and consecutive differences clipped to
the rate bounds (sequentially from the last context value). **Monotone coupling** (optional, paired sampling):
x̂₀^drought ← min(x̂₀^drought, x̂₀^normal + 0.01) at every step; reported separately because it forces Δ ≤ 0.01.

## 5 Counterfactual estimation
For an initial scene, both scenarios are sampled with **common random numbers** (member i uses the same initial
noise in both branches; normal member i uses analog year i mod n). Δ = x^drought − x^normal per member;
reported as ensemble mean, s.d., 5th/95th percentiles per pixel and time, the area-and-horizon mean, and the
fraction of pixels whose 95th percentile of Δ is below zero ("significant loss").

## 6 Validation framework
1. **Proxy.** (a) Back-test of factual forecasts on non-drought test windows (MAE, RMSE, SSIM, sharpness).
   (b) Cross-year proxy: the normal-scenario forecast *change* from the forecast start is compared with the mean
   observed change at the same location and bins in other non-drought years (MAE), against persistence (zero
   change), climatology change and the factual forecast. Weakness: other years differ in more than weather; the
   climatology baseline partly shares those years.
2. **Natural experiment.** Treated: drought-labelled test windows. Controls: non-drought windows in the same year
   and bins at other held-out (validation/test) locations; nearest neighbours (up to 3, caliper 0.5 in
   standardised distance) on context NDVI level and trend, elevation, latitude and land-cover fractions, with
   covariate scaling fitted on training windows. Balance: standardised mean differences. Observed effect: DiD =
   (post − pre)_treated − mean(post − pre)_controls, pre = last two context bins, post = horizon mean.
   Compared with the model's observed-minus-normal effect (bias, correlation, coverage of the 90 % member interval).
   Weakness: unobserved confounders; DiD itself is noisy, so coverage is not expected to reach nominal.
3. **Placebos.** (a) Two different normal analog assignments with CRN (only windows with ≥ 2 analogs, so the test
   can fail), and identical normal weather with independent noise; Δ should be ≈ 0. (b) Temporal: AUC of the
   estimated observed-minus-normal loss for separating drought from non-drought windows. (c) Permutation:
   drought labels are reassigned across locations; the drought-minus-normal contrast should vanish.
4. **Calibration.** Fair CRPS, central 50/80/90 % interval coverage, spread–skill ratio and rank histograms on
   factual test forecasts (20 members); DiD coverage of the Δ interval.
5. **Constraint violations.** Fraction of pixel-timesteps violating rate bounds, envelope, NDVI range, and
   monotone ordering (forecast under w′ above forecast under w by > 0.01), with mean violation size.

## 7 Metrics
Masked MAE/RMSE in NDVI; SSIM on frames with ≥ 70 % valid pixels; sharpness = mean absolute spatial gradient and
share of spectral power above 0.25 cycles/pixel, prediction vs observation on frames with ≥ 95 % valid pixels; <!--noaudit: code constants-->
bootstrap 95 % intervals resampling locations (1000 draws); paired bootstrap for differences between variants.
