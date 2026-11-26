# Audit log

## Red-team review 1 (code, before any real-data training) — 2026-11-26

Independent fresh-context review of `src/` and `scripts/` (read-only; confirmed suspicions on the real-data
test fixture). Findings and actions:

| # | Severity | Finding | Action |
|---|---|---|---|
| C1 | critical | C4 "credible effects" verdict could pass without observations: scenario/noise placebos cannot fail by exchangeability; temporal AUC and permutation test use the drought flag that is itself a model input (circular). | C4 now counts only observation-based checks: natural experiment (DiD), new model-vs-observed anomaly agreement (`framework.anomaly_agreement`), cross-year proxy. Placebos/AUC/permutation reported as sanity checks only. |
| M1 | major | C2 "more physical" guaranteed by construction (projection zeroes rate violations; guidance uses the same envelope as the metric); no reference for over-constraint. | C2 judged on envelope + monotone violations, requires the loss-only model to improve on unconstrained diffusion (≥10 % relative), and reports violation rates of the observations themselves (`observed` row). |
| M2 | major | "Normal" analogs and natural-experiment controls only excluded drought-labelled windows; could be dry (s up to 1.56). | Normal = near-typical: \|mean s\| < `normal_max_s` and ≤ `normal_max_stressed_days` stressed days; same rule for controls; s distribution reported. Tests added. | <!--noaudit: review note on test fixture-->
| M3 | major | Lower phenology bound applied only to the normal branch, so guided Δ partly manufactured by guidance. | Effects also computed for the unguided physics model (`diff_phys`); summary reports Δ for every forecaster side by side. |
| M4 | major | First rate step used forward-filled last value as if 5 days old; never-observed pixels used a fill value. | `last_gap` per pixel: first-step rate uses the true time since the last valid observation; never-observed pixels excluded. Test added. |
| M5 | major | Sharpness verdict rewarded noise (larger ratio always "wins"). | Verdict uses closeness to observed sharpness (\|log ratio\|). |
| m | minor | Unequal ensemble sizes; γ swept without projection; uncapped neighbour analogs; perturbed fallback may not reach threshold; DiD pixel-set mismatch; covariate scaling on 64-px patches vs 32-px units; SSIM inflated by fill; envelope fitted on uncleaned frames. | All test variants 12 members; γ swept with +P; neighbours ≤ 1.5°; fallback shifted until s ≥ threshold + 0.25; DiD and model effect on a common pixel mask; covariate scaling on quadrants; SSIM over windows centred on valid pixels; envelope on frames ≥ 80 % valid. | <!--noaudit: code constants-->

Checked and found correct by the reviewer: leakage controls and tests, masked losses/metrics, NDVI unit
conversion in physics terms, CRN implementation, diffusion/DDIM/guidance maths, CRPS/coverage/spread-skill/rank
histogram, bootstraps, DiD formula and matching, factual context weather in scenarios.

All selection and verdict rules were (re)fixed in code before any model was trained on the full real dataset.

## Red-team review 2 (results and claims, fast run) — 2026-11-26

| # | Severity | Finding | Action |
|---|---|---|---|
| C1 | critical | Physics diffusion's drought anomaly ≈2.7× the observed anomaly; summary did not say so. | RESULTS_SUMMARY now adds the observed reference and per-model ratios, states this is model bias; LIMITATIONS has a "Drought-effect bias" section. |
| C2 | critical | "More physically consistent" over-claimed: guided/projected rates zero by construction and below observations; monotone metric in-sample. | Verdict capped at "partly supported" with explicit over-constraint wording; envelope and monotone reported separately; in-sample note added. |
| M1 | major | CRPS flatters ensembles; MAE: deterministic best; physics costs accuracy vs unconstrained diffusion. | MAE paired tests added to the accuracy evidence with the ensemble caveat. |
| M2 | major | Sharpness gain vs unconstrained diffusion marginal (≈4 %). | ≥10 % margin required -> "no meaningful difference". |
| M3 | major | Small samples (10 test locations, 11 drought windows, 5 matched pairs, SMD 0.38), uncorrected tests. | "Statistical caveats" section; natural experiment marked uninformative under n < 10 or SMD > 0.25. |
| M4 | major | Guidance scale at grid edge; calibration cost. | Stated in caveats and LIMITATIONS. |
| M5 | major | Docs said preset cpu, 270 test windows, 8000 steps, 20 members, 160 locations. | Preset and window counts read from provenance/per-window files; LIMITATIONS/METHODS distinguish planned design from the reported run. |
| mi | minor | Rate bounds vs observed 2.4 % exceedance; coupling identical to uncoupled; overconfident calibration; single seed. | Explained in LIMITATIONS; calibration verdict says "overconfident"; single-seed caveat. |

Verdict-rule changes after this review only make verdicts stricter.

## Owner review of the fast results — 2026-11-26

| Finding | Action |
|---|---|
| Abstract stated a CRPS win over the deterministic network without noting that it wins on MAE and SSIM. | Abstract now reports MAE and SSIM (deterministic best) and explains that CRPS favours ensembles over single forecasts. |
| λ sweep never ran; guidance scale at the grid edge. | Summary and paper state "λ not tuned (config defaults)" and that γ is the largest grid value (optimum not identified). |
| AUDIT.md left the number audit unchecked. | Audit run and ticked (below). |
| References not marked for checking; plain article skeleton. | refs.bib header requires authors to re-check every entry; [VERIFY] notes on entries with truncated author lists/unconfirmed details; TODO in main.tex to port to the Springer LNEE template. |
| Tiny sample (10 test locations, 11 drought windows at 6 locations). | Abstract, conclusion and summary state that no conclusion about drought effects is solid yet (counts generated from results). |

## Medium data and validation-only tuning — 2026-11-26 (owner request)
- Owner asked to "train so that all claims are supported". Declined as stated (that would select on test results);
  instead weaknesses were addressed with VALIDATION-only selection and pre-registered rules, and test was evaluated once.
- Medium run: 80 locations (download cut-off 18:15), re-split 56/12/12 before training, 3000 steps (`results/medium/`).
- λ sweep {0.3, 1, 3} on validation (pipeline rule) -> 3; γ -> 0.1 (grid edge); stochastic DDIM η {0, 0.5, 1} on
  validation -> 0 (η > 0 lowered validation coverage). Main reported run = `medium_tuned` (promoted to `results/`);
  fast run archived in `results/archive_fast69/`.
- Outcome: verdicts unchanged or stricter (tuning traded accuracy for fewer violations; calibration not improved).
- The summary's interpretive sentence about the drought-effect bias was fixed text written from the fast run and
  became wrong for the medium run; it is now generated from the per-model ratios.

## Pending (final audit, Section 11 of the brief)
- [x] Clean-checkout test (2026-11-26): fresh `git clone` + fresh venv (torch CPU + requirements.lock) -> `tasks.py test` 32 passed, `tasks.py smoke` OK. It caught a critical packaging bug: `.gitignore` rule `data/` also ignored `src/cfveg/data/`, so the data package had never been committed; fixed by anchoring to `/data/`.
- [x] Number audit (`scripts/audit_numbers.py`): OK on 2026-11-26 over docs/*.md and paper/*.tex; re-run automatically by `scripts/promote_results.py` and `tasks.py paper-assets` after every results change
- [x] Leakage audit re-run: tests/test_leakage.py passes; review 2 confirmed disjoint 49/10/10 locations (test ≥0.47° from train), disjoint years, train-only fitted objects, guidance chosen on validation <!--noaudit-->
- [x] Red-team review 2 (results and claims) — see above
- [x] Claim check in RESULTS_SUMMARY.md (generated; every abstract claim has a verdict)
