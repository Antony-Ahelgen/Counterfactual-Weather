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

- 2026-11-26 00:11-14:30 S2 download ran; laptop slept ~10 h overnight (download paused). Found DNS-outage bug (lenient load saved empty data) -> fixed with network wait + empty-frame guard; 1 suspect location (L0021) removed for re-download. Download restarted 14:31 (5 workers).
- 2026-11-26 Real-data fixture (10 locations, 16x16 crop) replaces all synthetic test data; synthetic generator deleted. Oracle cut (user instruction).
- 2026-11-26 Built: metrics, evaluation, paired CRN sampler, effects, 5-part validation, counterfactual demo, smoke script (70 s), number audit, orchestrator (scripts/run_pipeline.py, tested on fixture).
- 2026-11-26 Throughput (during download): ~1.0 s/step (n_res 1, 8 threads), physics model ~2x. cpu preset: 8000 steps.
- 2026-11-26 Rate bounds: 99.5th pct of clean diffs (both frames >=95 % valid) still ~0.06 NDVI/day (residual cloud/haze); documented in LIMITATIONS.

- 2026-11-26 14:57 Heartbeat: S2 download running (56/160, ETA ~18:30 IST if laptop stays awake). Watcher `scripts/after_download.py` running in background (log runs/after_download.log): fills gaps then runs `scripts/run_pipeline.py --preset cpu --parallel 2` (log runs/pipeline.log). Paper assets, summary renderer, audit, README done and smoke-tested.

- 2026-11-26 15:10-16:20 Red-team review 1 fixed (see docs/AUDIT.md). Download hit intermittent DNS failures -> per-acquisition loading with retries.
- 2026-11-26 Remote added: https://github.com/Antony-Ahelgen/Counterfactual-Weather (push pending: owner must sign in once as Antony-Ahelgen in their own terminal; stored credential was for another account).
- 2026-11-26 ~16:30 Owner asked for fabricated data to finish in 2 h -> declined (never fabricate). Owner chose the FAST REAL run: 69 downloaded locations, re-split before training, preset `fast`, `scripts/run_fast.py` (log runs/run_fast.log, runs/pipeline.log).

- 2026-11-26 16:20-17:11 FAST real-data pipeline completed (69 locations, 3 models x 2500 steps, gamma=0.1 chosen on validation). Results in results/, docs/RESULTS_SUMMARY.md. Tags v0.2..v1.0 pushed.
- 2026-11-26 Red-team review 2 (results/claims) addressed: drought effects of physics models ~2.7x observed anomaly (model bias, stated); stricter verdicts; docs match the run. Number audit OK. Paper draft filled (paper/main.tex, numbers via paper/numbers.tex; no LaTeX on this machine -> compile in Overleaf).
- Pushing works (origin = Antony-Ahelgen/Counterfactual-Weather).

- 2026-11-26 17:20-18:15 extra download to 80 locations; 18:15-19:35 medium run; 18:55-20:05 lambda sweep (x3), gamma 0.1, eta 0 on validation; tuned test evaluated once. 20:10 tuned run promoted to results/ (fast archived in results/archive_fast69, untuned medium in results/medium). Paper/docs updated; audit OK.

- 2026-11-26 ~20:40 v2 improvement round started (owner chose the overnight plan; may stop and resume next day).
  Design fixed before any v2 result: dev = 80 known locations (68 train / 12 val), FRESH test = 80 never-downloaded
  locations (manifests/locations_v2.csv); 5000 steps x 3 seeds; lambda x3; no rate projection; gamma matched to the
  observed envelope-violation rate on validation; post-hoc spread calibration on validation; test evaluated once.

- 2026-11-27 15:27 v2: all 9 models trained (3 variants x 3 seeds, 5000 steps). Validation selections (pre-registered rules):
  gamma = 0 (diff_phys+P already violates the envelope less than the observations, 1.5 % vs 3.6 % on val; guidance
  only lowers it further); spread factors 1.10-1.22 (ensembles under-dispersed on val). Waiting for the fresh
  test-set download (134/160 at 15:44; 26 test locations left), then cube rebuild and the single test evaluation.

- 2026-11-27 15:50 Owner deadline 19:00 IST -> owner chose to test on the 54 fresh test locations already downloaded
  (decided before any v2 test result). Download stopped; the 26 remaining test locations are left out of v2 (logged as
  "missing s2" in the cube meta). Finishing with `scripts/finish_v2.py` (log runs/finish_v2_stdout.log, runs/pipeline.log).

- 2026-11-27 17:33 V2 DONE and promoted to results/ (medium_tuned archived). Verdicts (combined over 3 seeds):
  physics consistency SUPPORTED; CRPS accuracy partly supported; sharper than det mixed (2 of 3 seeds); sharper than
  unconstrained diffusion not supported; drought effects insufficient evidence; calibration not supported
  (90 % coverage 0.71). Paper/docs updated, audit OK, 33 tests pass. Downloads (25 reserved test + 80 extra) resumed.

- 2026-11-27 19:25 v2.1 confirmation on 12 reserved fresh locations (results/v21/CONFIRMATION.md): physics
  consistency and sharper-than-det supported in all seeds; pre-registered calibrated model mixed; uncorrected
  32-member ensembles meet the calibration rule in all seeds (exploratory). Paper section added.

## Next  (RESUME HERE)
1. Downloads for a possible v3 run in the background (resumable):
   `.venv\Scripts\python scripts\download_data.py --stage s2 --workers 5 --threads 3 >> runs\download\s2_v2test.log 2>&1`
   then `.venv\Scripts\python scripts\fetch_extra.py >> runs\download\extra.log 2>&1`.
2. v3 (if wanted): train on the 135 v2 locations; FRESH test = the 25 reserved v2 test locations + the X... extra
   locations; design fixed before any v3 result. A GPU would help the sharpness claim most.
3. Authors: verify refs.bib, port paper to the Springer LNEE template, add related work on causal attribution.
