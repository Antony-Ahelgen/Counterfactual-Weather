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

## Next (optional, if more time)
1. Full run on all 160 locations (download remaining ~80, then `CFVEG_RESULTS=results/full python scripts/after_download.py`).
2. Multiple seeds; larger ensembles; per-constraint retraining ablations; finer guidance grid beyond 0.1.
3. Authors: verify refs.bib, port paper to the Springer LNEE template, add related work on causal attribution.
