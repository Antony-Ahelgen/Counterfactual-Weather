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

## Next
1. When `runs/download/s2.log` reaches 160/160 (or ends): re-run `python scripts/download_data.py --stage s2 --workers 5` to fill any failures, then
   `python scripts/run_pipeline.py --preset cpu --parallel 2 > runs/pipeline_stdout.log 2>&1` (resumable; stages skip when outputs exist).
2. While it runs: paper_assets.py (figures + LaTeX tables), render_summary.py (RESULTS_SUMMARY + claim check), red-team review, AUDIT.md.
3. Tags: v0.1-data after build_data; v0.2-baselines/v0.3-diffusion after stage 2; v0.4-physics after stage 4; v0.5/v0.6 after stage 6.
