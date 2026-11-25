# Environment (recon 2026-11-25 22:13 IST)

| Item | Value |
|---|---|
| OS | Windows 11 Home 10.0.26200 |
| CPU | 13th Gen Intel Core i7-13620H, 16 logical threads |
| RAM | 31.6 GB |
| GPU | Intel UHD Graphics (no CUDA). `nvidia-smi` absent. |
| Disk | 663 GB free on C: |
| Python | 3.14.2 (venv `.venv`); brief asked 3.11, not installed |
| PyTorch | 2.15.0+cpu (CPU wheel index) |
| git | present; `gh` CLI **not installed**; `make` not installed (use `tasks.py`) |
| CDS key | `~/.cdsapirc` absent (checked with `test -f` only) |

## Network
Planetary Computer STAC 200, CDS 200, Hugging Face 200, Zenodo 200, NASA POWER 200, PyPI 200, Open-Meteo archive OK. `storage.googleapis.com` root 400 (normal for bucket root).

## Notable constraints
- **Windows Application Control blocks `numcodecs/vlen` DLL** → zarr unusable. Cubes stored as NetCDF/NPZ.
- Project lives outside OneDrive at `C:\Users\senth\cfveg-project\counterfactual-vegetation-diffusion` (user choice) to avoid syncing data/venv.
- No GPU → `cpu` preset (see PLAN.md). Training throughput and chosen thread settings are logged below once measured.

## Throughput (filled in after first training run)
