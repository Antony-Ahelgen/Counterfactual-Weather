# Counterfactual Vegetation Forecasting under Drought with Physics-Constrained Diffusion

Research code. Status: work in progress (see `PROGRESS.md`).

## Quick start (Windows, no make)
```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.lock
.venv/Scripts/python -m pip install -e . --no-deps
.venv/Scripts/python tasks.py test            # unit tests
.venv/Scripts/python tasks.py smoke           # end-to-end smoke run on tiny data (CPU, minutes)
.venv/Scripts/python tasks.py data --preset cpu
.venv/Scripts/python tasks.py train --preset cpu
.venv/Scripts/python tasks.py eval --preset cpu
.venv/Scripts/python tasks.py paper-assets --preset cpu
```
On Linux/macOS, `make test|data|train|eval|paper-assets PRESET=cpu PY=.venv/bin/python` does the same.
PyTorch CPU wheel: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.
