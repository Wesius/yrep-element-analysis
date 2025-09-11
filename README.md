# YREP Spectral Analysis

Repo for running spectral analysis on various spectrometer outputs.

## Setup (uv)

```bash
# Create venv and install deps
uv sync

# Activate venv (Windows)
./.venv/Scripts/activate

# Activate venv (Mac)
source .venv/bin/activate
```

## Run

```bash
python run_with_library.py
# or (if outside of venv)
uv run run_with_library.py
```

For details, refer to `/yrep-spectral-analysis/README.md`

Outputs will be written under `plots/`.
