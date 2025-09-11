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


# Example outputs:

## Background Subtraction
<img width="3714" height="2799" alt="04_background_subtraction" src="https://github.com/user-attachments/assets/f190e1e4-68a1-47a9-9866-186ac5487f43" />

## arPLS + Rolling Percentile Filter
<img width="5509" height="2799" alt="05_continuum_removal" src="https://github.com/user-attachments/assets/20bf997b-9bf7-4625-acc9-aad487b8ae62" />

## Expected Band Reonstruction (NIST Database)
<img width="3688" height="2277" alt="08_band_regions" src="https://github.com/user-attachments/assets/96fa6be2-c4a9-4c3b-a7f5-027bdc80a04c" />

## NLSS Reconstruction (2010 Penny)
<img width="5060" height="3843" alt="10_nnls_fitting" src="https://github.com/user-attachments/assets/e012255d-6a14-4777-ab0d-c53978ea944c" />

## NLSS Reconstruction (Pb)
<img width="5032" height="3843" alt="10_nnls_fitting" src="https://github.com/user-attachments/assets/e8f54d08-2159-410f-8bff-77d17c1f72ee" />

