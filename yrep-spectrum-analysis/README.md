## yrep_spectrum_analysis

Spectrum-analysis pipeline for emission spectra: preprocessing → template generation → NNLS-based species detection. 

Public API: `AnalysisConfig`, `Spectrum`, `analyze`.

### Quickstart (see `run_with_library.py`)

```python
from pathlib import Path

from yrep_spectrum_analysis import AnalysisConfig, analyze
from yrep_spectrum_analysis.utils import load_batch, load_references, group_spectra

base = Path(__file__).resolve().parent

# 1) Load reference line lists (CSV directory with wavelength/species/intensity)
refs = load_references(base / "data" / "lists")

# 2) Load spectra (directories of .txt files)
meas_root = base / "data" / "StandardsTest" / "Copper" / "Copper"
bg_root = base / "data" / "StandardsTest" / "Copper" / "BG"
measurements, backgrounds = load_batch(meas_root, bg_root)

# (Optional) Group similar spectra and analyze each group
groups = group_spectra(measurements)

cfg = AnalysisConfig(
    fwhm_nm=0.75,
    grid_step_nm=None,
    species=["Na", "K", "Ca", "Cu", "Zn", "Pb"],
    baseline_strength=0.5,
    continuum_strategy="both", # If data is good, try "arpls"
    regularization=0.0,
    min_bands_required=5,
    presence_threshold=0,  # minimum FVE to display
    top_k=0, # all top elements
    min_wavelength_nm=300, # yrep spectrometer most accurate in this range for some reason
    max_wavelength_nm=600,
    auto_trim_left=False, # normally better to do manually above
    auto_trim_right=False,
    align_background=False, # works occasionally
    # depending on data quality, the following can help a lot. try these values: 
    search_shift=False, 
    shift_search_iterations=0, # 10
    shift_search_spread=0, # absolute nm window
    search_fwhm=False,
    fwhm_search_iterations=0, # 10
    fwhm_search_spread=0, # 1 (nm)
)

for i, group in enumerate(groups, start=1):
    result = analyze(
        measurements=group,
        references=refs,
        backgrounds=backgrounds,
        config=cfg,
        visualize=True,
        viz_output_dir=str(base / "plots" / "copper" / f"group_{i:02d}"),
        viz_show=False,
    )
    print("R²=", float(result.metrics.get("fit_R2", 0.0)))
    for d in result.detections:
        print(d["species"], d.get("score", d.get("fve", 0.0)), d.get("bands_hit", 0))
```

`analyze(...)` returns an `AnalysisResult` with:
- `metrics`: `{"fit_R2": float}`
- `detections`: list of dicts with `species`, `score` (FVE), `fve`, `coeff`, `bands_hit`
- `detection`: detailed single-run `DetectionResult`

Visualizer files are saved when `visualize=True` to `viz_output_dir`.

### Configuration reference

- **AnalysisConfig**
  - Instrument and core
    - **fwhm_nm (float, default 2.0)**: Full width at half maximum used to broaden reference lines when building Gaussian templates; also sizes band regions.
    - **grid_step_nm (float | None, default None)**: Uniform interpolation step for the working wavelength grid. If `None`, derived from data (median positive spacing) with a minimum of ~1000 samples.
    - **average_n_points (int | None, default None)**: Averaging grid points when combining multiple spectra. If `None`, defaults to 1000 internally.
    - **species (list[str] | None)**: Optional whitelist of species symbols (case-insensitive base symbols) to consider.
  - Tweaks
    - **baseline_strength (float, default 0.5)**: 0..1 knob that scales continuum removal windows and smoothing. Larger = broader/lower baseline.
    - **continuum_strategy ("arpls" | "rolling" | "both", default "both")**: Continuum removal mode. `arpls`: robust ARPLS baseline subtraction. `rolling`: upper-envelope subtraction. `both`: ARPLS then rolling envelope (recommended).
    - **regularization (float, default 0.0)**: Ridge λ for the NNLS fit (Tikhonov via augmentation). `0.0` disables; ~1e-2 light, ~1e-1 stronger.
    - **min_bands_required (int, default 2)**: Minimum corroborated band regions per species required to count as detected (when band info is available).
    - **presence_threshold (float | None, default None)**: FVE threshold (fraction of total variance explained) to accept a species. If `None`, defaults to `0.02`.
    - **top_k (int, default 5)**: Keep top-K detections by score. `<= 0` keeps all.
  - Preprocessing trims
    - **min_wavelength_nm (float | None)** / **max_wavelength_nm (float | None)**: Explicit wavelength limits to keep.
    - **auto_trim_left (bool, default False)** / **auto_trim_right (bool, default False)**: Auto-detect and trim off extreme left/right spikes using slope heuristics.
  - Background handling
    - **align_background (bool, default False)**: If true, registers background to measurement via phase correlation before subtraction. If false, subtracts as-is.
    - **background_fn (callable | None)**: Advanced override for background subtraction/registration. Signature `(wl, y_meas, wl_bg, y_bg, config) -> (y_sub, params_dict)`.
  - Continuum override
    - **continuum_fn (callable | None)**: Advanced override for continuum removal. Signature `(wl, y, strength) -> (y_cr, baseline)`.
  - Search controls (coarse optimization inside detection)
    - **search_shift (bool, default True)**: Enable iterative coarse wavelength alignment of the observed signal to templates.
    - **shift_search_iterations (int, default 1)**: Number of refinement iterations; the shift step reduces each iteration.
    - **shift_search_spread (float, default 0.0)**: Absolute nm window for shift search: search range = `±shift_search_spread`. Set `<= 0` to disable.
    - **search_fwhm (bool, default True)**: Enable FWHM search by rebuilding templates around the current `fwhm_nm`.
    - **fwhm_search_iterations (int, default 1)**: Iterative narrowing of the candidate bracket around `fwhm_nm`.
    - **fwhm_search_spread (float, default 0.5)**: Relative bracket half-width for the first iteration: `width = fwhm_nm × fwhm_search_spread`.
  - Masking
    - **mask (list[tuple[float, float]] | None, default None)**: Post-preprocessing wavelength intervals to zero out (e.g., instrument artifacts).

### API surface

```python
from yrep_spectrum_analysis import AnalysisConfig, Spectrum, analyze
```

`analyze(measurements, references, *, backgrounds=None, config=None, visualize=False, viz_output_dir=None, viz_show=False)`
- `measurements`: sequence of objects with `.wavelength` and `.intensity` arrays (`Spectrum` or similar)
- `references`: DataFrame or tuple/dict of `(species, wavelength_nm, intensity)`; see `utils.load_references`
- `backgrounds`: optional sequence of background spectra
- `visualize`: generate and save plots
- `viz_output_dir`: output directory for plots (default `./debug_plots`)
- `viz_show`: show plots interactively

Utilities used in the example:
- `utils.load_references(lists_dir)`: reads CSVs with line lists and returns a normalized DataFrame
- `utils.load_batch(meas_root, bg_root)`: reads `.txt` spectra into `Spectrum` objects
- `utils.group_spectra(measurements)`: clusters similar spectra to process them as groups

### Notes

- Species names are normalized to base symbols (e.g., "Fe I" → "FE").
- When `top_k <= 0`, all detections are returned in score-descending order.
- Visualization saves numbered PNGs for each step of the pipeline.


