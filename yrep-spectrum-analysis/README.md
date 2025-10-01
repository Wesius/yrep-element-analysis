## yrep_spectrum_analysis

Composable spectral analysis toolkit built around simple dataclasses and pure functions.

### Core dataclasses
- `Signal`: 1-D wavelength/intensity arrays plus a lightweight `meta` dict.
- `References`: per-species raw line lists, each stored as `(wavelength_nm, intensity)` arrays.
- `Templates`: Gaussian-broadened matrix aligned to a signal grid with companion species/band metadata.
- `DetectionResult`: processed signal + detections list (each detection holds per-species `meta`).

### Quickstart (see `run_with_library.py`)

```python
from pathlib import Path

from yrep_spectrum_analysis import (
    average_signals,
    build_templates,
    fwhm_search,
    continuum_remove_arpls,
    continuum_remove_rolling,
    detect_nnls,
    resample,
    shift_search,
    subtract_background,
    trim,
)
from yrep_spectrum_analysis.utils import (
    expand_species_filter,
    group_signals,
    is_junk_group,
    load_references,
    load_signals_from_dir,
)

base = Path(__file__).resolve().parent
refs = load_references(base / "data" / "lists", element_only=False)
measurements = load_signals_from_dir(base / "data" / "StandardsTest" / "Copper" / "Copper")
backgrounds = load_signals_from_dir(base / "data" / "StandardsTest" / "Copper" / "BG")
species_filter = expand_species_filter(refs.lines.keys(), ["CU", "ZN", "PB"])

for group in group_signals(measurements):
    if is_junk_group(group):
        continue

    signal = average_signals(group, n_points=1200)
    background = average_signals(backgrounds, n_points=1200) if backgrounds else None

    processed = trim(signal, min_nm=300, max_nm=600)
    processed = resample(processed, n_points=1500)
    if background is not None:
        processed = subtract_background(processed, background, align=False)
    processed = continuum_remove_arpls(processed, strength=0.5)
    processed = continuum_remove_rolling(processed, strength=0.5)

    templates = build_templates(
        processed,
        references=refs,
        fwhm_nm=0.75,
        species_filter=species_filter,
    )
    templates = fwhm_search(
        processed,
        references,
        initial_fwhm_nm=0.75,
        spread_nm=0.2,
        iterations=3,
        species_filter=species_filter,
    )
    processed = shift_search(processed, templates, spread_nm=0.5, iterations=3)
    result = detect_nnls(processed, templates, presence_threshold=0.02, min_bands=5)

    print(result.meta.get("fit_R2"))
    for det in result.detections:
        print(det.species, det.score, det.meta.get("bands_hit"))
```

### Building custom pipelines

Every preprocessing stage is a `Signal -> Signal` function. Compose them explicitly to suit your dataset.

```python
history = []
signal = average_signals(group, n_points=1200)
history.append(("input", signal))

processed = trim(signal, min_nm=300, max_nm=600)
history.append(("trim", processed))
processed = resample(processed, n_points=1500)
history.append(("resample", processed))
processed = subtract_background(processed, background, align=False)
history.append(("background", processed))
processed = continuum_remove_arpls(processed, strength=0.5)
history.append(("continuum_arpls", processed))
processed = continuum_remove_rolling(processed, strength=0.5)
history.append(("continuum_rolling", processed))

templates = build_templates(processed, references, fwhm_nm=0.75)
templates = fwhm_search(
    processed,
    references,
    initial_fwhm_nm=0.75,
    spread_nm=0.2,
    iterations=3,
)
processed = shift_search(processed, templates, spread_nm=0.5, iterations=3)
history.append(("shift", processed))

result = detect_nnls(processed, templates, presence_threshold=0.02, min_bands=5)
```

### Utilities
- `utils.load_signals_from_dir(path)` → list of `Signal` instances from `.txt` spectra.
- `utils.group_signals(signals)` → cosine-similarity clusters for batch processing.
- `utils.signal_quality(signal)` / `utils.is_junk_group(signals)` → quick QC heuristics.
- `utils.expand_species_filter(ref_species, requested)` → expand base elements to ionized labels present in references.

### Visualization

Capture whichever stages you need (as shown above) and plot them with your preferred tool; each step simply returns a new `Signal`.

### Migration Notes
- The legacy `AnalysisConfig` and `PreprocessResult` types have been removed.
- Old pipeline helpers (`_preprocess.py`, `_detect.py`, `_templates.py`) were replaced by pure functions under `yrep_spectrum_analysis.pipeline`.
