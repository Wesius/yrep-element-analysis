"""Public API for yrep_spectrum_analysis (minimal surface).

Exports:
- AnalysisConfig: configuration object controlling instrument and presets
- Spectrum: typed container for wavelength/intensity arrays
- analyze: full pipeline (preprocess → templates/bands → detect)
"""

from .types import AnalysisConfig, Spectrum, Instrument
from .api import analyze

__all__ = [
    "AnalysisConfig",
    "Instrument",
    "Spectrum",
    "analyze",
]
