"""Public API for yrep_spectrum_analysis (minimal surface).

Exports:
- AnalysisConfig: configuration object controlling analysis parameters (incl. fwhm/grid)
- Spectrum: typed container for wavelength/intensity arrays
- analyze: full pipeline (preprocess → templates/bands → detect)
"""

from .types import AnalysisConfig, Spectrum
from .api import analyze

__all__ = [
    "AnalysisConfig",
    "Spectrum",
    "analyze",
]
