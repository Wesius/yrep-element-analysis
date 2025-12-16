from .stages import (
    average_signals,
    trim,
    mask,
    resample,
    subtract_background,
    continuum_remove_arpls,
    continuum_remove_rolling,
    fwhm_search,
    build_templates,
    shift_search,
    detect_nnls,
)
from .decomposition import (
    analyze_pca,
    analyze_ica,
    analyze_mcr,
    identify_components,
)

__all__ = [
    "average_signals",
    "trim",
    "mask",
    "resample",
    "subtract_background",
    "continuum_remove_arpls",
    "continuum_remove_rolling",
    "fwhm_search",
    "build_templates",
    "shift_search",
    "detect_nnls",
    "analyze_pca",
    "analyze_ica",
    "analyze_mcr",
    "identify_components",
]
