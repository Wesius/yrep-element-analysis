"""Composable spectral analysis pipeline."""

from .pipeline import (
    average_signals,
    build_templates,
    continuum_remove_arpls,
    continuum_remove_rolling,
    fwhm_search,
    detect_nnls,
    mask,
    resample,
    shift_search,
    subtract_background,
    trim,
    analyze_pca,
    analyze_ica,
    analyze_mcr,
    identify_components,
)
from .types import Detection, DetectionResult, References, Signal, Templates

__all__ = [
    "average_signals",
    "build_templates",
    "continuum_remove_arpls",
    "continuum_remove_rolling",
    "fwhm_search",
    "detect_nnls",
    "mask",
    "resample",
    "shift_search",
    "subtract_background",
    "trim",
    "analyze_pca",
    "analyze_ica",
    "analyze_mcr",
    "identify_components",
    "Detection",
    "DetectionResult",
    "References",
    "Signal",
    "Templates",
]
