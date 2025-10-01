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
    "Detection",
    "DetectionResult",
    "References",
    "Signal",
    "Templates",
]
