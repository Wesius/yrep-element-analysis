from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Union
import numpy as np


class SpectrumLike(Protocol):
    @property
    def wavelength(self) -> np.ndarray: ...

    @property
    def intensity(self) -> np.ndarray: ...


@dataclass(frozen=True)
class Spectrum:
    wavelength: np.ndarray
    intensity: np.ndarray
    metadata: Optional[Dict[str, str]] = None

    def __post_init__(self):
        if self.wavelength.shape != self.intensity.shape:
            raise ValueError("wavelength and intensity must have same shape")
        if self.wavelength.size == 0:
            raise ValueError("spectrum cannot be empty")


@dataclass(frozen=True)
class Instrument:
    fwhm_nm: float = 2.0
    grid_step_nm: Optional[float] = None
    max_shift_nm: float = 3.0


# Optional user overrides (advanced), each receives arrays and returns arrays
BackgroundFn = Callable[
    [np.ndarray, np.ndarray, np.ndarray, np.ndarray, Instrument],
    Tuple[np.ndarray, Dict[str, float]],
]
ContinuumFn = Callable[[np.ndarray, np.ndarray, float], Tuple[np.ndarray, np.ndarray]]


@dataclass
class AnalysisConfig:
    # Core
    instrument: Instrument = field(default_factory=Instrument)
    species: Optional[List[str]] = None  # optional species filter

    # Tweaks (coarse knobs)
    baseline_strength: float = 0.5  # 0..1
    # Continuum removal strategy: "arpls" (ARPLS subtraction), "rolling" (upper-envelope/percentile), or "both" (ARPLS then rolling)
    continuum_strategy: str = "both"
    regularization: float = 0.0  # ridge λ; 0.0 (none), ~1e-2 (light), ~1e-1 (strong)
    min_bands_required: int = 2
    presence_threshold: Optional[float] = None  # defaults to 0.02 if None
    top_k: int = 5

    # Preprocessing trims (flattened settings)
    min_wavelength_nm: Optional[float] = None
    max_wavelength_nm: Optional[float] = None
    auto_trim_left: bool = False
    auto_trim_right: bool = False

    # Background handling
    align_background: bool = (False) # if True, register (shift) background before subtraction

    # Optional advanced overrides
    background_fn: Optional[BackgroundFn] = None
    continuum_fn: Optional[ContinuumFn] = None

    # Search controls
    # Coarse wavelength shift alignment against templates
    search_shift: bool = True
    shift_search_iterations: int = 1
    # Optional FWHM optimization (templates rebuilt per candidate)
    search_fwhm: bool = True
    fwhm_search_iterations: int = 1
    # Search spreads
    # For shift (nm): search in [-max_shift_nm*shift_search_spread, +...]
    shift_search_spread: float = 1.0
    # For FWHM (relative): bracket width = fwhm_nm * fwhm_search_spread
    fwhm_search_spread: float = 0.5

    # Post-preprocessing wavelength mask: list of (start_nm, end_nm) intervals to zero
    mask: Optional[List[Tuple[float, float]]] = None


@dataclass
class PreprocessResult:
    wl_grid: np.ndarray
    y_meas: np.ndarray
    y_sub: np.ndarray
    y_cr: np.ndarray
    baseline: np.ndarray
    # Optional intermediates for visualization
    y_div: Optional[np.ndarray] = None
    baseline_div: Optional[np.ndarray] = None
    y_bg_interp: Optional[np.ndarray] = None
    avg_meas: Optional[Tuple[np.ndarray, np.ndarray]] = None
    avg_bg: Optional[Tuple[np.ndarray, np.ndarray]] = None


@dataclass
class DetectionResult:
    wl_grid: np.ndarray
    y_cr: np.ndarray
    y_fit: np.ndarray
    coeffs: np.ndarray
    species_order: List[str]
    present: List[Dict[str, Union[str, float, int]]]
    per_species_scores: Dict[str, float]
    fit_R2: float


@dataclass
class AnalysisResult:
    detections: List[Dict[str, Union[str, float, int]]]
    detection: DetectionResult
    metrics: Dict[str, Union[float, Dict[str, float]]]
