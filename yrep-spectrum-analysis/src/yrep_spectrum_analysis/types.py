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

    @property
    def sigma_nm(self) -> float:
        return float(self.fwhm_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))


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
    mode: str = "balanced"  # fast | balanced | accurate
    sensitivity: str = "medium"  # low | medium | high
    species: Optional[List[str]] = None  # optional species filter

    # Tweaks (coarse knobs)
    baseline_strength: float = 0.5  # 0..1
    regularization: str = "none"  # none | light | strong
    min_bands_required: int = 2
    presence_threshold: Optional[float] = None  # derived from sensitivity if None
    top_k: int = 5
    return_arrays: bool = True

    # Preprocessing trims
    min_wavelength_nm: Optional[float] = (
        None  # drop data below this wavelength if provided
    )
    auto_trim_left: bool = (
        False  # if True and min_wavelength_nm is None, auto-trim steep left spike
    )

    # Background handling
    align_background: bool = (
        False  # if True, register (shift) background before subtraction
    )

    # Optional advanced overrides
    background_fn: Optional[BackgroundFn] = None
    continuum_fn: Optional[ContinuumFn] = None


@dataclass
class PreprocessResult:
    wl_grid: np.ndarray
    y_meas: np.ndarray
    y_sub: np.ndarray
    y_cr: np.ndarray
    baseline: np.ndarray
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
