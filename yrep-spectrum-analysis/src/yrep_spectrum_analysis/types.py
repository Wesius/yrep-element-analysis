from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple
import numpy as np


@dataclass
class Signal:
    """Single spectrum snapshot carried between preprocessing stages."""
    wavelength: np.ndarray  # shape: (n_samples,)
    intensity: np.ndarray   # shape: (n_samples,)
    meta: Dict[str, Any] = field(default_factory=dict)

    def copy(self, *, deep: bool = True) -> "Signal":
        wl = np.copy(self.wavelength) if deep else self.wavelength
        iy = np.copy(self.intensity) if deep else self.intensity
        meta = {k: v for k, v in self.meta.items()} if deep else self.meta
        return Signal(wavelength=wl, intensity=iy, meta=meta)


@dataclass
class References:
    """Line lists grouped by species.

    Each entry maps to a tuple of (wavelength_nm, intensity) arrays, both 1-D.
    """
    lines: Dict[str, Tuple[np.ndarray, np.ndarray]]
    meta: Dict[str, Any] = field(default_factory=dict)

    def species(self) -> List[str]:
        return list(self.lines.keys())


@dataclass
class Templates:
    """Gaussian-broadened templates aligned to a signal's grid."""
    matrix: np.ndarray                 # shape: (n_samples, n_species)
    species: List[str]
    bands: Dict[str, List[Tuple[float, float]]]
    meta: Dict[str, Any] = field(default_factory=dict)

    def as_tuple(self) -> Tuple[np.ndarray, List[str], Dict[str, List[Tuple[float, float]]]]:
        return self.matrix, self.species, self.bands


@dataclass
class Detection:
    species: str
    score: float
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    signal: Signal
    detections: List[Detection]
    meta: Dict[str, Any] = field(default_factory=dict)
