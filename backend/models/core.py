"""Core data models for spectral analysis."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Signal(BaseModel):
    """Single spectrum with wavelength and intensity arrays.

    Represents a spectral measurement from a spectrometer, containing
    matched arrays of wavelength (in nanometers) and intensity values.
    """
    wavelength: List[float] = Field(
        description="Wavelength values in nanometers"
    )
    intensity: List[float] = Field(
        description="Intensity values (arbitrary units)"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (source file, acquisition params, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "wavelength": [300.0, 300.5, 301.0, 301.5, 302.0],
                "intensity": [0.1, 0.15, 0.3, 0.25, 0.12],
                "meta": {"source": "sample_001.txt", "integration_time_ms": 100}
            }
        }


class Detection(BaseModel):
    """Single species detection result.

    Represents the detection of a chemical species (element or compound)
    in a spectrum, including confidence score and diagnostic metadata.
    """
    species: str = Field(
        description="Detected species identifier (e.g., 'Cu', 'Fe I', 'CuO')"
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Detection confidence score (0-1, higher = more confident)"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detection metadata (bands_hit, wavelengths matched, etc.)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "species": "Cu",
                "score": 0.92,
                "meta": {"bands_hit": 12, "primary_wavelength": 324.7}
            }
        }


class DetectionResult(BaseModel):
    """Complete detection analysis result.

    Contains the processed signal along with all detected species
    and analysis metadata like fit quality metrics.
    """
    signal: Signal = Field(
        description="The analyzed signal after preprocessing"
    )
    detections: List[Detection] = Field(
        default_factory=list,
        description="List of detected species"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis metadata (fit_R2, processing_time, etc.)"
    )


class FileInfo(BaseModel):
    """Information about a single file."""
    name: str = Field(description="File name")
    path: str = Field(description="Full file path")
    size: int = Field(description="File size in bytes")
    is_dir: bool = Field(description="Whether this is a directory")
    extension: Optional[str] = Field(
        default=None,
        description="File extension (without dot)"
    )


class DirectoryListing(BaseModel):
    """Directory contents listing."""
    path: str = Field(description="Directory path")
    files: List[FileInfo] = Field(
        default_factory=list,
        description="List of files and subdirectories"
    )
    parent: Optional[str] = Field(
        default=None,
        description="Parent directory path"
    )
