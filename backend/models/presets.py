"""Preset workflow models."""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class PresetParameter(BaseModel):
    """User-configurable parameter for a preset.

    Presets expose certain parameters for user customization while
    keeping the underlying pipeline structure fixed.
    """
    name: str = Field(description="Parameter name/key")
    label: str = Field(description="Human-readable label")
    type: Literal["string", "number", "boolean", "select", "file", "directory"] = Field(
        description="Parameter type"
    )
    default: Any = Field(description="Default value")
    description: Optional[str] = Field(
        default=None,
        description="Help text for this parameter"
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="Valid options for select type"
    )
    required: bool = Field(
        default=True,
        description="Whether this parameter must be provided"
    )
    group: Optional[str] = Field(
        default=None,
        description="Parameter group for UI organization"
    )


class Preset(BaseModel):
    """Predefined workflow preset.

    Presets are pre-configured pipelines for common analysis tasks.
    Users can run them with minimal configuration.
    """
    id: str = Field(description="Unique preset identifier")
    name: str = Field(description="Display name")
    description: str = Field(description="What this preset does")
    category: str = Field(
        default="General",
        description="Preset category"
    )
    icon: Optional[str] = Field(
        default=None,
        description="Icon identifier for UI"
    )
    parameters: List[PresetParameter] = Field(
        default_factory=list,
        description="User-configurable parameters"
    )
    # Educational content
    explanation: str = Field(
        default="",
        description="Detailed explanation of the workflow"
    )
    use_cases: List[str] = Field(
        default_factory=list,
        description="Example use cases"
    )
    # Internal pipeline template
    pipeline_template: Dict[str, Any] = Field(
        default_factory=dict,
        description="Pipeline graph template with parameter placeholders"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "id": "basic_detection",
                "name": "Basic Element Detection",
                "description": "Detect elements in a single spectrum file",
                "category": "Detection",
                "parameters": [
                    {
                        "name": "signal_path",
                        "label": "Spectrum File",
                        "type": "file",
                        "default": "",
                        "description": "Path to spectrum file (.txt)",
                        "required": True
                    },
                    {
                        "name": "species_filter",
                        "label": "Target Elements",
                        "type": "string",
                        "default": "Cu,Fe,Pb",
                        "description": "Comma-separated element symbols to search for"
                    }
                ],
                "explanation": "This preset loads a spectrum, applies standard preprocessing, and detects specified elements using NNLS fitting.",
                "use_cases": ["Quick element identification", "Sample screening", "Teaching spectroscopy basics"]
            }
        }


class PresetList(BaseModel):
    """Collection of available presets."""
    presets: List[Preset] = Field(
        default_factory=list,
        description="Available presets"
    )
    categories: List[str] = Field(
        default_factory=list,
        description="Preset categories in display order"
    )


class PresetExecutionRequest(BaseModel):
    """Request to execute a preset with parameters."""
    preset_id: str = Field(description="Preset identifier to execute")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameter values"
    )
    workspace_root: Optional[str] = Field(
        default=None,
        description="Root directory for resolving paths"
    )
