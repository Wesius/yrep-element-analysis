"""Node definition models for the pipeline builder."""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class ConfigField(BaseModel):
    """Configuration field definition for a node.

    Describes a single configurable parameter for a pipeline node,
    including type information and UI hints.
    """
    name: str = Field(description="Field name/key")
    type: Literal["string", "number", "boolean", "select", "file", "directory", "json"] = Field(
        description="Field data type"
    )
    default: Any = Field(
        default=None,
        description="Default value"
    )
    label: Optional[str] = Field(
        default=None,
        description="Human-readable label"
    )
    description: Optional[str] = Field(
        default=None,
        description="Help text explaining the parameter"
    )
    options: Optional[List[str]] = Field(
        default=None,
        description="Valid options for select-type fields"
    )
    min: Optional[float] = Field(
        default=None,
        description="Minimum value for number fields"
    )
    max: Optional[float] = Field(
        default=None,
        description="Maximum value for number fields"
    )
    step: Optional[float] = Field(
        default=None,
        description="Step increment for number fields"
    )


class PortDefinition(BaseModel):
    """Input or output port definition.

    Ports are connection points on nodes. Inputs receive data from
    upstream nodes; outputs send data to downstream nodes.
    """
    name: str = Field(description="Port name (e.g., 'Signal', 'References')")
    type: str = Field(description="Data type expected/produced")
    optional: bool = Field(
        default=False,
        description="Whether this input can be left unconnected"
    )
    multi: bool = Field(
        default=False,
        description="Whether this input accepts multiple connections"
    )


class NodeDefinition(BaseModel):
    """Complete definition of a pipeline node type.

    Defines a reusable processing step that can be added to pipelines.
    Includes full metadata for UI rendering and educational content.
    """
    identifier: str = Field(
        description="Unique identifier (e.g., 'trim', 'detect_nnls')"
    )
    title: str = Field(
        description="Display title (e.g., 'Trim Wavelength Range')"
    )
    category: str = Field(
        description="Category for grouping (e.g., 'Preprocess', 'Detection')"
    )
    inputs: List[PortDefinition] = Field(
        default_factory=list,
        description="Input port definitions"
    )
    outputs: List[PortDefinition] = Field(
        default_factory=list,
        description="Output port definitions"
    )
    config_fields: List[ConfigField] = Field(
        default_factory=list,
        description="Configurable parameters"
    )
    default_config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Default configuration values"
    )
    # Educational content
    description: str = Field(
        default="",
        description="Brief description of what this node does"
    )
    explanation: str = Field(
        default="",
        description="Detailed educational explanation of the processing step"
    )
    tips: List[str] = Field(
        default_factory=list,
        description="Usage tips and best practices"
    )
    related_nodes: List[str] = Field(
        default_factory=list,
        description="Related node identifiers for discovery"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "identifier": "trim",
                "title": "Trim Wavelength Range",
                "category": "Preprocess",
                "description": "Restrict signal to a specific wavelength range",
                "explanation": "Trimming removes data points outside a specified wavelength range. This is useful for focusing analysis on regions of interest and removing noisy edge data.",
                "inputs": [{"name": "Signal", "type": "Signal", "optional": False, "multi": False}],
                "outputs": [{"name": "Signal", "type": "Signal"}],
                "config_fields": [
                    {"name": "min_nm", "type": "number", "default": 300.0, "label": "Min Wavelength (nm)"},
                    {"name": "max_nm", "type": "number", "default": 600.0, "label": "Max Wavelength (nm)"}
                ],
                "tips": ["Use trim early in your pipeline to reduce processing time", "Check your spectrometer's valid range to avoid edge artifacts"]
            }
        }


class NodeInstance(BaseModel):
    """Instance of a node in a pipeline graph.

    Represents a specific node placed on the canvas with its
    position and configured parameters.
    """
    id: str = Field(description="Unique instance ID within the graph")
    identifier: str = Field(description="Node type identifier")
    config: Dict[str, Any] = Field(
        default_factory=dict,
        description="Instance configuration values"
    )
    position: Dict[str, float] = Field(
        default_factory=lambda: {"x": 0, "y": 0},
        description="Canvas position {x, y}"
    )
