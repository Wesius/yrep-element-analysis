"""Visualization data models."""

from typing import Any, Dict, List, Optional, Literal
from pydantic import BaseModel, Field


class PlotSeries(BaseModel):
    """Single data series for plotting.

    Represents one line/trace in a plot, with x/y data
    and styling options.
    """
    name: str = Field(description="Series name for legend")
    x: List[float] = Field(description="X-axis values")
    y: List[float] = Field(description="Y-axis values")
    type: Literal["line", "scatter", "bar", "area"] = Field(
        default="line",
        description="Plot type"
    )
    color: Optional[str] = Field(
        default=None,
        description="Series color (CSS color string)"
    )
    opacity: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Series opacity"
    )
    line_width: float = Field(
        default=1.5,
        description="Line width in pixels"
    )
    visible: bool = Field(
        default=True,
        description="Whether series is initially visible"
    )


class PlotAnnotation(BaseModel):
    """Annotation marker on a plot."""
    x: float = Field(description="X position")
    y: float = Field(description="Y position")
    text: str = Field(description="Annotation text")
    color: Optional[str] = Field(default=None)


class PlotData(BaseModel):
    """Complete plot data for frontend rendering.

    Contains all data and configuration needed to render
    an interactive plot in the frontend.
    """
    title: str = Field(default="", description="Plot title")
    x_label: str = Field(
        default="Wavelength (nm)",
        description="X-axis label"
    )
    y_label: str = Field(
        default="Intensity (a.u.)",
        description="Y-axis label"
    )
    series: List[PlotSeries] = Field(
        default_factory=list,
        description="Data series to plot"
    )
    annotations: List[PlotAnnotation] = Field(
        default_factory=list,
        description="Plot annotations"
    )
    x_range: Optional[List[float]] = Field(
        default=None,
        description="Fixed x-axis range [min, max]"
    )
    y_range: Optional[List[float]] = Field(
        default=None,
        description="Fixed y-axis range [min, max]"
    )
    show_legend: bool = Field(
        default=True,
        description="Whether to show legend"
    )
    show_grid: bool = Field(
        default=True,
        description="Whether to show grid lines"
    )


class VisualizationRequest(BaseModel):
    """Request to generate a visualization."""
    type: Literal[
        "signal",
        "comparison",
        "detection",
        "templates",
        "preprocessing_stages"
    ] = Field(description="Visualization type")
    node_id: Optional[str] = Field(
        default=None,
        description="Node ID to visualize output from"
    )
    data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Direct data to visualize"
    )
    options: Dict[str, Any] = Field(
        default_factory=dict,
        description="Visualization options"
    )


class VisualizationResponse(BaseModel):
    """Response containing visualization data."""
    plot: PlotData = Field(description="Plot data for rendering")
    summary: Optional[str] = Field(
        default=None,
        description="Text summary of the visualization"
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )
