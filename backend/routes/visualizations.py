"""Visualization API routes."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException

from backend.models.visualization import (
    PlotData,
    PlotSeries,
    PlotAnnotation,
    VisualizationRequest,
    VisualizationResponse,
)
from backend.models.core import Signal, DetectionResult

router = APIRouter()


def _signal_to_plot(
    signal_data: Dict[str, Any],
    title: str = "",
    normalize: bool = False,
    color: Optional[str] = None,
) -> PlotData:
    """Convert signal data to plot format."""
    wavelength = signal_data.get("wavelength", [])
    intensity = signal_data.get("intensity", [])

    if normalize and intensity:
        max_val = max(abs(v) for v in intensity) if intensity else 1.0
        if max_val > 0:
            intensity = [v / max_val for v in intensity]

    series = [PlotSeries(
        name="Signal",
        x=wavelength,
        y=intensity,
        type="line",
        color=color or "#4AA0E0",
        line_width=1.5,
    )]

    return PlotData(
        title=title,
        x_label="Wavelength (nm)",
        y_label="Intensity (a.u.)" if not normalize else "Normalized Intensity",
        series=series,
    )


def _detection_to_plot(
    result_data: Dict[str, Any],
    show_signal: bool = True,
) -> PlotData:
    """Convert detection result to annotated plot."""
    signal = result_data.get("signal", {})
    detections = result_data.get("detections", [])

    series = []

    if show_signal:
        series.append(PlotSeries(
            name="Signal",
            x=signal.get("wavelength", []),
            y=signal.get("intensity", []),
            type="line",
            color="#4AA0E0",
            line_width=1.5,
        ))

    # Create annotations for detections
    annotations = []
    for det in detections:
        species = det.get("species", "?")
        score = det.get("score", 0)
        meta = det.get("meta", {})

        # If we have a primary wavelength, annotate it
        primary_wl = meta.get("primary_wavelength")
        if primary_wl and signal.get("wavelength") and signal.get("intensity"):
            # Find intensity at this wavelength (approximate)
            wls = signal["wavelength"]
            ints = signal["intensity"]
            if wls and ints:
                # Find closest wavelength
                closest_idx = min(range(len(wls)), key=lambda i: abs(wls[i] - primary_wl))
                y_pos = ints[closest_idx] if closest_idx < len(ints) else 0

                annotations.append(PlotAnnotation(
                    x=primary_wl,
                    y=y_pos,
                    text=f"{species} ({score:.2f})",
                    color="#FF6B6B" if score > 0.5 else "#FFE66D",
                ))

    return PlotData(
        title="Detection Results",
        series=series,
        annotations=annotations,
        show_legend=True,
    )


def _comparison_plot(
    signals: List[Dict[str, Any]],
    labels: Optional[List[str]] = None,
    title: str = "Signal Comparison",
) -> PlotData:
    """Create comparison plot of multiple signals."""
    colors = ["#4AA0E0", "#FF6B6B", "#4ECDC4", "#FFE66D", "#95E1D3", "#F38181"]

    series = []
    for i, signal in enumerate(signals):
        label = labels[i] if labels and i < len(labels) else f"Signal {i+1}"
        color = colors[i % len(colors)]

        series.append(PlotSeries(
            name=label,
            x=signal.get("wavelength", []),
            y=signal.get("intensity", []),
            type="line",
            color=color,
            line_width=1.5,
        ))

    return PlotData(
        title=title,
        series=series,
        show_legend=True,
    )


@router.post("/signal", response_model=VisualizationResponse)
async def visualize_signal(
    signal: Signal,
    title: str = "",
    normalize: bool = False,
):
    """Generate visualization data for a signal.

    Returns plot data suitable for frontend rendering.
    """
    plot = _signal_to_plot(
        signal.model_dump(),
        title=title,
        normalize=normalize,
    )

    summary = f"Signal with {len(signal.wavelength)} points"
    if signal.wavelength:
        summary += f", range {min(signal.wavelength):.1f}-{max(signal.wavelength):.1f} nm"

    return VisualizationResponse(
        plot=plot,
        summary=summary,
    )


@router.post("/detection", response_model=VisualizationResponse)
async def visualize_detection(result: DetectionResult):
    """Generate visualization for detection results.

    Shows signal with detected species annotated.
    """
    plot = _detection_to_plot(result.model_dump())

    detection_count = len(result.detections)
    top_species = ", ".join(
        f"{d.species} ({d.score:.2f})"
        for d in sorted(result.detections, key=lambda x: x.score, reverse=True)[:3]
    )

    summary = f"{detection_count} species detected"
    if top_species:
        summary += f": {top_species}"

    return VisualizationResponse(
        plot=plot,
        summary=summary,
        meta={"detection_count": detection_count},
    )


@router.post("/compare", response_model=VisualizationResponse)
async def visualize_comparison(
    signals: List[Signal],
    labels: Optional[List[str]] = None,
    title: str = "Signal Comparison",
):
    """Generate comparison visualization of multiple signals.

    Overlays signals for visual comparison.
    """
    if not signals:
        raise HTTPException(status_code=400, detail="At least one signal required")

    plot = _comparison_plot(
        [s.model_dump() for s in signals],
        labels=labels,
        title=title,
    )

    return VisualizationResponse(
        plot=plot,
        summary=f"Comparison of {len(signals)} signals",
    )


@router.post("/from-data", response_model=VisualizationResponse)
async def visualize_from_data(request: VisualizationRequest):
    """Generate visualization from arbitrary data.

    Flexible endpoint for various visualization types.
    """
    viz_type = request.type
    data = request.data or {}
    options = request.options

    if viz_type == "signal":
        if "wavelength" not in data or "intensity" not in data:
            raise HTTPException(
                status_code=400,
                detail="Signal visualization requires 'wavelength' and 'intensity' in data"
            )
        plot = _signal_to_plot(
            data,
            title=options.get("title", ""),
            normalize=options.get("normalize", False),
        )
        return VisualizationResponse(plot=plot)

    elif viz_type == "detection":
        if "detections" not in data:
            raise HTTPException(
                status_code=400,
                detail="Detection visualization requires 'detections' in data"
            )
        plot = _detection_to_plot(data)
        return VisualizationResponse(plot=plot)

    elif viz_type == "comparison":
        signals = data.get("signals", [])
        if not signals:
            raise HTTPException(
                status_code=400,
                detail="Comparison visualization requires 'signals' array in data"
            )
        plot = _comparison_plot(
            signals,
            labels=data.get("labels"),
            title=options.get("title", "Comparison"),
        )
        return VisualizationResponse(plot=plot)

    elif viz_type == "preprocessing_stages":
        # Visualization showing multiple preprocessing stages
        stages = data.get("stages", [])
        if not stages:
            raise HTTPException(
                status_code=400,
                detail="Preprocessing visualization requires 'stages' array"
            )

        series = []
        colors = ["#888888", "#4AA0E0", "#4ECDC4", "#FF6B6B", "#FFE66D"]

        for i, stage in enumerate(stages):
            series.append(PlotSeries(
                name=stage.get("name", f"Stage {i+1}"),
                x=stage.get("wavelength", []),
                y=stage.get("intensity", []),
                type="line",
                color=colors[i % len(colors)],
                line_width=1.0 if i < len(stages) - 1 else 2.0,
                opacity=0.5 if i < len(stages) - 1 else 1.0,
            ))

        plot = PlotData(
            title=options.get("title", "Preprocessing Stages"),
            series=series,
            show_legend=True,
        )
        return VisualizationResponse(
            plot=plot,
            summary=f"{len(stages)} preprocessing stages",
        )

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown visualization type: {viz_type}"
        )


@router.get("/plot-options")
async def get_plot_options():
    """Get available plot customization options.

    Returns documentation of available options for visualizations.
    """
    return {
        "visualization_types": [
            {
                "type": "signal",
                "description": "Single signal plot",
                "options": ["title", "normalize", "color"],
            },
            {
                "type": "detection",
                "description": "Detection results with annotations",
                "options": ["show_signal", "highlight_threshold"],
            },
            {
                "type": "comparison",
                "description": "Multiple signals overlaid",
                "options": ["title", "labels"],
            },
            {
                "type": "preprocessing_stages",
                "description": "Show preprocessing pipeline stages",
                "options": ["title"],
            },
            {
                "type": "templates",
                "description": "Template matrix visualization",
                "options": ["species_filter", "normalize"],
            },
        ],
        "plot_options": {
            "x_label": "X-axis label (default: 'Wavelength (nm)')",
            "y_label": "Y-axis label (default: 'Intensity (a.u.)')",
            "x_range": "Fixed x-axis range [min, max]",
            "y_range": "Fixed y-axis range [min, max]",
            "show_legend": "Show legend (default: true)",
            "show_grid": "Show grid lines (default: true)",
        },
        "series_options": {
            "type": "Plot type: 'line', 'scatter', 'bar', 'area'",
            "color": "CSS color string",
            "opacity": "Opacity 0-1",
            "line_width": "Line width in pixels",
            "visible": "Initial visibility",
        },
    }
