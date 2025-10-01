"""Built-in templates and catalog helpers for the YREP GUI package."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable


RUN_WITH_LIBRARY_TEMPLATE: Dict[str, Any] = {
    "version": 1,
    "nodes": [
        {"id": 1, "identifier": "load_signal_batch", "config": {"directory": ""}, "position": [0, 0]},
        {"id": 2, "identifier": "group_signals", "config": {"grid_points": 1000}, "position": [320, 0]},
        {"id": 3, "identifier": "select_best_group", "config": {"quality_metric": "avg_quality", "min_quality": 0.0}, "position": [640, 0]},
        {"id": 4, "identifier": "average_signals", "config": {"n_points": 1200}, "position": [960, 0]},
        {"id": 5, "identifier": "trim", "config": {"min_nm": 300.0, "max_nm": 600.0}, "position": [1280, 0]},
        {"id": 6, "identifier": "resample", "config": {"n_points": 1500, "step_nm": 0.0}, "position": [1600, 0]},
        {"id": 7, "identifier": "load_signal_batch", "config": {"directory": ""}, "position": [960, -280]},
        {"id": 8, "identifier": "average_signals", "config": {"n_points": 1200}, "position": [1280, -280]},
        {"id": 9, "identifier": "subtract_background", "config": {"align": False}, "position": [1920, 0]},
        {"id": 10, "identifier": "continuum_remove_arpls", "config": {"strength": 0.5}, "position": [2240, 0]},
        {"id": 11, "identifier": "continuum_remove_rolling", "config": {"strength": 0.5}, "position": [2560, 0]},
        {"id": 12, "identifier": "load_references", "config": {"directory": "", "element_only": False}, "position": [2880, 320]},
        {"id": 13, "identifier": "build_templates", "config": {"fwhm_nm": 0.75, "species_filter": []}, "position": [2880, 0]},
        {"id": 14, "identifier": "shift_search", "config": {"spread_nm": 0.5, "iterations": 3}, "position": [3200, 0]},
        {"id": 15, "identifier": "detect_nnls", "config": {"presence_threshold": 0.02, "min_bands": 5}, "position": [3520, 0]},
        {"id": 16, "identifier": "plot_signal", "config": {"title": "Processed Signal", "normalize": False}, "position": [3840, 0]},
    ],
    "edges": [
        {"source": 1, "source_port": 0, "target": 2, "target_port": 0},
        {"source": 2, "source_port": 0, "target": 3, "target_port": 0},
        {"source": 3, "source_port": 0, "target": 4, "target_port": 0},
        {"source": 4, "source_port": 0, "target": 5, "target_port": 0},
        {"source": 5, "source_port": 0, "target": 6, "target_port": 0},
        {"source": 7, "source_port": 0, "target": 8, "target_port": 0},
        {"source": 6, "source_port": 0, "target": 9, "target_port": 0},
        {"source": 8, "source_port": 0, "target": 9, "target_port": 1},
        {"source": 9, "source_port": 0, "target": 10, "target_port": 0},
        {"source": 10, "source_port": 0, "target": 11, "target_port": 0},
        {"source": 11, "source_port": 0, "target": 13, "target_port": 0},
        {"source": 12, "source_port": 0, "target": 13, "target_port": 1},
        {"source": 11, "source_port": 0, "target": 14, "target_port": 0},
        {"source": 13, "source_port": 0, "target": 14, "target_port": 1},
        {"source": 14, "source_port": 0, "target": 15, "target_port": 0},
        {"source": 14, "source_port": 1, "target": 15, "target_port": 1},
        {"source": 15, "source_port": 0, "target": 16, "target_port": 0},
    ],
}


@dataclass(frozen=True)
class TemplateDetails:
    """Describes a template option surfaced to users."""

    title: str
    description: str
    payload: Dict[str, Any]


RUN_WITH_LIBRARY_DETAILS = TemplateDetails(
    title="Run With Library",
    description=(
        "Load a batch of signals, build and refine spectral templates, then run "
        "detection with default thresholds."
    ),
    payload=RUN_WITH_LIBRARY_TEMPLATE,
)

# The catalog keeps metadata alongside the raw payload so the UI can surface descriptions.
TEMPLATE_CATALOG: Dict[str, TemplateDetails] = {
    RUN_WITH_LIBRARY_DETAILS.title: RUN_WITH_LIBRARY_DETAILS,
}

# Backwards-compatible mapping of display name -> payload.
TEMPLATES: Dict[str, Dict[str, Any]] = {
    title: details.payload for title, details in TEMPLATE_CATALOG.items()
}


def iter_catalog() -> Iterable[TemplateDetails]:
    """Yield template definitions in a stable order for UI consumption."""

    return TEMPLATE_CATALOG.values()


def get_template_details(title: str) -> TemplateDetails | None:
    """Return metadata for a given template display title."""

    return TEMPLATE_CATALOG.get(title)


__all__ = [
    "TemplateDetails",
    "TEMPLATE_CATALOG",
    "TEMPLATES",
    "RUN_WITH_LIBRARY_DETAILS",
    "RUN_WITH_LIBRARY_TEMPLATE",
    "get_template_details",
    "iter_catalog",
]
