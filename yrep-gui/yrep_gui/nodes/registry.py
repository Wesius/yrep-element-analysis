"""Registry describing available pipeline nodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(slots=True)
class NodeDefinition:
    identifier: str
    title: str
    category: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    default_config: Dict[str, object]
    multi_input_ports: tuple[int, ...] = ()
    optional_input_ports: tuple[int, ...] = ()


CATEGORY_ORDER: List[str] = [
    "I/O",
    "Aggregate",
    "Preprocess",
    "Templates",
    "Alignment",
    "Detection",
    "Visualization",
]

_REGISTRY: Dict[str, NodeDefinition] = {
    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------
    "load_signal": NodeDefinition(
        identifier="load_signal",
        title="Load Signal",
        category="I/O",
        inputs=(),
        outputs=("Signal",),
        default_config={"path": ""},
    ),
    "load_signal_batch": NodeDefinition(
        identifier="load_signal_batch",
        title="Load Signal Batch",
        category="I/O",
        inputs=(),
        outputs=("SignalBatch",),
        default_config={"directory": ""},
    ),
    "load_references": NodeDefinition(
        identifier="load_references",
        title="Load References",
        category="I/O",
        inputs=(),
        outputs=("References",),
        default_config={"directory": "", "element_only": True},
    ),
    # ------------------------------------------------------------------
    # Aggregate / preprocess
    # ------------------------------------------------------------------
    "average_signals": NodeDefinition(
        identifier="average_signals",
        title="Average Signals",
        category="Aggregate",
        inputs=("SignalBatch",),
        outputs=("Signal",),
        default_config={"n_points": 1000},
        multi_input_ports=(0,),
    ),
    "trim": NodeDefinition(
        identifier="trim",
        title="Trim",
        category="Preprocess",
        inputs=("Signal",),
        outputs=("Signal",),
        default_config={"min_nm": 300.0, "max_nm": 600.0},
    ),
    "mask": NodeDefinition(
        identifier="mask",
        title="Mask Interval",
        category="Preprocess",
        inputs=("Signal",),
        outputs=("Signal",),
        default_config={"intervals": []},
    ),
    "resample": NodeDefinition(
        identifier="resample",
        title="Resample",
        category="Preprocess",
        inputs=("Signal",),
        outputs=("Signal",),
        default_config={"n_points": 1500, "step_nm": 0.0},
    ),
    "subtract_background": NodeDefinition(
        identifier="subtract_background",
        title="Subtract Background",
        category="Preprocess",
        inputs=("Signal", "Signal"),
        outputs=("Signal",),
        default_config={"align": False},
        optional_input_ports=(1,),
    ),
    "continuum_remove_arpls": NodeDefinition(
        identifier="continuum_remove_arpls",
        title="Continuum Remove (arPLS)",
        category="Preprocess",
        inputs=("Signal",),
        outputs=("Signal",),
        default_config={"strength": 0.5},
    ),
    "continuum_remove_rolling": NodeDefinition(
        identifier="continuum_remove_rolling",
        title="Continuum Remove (Rolling)",
        category="Preprocess",
        inputs=("Signal",),
        outputs=("Signal",),
        default_config={"strength": 0.5},
    ),
    # ------------------------------------------------------------------
    # Templates + detection
    # ------------------------------------------------------------------
    "build_templates": NodeDefinition(
        identifier="build_templates",
        title="Build Templates",
        category="Templates",
        inputs=("Signal", "References"),
        outputs=("Template",),
        default_config={"fwhm_nm": 0.75, "species_filter": [], "bands_kwargs": {}},
    ),
    "plot_signal": NodeDefinition(
        identifier="plot_signal",
        title="Plot Signal",
        category="Visualization",
        inputs=("Signal",),
        outputs=("Signal",),
        default_config={"title": "", "normalize": False},
    ),
    "shift_search": NodeDefinition(
        identifier="shift_search",
        title="Shift Search",
        category="Alignment",
        inputs=("Signal", "Template"),
        outputs=("Signal",),
        default_config={"spread_nm": 0.5, "iterations": 3},
    ),
    "detect_nnls": NodeDefinition(
        identifier="detect_nnls",
        title="NNLS Detect",
        category="Detection",
        inputs=("Signal", "Template"),
        outputs=("DetectionResult",),
        default_config={"presence_threshold": 0.02, "min_bands": 5},
    ),
}


def definitions() -> Iterable[NodeDefinition]:
    return _REGISTRY.values()


def category_order() -> List[str]:
    return list(CATEGORY_ORDER)


def grouped_definitions() -> Dict[str, List[NodeDefinition]]:
    grouped: Dict[str, List[NodeDefinition]] = {category: [] for category in CATEGORY_ORDER}
    for definition in _REGISTRY.values():
        grouped.setdefault(definition.category, []).append(definition)
    for defs in grouped.values():
        defs.sort(key=lambda d: d.title.lower())
    return grouped


def get(identifier: str) -> NodeDefinition:
    return _REGISTRY[identifier]
