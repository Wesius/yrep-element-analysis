"""Adapters that execute yrep-spectrum-analysis pipelines from graph descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable

from yrep_spectrum_analysis import (
    average_signals,
    build_templates,
    continuum_remove_arpls,
    continuum_remove_rolling,
    detect_nnls,
    mask,
    resample,
    shift_search,
    subtract_background,
    trim,
)


PipelineCallable = Callable[..., Any]



def _build_templates(
    signal: Any,
    references: Any,
    *,
    fwhm_nm: float,
    species_filter: Iterable[str] | None = None,
    bands_kwargs: Dict[str, Any] | None = None,
    **kwargs: Any,
):
    """Adapter that fills optional kwargs for the updated template builder."""

    return build_templates(
        signal,
        references,
        fwhm_nm=fwhm_nm,
        species_filter=species_filter,
        bands_kwargs=bands_kwargs or {},
        **kwargs,
    )


@dataclass(slots=True)
class PipelineNode:
    """Lightweight representation of a pipeline node definition."""

    identifier: str
    kind: str
    config: Dict[str, Any]


class PipelineRunner:
    """Converts node graphs into yrep-spectrum-analysis execution plans."""

    def __init__(self) -> None:
        self._available_steps: Dict[str, PipelineCallable] = {
            "average_signals": average_signals,
            "trim": trim,
            "mask": mask,
            "resample": resample,
            "subtract_background": subtract_background,
            "continuum_remove_arpls": continuum_remove_arpls,
            "continuum_remove_rolling": continuum_remove_rolling,
            "build_templates": _build_templates,
            "shift_search": shift_search,
            "detect_nnls": detect_nnls,
        }

    def list_steps(self) -> Iterable[str]:
        return tuple(self._available_steps)

    def get_callable(self, identifier: str) -> PipelineCallable:
        return self._available_steps[identifier]

    def run(self, graph: Iterable[PipelineNode]) -> None:
        raise NotImplementedError("Execution pipeline not yet implemented")

