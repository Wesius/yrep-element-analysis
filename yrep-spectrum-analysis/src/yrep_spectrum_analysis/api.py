from __future__ import annotations

from typing import Optional, Sequence

from .types import (
    AnalysisConfig,
    AnalysisResult,
    DetectionResult,
    PreprocessResult,
    SpectrumLike,
)
from ._preprocess import preprocess as _preprocess_impl
from ._templates import build_bands_index, build_templates, normalize_reference
from .utils import expand_species_filter
from ._detect import nnls_detect
from ._visualize import SpectrumVisualizer


def preprocess(
    measurements: Sequence[SpectrumLike],
    backgrounds: Optional[Sequence[SpectrumLike]],
    config: AnalysisConfig,
) -> PreprocessResult:
    return _preprocess_impl(measurements, backgrounds, config)


def detect(
    preprocessed: PreprocessResult, references, config: AnalysisConfig
) -> DetectionResult:
    ref = normalize_reference(references)
    # Expand species filter: if user passed base elements (e.g., "FE"), but
    # references include ionization (e.g., "FE I", "FE II"), expand to exact labels.
    species_filter = expand_species_filter(ref.species, config.species)

    S, names = build_templates(
        ref,
        preprocessed.wl_grid,
        fwhm_nm=config.fwhm_nm,
        species_filter=species_filter,
    )
    bands = build_bands_index(ref, names, fwhm_nm=config.fwhm_nm)
    coeffs, y_fit, present, per_species_scores, R2, _S_used = nnls_detect(
        preprocessed.wl_grid, preprocessed.y_cr, S, names, bands=bands, config=config, ref=ref
    )
    return DetectionResult(
        wl_grid=preprocessed.wl_grid,
        y_cr=preprocessed.y_cr,
        y_fit=y_fit,
        coeffs=coeffs,
        species_order=list(names),
        present=present if config.top_k <= 0 else present[: int(config.top_k)],
        per_species_scores=per_species_scores,
        fit_R2=R2,
    )


def analyze(
    measurements: Sequence[SpectrumLike],
    references,
    *,
    backgrounds: Optional[Sequence[SpectrumLike]] = None,
    config: Optional[AnalysisConfig] = None,
    visualize: bool = False,
    viz_output_dir: Optional[str] = None,
    viz_show: bool = False,
) -> AnalysisResult:
    cfg = config or AnalysisConfig()

    viz: Optional[SpectrumVisualizer] = None
    if visualize:
        viz = SpectrumVisualizer(
            output_dir=viz_output_dir or "./debug_plots",
            show_plots=viz_show,
            save_plots=True,
        )

    pre = preprocess(measurements, backgrounds, cfg)

    # Prepare references/templates once
    ref = normalize_reference(references)
    species_filter = expand_species_filter(ref.species, cfg.species)
    S, names = build_templates(ref, pre.wl_grid, fwhm_nm=cfg.fwhm_nm, species_filter=species_filter)
    bands = build_bands_index(ref, names, fwhm_nm=cfg.fwhm_nm)

    coeffs, y_fit, present, per_species_scores, R2, S_used = nnls_detect(
        pre.wl_grid, pre.y_cr, S, names, bands=bands, config=cfg, ref=ref
    )
    det = DetectionResult(
        wl_grid=pre.wl_grid,
        y_cr=pre.y_cr,
        y_fit=y_fit,
        coeffs=coeffs,
        species_order=list(names),
        present=present if cfg.top_k <= 0 else present[: int(cfg.top_k)],
        per_species_scores=per_species_scores,
        fit_R2=R2,
    )

    # Aggregate equals the single detection list here
    detections = list(det.present)

    result = AnalysisResult(
        detections=detections if cfg.top_k <= 0 else detections[: int(cfg.top_k)],
        detection=det,
        metrics={"fit_R2": float(R2)},
    )
    if visualize and viz is not None:
        viz.plot_full_analysis(
            measurements=measurements,
            backgrounds=backgrounds,
            pre=pre,
            ref=ref,
            templates=S_used,
            species_names=names,
            bands=bands,
            coeffs=coeffs,
            y_fit=y_fit,
            R2=R2,
            det=det,
            result=result,
            config=cfg,
        )
    return result
