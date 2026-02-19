#!/usr/bin/env python3
"""Run NNLS vs decomposition analysis on StandardsTest datasets only."""

from __future__ import annotations

from pathlib import Path

from yrep_spectrum_analysis import (
    average_signals,
    fwhm_search,
    continuum_remove_arpls,
    continuum_remove_rolling,
    detect_nnls,
    resample,
    shift_search,
    subtract_background,
    trim,
    analyze_pca,
    analyze_ica,
    analyze_mcr,
    identify_components,
)
from yrep_spectrum_analysis.types import Signal
from yrep_spectrum_analysis.utils import (
    expand_species_filter,
    group_signals,
    is_junk_group,
    load_references,
    load_signals_from_dir,
)
from yrep_spectrum_analysis.visualizations import (
    visualize_detection,
    visualize_preprocessing,
    visualize_templates,
)

# Config
RESAMPLE_POINTS = 1500
TRIM_RANGE = (300.0, 600.0)
CONTINUUM_STRENGTH = 0.5
DETECT_PARAMS = {"presence_threshold": 0.02, "min_bands": 3}
INITIAL_FWHM = 0.75
PLOT_DIR = Path(__file__).parent / "plots" / "standards_test"
VISUALIZE = True


def preprocess_signal(sig: Signal, background: Signal | None = None) -> Signal:
    """Standard preprocessing pipeline."""
    s = trim(sig, min_nm=TRIM_RANGE[0], max_nm=TRIM_RANGE[1])
    s = resample(s, n_points=RESAMPLE_POINTS)
    if background is not None:
        s = subtract_background(s, background, align=False)
    s = continuum_remove_arpls(s, strength=CONTINUUM_STRENGTH)
    s = continuum_remove_rolling(s, strength=CONTINUUM_STRENGTH)
    return s


def run_nnls_analysis(
    signals: list[Signal],
    backgrounds: list[Signal],
    references,
    species_filter: list[str] | None,
    label: str,
    plot_path_prefix: Path | None = None,
) -> dict:
    """Run NNLS detection on averaged signal."""
    avg_signal = average_signals(signals, n_points=1200)
    avg_bg = average_signals(backgrounds, n_points=1200) if backgrounds else None

    processed = trim(avg_signal, min_nm=TRIM_RANGE[0], max_nm=TRIM_RANGE[1])
    processed = resample(processed, n_points=RESAMPLE_POINTS)
    if avg_bg is not None:
        processed = subtract_background(processed, avg_bg, align=False)
    subtracted = processed

    arpls_corrected = continuum_remove_arpls(processed, strength=CONTINUUM_STRENGTH)
    arpls_baseline_intensity = processed.intensity - arpls_corrected.intensity
    arpls_baseline = Signal(
        wavelength=processed.wavelength,
        intensity=arpls_baseline_intensity,
        meta={},
    )

    rolling_corrected = continuum_remove_rolling(
        arpls_corrected, strength=CONTINUUM_STRENGTH
    )
    rolling_baseline_intensity = arpls_corrected.intensity - rolling_corrected.intensity
    rolling_baseline = Signal(
        wavelength=processed.wavelength,
        intensity=rolling_baseline_intensity,
        meta={},
    )

    processed = rolling_corrected

    if VISUALIZE and plot_path_prefix:
        visualize_preprocessing(
            original=avg_signal,
            background=avg_bg,
            subtracted=subtracted,
            arpls_baseline=arpls_baseline,
            rolling_baseline=rolling_baseline,
            final=processed,
            title="Preprocessing Steps",
            save_path=str(plot_path_prefix) + "_preprocessing.png",
            show=False,
        )

    templates = fwhm_search(
        processed,
        references,
        initial_fwhm_nm=INITIAL_FWHM,
        spread_nm=0.2,
        iterations=3,
        species_filter=species_filter,
    )
    if VISUALIZE and plot_path_prefix:
        visualize_templates(
            signal=processed,
            templates=templates,
            title="Optimized Templates",
            save_path=str(plot_path_prefix) + "_templates.png",
            show=False,
        )

    processed = shift_search(processed, templates, spread_nm=0.5, iterations=3)
    result = detect_nnls(
        processed,
        templates,
        presence_threshold=DETECT_PARAMS["presence_threshold"],
        min_bands=int(DETECT_PARAMS["min_bands"]),
    )
    if VISUALIZE and plot_path_prefix:
        visualize_detection(
            result=result,
            templates=templates,
            title="Detection Results",
            save_path=str(plot_path_prefix) + "_detection.png",
            show=False,
        )

    return {
        "label": label,
        "method": "NNLS",
        "r2": result.meta.get("fit_R2", 0.0),
        "detections": [(d.species, d.score, d.meta.get("bands_hit", 0)) for d in result.detections],
        "signal": processed,
        "templates": templates,
        "result": result,
    }


def run_decomposition_analysis(
    signals: list[Signal],
    backgrounds: list[Signal],
    references,
    label: str,
) -> dict:
    """Run PCA/ICA/MCR decomposition on individual signals."""
    avg_bg = average_signals(backgrounds, n_points=1200) if backgrounds else None

    processed_list = []
    for sig in signals:
        p = preprocess_signal(sig, avg_bg)
        processed_list.append(p)

    if len(processed_list) < 5:
        return {"label": label, "method": "Decomposition", "error": "Not enough signals (<5)"}

    wavelength = processed_list[0].wavelength

    pca_scores, pca_comps = analyze_pca(processed_list, n_components=min(5, len(processed_list) - 1))
    pca_ids = identify_components(pca_comps, wavelength, references, top_n=3)

    try:
        ica_sources = analyze_ica(processed_list, n_components=min(5, len(processed_list) - 1))
        ica_comps = ica_sources.T
        ica_ids = identify_components(ica_comps, wavelength, references, top_n=3)
    except Exception:
        ica_comps, ica_ids = None, None

    try:
        mcr_conc, mcr_spectra = analyze_mcr(processed_list, n_components=min(4, len(processed_list) - 1))
        mcr_ids = identify_components(mcr_spectra, wavelength, references, top_n=3)
    except Exception:
        mcr_spectra, mcr_ids = None, None

    return {
        "label": label,
        "method": "Decomposition",
        "n_signals": len(processed_list),
        "wavelength": wavelength,
        "pca": {"components": pca_comps, "scores": pca_scores, "ids": pca_ids},
        "ica": {"components": ica_comps, "ids": ica_ids} if ica_comps is not None else None,
        "mcr": {"spectra": mcr_spectra, "ids": mcr_ids} if mcr_spectra is not None else None,
    }


def load_standards_datasets(base: Path) -> list[tuple[str, list[Signal], list[Signal]]]:
    """Load Standards datasets."""
    datasets = []
    std_root = base / "data" / "StandardsTest"

    for std_name in ["Copper", "StandardA", "StandardB", "StandardC", "StandardD"]:
        std_dir = std_root / std_name
        if not std_dir.exists():
            continue

        bg_dir = std_dir / "BG"
        backgrounds = load_signals_from_dir(bg_dir) if bg_dir.exists() else []

        meas_dir = std_dir / std_name
        if not meas_dir.exists():
            meas_dir = std_dir / "StdB" if std_name == "StandardB" else None

        if meas_dir and meas_dir.exists():
            signals = load_signals_from_dir(meas_dir)
            if signals:
                datasets.append((f"Standards/{std_name}", signals, backgrounds))

    return datasets


def print_comparison(nnls_result: dict, decomp_result: dict) -> None:
    label = nnls_result.get("label")
    print("\n" + "=" * 60)
    print(f"  {label}")
    print("=" * 60)

    r2 = nnls_result.get("r2", 0.0)
    print(f"\n  NNLS Detection (R^2 = {r2:.4f}):")
    if nnls_result.get("detections"):
        for sp, sc, bands in nnls_result["detections"]:
            print(f"    {sp:<12} score={sc:.4f}  bands={bands}")
    else:
        print("    No detections")

    if "error" in decomp_result:
        print(f"\n  Decomposition skipped: {decomp_result['error']}")
        return

    print(f"\n  PCA Top Components (n={decomp_result['n_signals']} signals):")
    for i, matches in enumerate(decomp_result["pca"]["ids"][:3]):
        match_str = ", ".join([f"{sp}({sc:.2f})" for sp, sc in matches[:2]])
        print(f"    PC{i+1}: {match_str}")

    if decomp_result["ica"]:
        print("\n  ICA Top Components:")
        for i, matches in enumerate(decomp_result["ica"]["ids"][:3]):
            match_str = ", ".join([f"{sp}({sc:.2f})" for sp, sc in matches[:2]])
            print(f"    IC{i+1}: {match_str}")

    if decomp_result["mcr"]:
        print("\n  MCR-ALS Components:")
        for i, matches in enumerate(decomp_result["mcr"]["ids"]):
            match_str = ", ".join([f"{sp}({sc:.2f})" for sp, sc in matches[:2]])
            print(f"    MCR{i+1}: {match_str}")


def main() -> None:
    base = Path(__file__).resolve().parent
    lists_dir = base / "data" / "lists"

    print("Loading references...")
    references = load_references(lists_dir, element_only=False)

    species_filter = expand_species_filter(references.lines.keys(), [
        "Na", "K", "Ca", "Li", "Cu", "Ba", "Sr", "Al", "Mg", "Si", "Zn",
        "Pb", "Cd", "Ag", "Au", "Cr", "Mn", "Co", "Ni", "Ti", "Sn", "Fe",
    ])

    print("Loading Standards datasets...")
    std_datasets = load_standards_datasets(base)
    print(f"  Found {len(std_datasets)} Standards datasets")

    if VISUALIZE:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for label, signals, backgrounds in std_datasets:
        print(f"\nAnalyzing {label} ({len(signals)} signals)...")

        groups = group_signals(signals)
        good_signals = []
        for g in groups:
            if not is_junk_group(g):
                good_signals.extend(g)

        if len(good_signals) < 3:
            print(f"  Skipping - not enough good signals ({len(good_signals)})")
            continue

        plot_prefix = None
        if VISUALIZE:
            safe_label = label.replace("/", "_").replace(" ", "_")
            plot_prefix = PLOT_DIR / safe_label
            plot_prefix.parent.mkdir(parents=True, exist_ok=True)
        nnls = run_nnls_analysis(
            good_signals,
            backgrounds,
            references,
            species_filter,
            label,
            plot_path_prefix=plot_prefix,
        )
        decomp = run_decomposition_analysis(good_signals, backgrounds, references, label)

        results.append((nnls, decomp))
        print_comparison(nnls, decomp)

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"\nAnalyzed {len(results)} datasets")

    nnls_sorted = sorted(results, key=lambda x: x[0]["r2"], reverse=True)
    print("\nTop NNLS R^2 scores:")
    for nnls, _ in nnls_sorted[:5]:
        top_det = nnls["detections"][0] if nnls["detections"] else ("None", 0, 0)
        print(f"  {nnls['label']:30s}  R^2={nnls['r2']:.4f}  Top: {top_det[0]}")


if __name__ == "__main__":
    main()
