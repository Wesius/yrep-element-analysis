#!/usr/bin/env python3
"""Compare NNLS detection vs decomposition methods on Nov 5 and Standards data."""

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

# Config
RESAMPLE_POINTS = 1500
TRIM_RANGE = (300.0, 600.0)
CONTINUUM_STRENGTH = 0.5
DETECT_PARAMS = {"presence_threshold": 0.02, "min_bands": 3}
INITIAL_FWHM = 0.75


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
) -> dict:
    """Run NNLS detection on averaged signal."""
    avg_signal = average_signals(signals, n_points=1200)
    avg_bg = average_signals(backgrounds, n_points=1200) if backgrounds else None
    processed = preprocess_signal(avg_signal, avg_bg)
    
    templates = fwhm_search(
        processed, references,
        initial_fwhm_nm=INITIAL_FWHM,
        spread_nm=0.2, iterations=3,
        species_filter=species_filter,
    )
    processed = shift_search(processed, templates, spread_nm=0.5, iterations=3)
    result = detect_nnls(
        processed, templates,
        presence_threshold=DETECT_PARAMS["presence_threshold"],
        min_bands=int(DETECT_PARAMS["min_bands"]),
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
    
    # PCA
    pca_scores, pca_comps = analyze_pca(processed_list, n_components=min(5, len(processed_list) - 1))
    pca_ids = identify_components(pca_comps, wavelength, references, top_n=3)
    
    # ICA
    try:
        ica_sources = analyze_ica(processed_list, n_components=min(5, len(processed_list) - 1))
        ica_comps = ica_sources.T
        ica_ids = identify_components(ica_comps, wavelength, references, top_n=3)
    except Exception:
        ica_comps, ica_ids = None, None
    
    # MCR-ALS
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


def load_nov5_datasets(base: Path) -> list[tuple[str, list[Signal], list[Signal]]]:
    """Load all Nov 5 datasets."""
    datasets = []
    nov5_root = base / "data" / "nov5"
    bg_dir = nov5_root / "Background"
    backgrounds = load_signals_from_dir(bg_dir) if bg_dir.exists() else []
    
    # Coke Can parts
    coke_dir = nov5_root / "Coke_Can"
    for part in ["Coke_Can_Tab", "Coke_Can_Side", "Coke_Can_Bottom"]:
        part_dir = coke_dir / part
        if part_dir.exists():
            signals = load_signals_from_dir(part_dir)
            if signals:
                datasets.append((f"Nov5/Coke_{part.split('_')[-1]}", signals, backgrounds))
    
    # Foil parts
    foil_dir = nov5_root / "Foil"
    for i in range(1, 4):
        part_dir = foil_dir / f"Foil_{i}"
        if part_dir.exists():
            signals = load_signals_from_dir(part_dir)
            if signals:
                datasets.append((f"Nov5/Foil_{i}", signals, backgrounds))
    
    # Penny
    penny_dir = nov5_root / "Penny"
    if penny_dir.exists():
        for sub in penny_dir.iterdir():
            if sub.is_dir():
                signals = load_signals_from_dir(sub)
                if signals:
                    datasets.append((f"Nov5/Penny/{sub.name}", signals, backgrounds))
    
    return datasets


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
        
        # Find measurement directory
        meas_dir = std_dir / std_name
        if not meas_dir.exists():
            meas_dir = std_dir / "StdB" if std_name == "StandardB" else None
        
        if meas_dir and meas_dir.exists():
            signals = load_signals_from_dir(meas_dir)
            if signals:
                datasets.append((f"Standards/{std_name}", signals, backgrounds))
    
    return datasets


def print_comparison(nnls_result: dict, decomp_result: dict) -> None:
    """Print side-by-side comparison."""
    label = nnls_result["label"]
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    # NNLS Results
    print(f"\n  NNLS Detection (R² = {nnls_result['r2']:.4f}):")
    if nnls_result["detections"]:
        for species, score, bands in nnls_result["detections"][:8]:
            print(f"    {species:12s}  score={score:.4f}  bands={bands}")
    else:
        print("    No detections")
    
    # Decomposition Results
    if "error" in decomp_result:
        print(f"\n  Decomposition: {decomp_result['error']}")
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
    
    # Load datasets
    print("Loading Nov 5 datasets...")
    nov5_datasets = load_nov5_datasets(base)
    print(f"  Found {len(nov5_datasets)} Nov 5 datasets")
    
    print("Loading Standards datasets...")
    std_datasets = load_standards_datasets(base)
    print(f"  Found {len(std_datasets)} Standards datasets")
    
    all_datasets = nov5_datasets + std_datasets
    
    results = []
    for label, signals, backgrounds in all_datasets:
        print(f"\nAnalyzing {label} ({len(signals)} signals)...")
        
        # Filter out junk
        groups = group_signals(signals)
        good_signals = []
        for g in groups:
            if not is_junk_group(g):
                good_signals.extend(g)
        
        if len(good_signals) < 3:
            print(f"  Skipping - not enough good signals ({len(good_signals)})")
            continue
        
        # Run both analyses
        nnls = run_nnls_analysis(good_signals, backgrounds, references, species_filter, label)
        decomp = run_decomposition_analysis(good_signals, backgrounds, references, label)
        
        results.append((nnls, decomp))
        print_comparison(nnls, decomp)
    
    # Summary
    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"\nAnalyzed {len(results)} datasets")
    
    # Find best NNLS fits
    nnls_sorted = sorted(results, key=lambda x: x[0]["r2"], reverse=True)
    print("\nTop NNLS R² scores:")
    for nnls, _ in nnls_sorted[:5]:
        top_det = nnls["detections"][0] if nnls["detections"] else ("None", 0, 0)
        print(f"  {nnls['label']:30s}  R²={nnls['r2']:.4f}  Top: {top_det[0]}")


if __name__ == "__main__":
    main()
