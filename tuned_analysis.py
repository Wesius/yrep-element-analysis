#!/usr/bin/env python3
"""Tuned comparison of NNLS vs MCR-ALS on Nov 5 and Standards data."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from yrep_spectrum_analysis import (
    average_signals,
    build_templates,
    fwhm_search,
    continuum_remove_arpls,
    continuum_remove_rolling,
    detect_nnls,
    resample,
    shift_search,
    subtract_background,
    trim,
    analyze_pca,
    analyze_mcr,
    identify_components,
)
from yrep_spectrum_analysis.types import Signal, References
from yrep_spectrum_analysis.utils import (
    expand_species_filter,
    group_signals,
    is_junk_group,
    load_references,
    load_signals_from_dir,
)

from sklearn.decomposition import PCA
from pymcr.mcr import McrAR
from pymcr.constraints import ConstraintNonneg, ConstraintNorm


@dataclass
class NNLSConfig:
    trim_range: Tuple[float, float]
    resample_points: int
    continuum_strength: float
    initial_fwhm: float
    fwhm_spread: float
    fwhm_iterations: int
    shift_spread: float
    shift_iterations: int
    presence_threshold: float
    min_bands: int


@dataclass
class MCRConfig:
    n_components: int
    max_iter: int
    tol_increase: float
    use_simplisma_init: bool


# Tuned NNLS configurations to try
NNLS_CONFIGS = {
    "aggressive": NNLSConfig(
        trim_range=(300.0, 700.0),
        resample_points=2000,
        continuum_strength=0.0,  # No continuum removal - keeps more signal
        initial_fwhm=0.6,
        fwhm_spread=0.3,
        fwhm_iterations=5,
        shift_spread=0.8,
        shift_iterations=4,
        presence_threshold=0.0,
        min_bands=3,
    ),
    "balanced": NNLSConfig(
        trim_range=(300.0, 650.0),
        resample_points=1800,
        continuum_strength=0.2,
        initial_fwhm=0.75,
        fwhm_spread=0.25,
        fwhm_iterations=4,
        shift_spread=0.6,
        shift_iterations=3,
        presence_threshold=0.01,
        min_bands=4,
    ),
    "conservative": NNLSConfig(
        trim_range=(300.0, 600.0),
        resample_points=1500,
        continuum_strength=0.5,
        initial_fwhm=0.75,
        fwhm_spread=0.2,
        fwhm_iterations=3,
        shift_spread=0.5,
        shift_iterations=3,
        presence_threshold=0.02,
        min_bands=5,
    ),
}

# Tuned MCR configurations
MCR_CONFIGS = {
    "robust": MCRConfig(n_components=5, max_iter=200, tol_increase=20.0, use_simplisma_init=False),
    "tight": MCRConfig(n_components=4, max_iter=150, tol_increase=5.0, use_simplisma_init=False),
}


def preprocess_signal(sig: Signal, background: Signal | None, cfg: NNLSConfig) -> Signal:
    s = trim(sig, min_nm=cfg.trim_range[0], max_nm=cfg.trim_range[1])
    s = resample(s, n_points=cfg.resample_points)
    if background is not None:
        s = subtract_background(s, background, align=False)
    if cfg.continuum_strength > 0:
        s = continuum_remove_arpls(s, strength=cfg.continuum_strength)
        s = continuum_remove_rolling(s, strength=cfg.continuum_strength)
    return s


def run_nnls(
    signals: list[Signal],
    backgrounds: list[Signal],
    references: References,
    species_filter: list[str] | None,
    cfg: NNLSConfig,
) -> dict:
    avg_signal = average_signals(signals, n_points=1200)
    avg_bg = average_signals(backgrounds, n_points=1200) if backgrounds else None
    processed = preprocess_signal(avg_signal, avg_bg, cfg)
    
    templates = fwhm_search(
        processed, references,
        initial_fwhm_nm=cfg.initial_fwhm,
        spread_nm=cfg.fwhm_spread,
        iterations=cfg.fwhm_iterations,
        species_filter=species_filter,
    )
    processed = shift_search(
        processed, templates,
        spread_nm=cfg.shift_spread,
        iterations=cfg.shift_iterations,
    )
    result = detect_nnls(
        processed, templates,
        presence_threshold=cfg.presence_threshold,
        min_bands=cfg.min_bands,
    )
    
    best_fwhm = templates.meta.get("fwhm_search", {}).get("best_fwhm_nm", cfg.initial_fwhm)
    
    return {
        "r2": result.meta.get("fit_R2", 0.0),
        "detections": [(d.species, d.score, d.meta.get("bands_hit", 0)) for d in result.detections],
        "best_fwhm": best_fwhm,
        "signal": processed,
        "templates": templates,
    }


def run_mcr_tuned(
    signals: list[Signal],
    backgrounds: list[Signal],
    references: References,
    cfg_nnls: NNLSConfig,
    cfg_mcr: MCRConfig,
) -> dict:
    """Run MCR-ALS with tuned parameters."""
    avg_bg = average_signals(backgrounds, n_points=1200) if backgrounds else None
    
    processed_list = []
    for sig in signals:
        p = preprocess_signal(sig, avg_bg, cfg_nnls)
        processed_list.append(p)
    
    if len(processed_list) < cfg_mcr.n_components + 1:
        return {"error": f"Not enough signals ({len(processed_list)}) for {cfg_mcr.n_components} components"}
    
    wavelength = processed_list[0].wavelength
    X = np.stack([s.intensity for s in processed_list])
    
    # Better initialization using PCA
    pca = PCA(n_components=cfg_mcr.n_components)
    pca.fit(X)
    initial_spectra = np.abs(pca.components_)
    # Normalize rows
    for i in range(initial_spectra.shape[0]):
        norm = np.linalg.norm(initial_spectra[i])
        if norm > 0:
            initial_spectra[i] /= norm
    
    try:
        mcr = McrAR(
            c_regr="nnls",
            st_regr="nnls",
            c_constraints=[ConstraintNonneg()],
            st_constraints=[ConstraintNonneg(), ConstraintNorm(axis=1)],
            tol_increase=cfg_mcr.tol_increase,
            max_iter=cfg_mcr.max_iter,
        )
        mcr.fit(X, ST=initial_spectra)
        
        spectra = mcr.ST_opt_
        concentrations = mcr.C_opt_
        
        # Calculate reconstruction error (R²-like metric)
        X_reconstructed = concentrations @ spectra
        ss_tot = np.sum((X - X.mean()) ** 2)
        ss_res = np.sum((X - X_reconstructed) ** 2)
        r2_mcr = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        # Identify components
        ids = identify_components(spectra, wavelength, references, top_n=3)
        
        return {
            "r2": r2_mcr,
            "n_signals": len(processed_list),
            "spectra": spectra,
            "concentrations": concentrations,
            "ids": ids,
            "wavelength": wavelength,
        }
    except Exception as e:
        return {"error": str(e)}


def load_nov5_datasets(base: Path) -> list[tuple[str, list[Signal], list[Signal]]]:
    datasets = []
    nov5_root = base / "data" / "nov5"
    bg_dir = nov5_root / "Background"
    backgrounds = load_signals_from_dir(bg_dir) if bg_dir.exists() else []
    
    # Coke Can
    coke_dir = nov5_root / "Coke_Can"
    for part in ["Coke_Can_Tab", "Coke_Can_Side", "Coke_Can_Bottom"]:
        part_dir = coke_dir / part
        if part_dir.exists():
            signals = load_signals_from_dir(part_dir)
            if signals:
                datasets.append((f"Coke_{part.split('_')[-1]}", signals, backgrounds))
    
    # Foil - combine all
    foil_dir = nov5_root / "Foil"
    all_foil = []
    for i in range(1, 4):
        part_dir = foil_dir / f"Foil_{i}"
        if part_dir.exists():
            all_foil.extend(load_signals_from_dir(part_dir))
    if all_foil:
        datasets.append(("Foil_Combined", all_foil, backgrounds))
    
    # Penny - combine all
    penny_dir = nov5_root / "Penny"
    all_penny = []
    if penny_dir.exists():
        for sub in penny_dir.iterdir():
            if sub.is_dir():
                all_penny.extend(load_signals_from_dir(sub))
    if all_penny:
        datasets.append(("Penny_Combined", all_penny, backgrounds))
    
    return datasets


def load_standards_datasets(base: Path) -> list[tuple[str, list[Signal], list[Signal]]]:
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
                datasets.append((std_name, signals, backgrounds))
    
    return datasets


def main() -> None:
    base = Path(__file__).resolve().parent
    lists_dir = base / "data" / "lists"
    
    print("=" * 80)
    print("  TUNED NNLS vs MCR-ALS COMPARISON")
    print("=" * 80)
    
    print("\nLoading references...")
    references = load_references(lists_dir, element_only=False)
    
    # Broader species filter including Al
    species_filter = expand_species_filter(references.lines.keys(), [
        "Na", "K", "Ca", "Li", "Cu", "Ba", "Sr", "Al", "Mg", "Si", "Zn", 
        "Pb", "Cd", "Ag", "Au", "Cr", "Mn", "Co", "Ni", "Ti", "Sn", "Fe",
        "C", "N", "O", "H", "Ar",
    ])
    
    print("Loading datasets...")
    nov5_datasets = load_nov5_datasets(base)
    std_datasets = load_standards_datasets(base)
    
    all_datasets = std_datasets + nov5_datasets
    print(f"Found {len(std_datasets)} Standards + {len(nov5_datasets)} Nov5 datasets\n")
    
    # Results storage
    results = []
    
    for label, signals, backgrounds in all_datasets:
        # Filter junk
        groups = group_signals(signals)
        good_signals = []
        for g in groups:
            if not is_junk_group(g):
                good_signals.extend(g)
        
        if len(good_signals) < 5:
            print(f"{label}: Skipped (only {len(good_signals)} good signals)")
            continue
        
        print(f"\n{'='*60}")
        print(f"  {label} ({len(good_signals)} signals)")
        print(f"{'='*60}")
        
        # Try all NNLS configs
        best_nnls = {"r2": 0.0, "config": None, "result": None}
        print("\n  NNLS Results:")
        for cfg_name, cfg in NNLS_CONFIGS.items():
            try:
                result = run_nnls(good_signals, backgrounds, references, species_filter, cfg)
                r2 = result["r2"]
                top_det = result["detections"][0] if result["detections"] else ("None", 0, 0)
                print(f"    {cfg_name:12s}: R²={r2:.4f}  Top: {top_det[0]:8s} (score={top_det[1]:.3f}, bands={top_det[2]})")
                
                if r2 > best_nnls["r2"]:
                    best_nnls = {"r2": r2, "config": cfg_name, "result": result}
            except Exception as e:
                print(f"    {cfg_name:12s}: ERROR - {e}")
        
        # Try MCR configs with the best NNLS preprocessing
        best_mcr = {"r2": 0.0, "config": None, "result": None}
        print("\n  MCR-ALS Results:")
        
        best_nnls_cfg = NNLS_CONFIGS[best_nnls["config"]] if best_nnls["config"] else NNLS_CONFIGS["balanced"]
        
        for cfg_name, cfg in MCR_CONFIGS.items():
            try:
                result = run_mcr_tuned(good_signals, backgrounds, references, best_nnls_cfg, cfg)
                if "error" in result:
                    print(f"    {cfg_name:12s}: {result['error']}")
                    continue
                
                r2 = result["r2"]
                top_matches = []
                for i, ids in enumerate(result["ids"][:3]):
                    if ids:
                        top_matches.append(f"C{i+1}:{ids[0][0]}({ids[0][1]:.2f})")
                match_str = ", ".join(top_matches)
                print(f"    {cfg_name:12s}: R²={r2:.4f}  {match_str}")
                
                if r2 > best_mcr["r2"]:
                    best_mcr = {"r2": r2, "config": cfg_name, "result": result}
            except Exception as e:
                print(f"    {cfg_name:12s}: ERROR - {e}")
        
        # Store results
        results.append({
            "label": label,
            "n_signals": len(good_signals),
            "best_nnls": best_nnls,
            "best_mcr": best_mcr,
        })
    
    # Final Report
    print("\n")
    print("=" * 80)
    print("  FINAL REPORT")
    print("=" * 80)
    
    print("\n" + "-" * 80)
    print(f"{'Dataset':<20} | {'NNLS R²':>8} | {'NNLS Top Det':>15} | {'MCR R²':>8} | {'MCR Top Match':>15}")
    print("-" * 80)
    
    for r in results:
        nnls_r2 = r["best_nnls"]["r2"]
        nnls_det = "None"
        if r["best_nnls"]["result"] and r["best_nnls"]["result"]["detections"]:
            nnls_det = r["best_nnls"]["result"]["detections"][0][0]
        
        mcr_r2 = r["best_mcr"]["r2"]
        mcr_match = "None"
        if r["best_mcr"]["result"] and "ids" in r["best_mcr"]["result"]:
            ids = r["best_mcr"]["result"]["ids"]
            if ids and ids[0]:
                mcr_match = ids[0][0][0]
        
        print(f"{r['label']:<20} | {nnls_r2:>8.4f} | {nnls_det:>15} | {mcr_r2:>8.4f} | {mcr_match:>15}")
    
    print("-" * 80)
    
    # Summary stats
    nnls_r2s = [r["best_nnls"]["r2"] for r in results]
    mcr_r2s = [r["best_mcr"]["r2"] for r in results if r["best_mcr"]["r2"] > 0]
    
    print(f"\nNNLS:    Mean R² = {np.mean(nnls_r2s):.4f}, Min = {np.min(nnls_r2s):.4f}, Max = {np.max(nnls_r2s):.4f}")
    if mcr_r2s:
        print(f"MCR-ALS: Mean R² = {np.mean(mcr_r2s):.4f}, Min = {np.min(mcr_r2s):.4f}, Max = {np.max(mcr_r2s):.4f}")
    
    # Detailed detection report
    print("\n" + "=" * 80)
    print("  DETAILED DETECTIONS (Best NNLS Config)")
    print("=" * 80)
    
    for r in results:
        print(f"\n{r['label']} (config: {r['best_nnls']['config']}, R²={r['best_nnls']['r2']:.4f}):")
        if r["best_nnls"]["result"] and r["best_nnls"]["result"]["detections"]:
            for species, score, bands in r["best_nnls"]["result"]["detections"][:6]:
                print(f"    {species:12s}  score={score:.4f}  bands={bands}")
        else:
            print("    No detections")
    
    # MCR component identification
    print("\n" + "=" * 80)
    print("  MCR-ALS COMPONENT IDENTIFICATION")
    print("=" * 80)
    
    for r in results:
        if r["best_mcr"]["result"] and "ids" in r["best_mcr"]["result"]:
            print(f"\n{r['label']} (config: {r['best_mcr']['config']}, R²={r['best_mcr']['r2']:.4f}):")
            for i, ids in enumerate(r["best_mcr"]["result"]["ids"]):
                if ids:
                    match_str = ", ".join([f"{sp}({sc:.2f})" for sp, sc in ids[:3]])
                    print(f"    Component {i+1}: {match_str}")


if __name__ == "__main__":
    main()
