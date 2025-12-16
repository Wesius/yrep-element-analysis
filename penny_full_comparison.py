#!/usr/bin/env python3
"""Full comparison: NNLS vs PCA vs ICA vs MCR-ALS on pennies dataset."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from yrep_spectrum_analysis import (
    average_signals,
    fwhm_search,
    continuum_remove_arpls,
    continuum_remove_rolling,
    detect_nnls,
    resample,
    shift_search,
    trim,
    analyze_pca,
    analyze_ica,
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

# Config
TRIM_RANGE = (300.0, 600.0)
RESAMPLE_POINTS = 1500
CONTINUUM_STRENGTH = 0.0
FWHM = 0.75
SPECIES_FILTER = ["CU", "ZN", "C", "FE", "NI", "PB", "SN", "SI", "AL", "MG", "CA", "NA", "K", "CR", "MN"]

PLOT_DIR = Path(__file__).parent / "plots" / "pennies_full_comparison"


@dataclass
class MethodResult:
    name: str
    top_species: str
    top_score: float
    all_matches: List[Tuple[str, float]] = field(default_factory=list)
    r2: float = 0.0
    converged: bool = True
    error: str = ""


@dataclass 
class YearResult:
    year: str
    kind: str
    n_signals: int
    nnls: MethodResult
    pca: MethodResult
    ica: MethodResult
    mcr: MethodResult


def preprocess(sig: Signal) -> Signal:
    s = trim(sig, min_nm=TRIM_RANGE[0], max_nm=TRIM_RANGE[1])
    s = resample(s, n_points=RESAMPLE_POINTS)
    if CONTINUUM_STRENGTH > 0:
        s = continuum_remove_arpls(s, strength=CONTINUUM_STRENGTH)
        s = continuum_remove_rolling(s, strength=CONTINUUM_STRENGTH)
    return s


def run_nnls(signals: list[Signal], refs: References, sp_filter: list[str]) -> MethodResult:
    """Run NNLS template matching."""
    try:
        avg = average_signals(signals, n_points=1200)
        proc = preprocess(avg)
        
        templates = fwhm_search(proc, refs, initial_fwhm_nm=FWHM, spread_nm=0.2, iterations=3, species_filter=sp_filter)
        proc = shift_search(proc, templates, spread_nm=0.5, iterations=2)
        result = detect_nnls(proc, templates, presence_threshold=0.0, min_bands=5)
        
        r2 = result.meta.get("fit_R2", 0.0)
        detections = [(d.species, d.score) for d in result.detections[:5]]
        top = detections[0] if detections else ("None", 0.0)
        
        return MethodResult(name="NNLS", top_species=top[0], top_score=top[1], all_matches=detections, r2=r2)
    except Exception as e:
        return MethodResult(name="NNLS", top_species="Error", top_score=0.0, converged=False, error=str(e))


def run_pca(signals: list[Signal], refs: References, wavelength: np.ndarray) -> MethodResult:
    """Run PCA decomposition."""
    try:
        n_comp = min(5, len(signals) - 1)
        scores, components = analyze_pca(signals, n_components=n_comp)
        
        # Identify components
        ids = identify_components(components, wavelength, refs, top_n=3)
        
        # Get top match from first principal component
        top_match = ids[0][0] if ids and ids[0] else ("None", 0.0)
        all_matches = [(ids[i][0][0], ids[i][0][1]) for i in range(len(ids)) if ids[i]]
        
        # Calculate explained variance as pseudo-R²
        total_var = np.var(np.stack([s.intensity for s in signals]))
        reconstructed = scores @ components
        residual_var = np.var(np.stack([s.intensity for s in signals]) - reconstructed)
        r2 = 1 - residual_var / total_var if total_var > 0 else 0.0
        
        return MethodResult(name="PCA", top_species=top_match[0], top_score=top_match[1], 
                          all_matches=all_matches, r2=r2)
    except Exception as e:
        return MethodResult(name="PCA", top_species="Error", top_score=0.0, converged=False, error=str(e))


def run_ica(signals: list[Signal], refs: References, wavelength: np.ndarray) -> MethodResult:
    """Run ICA decomposition."""
    try:
        n_comp = min(5, len(signals) - 1)
        sources = analyze_ica(signals, n_components=n_comp)
        
        # sources shape: (M_wavelengths, n_components) - transpose for identify_components
        components = sources.T
        
        ids = identify_components(components, wavelength, refs, top_n=3)
        
        top_match = ids[0][0] if ids and ids[0] else ("None", 0.0)
        all_matches = [(ids[i][0][0], ids[i][0][1]) for i in range(len(ids)) if ids[i]]
        
        return MethodResult(name="ICA", top_species=top_match[0], top_score=top_match[1],
                          all_matches=all_matches, r2=0.0)  # ICA doesn't have natural R²
    except Exception as e:
        return MethodResult(name="ICA", top_species="Error", top_score=0.0, converged=False, error=str(e))


def run_mcr(signals: list[Signal], refs: References, wavelength: np.ndarray) -> MethodResult:
    """Run MCR-ALS decomposition."""
    try:
        n_comp = min(4, len(signals) - 2)
        if n_comp < 2:
            return MethodResult(name="MCR", top_species="N/A", top_score=0.0, 
                              converged=False, error="Not enough signals")
        
        X = np.stack([s.intensity for s in signals])
        conc, spectra = analyze_mcr(signals, n_components=n_comp)
        
        # Check for NaN in results
        if np.any(np.isnan(spectra)) or np.any(np.isnan(conc)):
            return MethodResult(name="MCR", top_species="NaN", top_score=0.0,
                              converged=False, error="NaN in results")
        
        ids = identify_components(spectra, wavelength, refs, top_n=3)
        
        # Calculate reconstruction R²
        X_fit = conc @ spectra
        ss_tot = np.sum((X - X.mean()) ** 2)
        ss_res = np.sum((X - X_fit) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        if np.isnan(r2) or r2 < 0:
            r2 = 0.0
            
        top_match = ids[0][0] if ids and ids[0] else ("None", 0.0)
        all_matches = [(ids[i][0][0], ids[i][0][1]) for i in range(len(ids)) if ids[i]]
        
        return MethodResult(name="MCR", top_species=top_match[0], top_score=top_match[1],
                          all_matches=all_matches, r2=r2, converged=(r2 > 0.5))
    except Exception as e:
        return MethodResult(name="MCR", top_species="Error", top_score=0.0, converged=False, error=str(e))


def discover_pennies(base: Path) -> List[Tuple[str, str, Path]]:
    """Return list of (year, kind, path) tuples."""
    pennies_root = base / "data" / "pennies"
    results = []
    
    for year_dir in sorted(pennies_root.iterdir()):
        if not year_dir.is_dir() or year_dir.name.startswith('.'):
            continue
        year = year_dir.name
        
        for meas_dir in sorted(year_dir.iterdir()):
            if not meas_dir.is_dir() or meas_dir.name.upper() == "BG":
                continue
            
            kind = "circ" if "circ" in meas_dir.name.lower() and "uncirc" not in meas_dir.name.lower() else "uncirc"
            results.append((year, kind, meas_dir))
    
    return results


def plot_method_comparison(results: List[YearResult], out_dir: Path):
    """Plot R² comparison across all methods."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for kind in ["circ", "uncirc"]:
        kind_results = sorted([r for r in results if r.kind == kind], key=lambda x: int(x.year))
        if not kind_results:
            continue
        
        years = [int(r.year) for r in kind_results]
        nnls_r2 = [r.nnls.r2 for r in kind_results]
        pca_r2 = [r.pca.r2 for r in kind_results]
        mcr_r2 = [r.mcr.r2 if r.mcr.converged else np.nan for r in kind_results]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(years, nnls_r2, 'o-', label='NNLS', color='tab:blue', linewidth=2, markersize=5)
        ax.plot(years, pca_r2, 's-', label='PCA', color='tab:green', linewidth=2, markersize=5)
        ax.plot(years, mcr_r2, '^-', label='MCR-ALS', color='tab:orange', linewidth=2, markersize=5)
        
        ax.axvline(x=1982, color='red', linestyle='--', alpha=0.5, label='1982 composition change')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title(f'Method Comparison: R² Over Time ({kind.title()})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"r2_all_methods_{kind}.png", dpi=150)
        plt.close()


def plot_detection_agreement(results: List[YearResult], out_dir: Path):
    """Heatmap showing which methods detect which species."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    methods = ["NNLS", "PCA", "ICA", "MCR"]
    species_of_interest = ["CU I", "ZN I", "CU II", "ZN II", "FE I"]
    
    # Count detections per method per species
    counts = {m: {s: 0 for s in species_of_interest} for m in methods}
    totals = {m: 0 for m in methods}
    
    for r in results:
        for method, result in [("NNLS", r.nnls), ("PCA", r.pca), ("ICA", r.ica), ("MCR", r.mcr)]:
            if result.converged and result.top_species in species_of_interest:
                counts[method][result.top_species] += 1
            totals[method] += 1
    
    # Build matrix
    matrix = np.zeros((len(methods), len(species_of_interest)))
    for i, m in enumerate(methods):
        for j, s in enumerate(species_of_interest):
            matrix[i, j] = counts[m][s] / totals[m] * 100 if totals[m] > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(matrix, xticklabels=species_of_interest, yticklabels=methods,
                annot=True, fmt='.1f', cmap='Blues', ax=ax, vmin=0, vmax=100)
    ax.set_title('Top Detection Frequency by Method (%)')
    ax.set_xlabel('Species')
    ax.set_ylabel('Method')
    
    plt.tight_layout()
    plt.savefig(out_dir / "detection_by_method.png", dpi=150)
    plt.close()


def plot_convergence_rate(results: List[YearResult], out_dir: Path):
    """Show MCR convergence rate over years."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    years = sorted(set(r.year for r in results))
    
    mcr_conv = []
    for year in years:
        year_results = [r for r in results if r.year == year]
        conv_rate = sum(1 for r in year_results if r.mcr.converged) / len(year_results) * 100
        mcr_conv.append(conv_rate)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar([int(y) for y in years], mcr_conv, color='tab:orange', alpha=0.7)
    ax.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='50% threshold')
    ax.set_xlabel('Year')
    ax.set_ylabel('MCR-ALS Convergence Rate (%)')
    ax.set_title('MCR-ALS Convergence Rate by Year')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / "mcr_convergence_rate.png", dpi=150)
    plt.close()


def main():
    base = Path(__file__).resolve().parent
    refs = load_references(base / "data" / "lists", element_only=False)
    sp_filter = expand_species_filter(refs.lines.keys(), SPECIES_FILTER)
    
    print("=" * 100)
    print("  PENNIES: FULL METHOD COMPARISON (NNLS vs PCA vs ICA vs MCR-ALS)")
    print("=" * 100)
    
    pennies = discover_pennies(base)
    print(f"\nFound {len(pennies)} penny measurement sets\n")
    
    results: List[YearResult] = []
    
    for year, kind, meas_dir in pennies:
        signals = load_signals_from_dir(meas_dir)
        if not signals:
            continue
        
        # Filter to best group
        groups = group_signals(signals)
        best_group = None
        best_size = 0
        for g in groups:
            if not is_junk_group(g) and len(g) >= 4 and len(g) > best_size:
                best_group = g
                best_size = len(g)
        
        if best_group is None or len(best_group) < 6:
            print(f"{year}/{kind}: Skipped (need 6+ signals, have {best_size})")
            continue
        
        # Preprocess all signals for decomposition methods
        processed = [preprocess(s) for s in best_group]
        wavelength = processed[0].wavelength
        
        print(f"{year}/{kind} ({len(best_group)} signals)...", end=" ", flush=True)
        
        # Run all methods
        nnls_result = run_nnls(best_group, refs, sp_filter)
        pca_result = run_pca(processed, refs, wavelength)
        ica_result = run_ica(processed, refs, wavelength)
        mcr_result = run_mcr(processed, refs, wavelength)
        
        print(f"NNLS:{nnls_result.top_species}({nnls_result.r2:.2f}) "
              f"PCA:{pca_result.top_species}({pca_result.r2:.2f}) "
              f"ICA:{ica_result.top_species} "
              f"MCR:{mcr_result.top_species}({'OK' if mcr_result.converged else 'FAIL'})")
        
        results.append(YearResult(
            year=year, kind=kind, n_signals=len(best_group),
            nnls=nnls_result, pca=pca_result, ica=ica_result, mcr=mcr_result
        ))
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_method_comparison(results, PLOT_DIR)
    plot_detection_agreement(results, PLOT_DIR)
    plot_convergence_rate(results, PLOT_DIR)
    
    # Summary table
    print("\n" + "=" * 100)
    print("  SUMMARY TABLE")
    print("=" * 100)
    
    print(f"\n{'Year':<6} {'Kind':<7} | {'NNLS':^12} | {'PCA':^12} | {'ICA':^12} | {'MCR':^12}")
    print("-" * 80)
    
    for r in sorted(results, key=lambda x: (int(x.year), x.kind)):
        nnls_str = f"{r.nnls.top_species[:6]:<6}({r.nnls.r2:.2f})"
        pca_str = f"{r.pca.top_species[:6]:<6}({r.pca.r2:.2f})"
        ica_str = f"{r.ica.top_species[:6]:<6}"
        mcr_str = f"{r.mcr.top_species[:6]:<6}" + ("(OK)" if r.mcr.converged else "(X)")
        
        print(f"{r.year:<6} {r.kind:<7} | {nnls_str:^12} | {pca_str:^12} | {ica_str:^12} | {mcr_str:^12}")
    
    # Aggregate statistics
    print("\n" + "=" * 100)
    print("  AGGREGATE STATISTICS")
    print("=" * 100)
    
    nnls_r2 = [r.nnls.r2 for r in results]
    pca_r2 = [r.pca.r2 for r in results]
    mcr_r2 = [r.mcr.r2 for r in results if r.mcr.converged]
    mcr_conv_rate = sum(1 for r in results if r.mcr.converged) / len(results) * 100
    
    print(f"\nNNLS:    R² mean={np.mean(nnls_r2):.4f}, std={np.std(nnls_r2):.4f}")
    print(f"PCA:     R² mean={np.mean(pca_r2):.4f}, std={np.std(pca_r2):.4f}")
    print(f"MCR-ALS: R² mean={np.mean(mcr_r2):.4f}, std={np.std(mcr_r2):.4f} (when converged)")
    print(f"         Convergence rate: {mcr_conv_rate:.1f}%")
    
    # Detection agreement
    print("\n  TOP DETECTION COUNTS:")
    for method_name, get_result in [("NNLS", lambda r: r.nnls), ("PCA", lambda r: r.pca), 
                                     ("ICA", lambda r: r.ica), ("MCR", lambda r: r.mcr)]:
        species_counts: Dict[str, int] = {}
        for r in results:
            res = get_result(r)
            if res.converged:
                sp = res.top_species
                species_counts[sp] = species_counts.get(sp, 0) + 1
        
        top3 = sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_str = ", ".join([f"{sp}:{cnt}" for sp, cnt in top3])
        print(f"  {method_name:6s}: {top_str}")
    
    print(f"\nPlots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
