#!/usr/bin/env python3
"""Compare NNLS vs MCR-ALS on pennies dataset across years."""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from yrep_spectrum_analysis import (
    average_signals,
    fwhm_search,
    continuum_remove_arpls,
    continuum_remove_rolling,
    detect_nnls,
    resample,
    shift_search,
    trim,
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

warnings.filterwarnings('ignore')

# Config matching penny_analysis.py
TRIM_RANGE = (300.0, 600.0)
RESAMPLE_POINTS = 1500
CONTINUUM_STRENGTH = 0.0
FWHM = 0.75
SPECIES_FILTER = ["CU", "ZN", "C", "FE", "NI", "PB", "SN", "SI", "AL", "MG", "CA", "NA", "K", "CR", "MN"]

PLOT_DIR = Path(__file__).parent / "plots" / "pennies_comparison"


@dataclass
class ComparisonResult:
    year: str
    kind: str  # circ or uncirc
    n_signals: int
    nnls_r2: float
    nnls_detections: List[Tuple[str, float]]
    mcr_r2: float
    mcr_components: List[List[Tuple[str, float]]]


def preprocess(sig: Signal) -> Signal:
    s = trim(sig, min_nm=TRIM_RANGE[0], max_nm=TRIM_RANGE[1])
    s = resample(s, n_points=RESAMPLE_POINTS)
    if CONTINUUM_STRENGTH > 0:
        s = continuum_remove_arpls(s, strength=CONTINUUM_STRENGTH)
        s = continuum_remove_rolling(s, strength=CONTINUUM_STRENGTH)
    return s


def run_nnls(signals: list[Signal], refs: References, sp_filter: list[str] | None) -> Tuple[float, List[Tuple[str, float]]]:
    """Run NNLS on averaged signal."""
    avg = average_signals(signals, n_points=1200)
    proc = preprocess(avg)
    
    templates = fwhm_search(proc, refs, initial_fwhm_nm=FWHM, spread_nm=0.2, iterations=3, species_filter=sp_filter)
    proc = shift_search(proc, templates, spread_nm=0.5, iterations=2)
    result = detect_nnls(proc, templates, presence_threshold=0.0, min_bands=5)
    
    r2 = result.meta.get("fit_R2", 0.0)
    detections = [(d.species, d.score) for d in result.detections[:5]]
    return r2, detections


def run_mcr(signals: list[Signal], refs: References, n_comp: int = 4) -> Tuple[float, List[List[Tuple[str, float]]]]:
    """Run MCR-ALS on individual signals."""
    processed = [preprocess(s) for s in signals]
    
    if len(processed) < n_comp + 2:
        return 0.0, []
    
    wavelength = processed[0].wavelength
    X = np.stack([s.intensity for s in processed])
    
    # PCA init
    pca = PCA(n_components=n_comp)
    pca.fit(X)
    init_st = np.abs(pca.components_)
    for i in range(init_st.shape[0]):
        n = np.linalg.norm(init_st[i])
        if n > 0:
            init_st[i] /= n
    
    try:
        mcr = McrAR(
            c_regr="nnls",
            st_regr="nnls",
            c_constraints=[ConstraintNonneg()],
            st_constraints=[ConstraintNonneg(), ConstraintNorm(axis=1)],
            tol_increase=50.0,
            max_iter=200,
        )
        mcr.fit(X, ST=init_st)
        
        spectra = mcr.ST_opt_
        conc = mcr.C_opt_
        
        if spectra is None or conc is None:
            return 0.0, []
        
        X_fit = conc @ spectra
        ss_tot = np.sum((X - X.mean()) ** 2)
        ss_res = np.sum((X - X_fit) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        if np.isnan(r2) or r2 < 0:
            r2 = 0.0
        
        ids = identify_components(spectra, wavelength, refs, top_n=3)
        return r2, ids
    except Exception:
        return 0.0, []


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


def plot_comparison_over_time(results: List[ComparisonResult], out_dir: Path):
    """Plot R² comparison over years."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Separate by kind
    for kind in ["circ", "uncirc"]:
        kind_results = [r for r in results if r.kind == kind]
        if not kind_results:
            continue
        
        years = [int(r.year) for r in kind_results]
        nnls_r2 = [r.nnls_r2 for r in kind_results]
        mcr_r2 = [r.mcr_r2 for r in kind_results]
        
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(years, nnls_r2, 'o-', label='NNLS R²', color='tab:blue', linewidth=2, markersize=6)
        ax.plot(years, mcr_r2, 's-', label='MCR-ALS R²', color='tab:orange', linewidth=2, markersize=6)
        
        ax.axvline(x=1982, color='red', linestyle='--', alpha=0.5, label='1982 (composition change)')
        ax.set_xlabel('Year', fontsize=12)
        ax.set_ylabel('R²', fontsize=12)
        ax.set_title(f'NNLS vs MCR-ALS R² Over Time ({kind.title()})', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(out_dir / f"r2_comparison_{kind}.png", dpi=150)
        plt.close()


def plot_detection_comparison(results: List[ComparisonResult], out_dir: Path):
    """Compare detected species between methods."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Aggregate top detections
    nnls_species_counts: Dict[str, int] = {}
    mcr_species_counts: Dict[str, int] = {}
    
    for r in results:
        # NNLS top detection
        if r.nnls_detections:
            sp = r.nnls_detections[0][0]
            nnls_species_counts[sp] = nnls_species_counts.get(sp, 0) + 1
        
        # MCR top component match
        if r.mcr_components and r.mcr_components[0]:
            sp = r.mcr_components[0][0][0]
            mcr_species_counts[sp] = mcr_species_counts.get(sp, 0) + 1
    
    # Plot comparison bar chart
    all_species = sorted(set(nnls_species_counts.keys()) | set(mcr_species_counts.keys()))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(all_species))
    width = 0.35
    
    nnls_vals = [nnls_species_counts.get(sp, 0) for sp in all_species]
    mcr_vals = [mcr_species_counts.get(sp, 0) for sp in all_species]
    
    ax.bar(x - width/2, nnls_vals, width, label='NNLS', color='tab:blue')
    ax.bar(x + width/2, mcr_vals, width, label='MCR-ALS', color='tab:orange')
    
    ax.set_xlabel('Species')
    ax.set_ylabel('Detection Count')
    ax.set_title('Top Detection Frequency: NNLS vs MCR-ALS')
    ax.set_xticks(x)
    ax.set_xticklabels(all_species, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(out_dir / "detection_comparison.png", dpi=150)
    plt.close()


def plot_component_heatmap(results: List[ComparisonResult], out_dir: Path):
    """Heatmap of MCR component identifications over years."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all species found
    all_species = set()
    for r in results:
        for comp in r.mcr_components:
            for sp, _ in comp:
                all_species.add(sp)
    
    # Focus on top species
    species_list = ["CU I", "ZN I", "CU II", "ZN II", "FE I", "NI I", "PB I", "CA I", "K I"]
    species_list = [s for s in species_list if s in all_species]
    
    for kind in ["circ", "uncirc"]:
        kind_results = sorted([r for r in results if r.kind == kind], key=lambda x: int(x.year))
        if not kind_results:
            continue
        
        years = [r.year for r in kind_results]
        
        # Build matrix: max score for each species in any component
        matrix = np.zeros((len(species_list), len(years)))
        for j, r in enumerate(kind_results):
            for comp in r.mcr_components:
                for sp, score in comp:
                    if sp in species_list:
                        i = species_list.index(sp)
                        matrix[i, j] = max(matrix[i, j], score)
        
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(matrix, xticklabels=years, yticklabels=species_list, 
                    cmap='YlOrRd', ax=ax, vmin=0, vmax=1)
        ax.set_title(f'MCR-ALS Component Scores by Year ({kind.title()})')
        ax.set_xlabel('Year')
        ax.set_ylabel('Species')
        
        plt.tight_layout()
        plt.savefig(out_dir / f"mcr_heatmap_{kind}.png", dpi=150)
        plt.close()


def main():
    base = Path(__file__).resolve().parent
    refs = load_references(base / "data" / "lists", element_only=False)
    sp_filter = expand_species_filter(refs.lines.keys(), SPECIES_FILTER)
    
    print("=" * 80)
    print("  PENNIES: NNLS vs MCR-ALS COMPARISON")
    print("=" * 80)
    
    pennies = discover_pennies(base)
    print(f"\nFound {len(pennies)} penny measurement sets\n")
    
    results: List[ComparisonResult] = []
    
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
        
        if best_group is None or len(best_group) < 5:
            print(f"{year}/{kind}: Skipped (no valid group)")
            continue
        
        print(f"{year}/{kind} ({len(best_group)} signals)...", end=" ", flush=True)
        
        # Run NNLS
        nnls_r2, nnls_det = run_nnls(best_group, refs, sp_filter)
        
        # Run MCR
        mcr_r2, mcr_comps = run_mcr(best_group, refs)
        
        nnls_top = nnls_det[0][0] if nnls_det else "None"
        mcr_top = mcr_comps[0][0][0] if mcr_comps and mcr_comps[0] else "None"
        
        print(f"NNLS R²={nnls_r2:.3f} ({nnls_top}), MCR R²={mcr_r2:.3f} ({mcr_top})")
        
        results.append(ComparisonResult(
            year=year,
            kind=kind,
            n_signals=len(best_group),
            nnls_r2=nnls_r2,
            nnls_detections=nnls_det,
            mcr_r2=mcr_r2,
            mcr_components=mcr_comps,
        ))
    
    # Generate plots
    print("\nGenerating comparison plots...")
    plot_comparison_over_time(results, PLOT_DIR)
    plot_detection_comparison(results, PLOT_DIR)
    plot_component_heatmap(results, PLOT_DIR)
    
    # Summary report
    print("\n" + "=" * 80)
    print("  SUMMARY REPORT")
    print("=" * 80)
    
    print(f"\n{'Year':<6} {'Kind':<8} {'N':>4} | {'NNLS R²':>8} {'NNLS Top':>10} | {'MCR R²':>8} {'MCR Top':>10}")
    print("-" * 75)
    
    for r in sorted(results, key=lambda x: (int(x.year), x.kind)):
        nnls_top = r.nnls_detections[0][0] if r.nnls_detections else "None"
        mcr_top = r.mcr_components[0][0][0] if r.mcr_components and r.mcr_components[0] else "None"
        print(f"{r.year:<6} {r.kind:<8} {r.n_signals:>4} | {r.nnls_r2:>8.4f} {nnls_top:>10} | {r.mcr_r2:>8.4f} {mcr_top:>10}")
    
    print("-" * 75)
    
    # Aggregate stats
    nnls_r2s = [r.nnls_r2 for r in results]
    mcr_r2s = [r.mcr_r2 for r in results if r.mcr_r2 > 0]
    
    print(f"\nNNLS:    Mean R²={np.mean(nnls_r2s):.4f}, Std={np.std(nnls_r2s):.4f}")
    if mcr_r2s:
        print(f"MCR-ALS: Mean R²={np.mean(mcr_r2s):.4f}, Std={np.std(mcr_r2s):.4f}")
    
    # Agreement check
    agree = 0
    total = 0
    for r in results:
        if r.nnls_detections and r.mcr_components and r.mcr_components[0]:
            total += 1
            nnls_top = r.nnls_detections[0][0]
            mcr_top = r.mcr_components[0][0][0]
            if nnls_top == mcr_top:
                agree += 1
    
    print(f"\nTop detection agreement: {agree}/{total} ({100*agree/total if total else 0:.1f}%)")
    
    print(f"\nPlots saved to: {PLOT_DIR}")


if __name__ == "__main__":
    main()
