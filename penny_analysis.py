#!/usr/bin/env python3
"""
Batch analyzer for pennies datasets under data/pennies.

Behavior:
- Discovers all measurement sets in data/pennies (both Circ_* and Uncirc_* across years)
- Disables grouping (analyzes each found measurement set as a single run)
- Uses the same config as run_with_library.py
- Enables visualizations only for the first 3 runs (saved to plots/pennies/...)
- After all runs finish, creates an aggregate summary visualization
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LogNorm

from yrep_spectrum_analysis import AnalysisConfig, analyze
from yrep_spectrum_analysis.types import Spectrum
from yrep_spectrum_analysis.utils import load_references, load_txt_spectrum, group_spectra, is_junk_group


# -------------------------
# Configuration (mirrors run_with_library.py)
# -------------------------
def build_config() -> AnalysisConfig:
    return AnalysisConfig(
        fwhm_nm=0.75,
        grid_step_nm=None,
        species=["Cu", "Zn", "C", "Fe", "Ni", "Pb", "Sn", "Si", "Al", "Mg", "Ca", "Na", "K", "Cr", "Mn"],
        baseline_strength=0,
        regularization=0.0,
        min_bands_required=5,
        presence_threshold=0,
        top_k=0,
        min_wavelength_nm=300,
        max_wavelength_nm=600,
        auto_trim_left=False,
        auto_trim_right=False,
        align_background=False,
        background_fn=None,
        continuum_fn=None,
        continuum_strategy = "arpls",
        search_shift=True,
        shift_search_iterations=2,
        shift_search_spread=0.5,  # absolute nm window
        search_fwhm=True,
        fwhm_search_iterations=2,
        fwhm_search_spread=0.5,
    )


# -------------------------
# Discovery
# -------------------------
@dataclass
class MeasurementSet:
    label: str
    year: str
    meas_root: Path
    bg_root: Path


def discover_pennies_sets(pennies_root: Path) -> List[MeasurementSet]:
    """Find all measurement sets under data/pennies.

    Strategy:
    - Treat any first-level subdirectory under a year directory (e.g., Circ_1980, Uncirc_1980)
      that contains .txt files as one measurement set.
    - Backgrounds: if a sibling BG directory exists at the year level, use it; otherwise
      pass a non-existent path which results in no backgrounds loaded.
    """
    discovered: List[MeasurementSet] = []
    if not pennies_root.exists():
        return discovered

    for year_dir in sorted([p for p in pennies_root.iterdir() if p.is_dir()]):
        year = year_dir.name
        bg_candidate = year_dir / "BG"
        for meas_dir in sorted([p for p in year_dir.iterdir() if p.is_dir()]):
            # Consider only non-BG directories which contain .txt files
            if meas_dir.name.upper() == "BG":
                continue
            txts = list(meas_dir.glob("*.txt"))
            if not txts:
                # Skip empty or non-matching directories to avoid accidental runs
                continue
            label = f"{year}/{meas_dir.name}"
            discovered.append(
                MeasurementSet(
                    label=label,
                    year=year,
                    meas_root=meas_dir,
                    bg_root=bg_candidate,  # may not exist; loader will handle gracefully
                )
            )

    return discovered


# -------------------------
# Aggregate reporting
# -------------------------
@dataclass
class RunResult:
    label: str
    r2: float
    detections: List[Dict[str, float]]


def save_aggregate_summary(results: Sequence[RunResult], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prep data
    labels = [r.label for r in results]
    r2_values = np.asarray([r.r2 for r in results], dtype=float)

    # Species frequency and score distribution
    species_counts: Dict[str, int] = {}
    detection_scores: List[float] = []
    for r in results:
        for d in r.detections:
            species = str(d.get("species", "?")).strip()
            score = float(d.get("score", d.get("fve", 0.0)))
            species_counts[species] = species_counts.get(species, 0) + 1
            detection_scores.append(score)

    # Top species by frequency
    top_items = sorted(species_counts.items(), key=lambda t: t[1], reverse=True)[:10]
    top_species = [t[0] for t in top_items]
    top_counts = np.asarray([t[1] for t in top_items], dtype=float)

    # Plot
    sns.set_palette("husl")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Panel A: R^2 per run
    ax = axes[0, 0]
    ax.barh(range(len(labels)), r2_values, color="tab:blue", alpha=0.8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("R²")
    ax.set_title("Fit Quality per Run (R²)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="x", alpha=0.3)

    # Panel B: Top detected species (by frequency)
    ax = axes[0, 1]
    if len(top_species) > 0:
        ax.barh(range(len(top_species)), top_counts, color="tab:green", alpha=0.8)
        ax.set_yticks(range(len(top_species)))
        ax.set_yticklabels(top_species)
        ax.set_xlabel("Detections (count)")
        ax.set_title("Top Species Across All Runs")
        ax.invert_yaxis()
        ax.grid(True, axis="x", alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No detections", ha="center", va="center", alpha=0.6)
        ax.axis("off")

    # Panel C: Detection score distribution
    ax = axes[1, 0]
    if detection_scores:
        sns.histplot(detection_scores, bins=30, kde=True, ax=ax, color="tab:purple")
        ax.set_xlabel("Score (FVE)")
        ax.set_title("Detection Score Distribution")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No detection scores", ha="center", va="center", alpha=0.6)
        ax.axis("off")

    # Panel D: Summary stats
    ax = axes[1, 1]
    num_runs = len(results)
    num_detections = int(sum(len(r.detections) for r in results))
    text = (
        f"Runs: {num_runs}\n"
        f"Total detections: {num_detections}\n"
        f"R² mean: {float(np.mean(r2_values)) if num_runs else 0.0:.3f}\n"
        f"R² std: {float(np.std(r2_values)) if num_runs else 0.0:.3f}\n"
        f"R² max: {float(np.max(r2_values)) if num_runs else 0.0:.3f}"
    )
    ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8))
    ax.axis("off")

    fig.suptitle("Pennies Analysis – Aggregate Summary", fontsize=16, fontweight="bold")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_dir / "00_pennies_summary.png", dpi=300)
    plt.close(fig)


def save_year_species_heatmap(
    per_kind_year_species_scores: Dict[str, Dict[str, Dict[str, List[float]]]],
    out_dir: Path,
) -> None:
    """Year × Species FVE heatmap for circ vs uncirc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Get union of all years and species
    all_years = sorted(set(
        y for kind_data in per_kind_year_species_scores.values()
        for y in kind_data.keys()
    ), key=lambda y: int(y))
    
    # Get top 20 species by total FVE across both conditions
    all_species_sums: Dict[str, float] = {}
    for kind_data in per_kind_year_species_scores.values():
        for year_data in kind_data.values():
            for sp, vals in year_data.items():
                all_species_sums[sp] = all_species_sums.get(sp, 0.0) + sum(vals)
    
    top_species = [sp for sp, _ in sorted(all_species_sums.items(), 
                                          key=lambda t: t[1], reverse=True)[:20]]
    
    if not all_years or not top_species:
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(48, 20))
    
    for ax, kind in [(ax1, "circ"), (ax2, "uncirc")]:
        # Build matrix
        matrix = np.zeros((len(top_species), len(all_years)))
        for i, sp in enumerate(top_species):
            for j, year in enumerate(all_years):
                vals = per_kind_year_species_scores.get(kind, {}).get(year, {}).get(sp, [])
                if vals:
                    matrix[i, j] = float(np.mean(vals))
                else:
                    matrix[i, j] = np.nan
        
        # Plot heatmap with log scale
        # Add small epsilon to avoid log(0)
        matrix_log = matrix.copy()
        matrix_log[matrix_log == 0] = 1e-10
        
        # Use LogNorm for better visibility of trace amounts
        vmin = 1e-6  # Minimum visible FVE
        vmax = np.nanmax(matrix) if np.any(np.isfinite(matrix)) else 1
        
        sns.heatmap(
            matrix,
            xticklabels=all_years,
            yticklabels=top_species,
            cmap="YlOrRd",
            cbar_kws={"label": "Mean FVE (log scale)"},
            ax=ax,
            norm=LogNorm(vmin=vmin, vmax=vmax),
            linewidths=0.5,
            linecolor='gray',
            annot=True,
            fmt=".4f",
            annot_kws={"fontsize": 8, "rotation": 90},
            mask=np.isnan(matrix),
        )
        ax.set_title(f"Year × Species FVE Heatmap ({kind.upper()})", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Species")
        
        # Rotate x labels
        ax.tick_params(axis='x', rotation=45)
    
    fig.suptitle("Species FVE Across Years: Circ vs Uncirc", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "02_year_species_heatmap.png", dpi=300)
    plt.close(fig)


def save_stacked_area_composition(
    per_kind_year_species_scores: Dict[str, Dict[str, Dict[str, List[float]]]],
    out_dir: Path,
) -> None:
    """Stacked area chart showing composition over time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    for ax, kind in [(ax1, "circ"), (ax2, "uncirc")]:
        kind_data = per_kind_year_species_scores.get(kind, {})
        if not kind_data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", alpha=0.6)
            ax.set_title(f"Composition Over Time ({kind.upper()})", fontweight="bold")
            continue
        
        # Get years in order
        years = sorted(kind_data.keys(), key=lambda y: int(y))
        x_years = np.array([int(y) for y in years])
        
        # Calculate total FVE per species
        species_totals: Dict[str, float] = {}
        for year_data in kind_data.values():
            for sp, vals in year_data.items():
                species_totals[sp] = species_totals.get(sp, 0.0) + sum(vals)
        
        # Get top 10 species
        top_species = [sp for sp, _ in sorted(species_totals.items(), 
                                             key=lambda t: t[1], reverse=True)[:10]]
        
        # Build matrix for stacked area
        matrix = []
        for sp in top_species:
            series = []
            for year in years:
                vals = kind_data.get(year, {}).get(sp, [])
                if vals:
                    series.append(float(np.sum(vals)))
                else:
                    series.append(0.0)
            matrix.append(series)
        
        # Normalize to percentages
        matrix = np.array(matrix)
        totals = np.sum(matrix, axis=0)
        totals[totals == 0] = 1.0  # Avoid division by zero
        matrix_pct = matrix / totals * 100
        
        # Plot stacked area with varied colors
        # Use tab20 colormap for more color variety
        colors = plt.cm.tab20(np.linspace(0, 1, len(top_species)))
        
        ax.stackplot(x_years, matrix_pct, labels=top_species, colors=colors, alpha=0.9)
        ax.set_title(f"Composition Over Time ({kind.upper()})", fontweight="bold")
        ax.set_xlabel("Year")
        ax.set_ylabel("Composition (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True, alpha=0.3)
    
    fig.suptitle("Species Composition Evolution: Circ vs Uncirc", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "03_stacked_area_composition.png", dpi=300)
    plt.close(fig)


def save_species_cooccurrence(
    all_results: List[RunResult],
    out_dir: Path,
    threshold: float = 0.0,
) -> None:
    """Species co-occurrence heatmap."""
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Split results by condition
    circ_results = [r for r in all_results if "circ_" in r.label.lower() and "uncirc" not in r.label.lower()]
    uncirc_results = [r for r in all_results if "uncirc" in r.label.lower()]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    for ax, results, kind in [(ax1, circ_results, "circ"), (ax2, uncirc_results, "uncirc")]:
        # Get all detected species (preserve full species labels)
        all_species = set()
        for r in results:
            for d in r.detections:
                if float(d.get("score", d.get("fve", 0.0))) >= threshold:
                    sp = str(d.get("species", "?")).strip()
                    all_species.add(sp)
        
        species_list = sorted(list(all_species))
        if not species_list:
            ax.text(0.5, 0.5, "No detections", ha="center", va="center", alpha=0.6)
            ax.set_title(f"Co-occurrence ({kind.upper()})", fontweight="bold")
            continue
        
        # Build co-occurrence matrix
        n = len(species_list)
        matrix = np.zeros((n, n))
        
        for r in results:
            detected = set()
            for d in r.detections:
                if float(d.get("score", d.get("fve", 0.0))) >= threshold:
                    sp = str(d.get("species", "?")).strip()
                    if sp in all_species:
                        detected.add(sp)
            
            # Update co-occurrence counts
            for i, sp1 in enumerate(species_list):
                for j, sp2 in enumerate(species_list):
                    if sp1 in detected and sp2 in detected:
                        matrix[i, j] += 1
        
        # Normalize by total runs
        if len(results) > 0:
            matrix = matrix / len(results)
        
        # Plot heatmap
        sns.heatmap(
            matrix,
            xticklabels=species_list,
            yticklabels=species_list,
            cmap="Blues",
            vmin=0,
            vmax=1,
            square=True,
            cbar_kws={"label": "Co-occurrence Frequency"},
            ax=ax,
            linewidths=0.5,
            linecolor='gray',
        )
        ax.set_title(f"Species Co-occurrence ({kind.upper()})", fontweight="bold")
        
        # Rotate labels
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', rotation=0)
    
    fig.suptitle("Species Co-occurrence Patterns: Circ vs Uncirc", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "04_species_cooccurrence.png", dpi=300)
    plt.close(fig)


def save_composition_over_time(
    per_year_species_scores: Dict[str, Dict[str, List[float]]],
    out_dir: Path,
    *,
    species_filter: Optional[List[str]] = None,
    title_suffix: str = "",
    filename_suffix: str = "",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Sort years numerically
    years = sorted(per_year_species_scores.keys(), key=lambda y: int(y))
    x_years = np.asarray([int(y) for y in years], dtype=int)

    # Collect complete species set
    species_all: List[str] = sorted(
        set(
            sp
            for y in years
            for sp in per_year_species_scores.get(y, {}).keys()
        )
    )
    if species_filter:
        species_all = [sp for sp in species_all if sp in set(species_filter)]
    if not species_all:
        return

    # Build matrix (year × species) of mean FVE
    fve_by_species: Dict[str, List[float]] = {}
    for sp in species_all:
        series: List[float] = []
        for y in years:
            vals = per_year_species_scores.get(y, {}).get(sp, [])
            if vals:
                series.append(float(np.mean(vals)))
            else:
                series.append(np.nan)
        fve_by_species[sp] = series

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    cmap = plt.get_cmap("tab20")
    num_colors = cmap.N
    for i, sp in enumerate(species_all):
        color = cmap(i % num_colors)
        y_vals = np.asarray(fve_by_species[sp], dtype=float)
        ax.plot(x_years, y_vals, label=sp, color=color, linewidth=2, alpha=0.85)

    title = "Composition Over Time (FVE by Species)"
    if title_suffix:
        title += f" – {title_suffix}"
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("FVE (fraction)")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    # Keep legend outside to avoid occlusion
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), ncol=1, fontsize=8)

    fig.tight_layout(rect=[0, 0, 0.8, 1])
    fname = "01_composition_over_time"
    if filename_suffix:
        fname += f"_{filename_suffix}"
    fig.savefig(out_dir / f"{fname}.png", dpi=300)
    plt.close(fig)


# -------------------------
# Main
# -------------------------
def _load_spectra_dir_filtered(root: Path) -> List[Spectrum]:
    specs: List[Spectrum] = []
    if not root.exists():
        return specs
    for fp in sorted(root.glob("*.txt")):
        # Skip non-spectral helper outputs commonly named *_average.txt, etc.
        name_l = fp.name.lower()
        if any(tag in name_l for tag in ["average", "avg_"]):
            continue
        try:
            wl, iy = load_txt_spectrum(fp)
        except Exception:
            # Not a spectral file in expected format; skip gracefully
            continue
        specs.append(Spectrum(wavelength=wl, intensity=iy))
    return specs


def main() -> None:
    base = Path(__file__).resolve().parent
    pennies_root = base / "data" / "pennies"
    lists_dir = base / "data" / "lists"

    print(f"Scanning {pennies_root}...")
    sets = discover_pennies_sets(pennies_root)
    if not sets:
        print("No measurement sets found under data/pennies.")
        return

    print("Discovered measurement sets:")
    for s in sets:
        meas_n = len(list(s.meas_root.glob("*.txt")))
        bg_n = len(list(s.bg_root.glob("*.txt"))) if s.bg_root.exists() else 0
        print(f"  - {s.label}: {meas_n} measurement files, {bg_n} background files")

    refs = load_references(lists_dir, element_only=False)
    cfg = build_config()

    all_results: List[RunResult] = []
    per_year_species_scores: Dict[str, Dict[str, List[float]]] = {}
    # Per-condition aggregates
    per_kind_year_species_scores: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        "circ": {},
        "uncirc": {},
    }
    species_counts_by_kind: Dict[str, Dict[str, int]] = {
        "circ": {},
        "uncirc": {},
    }
    # Sum of FVE across runs per species per condition (for robust top-5 selection)
    species_fve_sum_by_kind: Dict[str, Dict[str, float]] = {
        "circ": {},
        "uncirc": {},
    }
    # R² by condition
    r2_by_kind: Dict[str, List[float]] = {"circ": [], "uncirc": []}

    for idx, s in enumerate(sets, start=1):
        print(f"\nRunning analysis {idx}/{len(sets)}: {s.label}")
        meas = _load_spectra_dir_filtered(s.meas_root)
        bg = _load_spectra_dir_filtered(s.bg_root)
        print(f"   Loaded {len(meas)} measurement(s), {len(bg)} background(s)")
        
        # Group spectra
        groups = group_spectra(meas)
        print(f"   Found {len(groups)} spectral group(s)")
        
        if not groups:
            print("   No groups found, skipping this measurement set")
            continue
        
        # Analyze only non-junk groups to find best total TVE
        best_result = None
        best_tve = -1.0
        best_r2_sel = -1.0
        best_group_idx = -1
        non_junk_groups = []
        
        for g_idx, group in enumerate(groups):
            n_specs = len(group)
            if n_specs <= 3:
                print(f"   Group {g_idx + 1} (too small, {n_specs} spectra): skipped")
                continue

            is_junk = is_junk_group(group, debug=False)
            status = "JUNK" if is_junk else "OK"
            
            if is_junk:
                print(f"   Group {g_idx + 1} ({status}, {n_specs} spectra): skipped")
                continue
                
            print(f"   Analyzing group {g_idx + 1} ({status}, {n_specs} spectra)...")
            non_junk_groups.append((g_idx, group))
            
            try:
                # Run analysis without visualization to get R²
                temp_result = analyze(
                    measurements=group,
                    references=refs,
                    backgrounds=bg if bg else None,
                    config=cfg,
                    visualize=False,
                    viz_show=False,
                )
                
                temp_r2 = float(temp_result.metrics.get("fit_R2", 0.0))
                temp_det = getattr(temp_result, "detection", None)
                temp_tve = float(sum((getattr(temp_det, "per_species_scores", {}) or {}).values())) if temp_det else 0.0
                print(f"     R²={temp_r2:.4f}; total_TVE={temp_tve:.4f}")
                
                if temp_tve > best_tve:
                    best_tve = temp_tve
                    best_r2_sel = temp_r2
                    best_result = temp_result
                    best_group_idx = g_idx
                    
            except Exception as e:
                print(f"     Failed to analyze: {e}")
                continue
        
        if best_result is None:
            print("   No valid (non-junk) groups found or all analyses failed, skipping this measurement set")
            continue
        
        print(f"   Selected group {best_group_idx + 1} with highest total_TVE={best_tve:.4f} (R²={best_r2_sel:.4f})")
        
        # Use the best result
        result = best_result
        
        # Re-run visualization for selected group if needed
        visualize_this = (idx % 10 == 0)
        if visualize_this:
            viz_dir = base / "plots" / "pennies" / s.year / s.meas_root.name
            viz_dir.mkdir(parents=True, exist_ok=True)
            print("   Re-running best group with visualization...")
            
            result = analyze(
                measurements=groups[best_group_idx],
                references=refs,
                backgrounds=bg if bg else None,
                config=cfg,
                visualize=True,
                viz_output_dir=str(viz_dir),
                viz_show=False,
            )

        r2 = float(result.metrics.get("fit_R2", 0.0))
        print(f"   R²={r2:.4f}; detections={len(result.detections)}")
        if result.detections:
            print("      Detections:")
            for d in result.detections:
                score = float(d.get("score", d.get("fve", 0.0)))
                coeff = float(d.get("coeff", 0.0))
                bands = int(d.get("bands_hit", 0))
                print(f"        - {d['species']}: score={score:.4f}, coeff={coeff:.4f}, bands={bands}")
        else:
            print("      No detections above threshold")

        all_results.append(RunResult(label=s.label, r2=r2, detections=list(result.detections)))

        # Aggregate per-species FVE for composition-over-time
        det = getattr(result, "detection", None)
        if det is not None:
            scores_map = getattr(det, "per_species_scores", {}) or {}
            year_scores = per_year_species_scores.setdefault(s.year, {})
            for sp, fve in scores_map.items():
                sp_key = str(sp).strip()
                year_scores.setdefault(sp_key, []).append(float(fve))

            # Also aggregate by condition (circ/uncirc)
            name_l = s.meas_root.name.lower()
            kind = "circ" if name_l.startswith("circ") else ("uncirc" if name_l.startswith("uncirc") else "uncirc")
            kind_year_scores = per_kind_year_species_scores.setdefault(kind, {}).setdefault(s.year, {})
            for sp, fve in scores_map.items():
                sp_key = str(sp).strip()
                kind_year_scores.setdefault(sp_key, []).append(float(fve))
                species_fve_sum_by_kind[kind][sp_key] = species_fve_sum_by_kind[kind].get(sp_key, 0.0) + float(fve)

            # Count detected species (presence list) for top-5 selection per condition
            for d in result.detections:
                sp_key = str(d.get("species", "?")).strip()
                species_counts_by_kind[kind][sp_key] = species_counts_by_kind[kind].get(sp_key, 0) + 1
            # Track R² per condition
            r2_by_kind[kind].append(r2)

    # Aggregate summary visuals (after all runs)
    summary_dir = base / "plots" / "pennies" / "summary"
    save_aggregate_summary(all_results, summary_dir)

    # Determine top-5 species per condition by total FVE (more robust than truncated detections)
    def top_k_by_sum(sums: Dict[str, float], k: int = 5) -> List[str]:
        items = sorted(sums.items(), key=lambda t: t[1], reverse=True)
        return [sp for sp, _ in items[:k]]

    top5_circ = top_k_by_sum(species_fve_sum_by_kind.get("circ", {}), 5)
    top5_uncirc = top_k_by_sum(species_fve_sum_by_kind.get("uncirc", {}), 5)
    # Fallback to frequency if sums are empty
    if not top5_circ:
        top5_circ = [sp for sp, _ in sorted(species_counts_by_kind.get("circ", {}).items(), key=lambda t: t[1], reverse=True)[:5]]
    if not top5_uncirc:
        top5_uncirc = [sp for sp, _ in sorted(species_counts_by_kind.get("uncirc", {}).items(), key=lambda t: t[1], reverse=True)[:5]]

    # Always include copper and zinc if present in data; keep max 5
    def pin_species(current: List[str], sums: Dict[str, float]) -> List[str]:
        pinned = [
            sp for sp in sums.keys() if sp.strip().upper() in {"CU", "ZN"}
        ]
        have = set(current)
        for sp in pinned:
            if sp in sums and sp not in have:
                current.append(sp)
        # If exceeding 5, keep pinned and highest others by sum
        if len(current) > 5:
            # Sort by (is_pinned, sum) so pinned rank first, then by sum
            current_sorted = sorted(
                set(current), key=lambda s: ((s in pinned), sums.get(s, 0.0)), reverse=True
            )
            out: List[str] = []
            for s in current_sorted:
                if s not in out:
                    out.append(s)
                if len(out) >= 5:
                    break
            return out
        return current

    top5_circ = pin_species(top5_circ, species_fve_sum_by_kind.get("circ", {}))
    top5_uncirc = pin_species(top5_uncirc, species_fve_sum_by_kind.get("uncirc", {}))

    # Debug print to verify Zn totals (species-level)
    uncirc_sums = species_fve_sum_by_kind.get("uncirc", {})
    for sp_name, total in uncirc_sums.items():
        if sp_name.strip().upper() == "ZN":
            print(f"[DEBUG] Uncirc Zn total FVE (species-level key='{sp_name}'): {total:.6f}")

    # Generate composition-over-time plots for circ and uncirc (top-5 only)
    save_composition_over_time(
        per_kind_year_species_scores.get("circ", {}),
        summary_dir,
        species_filter=top5_circ,
        title_suffix="Circ",
        filename_suffix="circ",
    )
    save_composition_over_time(
        per_kind_year_species_scores.get("uncirc", {}),
        summary_dir,
        species_filter=top5_uncirc,
        title_suffix="Uncirc",
        filename_suffix="uncirc",
    )
    print(f"\nSaved aggregate summary to {summary_dir}")
    
    # Save additional visualizations
    save_year_species_heatmap(per_kind_year_species_scores, summary_dir)
    save_stacked_area_composition(per_kind_year_species_scores, summary_dir)
    save_species_cooccurrence(all_results, summary_dir)

    # -------------------------
    # Detailed metrics (printed)
    # -------------------------
    def _flatten_vals(per_year: Dict[str, Dict[str, List[float]]]) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {}
        for y in per_year.values():
            for sp, vals in y.items():
                out.setdefault(sp, []).extend([float(v) for v in vals])
        return out

    def print_metrics_for(kind: str) -> None:
        print(f"\n===== Detailed Metrics: {kind.upper()} =====")
        r2s = np.asarray(r2_by_kind.get(kind, []), dtype=float)
        if r2s.size:
            print(f"Runs: {r2s.size}; R² mean={np.mean(r2s):.4f}, median={np.median(r2s):.4f}, std={np.std(r2s):.4f}")
        else:
            print("No runs.")

        sums = species_fve_sum_by_kind.get(kind, {})
        if not sums:
            print("No species FVE data.")
            return
        # Composition shares
        total_sum = float(sum(sums.values())) or 1.0
        shares = {sp: float(val) / total_sum for sp, val in sums.items()}
        top_items = sorted(sums.items(), key=lambda t: t[1], reverse=True)[:10]
        print("Top species by total FVE:")
        for sp, tot in top_items:
            print(f"  - {sp}: total_FVE={tot:.4f}, share={shares.get(sp, 0.0):.3f}")

        # Mean FVE across all runs per species
        flat = _flatten_vals(per_kind_year_species_scores.get(kind, {}))
        sp_means = {sp: float(np.nanmean(vals)) if len(vals) else 0.0 for sp, vals in flat.items()}
        top_means = sorted(sp_means.items(), key=lambda t: t[1], reverse=True)[:10]
        print("Mean FVE per species across runs:")
        for sp, m in top_means:
            print(f"  - {sp}: mean_FVE={m:.4f}")

        # Focus on Cu/Zn if present
        for sp in ["CU", "ZN"]:
            if sp in sp_means:
                print(f"  * {sp}: mean_FVE={sp_means[sp]:.4f}, share={shares.get(sp, 0.0):.3f}")

    print_metrics_for("circ")
    print_metrics_for("uncirc")


if __name__ == "__main__":
    main()


