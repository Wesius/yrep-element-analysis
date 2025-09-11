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

from yrep_spectrum_analysis import AnalysisConfig, Instrument, analyze
from yrep_spectrum_analysis.types import Spectrum
from yrep_spectrum_analysis.utils import load_references, load_txt_spectrum


# -------------------------
# Configuration (mirrors run_with_library.py)
# -------------------------
def build_config() -> AnalysisConfig:
    return AnalysisConfig(
        instrument=Instrument(
            fwhm_nm=0.75,
            grid_step_nm=None,
            max_shift_nm=2,
        ),
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
        shift_search_iterations=5,
        shift_search_spread=3,
        search_fwhm=True,
        fwhm_search_iterations=5,
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

    refs = load_references(lists_dir)
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

        # Visualization for every 10th run
        visualize_this = (idx % 10 == 0)
        viz_dir = base / "plots" / "pennies" / s.year / s.meas_root.name
        if visualize_this:
            viz_dir.mkdir(parents=True, exist_ok=True)

        result = analyze(
            measurements=meas,
            references=refs,
            backgrounds=bg if bg else None,
            config=cfg,
            visualize=visualize_this,
            viz_output_dir=str(viz_dir) if visualize_this else None,
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
                sp_u = str(sp).strip().split()[0].upper()
                year_scores.setdefault(sp_u, []).append(float(fve))

            # Also aggregate by condition (circ/uncirc)
            name_l = s.meas_root.name.lower()
            kind = "circ" if name_l.startswith("circ") else ("uncirc" if name_l.startswith("uncirc") else "uncirc")
            kind_year_scores = per_kind_year_species_scores.setdefault(kind, {}).setdefault(s.year, {})
            for sp, fve in scores_map.items():
                sp_u = str(sp).strip().split()[0].upper()
                kind_year_scores.setdefault(sp_u, []).append(float(fve))
                species_fve_sum_by_kind[kind][sp_u] = species_fve_sum_by_kind[kind].get(sp_u, 0.0) + float(fve)

            # Count detected species (presence list) for top-5 selection per condition
            for d in result.detections:
                sp_u = str(d.get("species", "?")).strip().split()[0].upper()
                species_counts_by_kind[kind][sp_u] = species_counts_by_kind[kind].get(sp_u, 0) + 1
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
        pinned = ["CU", "ZN"]
        have = set(current)
        for sp in pinned:
            if sp in sums and sp not in have:
                current.append(sp)
        # If exceeding 5, keep pinned and highest others by sum
        if len(current) > 5:
            # Sort by (is_pinned, sum) so pinned rank first, then by sum
            current_sorted = sorted(set(current), key=lambda s: ((s in pinned), sums.get(s, 0.0)), reverse=True)
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

    # Debug print to verify ZN totals
    uncirc_sums = species_fve_sum_by_kind.get("uncirc", {})
    if "ZN" in uncirc_sums:
        print(f"[DEBUG] Uncirc ZN total FVE: {uncirc_sums['ZN']:.6f}")

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


