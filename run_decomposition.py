#!/usr/bin/env python3
"""Run the unsupervised decomposition analysis (PCA, ICA, MCR-ALS) across the dirt datasets."""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from yrep_spectrum_analysis import (
    average_signals,
    resample,
    trim,
    analyze_pca,
    analyze_ica,
    analyze_mcr,
    identify_components,
    subtract_background,
    continuum_remove_arpls,
    continuum_remove_rolling,
)
from yrep_spectrum_analysis.types import Signal
from yrep_spectrum_analysis.utils import (
    group_signals,
    is_junk_group,
    load_signals_from_dir,
    signal_quality,
    load_references,
)

# Visualization settings
PLOT_DIR = Path(__file__).parent / "plots/decomposition"

# Processing Constants
RESAMPLE_POINTS = 1500
TRIM_RANGE = (300.0, 600.0)
CONTINUUM_STRENGTH = 0.5


def load_spectra_from_dir(root: Path) -> list[Signal]:
    """Load signals and filter out averages."""
    signals = load_signals_from_dir(root)
    filtered: list[Signal] = []
    for sig in signals:
        name_l = str(sig.meta.get("file", "")).lower()
        if any(tag in name_l for tag in ("average", "avg_")):
            continue
        filtered.append(sig)
    return filtered


def prepare_datasets(base: Path) -> list[tuple[str, list[Signal], list[Signal]]]:
    """Detect and load all available dirt datasets."""
    datasets: list[tuple[str, list[Signal], list[Signal]]] = []
    dirt_root = base / "data" / "dirt"
    legacy_dir = dirt_root / "dirt"
    if legacy_dir.exists():
        meas = load_spectra_from_dir(legacy_dir)
        if meas:
            datasets.append(("legacy_dirt", meas, []))

    data_27_dir = dirt_root / "Data_27_Dec"
    if data_27_dir.exists():
        bg_root = data_27_dir / "BG"
        bg_all = load_spectra_from_dir(bg_root) if bg_root.exists() else []
        bg_na = load_spectra_from_dir(bg_root / "BGNA") if bg_root.exists() else []
        bg_palet = load_spectra_from_dir(bg_root / "BGPalet") if bg_root.exists() else []

        # Add pure backgrounds as a dataset to analyze their variation too
        if bg_all:
            datasets.append(("Data_27_Dec/BG", bg_all, []))

        dirt_dir = data_27_dir / "DIRT"
        meas_dirt = load_spectra_from_dir(dirt_dir)
        if meas_dirt:
            bg_for_dirt = bg_na or bg_all
            datasets.append(("Data_27_Dec/DIRT", meas_dirt, bg_for_dirt))

        kbr_dir = data_27_dir / "KBr"
        meas_kbr = load_spectra_from_dir(kbr_dir)
        if meas_kbr:
            bg_for_kbr = bg_palet or bg_all
            datasets.append(("Data_27_Dec/KBr", meas_kbr, bg_for_kbr))

    # Add Standards datasets
    standards_root = base / "data" / "StandardsTest"
    if standards_root.exists():
        for std_dir in standards_root.glob("Standard*"):
            if std_dir.is_dir():
                meas = load_spectra_from_dir(std_dir)
                if meas:
                    datasets.append((f"Standards/{std_dir.name}", meas, []))

    return datasets


def preprocess_signals(signals: list[Signal]) -> list[Signal]:
    """Standard preprocessing for decomposition: trim, resample, and continuum removal."""
    if not signals:
        return []

    # Align all signals to a common grid first
    # We'll pick the first signal's range (after trim) as the master grid
    # Note: In a real scenario, we might want to find the common overlap.
    
    processed_list = []
    for sig in signals:
        s = trim(sig, min_nm=TRIM_RANGE[0], max_nm=TRIM_RANGE[1])
        s = resample(s, n_points=RESAMPLE_POINTS)
        s = continuum_remove_arpls(s, strength=CONTINUUM_STRENGTH)
        s = continuum_remove_rolling(s, strength=CONTINUUM_STRENGTH)
        processed_list.append(s)
        
    return processed_list


def plot_components(
    wavelength: np.ndarray,
    components: np.ndarray,
    method_name: str,
    dataset_name: str,
    save_path: Path,
    identifications: list[list[tuple[str, float]]] | None = None,
):
    """Plot the spectral components found by a method."""
    n_comps = components.shape[0]
    fig, axes = plt.subplots(n_comps, 1, figsize=(10, 2 * n_comps), sharex=True)
    if n_comps == 1:
        axes = [axes]
    
    for i, ax in enumerate(axes):
        ax.plot(wavelength, components[i], color='tab:blue', linewidth=1)
        ax.set_ylabel(f"Comp {i+1}")
        
        # Add identification labels
        if identifications and i < len(identifications):
            matches = identifications[i]
            label_str = ", ".join([f"{sp} ({sc:.2f})" for sp, sc in matches])
            ax.set_title(f"Matches: {label_str}", fontsize=9, loc='right')
            
        ax.grid(True, alpha=0.3)
        
    axes[-1].set_xlabel("Wavelength (nm)")
    fig.suptitle(f"{method_name} Components - {dataset_name}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def plot_pca_scatter(
    scores: np.ndarray,
    dataset_name: str,
    save_path: Path
):
    """Plot PC1 vs PC2 scatter plot."""
    plt.figure(figsize=(8, 6))
    plt.scatter(scores[:, 0], scores[:, 1], alpha=0.7, c='tab:purple', edgecolor='k')
    plt.xlabel("PC1 (Matrix/Background)")
    plt.ylabel("PC2 (Variation)")
    plt.title(f"PCA Score Plot (PC1 vs PC2) - {dataset_name}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def main() -> None:
    base = Path(__file__).resolve().parent
    
    # Load References for Identification
    print("Loading reference lines for component identification...")
    lists_dir = base / "data" / "lists"
    references = load_references(lists_dir, element_only=False)
    
    datasets = prepare_datasets(base)
    
    if not datasets:
        raise RuntimeError("No dirt datasets detected.")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    print(f"Found {len(datasets)} datasets to analyze.")

    for label, measurements, _ in datasets:
        safe_label = label.replace("/", "_").replace(" ", "_")
        print(f"\nProcessing {label}…")
        
        if len(measurements) < 5:
            print("   Not enough measurements for meaningful decomposition (<5). Skipping.")
            continue

        # Preprocess all signals to be uniform
        # Note: We don't average here! We need the variance between shots.
        print("   Preprocessing signals (Trim -> Resample -> Continuum Removal)...")
        processed_signals = preprocess_signals(measurements)
        
        # Extract wavelength grid from the first signal
        wavelength = processed_signals[0].wavelength

        # ----------------------------------------------------------------
        # 1. PCA Analysis
        # ----------------------------------------------------------------
        print("   Running PCA...")
        n_pca = 5
        pca_scores, pca_comps = analyze_pca(processed_signals, n_components=n_pca)
        
        # Identify components
        pca_ids = identify_components(pca_comps, wavelength, references, top_n=3)
        
        plot_components(
            wavelength, 
            pca_comps, 
            "PCA", 
            label, 
            PLOT_DIR / f"{safe_label}_PCA_components.png",
            identifications=pca_ids
        )
        plot_pca_scatter(
            pca_scores,
            label,
            PLOT_DIR / f"{safe_label}_PCA_scatter.png"
        )
        print(f"      PCA done. Saved plots to {PLOT_DIR}")

        # ----------------------------------------------------------------
        # 2. ICA Analysis
        # ----------------------------------------------------------------
        print("   Running ICA...")
        n_ica = 5
        ica_sources = analyze_ica(processed_signals, n_components=n_ica)
        # ICA returns sources as columns (wavelengths x components), we transpose for plotting
        ica_comps = ica_sources.T
        
        # Identify components
        ica_ids = identify_components(ica_comps, wavelength, references, top_n=3)
        
        plot_components(
            wavelength,
            ica_comps,
            "ICA",
            label,
            PLOT_DIR / f"{safe_label}_ICA_components.png",
            identifications=ica_ids
        )
        print(f"      ICA done. Saved plots to {PLOT_DIR}")

        # ----------------------------------------------------------------
        # 3. MCR-ALS Analysis
        # ----------------------------------------------------------------
        print("   Running MCR-ALS (this may take a moment)...")
        n_mcr = 4
        try:
            mcr_conc, mcr_spectra = analyze_mcr(processed_signals, n_components=n_mcr)
            # MCR spectra are typically returned as (components x wavelengths)
            
            # Identify components
            mcr_ids = identify_components(mcr_spectra, wavelength, references, top_n=3)
            
            plot_components(
                wavelength,
                mcr_spectra,
                "MCR-ALS",
                label,
                PLOT_DIR / f"{safe_label}_MCR_components.png",
                identifications=mcr_ids
            )
            print(f"      MCR-ALS done. Saved plots to {PLOT_DIR}")
            
        except Exception as e:
            print(f"      MCR-ALS failed: {e}")

    print(f"\nAll analyses complete. Check {PLOT_DIR} for results.")


if __name__ == "__main__":
    main()
