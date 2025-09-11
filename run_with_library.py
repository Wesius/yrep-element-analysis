#!/usr/bin/env python3
"""
Minimal demo using yrep_spectrum_analysis library.

Reads existing dataset files into arrays, runs analyze(), prints concise results.
This is a demonstration of simplicity: no file writes, no plotting.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np


from yrep_spectrum_analysis import AnalysisConfig, analyze
from yrep_spectrum_analysis.utils import (
    load_batch,
    load_references,
    group_spectra,
    is_junk_group,
    spectrum_quality,
    average_spectra
)


# Analysis configuration (explicitly sets all used fields)
CFG = AnalysisConfig(
    fwhm_nm=0.75,
    grid_step_nm=None,  # use data-driven grid if None
    species=[
        "Na", "K", "Ca", "Li", "Cu", "Ba", "Sr", "Hg", "O", "N", "Al", "Mg", "Si", "Zn", "Pb", "Cd", "Ag", "Au", "Cr", "Mn", "Co", "Ni", "Ti", "Sn", "Sb", "As", "Se", "C", "B", "Fe", "H", "Ar"
    ],
    # Tweaks
    baseline_strength=0.5,
    # Ridge regularization strength (λ). 0 disables; ~1e-2 light, ~1e-1 strong.
    regularization=0.0,
    min_bands_required=5,
    presence_threshold=0,  # default threshold (FVE fraction)
    top_k=3,
    # Trimming controls (flattened)
    min_wavelength_nm=300,
    max_wavelength_nm=600,
    auto_trim_left=False,
    auto_trim_right=False,
    # Background handling and optional overrides
    align_background=False,
    background_fn=None,
    continuum_fn=None,
    # Search controls
    search_shift=True,
    shift_search_iterations=3,
    shift_search_spread=0.5,  # absolute nm window
    search_fwhm=True,
    fwhm_search_iterations=3,
    fwhm_search_spread=0.5,
)


def main() -> None:
    base = Path(__file__).resolve().parent
    lists_dir = base / "data" / "lists"
    refs = load_references(lists_dir)

    for std in ["Copper", "StandardA", "StandardB", "StandardC"]: #     for std in ["StandardA", "StandardB", "StandardC", "StandardD"]:
        print(f"\nProcessing {std}...")
        std_dir = base / "data" / "StandardsTest" / std
        meas_root = (
            (std_dir / std)
            if (std_dir / std).exists()
            else (std_dir / ("StdB" if std == "StandardB" else std))
        )
        
        bg_root = std_dir / "BG"
        meas, bg = load_batch(meas_root, bg_root)
        groups = group_spectra(meas)
        print(f"   Split into {len(groups)} group(s)")
        for gi, group in enumerate(groups, start=1):
            output_dir = base / "plots" / std.lower() / f"group_{gi:02d}"
            output_dir.mkdir(parents=True, exist_ok=True)

            junk = is_junk_group(group)
            # Quality: mean per-spectrum and average-of-group quality
            q_specs = [spectrum_quality(s) for s in group]
            try:
                avg = average_spectra(group, n_points=1000)
                q_avg = spectrum_quality(avg)
            except Exception:
                q_avg = 0.0
            print(f"   Group {gi}: junk={junk}; quality_mean={np.mean(q_specs):.3f}; quality_avg={q_avg:.3f}")

            result = analyze(
                measurements=group,
                references=refs,
                backgrounds=bg,
                config=CFG,
                visualize=True,
                viz_output_dir=str(output_dir),
                viz_show=False,
            )

            r2 = float(result.metrics.get("fit_R2", 0.0))
            print(f"   Group {gi}: R²={r2:.4f}; detections={len(result.detections)}")
            if result.detections:
                print("      Detections:")
                for d in result.detections:
                    print(
                        f"        - {d['species']}: score={d.get('score', d.get('fve', 0.0)):.4f}, "
                        f"coeff={d.get('coeff', 0.0):.4f}, bands={d.get('bands_hit', 0)}"
                    )
            else:
                print("      No detections above threshold")


if __name__ == "__main__":
    main()
