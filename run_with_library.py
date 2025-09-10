#!/usr/bin/env python3
"""
Minimal demo using yrep_spectrum_analysis library.

Reads existing dataset files into arrays, runs analyze(), prints concise results.
This is a demonstration of simplicity: no file writes, no plotting.
"""

from __future__ import annotations

from pathlib import Path


from yrep_spectrum_analysis import AnalysisConfig, analyze, Instrument
from yrep_spectrum_analysis.utils import load_batch, load_references, group_spectra


def main() -> None:
    base = Path(__file__).resolve().parent
    lists_dir = base / "data" / "lists"
    refs = load_references(lists_dir)

    cfg = AnalysisConfig(
        instrument=Instrument(
            fwhm_nm=2,
            max_shift_nm=0.6,  # Used with shift search
        ),
        mode="accurate",  # Uses registration + robust scale for background
        sensitivity="high",  # Raises presence threshold to ~0.05
        min_bands_required=2,  # Makes single-line "hits" fail
        presence_threshold=0.001,  # explicit, stricter than "low"; 0.08–0.12 is sensible here
        auto_trim_left=True,
        align_background=False,
        top_k=3,
        species=[
            "Na",
            "K",
            "Ca",
            "Li",
            "Cu",
            "Ba",
            "Sr",
            "Hg",
            "O",
            "N",
            "Al",
            "Mg",
            "Si",
            "Zn",
            "Pb",
            "Cd",
            "Ag",
            "Au",
            "Cr",
            "Mn",
            "Co",
            "Ni",
            "Ti",
            "Sn",
            "Sb",
            "As",
            "Se",
            "C",
            "B",
            "Fe",
            "H",
            "Ar",
        ],
    )

    for std in ["StandardA"]:
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
            # Create per-group output directory
            output_dir = base / "plots" / std.lower() / f"group_{gi:02d}"
            output_dir.mkdir(parents=True, exist_ok=True)

            result = analyze(
                measurements=group,
                references=refs,
                backgrounds=bg,
                config=cfg,
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
