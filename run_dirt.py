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
from yrep_spectrum_analysis.types import Spectrum
from yrep_spectrum_analysis.utils import (
    load_txt_spectrum,
    load_references,
    group_spectra,
    is_junk_group,
    spectrum_quality,
    average_spectra
)


def load_spectra_from_dir(root: Path) -> list[Spectrum]:
    """Recursively load spectra from ``root``, ignoring helper README files."""
    if not root.exists():
        return []

    spectra: list[Spectrum] = []
    for path in sorted(root.rglob("*.txt")):
        if not path.is_file():
            continue
        name_lower = path.name.lower()
        if name_lower.startswith("readme"):
            continue
        try:
            wl, iy = load_txt_spectrum(path)
        except ValueError:
            try:
                data = np.loadtxt(path, delimiter="\t")
            except Exception:
                try:
                    data = np.loadtxt(path)
                except Exception:
                    print(f"   Skipping {path.relative_to(root)}: no spectral data detected")
                    continue
            if data.ndim == 1:
                if data.size % 2 != 0:
                    print(f"   Skipping {path.relative_to(root)}: unsupported data shape")
                    continue
                data = data.reshape(-1, 2)
            if data.shape[1] < 2:
                print(f"   Skipping {path.relative_to(root)}: not enough columns")
                continue
            wl = data[:, 0]
            iy = data[:, 1]
        spectra.append(
            Spectrum(
                wavelength=np.asarray(wl, dtype=float),
                intensity=np.asarray(iy, dtype=float),
                metadata={"file": str(path.relative_to(root))},
            )
        )
    return spectra


def prepare_datasets(base: Path) -> list[tuple[str, list[Spectrum], list[Spectrum]]]:
    """Discover and load available dirt datasets beneath the analysis root."""
    datasets: list[tuple[str, list[Spectrum], list[Spectrum]]] = []

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

    return datasets


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
    max_wavelength_nm=1000,
    auto_trim_left=False,
    auto_trim_right=False,
    # Background handling and optional overrides
    align_background=False,
    background_fn=None,
    continuum_fn=None,
    mask = [(545, 570), (585, 640)],
    # Search controls
    search_shift=False,
    shift_search_iterations=3,
    shift_search_spread=0.5,  # absolute nm window
    search_fwhm=False,
    fwhm_search_iterations=3,
    fwhm_search_spread=0.5,
)


def main() -> None:
    base = Path(__file__).resolve().parent
    lists_dir = base / "data" / "lists"
    refs = load_references(lists_dir, element_only=False)

    datasets = prepare_datasets(base)
    if not datasets:
        raise RuntimeError("No dirt datasets detected.")

    for label, meas, bg in datasets:
        print(f"\nProcessing {label}...")
        if not meas:
            print("   No spectra found; skipping.")
            continue

        if bg:
            print(f"   Loaded {len(meas)} measurements and {len(bg)} backgrounds")
        else:
            print(f"   Loaded {len(meas)} measurements; no backgrounds provided")

        groups = group_spectra(meas)
        print(f"   Split into {len(groups)} group(s)")
        plot_root = base / "plots"
        for gi, group in enumerate(groups, start=1):
            output_dir = plot_root / label.replace("/", "_").lower() / f"group_{gi:02d}"

            junk = is_junk_group(group)
            q_specs = [spectrum_quality(s) for s in group]
            try:
                avg = average_spectra(group, n_points=1000)
                q_avg = spectrum_quality(avg)
            except Exception:
                q_avg = 0.0
            print(f"   Group {gi}: junk={junk}; quality_mean={np.mean(q_specs):.3f}; quality_avg={q_avg:.3f}")

            if junk:
                print(f"   Group {gi}: Skipping analysis (identified as junk)")
                

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
