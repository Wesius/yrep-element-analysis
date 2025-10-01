#!/usr/bin/env python3
"""Minimal demo using the composable yrep_spectrum_analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
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
)
from yrep_spectrum_analysis.types import Signal
from yrep_spectrum_analysis.utils import (
    expand_species_filter,
    group_signals,
    is_junk_group,
    load_references,
    load_signals_from_dir,
    signal_quality,
)


def discover_groups(base: Path, standard: str) -> tuple[list[list[Signal]], list[Signal]]:
    std_dir = base / "data" / "StandardsTest" / standard
    meas_root = (
        (std_dir / standard)
        if (std_dir / standard).exists()
        else (std_dir / ("StdB" if standard == "StandardB" else standard))
    )
    bg_root = std_dir / "BG"
    measurements = load_signals_from_dir(meas_root)
    backgrounds = load_signals_from_dir(bg_root)
    groups = group_signals(measurements)
    return groups, backgrounds


def describe_group(group: list[Signal]) -> tuple[bool, float, float]:
    junk = is_junk_group(group)
    qualities = [signal_quality(sig) for sig in group]
    try:
        avg = average_signals(group, n_points=1200)
        avg_quality = signal_quality(avg)
    except Exception:
        avg_quality = 0.0
    return junk, float(np.mean(qualities)), avg_quality


AVERAGE_POINTS = 1200
RESAMPLE_POINTS = 1500
TRIM_RANGE = (300.0, 600.0)
CONTINUUM_STRENGTH = 0.5
SHIFT_PARAMS = {"spread_nm": 0.5, "iterations": 3}
DETECT_PARAMS = {"presence_threshold": 0.02, "min_bands": 5}
INITIAL_FWHM = 0.75
FWHM_SEARCH = {"enabled": True, "spread_nm": 0.2, "iterations": 3}


def run_pipeline(
    measurements: list[Signal],
    backgrounds: list[Signal],
    references,
    species_filter: list[str] | None,
) -> Tuple:
    signal = average_signals(measurements, n_points=AVERAGE_POINTS)
    background_signal = (
        average_signals(backgrounds, n_points=AVERAGE_POINTS)
        if backgrounds
        else None
    )
    history: List[Tuple[str, Signal]] = [("input", signal)]
    processed = trim(signal, min_nm=TRIM_RANGE[0], max_nm=TRIM_RANGE[1])
    history.append(("trim", processed))
    processed = resample(processed, n_points=RESAMPLE_POINTS)
    history.append(("resample", processed))
    if background_signal is not None:
        processed = subtract_background(processed, background_signal, align=False)
        history.append(("background", processed))
    processed = continuum_remove_arpls(processed, strength=CONTINUUM_STRENGTH)
    history.append(("continuum_arpls", processed))
    processed = continuum_remove_rolling(processed, strength=CONTINUUM_STRENGTH)
    history.append(("continuum_rolling", processed))
    templates = build_templates(
        processed,
        references=references,
        fwhm_nm=INITIAL_FWHM,
        species_filter=species_filter,
    )
    if FWHM_SEARCH["enabled"] and FWHM_SEARCH["iterations"] > 0 and FWHM_SEARCH["spread_nm"] > 0:
        templates = fwhm_search(
            processed,
            references,
            initial_fwhm_nm=INITIAL_FWHM,
            spread_nm=FWHM_SEARCH["spread_nm"],
            iterations=FWHM_SEARCH["iterations"],
            species_filter=species_filter,
        )
    if SHIFT_PARAMS["spread_nm"] > 0 and SHIFT_PARAMS["iterations"] > 0:
        processed = shift_search(
            processed,
            templates,
            spread_nm=SHIFT_PARAMS["spread_nm"],
            iterations=SHIFT_PARAMS["iterations"],
        )
        history.append(("shift", processed))
    return detect_nnls(
        processed,
        templates,
        presence_threshold=DETECT_PARAMS["presence_threshold"],
        min_bands=DETECT_PARAMS["min_bands"],
    ), history, templates


def save_stage_history(
    history: List[Tuple[str, Signal]],
    result,
    templates,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (label, sig) in enumerate(history, start=1):
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(sig.wavelength, sig.intensity, linewidth=1.0)
        ax.set_title(f"{label} (stage {idx:02d})")
        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"stage_{idx:02d}_{label}.png", dpi=200)
        plt.close(fig)

    coeff_map = result.meta.get("coefficients", {})
    coeff_vector = np.array([float(coeff_map.get(sp, 0.0)) for sp in templates.species], dtype=float)
    fit_curve = templates.matrix @ coeff_vector
    processed = result.signal

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(processed.wavelength, processed.intensity, label="Processed", linewidth=1.0)
    ax.plot(processed.wavelength, fit_curve, label="Template Fit", linewidth=0.8, alpha=0.7)
    ax.set_title("Detection Stage")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "stage_final_detection.png", dpi=200)
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parent
    lists_dir = base / "data" / "lists"
    references = load_references(lists_dir, element_only=False)

    species_filter = expand_species_filter(references.lines.keys(), [
        "Na", "K", "Ca", "Li", "Cu", "Ba", "Sr", "Hg", "O", "N", "Al",
        "Mg", "Si", "Zn", "Pb", "Cd", "Ag", "Au", "Cr", "Mn", "Co", "Ni",
        "Ti", "Sn", "Sb", "As", "Se", "C", "B", "Fe", "H", "Ar",
    ])

    for std in ["Copper", "StandardA", "StandardB", "StandardC", "StandardD"]:
        print(f"\nProcessing {std}…")
        groups, backgrounds = discover_groups(base, std)
        print(f"   Split into {len(groups)} group(s)")

        for gi, group in enumerate(groups, start=1):
            junk, q_mean, q_avg = describe_group(group)
            print(f"   Group {gi}: junk={junk}; quality_mean={q_mean:.3f}; quality_avg={q_avg:.3f}")
            if junk:
                print("      Skipping analysis (identified as junk)")
                continue

            result, history, templates = run_pipeline(group, backgrounds, references, species_filter)

            r2 = result.meta.get("fit_R2", 0.0)
            print(f"      R²={r2:.4f}; detections={len(result.detections)}")
            best_fwhm = templates.meta.get("fwhm_search", {}).get("best_fwhm_nm")
            if best_fwhm is not None:
                print(f"        Best FWHM ≈ {float(best_fwhm):.3f} nm")
            if result.detections:
                for det in result.detections:
                    coeff = det.meta.get("coeff")
                    bands_hit = det.meta.get("bands_hit")
                    print(
                        f"        - {det.species}: score={det.score:.4f}"
                        + (f", coeff={coeff:.4f}" if coeff is not None else "")
                        + (f", bands={bands_hit}" if bands_hit is not None else "")
                    )
            else:
                print("        No detections above threshold")

            history_dir = base / "plots" / std.lower() / f"group_{gi:02d}" / "pipeline_stages"
            save_stage_history(history, result, templates, history_dir)


if __name__ == "__main__":
    main()
