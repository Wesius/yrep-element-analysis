#!/usr/bin/env python3
"""Run the composable analysis pipeline across the dirt datasets."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from yrep_spectrum_analysis import (
    average_signals,
    build_templates,
    continuum_remove_arpls,
    continuum_remove_rolling,
    detect_nnls,
    fwhm_search,
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


def load_spectra_from_dir(root: Path) -> list[Signal]:
    signals = load_signals_from_dir(root)
    filtered: list[Signal] = []
    for sig in signals:
        name_l = str(sig.meta.get("file", "")).lower()
        if any(tag in name_l for tag in ("average", "avg_")):
            continue
        filtered.append(sig)
    return filtered


def prepare_datasets(base: Path) -> list[tuple[str, list[Signal], list[Signal]]]:
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


def describe_group(group: list[Signal]) -> tuple[bool, float, float]:
    junk = is_junk_group(group)
    q_specs = [signal_quality(sig) for sig in group]
    try:
        avg = average_signals(group, n_points=1200)
        q_avg = signal_quality(avg)
    except Exception:
        q_avg = 0.0
    return junk, float(np.mean(q_specs)), q_avg


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
):
    signal = average_signals(measurements, n_points=AVERAGE_POINTS)
    background_signal = (
        average_signals(backgrounds, n_points=AVERAGE_POINTS)
        if backgrounds
        else None
    )
    processed = trim(signal, min_nm=TRIM_RANGE[0], max_nm=TRIM_RANGE[1])
    processed = resample(processed, n_points=RESAMPLE_POINTS)
    if background_signal is not None:
        processed = subtract_background(processed, background_signal, align=False)
    processed = continuum_remove_arpls(processed, strength=CONTINUUM_STRENGTH)
    processed = continuum_remove_rolling(processed, strength=CONTINUUM_STRENGTH)
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
    result = detect_nnls(
        processed,
        templates,
        presence_threshold=DETECT_PARAMS["presence_threshold"],
        min_bands=DETECT_PARAMS["min_bands"],
    )
    return result, templates


def main() -> None:
    base = Path(__file__).resolve().parent
    lists_dir = base / "data" / "lists"
    references = load_references(lists_dir, element_only=False)

    species_filter = expand_species_filter(references.lines.keys(), [
        "Na", "K", "Ca", "Li", "Cu", "Ba", "Sr", "Hg", "O", "N", "Al",
        "Mg", "Si", "Zn", "Pb", "Cd", "Ag", "Au", "Cr", "Mn", "Co", "Ni",
        "Ti", "Sn", "Sb", "As", "Se", "C", "B", "Fe", "H", "Ar",
    ])

    datasets = prepare_datasets(base)
    if not datasets:
        raise RuntimeError("No dirt datasets detected.")

    for label, measurements, backgrounds in datasets:
        print(f"\nProcessing {label}…")
        if not measurements:
            print("   No spectra found; skipping.")
            continue
        if backgrounds:
            print(f"   Loaded {len(measurements)} measurements and {len(backgrounds)} backgrounds")
        else:
            print(f"   Loaded {len(measurements)} measurements; no backgrounds provided")

        groups = group_signals(measurements)
        print(f"   Split into {len(groups)} group(s)")

        for gi, group in enumerate(groups, start=1):
            junk, q_mean, q_avg = describe_group(group)
            print(f"   Group {gi}: junk={junk}; quality_mean={q_mean:.3f}; quality_avg={q_avg:.3f}")
            if junk:
                print("      Skipping analysis (identified as junk)")
                continue

            detection, templates = run_pipeline(group, backgrounds, references, species_filter)

            r2 = detection.meta.get("fit_R2", 0.0)
            print(f"      R²={r2:.4f}; detections={len(detection.detections)}")
            best_fwhm = templates.meta.get("fwhm_search", {}).get("best_fwhm_nm")
            if best_fwhm is not None:
                print(f"        Best FWHM ≈ {float(best_fwhm):.3f} nm")
            if detection.detections:
                for det in detection.detections:
                    coeff = det.meta.get("coeff")
                    bands_hit = det.meta.get("bands_hit")
                    print(
                        f"        - {det.species}: score={det.score:.4f}"
                        + (f", coeff={coeff:.4f}" if coeff is not None else "")
                        + (f", bands={bands_hit}" if bands_hit is not None else "")
                    )
            else:
                print("        No detections above threshold")


if __name__ == "__main__":
    main()
