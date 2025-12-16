#!/usr/bin/env python3
"""Run the composable analysis pipeline on the Coke can measurements (Nov 5)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

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
    is_junk_group,
    load_references,
    load_txt_spectrum,
    signal_quality,
)
from yrep_spectrum_analysis.visualizations import (
    visualize_detection,
    visualize_preprocessing,
    visualize_templates,
)

# Configuration tuned for the metal can measurements
VISUALIZE = True
PLOT_DIR = Path(__file__).parent / "plots/coke_can"
AVERAGE_POINTS = 1200
RESAMPLE_POINTS = 1500
TRIM_RANGE = (300.0, 600.0)
CONTINUUM_STRENGTH = 0.00001
SHIFT_PARAMS = {"spread_nm": 0.5, "iterations": 3}
DETECT_PARAMS = {"presence_threshold": 0.00002, "min_bands": 3}
INITIAL_FWHM = 0.75
FWHM_SEARCH = {"enabled": True, "spread_nm": 0.2, "iterations": 3}

# Expected elements for an aluminum beverage can and common contaminants
CAN_ELEMENTS = [
    "Al",
    "Mg",
    "Si",
    "Fe",
    "Mn",
    "Cu",
    "Zn",
    "Cr",
    "Ni",
    "Ti",
    "Na",
    "K",
    "Ca",
    "Ba",
    "Sr",
    "Pb",
    "Sn",
    "P",
    "S",
]


class ResultSummary(NamedTuple):
    dataset_name: str
    bg_name: str
    r2: float
    detections: list[str]
    detection_scores: list[tuple[str, float]]
    top_fve: list[tuple[str, float]]
    top_coeffs: list[tuple[str, float]]
    best_fwhm: float | None


def load_recursive(root: Path) -> list[Signal]:
    """Recursively load all .txt spectra in a directory, skipping any averages."""
    signals: list[Signal] = []
    if not root.exists():
        return signals

    for fp in sorted(root.rglob("*.txt")):
        if any(tag in fp.name.lower() for tag in ("average", "avg_")):
            continue
        try:
            wl, iy = load_txt_spectrum(fp)
            signals.append(Signal(wavelength=wl, intensity=iy, meta={"file": fp.name}))
        except ValueError:
            continue
    return signals


def load_runs(root: Path) -> dict[str, list[Signal]]:
    """Load spectra grouped by immediate subdirectories (Side, Bottom, Tab)."""
    runs: dict[str, list[Signal]] = {}
    if not root.exists():
        return runs

    for path in sorted(root.iterdir()):
        if path.is_dir():
            signals = load_recursive(path)
            if signals:
                runs[path.name] = signals
    return runs


def describe_group(group: list[Signal]) -> tuple[bool, float]:
    """Return junk status and group quality for the averaged signal."""
    junk = is_junk_group(group)
    try:
        avg = average_signals(group, n_points=AVERAGE_POINTS)
        q_avg = signal_quality(avg)
    except Exception:
        q_avg = 0.0
    return junk, q_avg


def run_pipeline(
    measurements: list[Signal],
    backgrounds: list[Signal],
    references,
    species_filter: list[str] | None,
    plot_path_prefix: Path | None = None,
) -> tuple[Any, Any, float]:
    """Run the preprocessing + detection pipeline for one run/background."""
    signal = average_signals(measurements, n_points=AVERAGE_POINTS)

    background_signal = None
    if backgrounds:
        background_signal = average_signals(backgrounds, n_points=AVERAGE_POINTS)

    processed = trim(signal, min_nm=TRIM_RANGE[0], max_nm=TRIM_RANGE[1])
    processed = resample(processed, n_points=RESAMPLE_POINTS)

    if background_signal:
        processed = subtract_background(processed, background_signal, align=False)

    subtracted = processed

    # Continuum removal (capture baselines for plotting)
    arpls_corrected = continuum_remove_arpls(processed, strength=CONTINUUM_STRENGTH)
    arpls_baseline_intensity = processed.intensity - arpls_corrected.intensity
    arpls_baseline = Signal(
        wavelength=processed.wavelength,
        intensity=arpls_baseline_intensity,
        meta={},
    )

    rolling_corrected = continuum_remove_rolling(
        arpls_corrected, strength=CONTINUUM_STRENGTH
    )
    rolling_baseline_intensity = arpls_corrected.intensity - rolling_corrected.intensity
    rolling_baseline = Signal(
        wavelength=processed.wavelength,
        intensity=rolling_baseline_intensity,
        meta={},
    )

    processed = rolling_corrected

    if VISUALIZE and plot_path_prefix:
        visualize_preprocessing(
            original=signal,
            background=background_signal,
            subtracted=subtracted,
            arpls_baseline=arpls_baseline,
            rolling_baseline=rolling_baseline,
            final=processed,
            title="Preprocessing Steps",
            save_path=str(plot_path_prefix) + "_preprocessing.png",
            show=False,
        )

    # Template matching and search
    templates = build_templates(
        processed,
        references=references,
        fwhm_nm=INITIAL_FWHM,
        species_filter=species_filter,
    )
    if FWHM_SEARCH["enabled"]:
        templates = fwhm_search(
            processed,
            references,
            initial_fwhm_nm=INITIAL_FWHM,
            spread_nm=FWHM_SEARCH["spread_nm"],
            iterations=FWHM_SEARCH["iterations"],
            species_filter=species_filter,
        )

    if VISUALIZE and plot_path_prefix:
        visualize_templates(
            signal=processed,
            templates=templates,
            title="Optimized Templates",
            save_path=str(plot_path_prefix) + "_templates.png",
            show=False,
        )

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

    if VISUALIZE and plot_path_prefix:
        visualize_detection(
            result=result,
            templates=templates,
            title="Detection Results",
            save_path=str(plot_path_prefix) + "_detection.png",
            show=False,
        )

    r2 = result.meta.get("fit_R2", 0.0)
    return result, templates, r2


def main() -> None:
    base = Path(__file__).resolve().parent
    data_root = base / "data" / "nov5"
    coke_dir = data_root / "Coke_Can"
    backgrounds_root = data_root / "Background"
    lists_dir = base / "data" / "lists"

    print("Loading references...")
    references = load_references(lists_dir, element_only=False)
    species_filter = expand_species_filter(references.lines.keys(), CAN_ELEMENTS)

    print("Loading coke can runs...")
    runs = load_runs(coke_dir)
    if not runs:
        print("No Coke can measurements found.")
        return

    background_sets = {
        "Background_1": load_recursive(backgrounds_root / "Background_1"),
        "Background_2": load_recursive(backgrounds_root / "Background_2"),
        "Background_3": load_recursive(backgrounds_root / "Background_3"),
    }

    if VISUALIZE:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[ResultSummary] = []

    for run_name, group_signals_list in runs.items():
        junk, q_avg = describe_group(group_signals_list)
        if junk:
            print(f"{run_name} looks like junk (quality={q_avg:.3f}), skipping.")
            continue

        print(f"\nProcessing {run_name}...")

        for bg_name, bg_files in background_sets.items():
            if not bg_files:
                print(f"  Background set {bg_name} empty, skipping.")
                continue

            plot_prefix = None
            if VISUALIZE:
                plot_prefix = PLOT_DIR / f"{run_name}_{bg_name}"

            try:
                detection, templates, r2 = run_pipeline(
                    group_signals_list,
                    bg_files,
                    references,
                    species_filter,
                    plot_path_prefix=plot_prefix,
                )
            except Exception as exc:  # keep going even if one combo fails
                print(f"  Error processing {run_name} + {bg_name}: {exc}")
                continue

            best_fwhm = templates.meta.get("fwhm_search", {}).get("best_fwhm_nm")
            det_species = [d.species for d in detection.detections]
            det_scores = [(d.species, float(d.score)) for d in detection.detections]

            coeff_map = detection.meta.get("coefficients", {})
            fve_map = detection.meta.get("per_species_fve", {})
            top_coeffs = sorted(
                ((sp, float(c)) for sp, c in coeff_map.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5]
            top_fve = sorted(
                ((sp, float(f)) for sp, f in fve_map.items()),
                key=lambda kv: kv[1],
                reverse=True,
            )[:5]

            score_str = (
                ", ".join(f"{sp} ({sc:.4f})" for sp, sc in det_scores[:5])
                if det_scores
                else "none"
            )
            fve_str = ", ".join(f"{sp} ({sc:.4f})" for sp, sc in top_fve)
            coeff_str = ", ".join(f"{sp} ({c:.4f})" for sp, c in top_coeffs)

            print(f"  {bg_name}: R²={r2:.4f}; detections={score_str}")
            print(f"     top FVE: {fve_str}")
            print(f"  top coeffs: {coeff_str}")

            results.append(
                ResultSummary(
                    dataset_name=run_name,
                    bg_name=bg_name,
                    r2=r2,
                    detections=det_species,
                    detection_scores=det_scores,
                    top_fve=top_fve,
                    top_coeffs=top_coeffs,
                    best_fwhm=best_fwhm,
                )
            )

    print("\n" + "=" * 80)
    print("SUMMARY OF COKE CAN RESULTS")
    print("=" * 80)
    print(f"{'Run':<18} | {'Background':<12} | {'R²':<8} | {'Detections (score)'}")
    print("-" * 80)

    results.sort(key=lambda x: x.r2, reverse=True)

    for res in results:
        det_str = ", ".join(
            f"{sp} ({sc:.4f})" for sp, sc in res.detection_scores[:3]
        )
        if len(res.detection_scores) > 3:
            det_str += "..."
        print(f"{res.dataset_name:<18} | {res.bg_name:<12} | {res.r2:.4f}   | {det_str}")
        contrib_str = ", ".join(f"{sp} ({sc:.4f})" for sp, sc in res.top_fve[:3])
        print(f"    top FVE -> {contrib_str}")

    if not results:
        print("No results to summarize.")
        return

    best = results[0]
    print("\n" + "=" * 80)
    print(f"BEST CONFIGURATION: {best.dataset_name} + {best.bg_name} (R²={best.r2:.4f})")
    print("=" * 80)

    best_run_signals = runs.get(best.dataset_name)
    best_bg_signals = background_sets.get(best.bg_name)

    if best_run_signals and best_bg_signals:
        print("Re-running best configuration to generate plots...")
        plot_prefix = PLOT_DIR / f"BEST_{best.dataset_name}_{best.bg_name}"
        run_pipeline(
            best_run_signals,
            best_bg_signals,
            references,
            species_filter,
            plot_path_prefix=plot_prefix,
        )
        print(f"Best run plots saved to {plot_prefix}_*.png")
    else:
        print("Could not reload data for best run to plot.")


if __name__ == "__main__":
    main()
