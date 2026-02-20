#!/usr/bin/env python3
"""Run the composable analysis pipeline on the Dec 3 mangroves datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple
import argparse

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
    filter_degraded_signals,
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

# Configuration
VISUALIZE = True
PLOT_DIR = Path(__file__).parent / "plots/mangroves_dec3"
AVERAGE_POINTS = 1200
RESAMPLE_POINTS = 1500
TRIM_RANGE = (300.0, 800.0)
CONTINUUM_STRENGTH = 0.00001
SHIFT_PARAMS = {"spread_nm": 0.5, "iterations": 3}
DETECT_PARAMS = {"presence_threshold": 0.00002, "min_bands": 3}
INITIAL_FWHM = 0.75
FWHM_SEARCH = {"enabled": True, "spread_nm": 0.2, "iterations": 3}

# Elements expected in mangrove samples and KBr matrices.
MANGROVE_ELEMENTS = [
    "Al", "Si", "Fe", "Ca", "Mg", "K", "Na", "Ti", "Mn", "P", "S",
    "Zn", "Cu", "Cr", "Ni", "Pb", "Ba", "Sr", "V", "Co", "Mo",
    "As", "Li", "Cd", "Br", "Cl",
]


class ResultSummary(NamedTuple):
    category: str
    dataset_name: str
    run_name: str
    r2: float
    detections: list[str]
    detection_scores: list[tuple[str, float]]
    top_fve: list[tuple[str, float]]
    top_coeffs: list[tuple[str, float]]
    best_fwhm: float | None


def load_recursive(root: Path) -> list[Signal]:
    """Recursively load all .txt spectra in a directory, skipping averages."""
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
    """Load spectra grouped by immediate subdirectories (Run_1, Run_2...)."""
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
        detected_species = [d.species for d in result.detections]
        visualize_templates(
            signal=processed,
            templates=templates,
            title="Optimized Templates",
            save_path=str(plot_path_prefix) + "_templates.png",
            show=False,
            species_subset=detected_species,
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


def sanitize_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_")


def collect_run_entries(category_root: Path) -> list[tuple[str, str, list[Signal], int, int]]:
    """Return (dataset_name, run_name, signals, kept, total) for runs."""
    entries: list[tuple[str, str, list[Signal], int, int]] = []
    if not category_root.exists():
        return entries

    for dataset_dir in sorted(category_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        runs = load_runs(dataset_dir)
        for run_name, signals in runs.items():
            filtered, kept, total = filter_degraded_signals(signals)
            if not filtered:
                continue
            entries.append((dataset_dir.name, run_name, filtered, kept, total))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Dec 3 mangroves analyses")
    parser.add_argument(
        "--category",
        default="all",
        choices=["all", "Cut_Leaves", "Dry_Prop", "Super_Dry_Prop"],
        help="Which category to run",
    )
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    data_root = base / "data" / "mangroves_dec3"
    lists_dir = base / "data" / "lists"

    categories = {
        "Cut_Leaves": data_root / "Cut_Leaves",
        "Dry_Prop": data_root / "Dry_Prop",
        "Super_Dry_Prop": data_root / "Super_Dry_Prop",
    }

    if args.category != "all":
        categories = {args.category: categories[args.category]}

    print("Loading references...")
    references = load_references(lists_dir, element_only=False)
    species_filter = expand_species_filter(references.lines.keys(), MANGROVE_ELEMENTS)

    if VISUALIZE:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[ResultSummary] = []
    kept_log: list[tuple[str, int, int]] = []

    for category, category_root in categories.items():
        run_entries = collect_run_entries(category_root)
        if not run_entries:
            print(f"No measurements found for {category}.")
            continue

        print(f"\nProcessing category: {category}")

        by_dataset: dict[str, list[tuple[str, list[Signal], int, int]]] = {}
        for dataset_name, run_name, signals, kept, total in run_entries:
            by_dataset.setdefault(dataset_name, []).append(
                (run_name, signals, kept, total)
            )

        entries: list[tuple[str, str, list[Signal], int, int]] = []
        for dataset_name, runs in by_dataset.items():
            for run_name, signals, kept, total in runs:
                entries.append((dataset_name, run_name, signals, kept, total))
            avg_signals: list[Signal] = []
            avg_kept = 0
            avg_total = 0
            for _, signals, kept, total in runs:
                avg_signals.extend(signals)
                avg_kept += kept
                avg_total += total
            if avg_signals:
                entries.append((dataset_name, "AVG", avg_signals, avg_kept, avg_total))

        cat_signals: list[Signal] = []
        cat_kept = 0
        cat_total = 0
        for _, _, signals, kept, total in run_entries:
            cat_signals.extend(signals)
            cat_kept += kept
            cat_total += total
        if cat_signals:
            entries.append(("ALL", "AVG", cat_signals, cat_kept, cat_total))

        for dataset_name, run_name, signals, kept, total in entries:
            if run_name != "AVG" and kept < total:
                print(
                    f"  {dataset_name}/{run_name}: using first {kept} of "
                    f"{total} shots (degradation cutoff)."
                )
            if run_name != "AVG":
                kept_log.append((f"{category}/{dataset_name}/{run_name}", total, kept))
            junk, q_avg = describe_group(signals)
            if junk:
                print(
                    f"  {dataset_name}/{run_name} looks like junk (quality={q_avg:.3f}), "
                    "skipping."
                )
                continue

            print(f"  {dataset_name}/{run_name}...")

            plot_prefix = None
            if VISUALIZE:
                safe = sanitize_label(run_name)
                plot_prefix = PLOT_DIR / category / dataset_name / safe
                plot_prefix.parent.mkdir(parents=True, exist_ok=True)

            try:
                detection, templates, r2 = run_pipeline(
                    signals,
                    backgrounds=[],
                    references=references,
                    species_filter=species_filter,
                    plot_path_prefix=plot_prefix,
                )
            except Exception as exc:
                print(f"    Error processing {dataset_name}/{run_name}: {exc}")
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

            print(f"    R^2={r2:.4f}; detections={score_str}")
            print(f"    top FVE: {fve_str}")
            print(f"    top coeffs: {coeff_str}")

            results.append(
                ResultSummary(
                    category=category,
                    dataset_name=dataset_name,
                    run_name=run_name,
                    r2=r2,
                    detections=det_species,
                    detection_scores=det_scores,
                    top_fve=top_fve,
                    top_coeffs=top_coeffs,
                    best_fwhm=best_fwhm,
                )
            )

    if not results:
        print("No results to summarize.")
        return

    results.sort(key=lambda x: (x.category, x.dataset_name, -x.r2))

    grouped: dict[tuple[str, str], list[ResultSummary]] = {}
    for res in results:
        grouped.setdefault((res.category, res.dataset_name), []).append(res)

    for (category, dataset_name), group in grouped.items():
        print("\n" + "=" * 80)
        print(f"SUMMARY: {category}/{dataset_name}")
        print("=" * 80)
        print(f"{'Run':<18} | {'R^2':<8} | Detections (score)")
        print("-" * 80)
        for res in group:
            det_str = ", ".join(
                f"{sp} ({sc:.4f})" for sp, sc in res.detection_scores[:3]
            )
            if len(res.detection_scores) > 3:
                det_str += "..."
            print(f"{res.run_name:<18} | {res.r2:.4f}   | {det_str}")
        best = max(group, key=lambda x: x.r2)
        print(
            f"BEST RUN: {best.category}/{best.dataset_name}/{best.run_name} "
            f"(R^2={best.r2:.4f})"
        )

    if VISUALIZE:
        print("\nPlots saved under plots/mangroves_dec3/<Category>/<Dataset>/<Run>_*.png")

    if kept_log:
        kept_path = PLOT_DIR / "kept_runs.txt"
        kept_path.parent.mkdir(parents=True, exist_ok=True)
        with kept_path.open("w") as f:
            f.write("run kept/total\n")
            for run_name, total, kept in kept_log:
                f.write(f"{run_name} {kept}/{total}\n")
        print(f"Wrote kept-run log to {kept_path}")


if __name__ == "__main__":
    main()
