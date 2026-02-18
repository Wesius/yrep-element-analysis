#!/usr/bin/env python3
"""Run the composable analysis pipeline on SandLead datasets."""

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
PLOT_DIR = Path(__file__).parent / "plots" / "sandlead"
AVERAGE_POINTS = 1200
RESAMPLE_POINTS = 1500
TRIM_RANGE = (300.0, 800.0)
CONTINUUM_STRENGTH = 0.00001
SHIFT_PARAMS = {"spread_nm": 0.5, "iterations": 3}
DETECT_PARAMS = {"presence_threshold": 0.00002, "min_bands": 3}
INITIAL_FWHM = 0.75
FWHM_SEARCH = {"enabled": True, "spread_nm": 0.2, "iterations": 3}

SANDLEAD_ELEMENTS = [
    "Al", "Si", "Fe", "Ca", "Mg", "K", "Na", "Ti", "Mn", "P", "S",
    "Zn", "Cu", "Cr", "Ni", "Pb", "Ba", "Sr", "V", "Co", "Mo",
    "As", "Li", "Cd",
]


class ResultSummary(NamedTuple):
    category: str
    sample_name: str
    bg_name: str
    r2: float
    detections: list[str]
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


def describe_group(group: list[Signal]) -> tuple[bool, float]:
    """Return junk status and quality."""
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
    """Run the pipeline on a group of measurements + background."""
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
        min_bands=int(DETECT_PARAMS["min_bands"]),
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
    data_root = base / "data" / "SandLead"
    lists_dir = base / "data" / "lists"

    categories = {
        "FiftyPercent": data_root / "FiftyPercent",
        "ThirtySevenandaHalf": data_root / "ThirtySevenandaHalf",
    }

    print("Loading references...")
    references = load_references(lists_dir, element_only=False)
    species_filter = expand_species_filter(references.lines.keys(), SANDLEAD_ELEMENTS)

    if VISUALIZE:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)

    results: list[ResultSummary] = []
    kept_log: list[tuple[str, int, int]] = []

    for category, cat_root in categories.items():
        if not cat_root.exists():
            print(f"Warning: {category} not found.")
            continue

        print(f"\nAnalyzing {category}...")
        samples: dict[str, list[Signal]] = {}
        for sample_dir in sorted(cat_root.iterdir()):
            if sample_dir.is_dir():
                signals = load_recursive(sample_dir)
                if signals:
                    samples[sample_dir.name] = signals

        if not samples:
            print(f"  No samples found for {category}.")
            continue

        # Add category-level average
        all_signals: list[Signal] = []
        for sigs in samples.values():
            all_signals.extend(sigs)
        samples_with_avg = dict(samples)
        if all_signals:
            samples_with_avg["AVG"] = all_signals

        for sample_name, signals in samples_with_avg.items():
            filtered = filter_degraded_signals(signals)
            if not filtered:
                print(f"  {sample_name} has no usable signals after cutoff, skipping.")
                continue
            if len(filtered) < len(signals):
                print(
                    f"  {sample_name}: using first {len(filtered)} of "
                    f"{len(signals)} shots (degradation cutoff)."
                )
            if sample_name != "AVG":
                kept_log.append(
                    (f"{category}/{sample_name}", len(signals), len(filtered))
                )

            junk, q_avg = describe_group(filtered)
            if junk:
                print(f"  {sample_name} is junk (quality={q_avg:.3f}), skipping.")
                continue

            print(f"  Processing {sample_name}...")
            plot_prefix = None
            if VISUALIZE:
                plot_prefix = PLOT_DIR / category / sample_name / "no_bg"
                plot_prefix.parent.mkdir(parents=True, exist_ok=True)

            try:
                detection, templates, r2 = run_pipeline(
                    filtered,
                    backgrounds=[],
                    references=references,
                    species_filter=species_filter,
                    plot_path_prefix=plot_prefix,
                )
            except Exception as exc:
                print(f"    Error processing {category} {sample_name}: {exc}")
                continue

            best_fwhm = templates.meta.get("fwhm_search", {}).get("best_fwhm_nm")
            det_species = [d.species for d in detection.detections]

            results.append(
                ResultSummary(
                    category=category,
                    sample_name=sample_name,
                    bg_name="no_bg",
                    r2=r2,
                    detections=det_species,
                    best_fwhm=best_fwhm,
                )
            )

        # Mark best run per category with BEST prefix plots
        category_results = [r for r in results if r.category == category]
        if category_results:
            best = max(category_results, key=lambda r: r.r2)
            best_signals = samples_with_avg.get(best.sample_name)
            if best_signals:
                filtered = filter_degraded_signals(best_signals)
                if filtered:
                    plot_prefix = PLOT_DIR / category / best.sample_name / "no_bg" / "BEST_"
                    plot_prefix.parent.mkdir(parents=True, exist_ok=True)
                    run_pipeline(
                        filtered,
                        backgrounds=[],
                        references=references,
                        species_filter=species_filter,
                        plot_path_prefix=plot_prefix,
                    )

    print("\n" + "=" * 80)
    print("SUMMARY OF RESULTS")
    print("=" * 80)
    print(f"{'Category':<18} | {'Sample':<12} | {'R^2':<8} | Detections")
    print("-" * 80)

    results.sort(key=lambda x: x.r2, reverse=True)
    for res in results:
        det_str = ", ".join(res.detections[:3]) + ("..." if len(res.detections) > 3 else "")
        print(f"{res.category:<18} | {res.sample_name:<12} | {res.r2:.4f}   | {det_str}")

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
