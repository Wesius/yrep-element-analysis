#!/usr/bin/env python3
"""Run the composable analysis pipeline across the dirt datasets with various backgrounds."""

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

# Configuration
VISUALIZE = True
PLOT_DIR = Path(__file__).parent / "plots/dirt_comparison"
AVERAGE_POINTS = 1200
RESAMPLE_POINTS = 1500
TRIM_RANGE = (300.0, 800.0)
CONTINUUM_STRENGTH = 0.00001
SHIFT_PARAMS = {"spread_nm": 0.5, "iterations": 3}
DETECT_PARAMS = {"presence_threshold": 0.00002, "min_bands": 3}
INITIAL_FWHM = 0.75
FWHM_SEARCH = {"enabled": True, "spread_nm": 0.2, "iterations": 3}

# Elements commonly found in soil/dirt
DIRT_ELEMENTS = [
    "Al", "Si", "Fe", "Ca", "Mg", "K", "Na", "Ti", "Mn", "P", "S", "Zn", "Cu",
    "Cr", "Ni", "Pb", "Ba", "Sr", "V", "Co", "Mo", "As", "Li", "Cd"
]

class ResultSummary(NamedTuple):
    dataset_name: str
    run_name: str
    bg_name: str
    r2: float
    detections: list[str]
    best_fwhm: float | None


def load_recursive(root: Path) -> list[Signal]:
    """recursively load all .txt spectra in a directory."""
    signals: list[Signal] = []
    if not root.exists():
        return signals
    
    # Filter out average files if present
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
    """
    Load spectra grouped by subdirectories (Runs).
    Returns dict like: {"Run1": [sig, sig...], "Run2": [sig, sig...]}
    """
    runs: dict[str, list[Signal]] = {}
    if not root.exists():
        return runs
        
    # List only directories
    for path in sorted(root.iterdir()):
        if path.is_dir():
            # Load signals in this run directory (non-recursive to keep runs separate? 
            # actually, runs might have subfolders? simpler to use recursive per run folder)
            signals = load_recursive(path)
            if signals:
                runs[path.name] = signals
    return runs


def describe_group(group: list[Signal]) -> tuple[bool, float]:
    """Return junk status and quality."""
    junk = is_junk_group(group)
    try:
        avg = average_signals(group, n_points=1200)
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
    
    # 1. Preprocessing
    signal = average_signals(measurements, n_points=AVERAGE_POINTS)
    
    background_signal = None
    if backgrounds:
        background_signal = average_signals(backgrounds, n_points=AVERAGE_POINTS)
        
    processed = trim(signal, min_nm=TRIM_RANGE[0], max_nm=TRIM_RANGE[1])
    processed = resample(processed, n_points=RESAMPLE_POINTS)
    
    if background_signal:
        processed = subtract_background(processed, background_signal, align=False)
        
    # Capture state for visualization
    subtracted = processed
    
    # Continuum Removal
    # Create copies or temporary signals to capture baselines
    # Note: arpls returns the CORRECTED signal, not the baseline. 
    # We need to calculate baseline = original - corrected
    
    arpls_corrected = continuum_remove_arpls(processed, strength=CONTINUUM_STRENGTH)
    arpls_baseline_intensity = processed.intensity - arpls_corrected.intensity
    arpls_baseline = Signal(wavelength=processed.wavelength, intensity=arpls_baseline_intensity, meta={})
    
    rolling_corrected = continuum_remove_rolling(arpls_corrected, strength=CONTINUUM_STRENGTH)
    rolling_baseline_intensity = arpls_corrected.intensity - rolling_corrected.intensity
    rolling_baseline = Signal(wavelength=processed.wavelength, intensity=rolling_baseline_intensity, meta={})
    
    processed = rolling_corrected # Final result
    
    if VISUALIZE and plot_path_prefix:
        visualize_preprocessing(
            original=signal, # Use the averaged original signal
            background=background_signal,
            subtracted=subtracted,
            arpls_baseline=arpls_baseline,
            rolling_baseline=rolling_baseline,
            final=processed,
            title="Preprocessing Steps",
            save_path=str(plot_path_prefix) + "_preprocessing.png",
            show=False,
        )

    # 2. Template Matching
    templates = build_templates(
        processed,
        references=references,
        fwhm_nm=INITIAL_FWHM,
        species_filter=species_filter,
    )
    templates = fwhm_search(
        processed,
        references,
        initial_fwhm_nm=INITIAL_FWHM,
        spread_nm=FWHM_SEARCH["spread_nm"],
        iterations=int(FWHM_SEARCH["iterations"]),
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

    # 3. Detection
    processed = shift_search(
        processed,
        templates,
        spread_nm=SHIFT_PARAMS["spread_nm"],
        iterations=int(SHIFT_PARAMS["iterations"]),
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
    data_27 = base / "data" / "dirt" / "Data_27_Dec"
    lists_dir = base / "data" / "lists"
    
    # Load References
    print("Loading references...")
    references = load_references(lists_dir, element_only=False)
    species_filter = expand_species_filter(references.lines.keys(), DIRT_ELEMENTS)
    
    # Load Datasets (Grouped by Runs)
    print("Loading datasets...")
    # Dictionaries mapping DatasetName -> {RunName -> [Signals]}
    datasets_with_runs = {
        "Mix_NoHeat": load_runs(data_27 / "MIX" / "MIXNOHEAT"),
    }
    
    # Backgrounds (Aggregated)
    backgrounds = {
        "BGNA": load_recursive(data_27 / "BG" / "BGNA"),
        "BGPalet": load_recursive(data_27 / "BG" / "BGPalet"),
        "KBrHeat": load_recursive(data_27 / "KBr" / "KBrHeat"),
        "KBrNoHeat": load_recursive(data_27 / "KBr" / "KBrNoHeat"),
    }
    
    if VISUALIZE:
        PLOT_DIR.mkdir(parents=True, exist_ok=True)
        
    results: list[ResultSummary] = []
    
    # Iterate Datasets
    for ds_name, runs_dict in datasets_with_runs.items():
        if not runs_dict:
            print(f"Warning: No runs found for {ds_name}")
            continue
            
        print(f"\nAnalyzing {ds_name}...")
        
        # Iterate Runs
        for run_name, group_signals_list in runs_dict.items():
            # Filter junk
            junk, q_avg = describe_group(group_signals_list)
            if junk:
                print(f"  {run_name} is junk (quality={q_avg:.3f}), skipping.")
                continue
                
            print(f"  Processing {run_name}...")
            
            # Iterate Backgrounds
            for bg_name, bg_files in backgrounds.items():
                if not bg_files:
                    continue

                plot_prefix = None
                if VISUALIZE:
                    # Structure: plots/dirt_comparison/Mix_Heat_Run1_BGNA
                    plot_prefix = PLOT_DIR / f"{ds_name}_{run_name}_{bg_name}"

                try:
                    detection, templates, r2 = run_pipeline(
                        group_signals_list, bg_files, references, species_filter, plot_path_prefix=plot_prefix
                    )
                    
                    best_fwhm = templates.meta.get("fwhm_search", {}).get("best_fwhm_nm")
                    det_species = [d.species for d in detection.detections]
                    
                    res = ResultSummary(
                        dataset_name=ds_name,
                        run_name=run_name,
                        bg_name=bg_name,
                        r2=r2,
                        detections=det_species,
                        best_fwhm=best_fwhm
                    )
                    results.append(res)
                    # print(f"    [{bg_name}] R²={r2:.4f}")
                    
                except Exception as e:
                    print(f"    Error processing {ds_name} {run_name} + {bg_name}: {e}")

    # Summary
    print("\n" + "="*80)
    print("SUMMARY OF RESULTS")
    print("="*80)
    print(f"{'Dataset':<15} | {'Run':<10} | {'Background':<12} | {'R²':<8} | {'Detections'}")
    print("-" * 80)
    
    # Sort by R2 descending
    results.sort(key=lambda x: x.r2, reverse=True)
    
    for res in results:
        det_str = ", ".join(res.detections[:3]) + ("..." if len(res.detections) > 3 else "")
        print(f"{res.dataset_name:<15} | {res.run_name:<10} | {res.bg_name:<12} | {res.r2:.4f}   | {det_str}")
        
    if results:
        best = results[0]
        print("\n" + "="*80)
        print(f"BEST CONFIGURATION: {best.dataset_name} {best.run_name} + {best.bg_name} (R²={best.r2:.4f})")
        print("="*80)

        # Re-run the best configuration to visualize/plot if needed
        print("\nRe-running best configuration to generate specific plot...")
        
        # Find the signals for the best run
        best_run_signals = None
        if best.dataset_name in datasets_with_runs:
             best_run_signals = datasets_with_runs[best.dataset_name].get(best.run_name)

        # Find the background signals
        best_bg_signals = backgrounds.get(best.bg_name)

        if best_run_signals and best_bg_signals:
            plot_prefix = PLOT_DIR / f"BEST_{best.dataset_name}_{best.run_name}_{best.bg_name}"
            run_pipeline(
                best_run_signals, 
                best_bg_signals, 
                references, 
                species_filter, 
                plot_path_prefix=plot_prefix
            )
            print(f"Best run plots saved to {plot_prefix}_*.png")
        else:
            print("Could not reload data for best run to plot.")

if __name__ == "__main__":
    main()
