import sys
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox

from yrep_spectrum_analysis import AnalysisConfig, analyze
from yrep_spectrum_analysis.utils import (
    load_batch,
    load_references,
    group_spectra,
)

# Resolve resource root even when bundled with PyInstaller (where _MEIPASS is set).
_BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
_DATA_DIR = _BASE_DIR.parent / "data"

_CFG = AnalysisConfig(
    fwhm_nm=0.75,
    grid_step_nm=None,
    species=[
        "Na", "K", "Ca", "Li", "Cu", "Ba", "Sr", "Hg", "O", "N", "Al", "Mg", "Si",
        "Zn", "Pb", "Cd", "Ag", "Au", "Cr", "Mn", "Co", "Ni", "Ti", "Sn", "Sb",
        "As", "Se", "C", "B", "Fe", "H", "Ar",
    ],
    baseline_strength=0.5,
    regularization=0.0,
    min_bands_required=5,
    presence_threshold=0,
    top_k=3,
    min_wavelength_nm=300,
    max_wavelength_nm=600,
    auto_trim_left=False,
    auto_trim_right=False,
    align_background=False,
    background_fn=None,
    continuum_fn=None,
    search_shift=True,
    shift_search_iterations=3,
    shift_search_spread=0.5,
    search_fwhm=True,
    fwhm_search_iterations=3,
    fwhm_search_spread=0.5,
)

_REFS = load_references(_DATA_DIR / "lists", element_only=False)


def _analyze_standard(std_name: str) -> list[str]:
    std_dir = _DATA_DIR / "StandardsTest" / std_name
    meas_root = std_dir / std_name
    if not meas_root.exists():
        fallback = "StdB" if std_name == "StandardB" else std_name
        meas_root = std_dir / fallback
    bg_root = std_dir / "BG"

    measurements, backgrounds = load_batch(meas_root, bg_root)
    outputs: list[str] = []
    for idx, group in enumerate(group_spectra(measurements), start=1):
        outputs.append(f"Group {idx}")
        result = analyze(
            measurements=group,
            references=_REFS,
            backgrounds=backgrounds,
            config=_CFG,
            visualize=False,
            viz_output_dir=None,
            viz_show=False,
        )
        r2 = float(result.metrics.get("fit_R2", 0.0))
        outputs.append(f"  R²={r2:.4f} | detections={len(result.detections)}")
        for det in result.detections:
            score = float(det.get("score", det.get("fve", 0.0)))
            coeff = float(det.get("coeff", 0.0))
            bands = int(det.get("bands_hit", 0))
            outputs.append(
                f"    - {det['species']} | score={score:.4f} | coeff={coeff:.4f} | bands={bands}"
            )
    if not outputs:
        outputs.append("No measurements found.")
    return outputs


class App(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Y-REP Spectrum Scaffold")
        self.geometry("600x400")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        top = ttk.Frame(self)
        top.grid(row=0, column=0, sticky="ew", padx=12, pady=(12, 6))
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Standard").grid(row=0, column=0, sticky="w")
        self.std_var = tk.StringVar(value="Copper")
        ttk.Entry(top, textvariable=self.std_var).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.run_btn = ttk.Button(top, text="Run", command=self._on_run)
        self.run_btn.grid(row=0, column=2, padx=(12, 0))

        self.output = tk.Text(self, state="disabled", wrap="word")
        self.output.grid(row=1, column=0, sticky="nsew", padx=12, pady=(0, 12))

    def _on_run(self) -> None:
        std = self.std_var.get().strip()
        if not std:
            messagebox.showwarning("Input required", "Please enter a standard name.")
            return
        self.run_btn.config(state="disabled")
        self._write_lines([f"Running analysis for {std}..."])
        threading.Thread(target=self._worker, args=(std,), daemon=True).start()

    def _worker(self, std: str) -> None:
        try:
            lines = _analyze_standard(std)
        except Exception as exc:  # noqa: BLE001 - surface any issue to the user
            self.after(0, self._handle_error, exc)
            return
        self.after(0, self._handle_success, std, lines)

    def _handle_success(self, std: str, lines: list[str]) -> None:
        self.run_btn.config(state="normal")
        self._write_lines([f"Finished analysis for {std}."] + lines)

    def _handle_error(self, exc: Exception) -> None:
        self.run_btn.config(state="normal")
        messagebox.showerror("Analysis failed", str(exc))

    def _write_lines(self, lines: list[str]) -> None:
        self.output.config(state="normal")
        self.output.delete("1.0", tk.END)
        self.output.insert(tk.END, "\n".join(lines))
        self.output.config(state="disabled")


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
