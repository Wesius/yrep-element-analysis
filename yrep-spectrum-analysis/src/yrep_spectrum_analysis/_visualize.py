from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns

from .types import DetectionResult, SpectrumLike
from ._templates import RefLines

# Set up matplotlib style
plt.style.use("default")
sns.set_palette("husl")


class SpectrumVisualizer:
    """Comprehensive visualization for all spectrum analysis steps."""

    def __init__(
        self,
        output_dir: Optional[str] = None,
        show_plots: bool = True,
        save_plots: bool = True,
    ):
        self.output_dir = Path(output_dir) if output_dir else Path("./plots")
        self.show_plots = show_plots
        self.save_plots = save_plots
        self.output_dir.mkdir(exist_ok=True, parents=True)

        # Plot counter for sequential numbering
        self._plot_counter = 0

    def _get_next_filename(self, base_name: str) -> str:
        """Generate sequential filenames for plots."""
        self._plot_counter += 1
        return f"{self._plot_counter:02d}_{base_name}"

    def _save_and_show(self, fig: plt.Figure, filename: str, title: str = ""):
        """Handle saving and showing plots."""
        # Apply title before layout so it is included in saved figures
        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold")

        # Do not force a particular layout engine here. We'll save differently
        # depending on whether constrained layout is active.

        if self.save_plots:
            filepath = self.output_dir / f"{filename}.png"
            is_constrained = False
            try:
                is_constrained = bool(fig.get_constrained_layout())
            except Exception:
                is_constrained = False

            if is_constrained:
                # Save normally when constrained layout is active
                fig.savefig(filepath, dpi=300, facecolor="white")
            else:
                # Use tight bounding box when not using constrained layout
                fig.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")

        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_raw_spectra(
        self,
        measurements: Sequence[SpectrumLike],
        backgrounds: Optional[Sequence[SpectrumLike]] = None,
        title: str = "Raw Input Spectra",
    ):
        """Step 1: Plot raw input spectra."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # Plot measurements
        ax1.set_title("Measurement Spectra", fontweight="bold")
        for i, spec in enumerate(measurements):
            wl = np.asarray(spec.wavelength)
            intensity = np.asarray(spec.intensity)
            ax1.plot(wl, intensity, alpha=0.7, label=f"Measurement {i + 1}")

        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Intensity")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Plot backgrounds if available
        if backgrounds:
            ax2.set_title("Background Spectra", fontweight="bold")
            for i, spec in enumerate(backgrounds):
                wl = np.asarray(spec.wavelength)
                intensity = np.asarray(spec.intensity)
                ax2.plot(wl, intensity, alpha=0.7, label=f"Background {i + 1}")
        else:
            ax2.set_title("No Background Spectra Provided", fontweight="bold")
            ax2.text(
                0.5,
                0.5,
                "No background data",
                ha="center",
                va="center",
                transform=ax2.transAxes,
                fontsize=14,
                alpha=0.6,
            )

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Intensity")
        if backgrounds:
            ax2.legend()
        ax2.grid(True, alpha=0.3)

        self._save_and_show(fig, self._get_next_filename("raw_spectra"), title)

    def plot_averaged_spectra(
        self,
        avg_meas: Tuple[np.ndarray, np.ndarray],
        avg_bg: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        wl_grid: Optional[np.ndarray] = None,
        title: str = "Averaged Spectra on Common Grid",
    ):
        """Step 2: Plot averaged spectra."""
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))

        # Plot averaged measurement
        wl_meas, int_meas = avg_meas
        ax.plot(
            wl_meas,
            int_meas,
            "b-",
            linewidth=2,
            label="Averaged Measurement",
            alpha=0.8,
        )

        # Plot averaged background if available
        if avg_bg:
            wl_bg, int_bg = avg_bg
            ax.plot(
                wl_bg, int_bg, "r-", linewidth=2, label="Averaged Background", alpha=0.8
            )

        # Show interpolation grid if provided
        if wl_grid is not None:
            ax.axvline(
                float(wl_grid[0]),
                color="gray",
                linestyle="--",
                alpha=0.5,
                label="Grid boundaries",
            )
            ax.axvline(float(wl_grid[-1]), color="gray", linestyle="--", alpha=0.5)

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Intensity")
        ax.legend()
        ax.grid(True, alpha=0.3)

        self._save_and_show(fig, self._get_next_filename("averaged_spectra"), title)

    def plot_background_subtraction(
        self,
        wl_grid: np.ndarray,
        y_meas: np.ndarray,
        y_bg: Optional[np.ndarray],
        y_sub: np.ndarray,
        bg_params: Optional[Dict[str, float]] = None,
        title: str = "Background Subtraction",
    ):
        """Step 3: Plot background subtraction process."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Original measurement and background
        axes[0, 0].set_title("Original Spectra", fontweight="bold")
        axes[0, 0].plot(wl_grid, y_meas, "b-", label="Measurement", linewidth=2)
        if y_bg is not None:
            axes[0, 0].plot(wl_grid, y_bg, "r-", label="Background", linewidth=2)
        axes[0, 0].set_xlabel("Wavelength (nm)")
        axes[0, 0].set_ylabel("Intensity")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Background alignment info
        axes[0, 1].set_title("Background Registration", fontweight="bold")
        if bg_params and y_bg is not None:
            shift_nm = bg_params.get("bg_shift_nm", 0)
            scale_a = bg_params.get("bg_scale_a", 1)
            offset_b = bg_params.get("bg_offset_b", 0)

            # Show aligned background
            y_bg_aligned = scale_a * y_bg + offset_b
            axes[0, 1].plot(wl_grid, y_bg, "r--", alpha=0.5, label="Original BG")
            label_str = (
                "Aligned BG"
                if (
                    abs(shift_nm) > 1e-9
                    or abs(scale_a - 1.0) > 1e-9
                    or abs(offset_b) > 1e-9
                )
                else "BG (no alignment)"
            )
            axes[0, 1].plot(
                wl_grid,
                y_bg_aligned,
                "r-",
                label=f"{label_str} (shift={shift_nm:.2f}nm)",
            )
            axes[0, 1].plot(wl_grid, y_meas, "b-", alpha=0.7, label="Measurement")

            # Add text with parameters
            param_text = f"Shift: {shift_nm:.2f} nm\nScale: {scale_a:.3f}\nOffset: {offset_b:.1f}"
            axes[0, 1].text(
                0.02,
                0.98,
                param_text,
                transform=axes[0, 1].transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )
        else:
            axes[0, 1].text(
                0.5,
                0.5,
                "No background subtraction\nperformed",
                ha="center",
                va="center",
                transform=axes[0, 1].transAxes,
                fontsize=12,
                alpha=0.6,
            )

        axes[0, 1].set_xlabel("Wavelength (nm)")
        axes[0, 1].set_ylabel("Intensity")
        handles, labels = axes[0, 1].get_legend_handles_labels()
        if labels:
            axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Subtracted result
        axes[1, 0].set_title("Background Subtracted", fontweight="bold")
        axes[1, 0].plot(wl_grid, y_sub, "g-", linewidth=2, label="Subtracted")
        axes[1, 0].axhline(0, color="k", linestyle="-", alpha=0.3)
        axes[1, 0].set_xlabel("Wavelength (nm)")
        axes[1, 0].set_ylabel("Intensity")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Comparison overlay
        axes[1, 1].set_title("Before vs After", fontweight="bold")
        axes[1, 1].plot(wl_grid, y_meas, "b-", alpha=0.7, label="Original")
        axes[1, 1].plot(wl_grid, y_sub, "g-", linewidth=2, label="Subtracted")
        axes[1, 1].set_xlabel("Wavelength (nm)")
        axes[1, 1].set_ylabel("Intensity")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        self._save_and_show(
            fig, self._get_next_filename("background_subtraction"), title
        )

    def plot_continuum_removal(
        self,
        wl_grid: np.ndarray,
        y_sub: np.ndarray,
        baseline: np.ndarray,
        y_cr: np.ndarray,
        title: str = "Continuum Removal",
    ):
        """Step 4: Plot continuum removal process."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Original with baseline
        axes[0, 0].set_title("Original + Baseline", fontweight="bold")
        axes[0, 0].plot(
            wl_grid, y_sub, "b-", label="Background Subtracted", linewidth=2
        )
        axes[0, 0].plot(wl_grid, baseline, "r-", label="Fitted Baseline", linewidth=2)
        axes[0, 0].fill_between(
            wl_grid, baseline, y_sub, alpha=0.3, color="yellow", label="Continuum"
        )
        axes[0, 0].set_xlabel("Wavelength (nm)")
        axes[0, 0].set_ylabel("Intensity")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Baseline only
        axes[0, 1].set_title("Fitted Baseline", fontweight="bold")
        axes[0, 1].plot(wl_grid, baseline, "r-", linewidth=3)
        axes[0, 1].set_xlabel("Wavelength (nm)")
        axes[0, 1].set_ylabel("Baseline Intensity")
        axes[0, 1].grid(True, alpha=0.3)

        # Continuum removed
        axes[1, 0].set_title("Continuum Removed", fontweight="bold")
        axes[1, 0].plot(wl_grid, y_cr, "g-", linewidth=2, label="Continuum Removed")
        axes[1, 0].axhline(0, color="k", linestyle="-", alpha=0.3)
        axes[1, 0].set_xlabel("Wavelength (nm)")
        axes[1, 0].set_ylabel("Normalized Intensity")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Before vs After (use twin y-axes so "After" is visible on normalized scale)
        axes[1, 1].set_title("Before vs After", fontweight="bold")
        ax_before = axes[1, 1]
        ax_after = ax_before.twinx()
        line_before, = ax_before.plot(wl_grid, y_sub, "b-", alpha=0.7, label="Before")
        line_after, = ax_after.plot(wl_grid, y_cr, "g-", linewidth=2, label="After")
        ax_before.set_xlabel("Wavelength (nm)")
        ax_before.set_ylabel("Intensity")
        ax_after.set_ylabel("Normalized Intensity")
        ax_before.grid(True, alpha=0.3)
        # Combined legend
        lines = [line_before, line_after]
        labels = [line.get_label() for line in lines]
        ax_before.legend(lines, labels, loc="best")

        self._save_and_show(fig, self._get_next_filename("continuum_removal"), title)

    def plot_reference_lines(
        self,
        ref_lines: RefLines,
        species_filter: Optional[List[str]] = None,
        title: str = "Reference Line Database",
    ):
        """Step 6: Plot reference line database."""
        # Normalize by base species and scale intensities for visibility
        species_all = [str(s).strip().split()[0].upper() for s in ref_lines.species]
        species_set = set(species_all)
        if species_filter:
            filt = {str(s).strip().split()[0].upper() for s in species_filter}
            species_set = species_set.intersection(filt)

        species_list = sorted(list(species_set))[:12]  # cap for readability

        fig, ax = plt.subplots(1, 1, figsize=(15, 8))
        cmap_tab20 = plt.get_cmap("tab20")
        colors = cmap_tab20(np.linspace(0, 1, max(1, len(species_list))))
        y_offset = 0.0
        for i, sp in enumerate(species_list):
            mask = np.array([s == sp for s in species_all])
            wl = np.asarray(ref_lines.wavelength_nm[mask], dtype=float)
            inten = np.asarray(ref_lines.intensity[mask], dtype=float)
            if wl.size == 0:
                continue
            # Normalize intensities per species to max=1
            max_i = float(np.max(inten)) if np.any(np.isfinite(inten)) else 1.0
            if max_i <= 0:
                max_i = 1.0
            inten_norm = inten / max_i
            scale = 0.9  # vertical scale per species row
            ax.vlines(
                wl,
                y_offset,
                y_offset + scale * inten_norm,
                colors=colors[i % len(colors)],
                alpha=0.9,
                linewidth=2,
            )
            ax.text(
                float(np.median(wl)),
                y_offset + 1.05 * scale,
                sp,
                ha="center",
                va="bottom",
                fontweight="bold",
                color=colors[i % len(colors)],
            )
            y_offset += 1.4

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Species (stacked)")
        ax.set_title(
            f"Reference Lines ({len(species_list)} species)", fontweight="bold"
        )
        ax.grid(True, alpha=0.25)
        ax.set_ylim(-0.2, y_offset + 0.2)

        self._save_and_show(fig, self._get_next_filename("reference_lines"), title)

    def plot_templates(
        self,
        wl_grid: np.ndarray,
        templates: np.ndarray,
        species_names: List[str],
        title: str = "Generated Gaussian Templates",
    ):
        """Step 7: Plot generated templates."""
        n_species = len(species_names)
        n_cols = min(3, n_species)
        n_rows = (n_species + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows)
        )
        # Normalize axes to a flat list for consistent indexing
        if isinstance(axes, np.ndarray):
            axes = axes.ravel().tolist()
        else:
            axes = [axes]

        for i, species in enumerate(species_names):
            ax = axes[i]
            template = templates[:, i]

            ax.plot(wl_grid, template, linewidth=2, label=species)
            ax.fill_between(wl_grid, 0, template, alpha=0.3)
            ax.set_title(f"{species}", fontweight="bold")
            ax.set_xlabel("Wavelength (nm)")
            ax.set_ylabel("Normalized Intensity")
            ax.grid(True, alpha=0.3)

            # Add peak markers
            peaks = np.where(template > 0.1 * np.max(template))[0]
            if len(peaks) > 0:
                ax.scatter(
                    wl_grid[peaks],
                    template[peaks],
                    color="red",
                    s=30,
                    alpha=0.7,
                    zorder=5,
                )

        # Remove unused subplots to avoid constrained_layout/tight_layout warnings
        for j in range(n_species, len(axes)):
            try:
                fig.delaxes(axes[j])
            except Exception:
                # Fallback if delaxes is unavailable
                axes[j].remove()

        self._save_and_show(fig, self._get_next_filename("templates"), title)

    def plot_band_regions(
        self,
        wl_grid: np.ndarray,
        bands: Dict[str, List[Tuple[float, float]]],
        templates: Optional[np.ndarray] = None,
        species_names: Optional[List[str]] = None,
        title: str = "Species Band Regions",
    ):
        """Step 8: Plot band regions for species."""
        fig, ax = plt.subplots(1, 1, figsize=(15, 8))

        cmap_tab20 = plt.get_cmap("tab20")
        colors = cmap_tab20(np.linspace(0, 1, len(bands)))
        y_offset = 0

        for i, (species, band_list) in enumerate(bands.items()):
            color = colors[i]

            # Plot template if available
            if (
                templates is not None
                and species_names is not None
                and species in species_names
            ):
                idx = species_names.index(species)
                template = templates[:, idx]
                ax.plot(
                    wl_grid, template + y_offset, color=color, linewidth=2, alpha=0.7
                )

            # Plot band regions
            for j, (start, end) in enumerate(band_list):
                rect = patches.Rectangle(
                    (start, y_offset - 0.05),
                    end - start,
                    0.1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.3,
                )
                ax.add_patch(rect)

                # Label first band
                if j == 0:
                    ax.text(
                        start + (end - start) / 2,
                        y_offset + 0.15,
                        species,
                        ha="center",
                        va="bottom",
                        fontweight="bold",
                        color=color,
                    )

            y_offset += 0.5

        ax.set_xlabel("Wavelength (nm)")
        ax.set_ylabel("Species (stacked)")
        ax.set_title("Band Regions for Species Detection", fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.2, y_offset + 0.2)

        self._save_and_show(fig, self._get_next_filename("band_regions"), title)

    def plot_wavelength_shift_optimization(
        self,
        wl_grid: np.ndarray,
        y_original: np.ndarray,
        y_shifted: np.ndarray,
        best_shift: float,
        title: str = "Wavelength Shift Optimization",
    ):
        """Step 9: Plot wavelength shift optimization."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Original vs shifted
        axes[0].set_title("Wavelength Alignment", fontweight="bold")
        axes[0].plot(
            wl_grid, y_original, "b-", label="Original", alpha=0.7, linewidth=2
        )
        axes[0].plot(
            wl_grid,
            y_shifted,
            "r-",
            label=f"Shifted ({best_shift:+.2f} nm)",
            linewidth=2,
        )
        axes[0].set_xlabel("Wavelength (nm)")
        axes[0].set_ylabel("Intensity")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Difference
        axes[1].set_title("Difference (Shifted - Original)", fontweight="bold")
        diff = y_shifted - y_original
        axes[1].plot(wl_grid, diff, "g-", linewidth=2)
        axes[1].axhline(0, color="k", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Wavelength (nm)")
        axes[1].set_ylabel("Intensity Difference")
        axes[1].grid(True, alpha=0.3)

        # Shift info
        axes[2].set_title("Shift Parameters", fontweight="bold")
        axes[2].text(
            0.5,
            0.6,
            f"Optimal Shift: {best_shift:+.3f} nm",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
            fontsize=16,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

        shift_direction = (
            "Right (red-shift)"
            if best_shift > 0
            else "Left (blue-shift)"
            if best_shift < 0
            else "None"
        )
        axes[2].text(
            0.5,
            0.4,
            f"Direction: {shift_direction}",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
            fontsize=12,
        )

        axes[2].set_xlim(0, 1)
        axes[2].set_ylim(0, 1)
        axes[2].axis("off")

        self._save_and_show(fig, self._get_next_filename("wavelength_shift"), title)

    def plot_nnls_fitting(
        self,
        wl_grid: np.ndarray,
        y_observed: np.ndarray,
        y_fitted: np.ndarray,
        coeffs: np.ndarray,
        species_names: List[str],
        fit_R2: float,
        templates: Optional[np.ndarray] = None,
        title: str = "NNLS Fitting Results",
    ):
        """Step 10: Plot NNLS fitting results."""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 3, figure=fig)

        # Main fit plot
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.plot(
            wl_grid, y_observed, "b-", linewidth=2, label="Observed", alpha=0.8
        )
        ax_main.plot(wl_grid, y_fitted, "r-", linewidth=2, label="Fitted")
        ax_main.fill_between(
            wl_grid, y_observed, y_fitted, alpha=0.3, color="gray", label="Residual"
        )
        ax_main.set_title(f"NNLS Fit (R² = {fit_R2:.4f})", fontweight="bold")
        ax_main.set_xlabel("Wavelength (nm)")
        ax_main.set_ylabel("Intensity")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        # Residuals
        ax_resid = fig.add_subplot(gs[1, 0])
        residuals = y_observed - y_fitted
        ax_resid.plot(wl_grid, residuals, "g-", linewidth=2)
        ax_resid.axhline(0, color="k", linestyle="--", alpha=0.5)
        ax_resid.set_title("Residuals", fontweight="bold")
        ax_resid.set_xlabel("Wavelength (nm)")
        ax_resid.set_ylabel("Residual")
        ax_resid.grid(True, alpha=0.3)

        # Coefficient bar plot
        ax_coeff = fig.add_subplot(gs[1, 1])
        significant_mask = coeffs > 0.001 * np.max(
            coeffs
        )  # Show only significant coefficients
        sig_coeffs = coeffs[significant_mask]
        sig_names = [
            species_names[i] for i in range(len(species_names)) if significant_mask[i]
        ]

        if len(sig_coeffs) > 0:
            bars = ax_coeff.barh(range(len(sig_coeffs)), sig_coeffs)
            ax_coeff.set_yticks(range(len(sig_coeffs)))
            ax_coeff.set_yticklabels(sig_names)
            ax_coeff.set_title("Significant Coefficients", fontweight="bold")
            ax_coeff.set_xlabel("Coefficient Value")
            ax_coeff.grid(True, alpha=0.3)

            # Color bars by magnitude
            max_coeff = np.max(sig_coeffs)
            cmap_viridis = plt.get_cmap("viridis")
            for i, bar in enumerate(bars):
                bar.set_color(cmap_viridis(sig_coeffs[i] / max_coeff))
        else:
            ax_coeff.text(
                0.5,
                0.5,
                "No significant\ncoefficients",
                ha="center",
                va="center",
                transform=ax_coeff.transAxes,
                fontsize=12,
                alpha=0.6,
            )
            ax_coeff.set_title("Coefficients", fontweight="bold")

        # Fit quality metrics
        ax_metrics = fig.add_subplot(gs[1, 2])
        rmse = np.sqrt(np.mean(residuals**2))
        mae = np.mean(np.abs(residuals))
        max_error = np.max(np.abs(residuals))

        metrics_text = f"R² = {fit_R2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nMax Error = {max_error:.4f}"
        ax_metrics.text(
            0.5,
            0.5,
            metrics_text,
            ha="center",
            va="center",
            transform=ax_metrics.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )
        ax_metrics.set_title("Fit Quality", fontweight="bold")
        ax_metrics.axis("off")

        # Individual contributions (if templates available)
        if templates is not None and len(sig_names) > 0:
            ax_contrib = fig.add_subplot(gs[2, :])
            ax_contrib.plot(
                wl_grid, y_observed, "k-", linewidth=2, label="Observed", alpha=0.7
            )

            cmap_tab10 = plt.get_cmap("tab10")
            colors = cmap_tab10(np.linspace(0, 1, len(sig_names)))
            y_cumulative = np.zeros_like(wl_grid)

            for i, (name, coeff) in enumerate(zip(sig_names, sig_coeffs)):
                if name in species_names:
                    idx = species_names.index(name)
                    contribution = templates[:, idx] * coeff
                    y_cumulative += contribution
                    ax_contrib.fill_between(
                        wl_grid,
                        y_cumulative - contribution,
                        y_cumulative,
                        alpha=0.7,
                        color=colors[i],
                        label=f"{name} ({coeff:.3f})",
                    )

            ax_contrib.set_title("Individual Species Contributions", fontweight="bold")
            ax_contrib.set_xlabel("Wavelength (nm)")
            ax_contrib.set_ylabel("Cumulative Intensity")
            ax_contrib.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax_contrib.grid(True, alpha=0.3)

        self._save_and_show(fig, self._get_next_filename("nnls_fitting"), title)

    def plot_detection_results(
        self,
        detection_result: DetectionResult,
        bands: Optional[Dict[str, List[Tuple[float, float]]]] = None,
        title: str = "Species Detection Results",
    ):
        """Step 11: Plot final detection results."""
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 2, figure=fig)

        # Main spectrum with detections
        ax_main = fig.add_subplot(gs[0, :])
        ax_main.plot(
            detection_result.wl_grid,
            detection_result.y_cr,
            "b-",
            linewidth=2,
            label="Observed",
            alpha=0.8,
        )
        ax_main.plot(
            detection_result.wl_grid,
            detection_result.y_fit,
            "r-",
            linewidth=2,
            label="Fitted",
        )

        # Highlight bands for detected species
        if bands and detection_result.present:
            cmap_set3 = plt.get_cmap("Set3")
            colors = cmap_set3(np.linspace(0, 1, len(detection_result.present)))
            for i, detection in enumerate(detection_result.present):
                species = detection["species"]
                if species in bands:
                    for start, end in bands[species]:
                        ax_main.axvspan(
                            start,
                            end,
                            alpha=0.2,
                            color=colors[i],
                            label=f"{species} bands" if i == 0 else "",
                        )

        ax_main.set_title(
            f"Species Detection Results (R² = {detection_result.fit_R2:.4f})",
            fontweight="bold",
        )
        ax_main.set_xlabel("Wavelength (nm)")
        ax_main.set_ylabel("Intensity")
        ax_main.legend()
        ax_main.grid(True, alpha=0.3)

        # Detection scores
        ax_scores = fig.add_subplot(gs[1, 0])
        if detection_result.present:
            species_names = [d["species"] for d in detection_result.present]
            scores = [d["score"] for d in detection_result.present]

            bars = ax_scores.barh(range(len(species_names)), scores)
            ax_scores.set_yticks(range(len(species_names)))
            ax_scores.set_yticklabels(species_names)
            ax_scores.set_title("Detection Scores", fontweight="bold")
            ax_scores.set_xlabel("Score (FVE)")
            ax_scores.grid(True, alpha=0.3)

            # Color bars by score
            max_score = max(scores)
            cmap_viridis = plt.get_cmap("viridis")
            for i, bar in enumerate(bars):
                bar.set_color(cmap_viridis(scores[i] / max_score))
        else:
            ax_scores.text(
                0.5,
                0.5,
                "No species\ndetected",
                ha="center",
                va="center",
                transform=ax_scores.transAxes,
                fontsize=14,
                alpha=0.6,
            )
            ax_scores.set_title("Detection Scores", fontweight="bold")

        # Band hits
        ax_bands = fig.add_subplot(gs[1, 1])
        if detection_result.present:
            species_names = [d["species"] for d in detection_result.present]
            band_hits = [d.get("bands_hit", 0) for d in detection_result.present]

            bars = ax_bands.barh(range(len(species_names)), band_hits)
            ax_bands.set_yticks(range(len(species_names)))
            ax_bands.set_yticklabels(species_names)
            ax_bands.set_title("Band Hits", fontweight="bold")
            ax_bands.set_xlabel("Number of Bands")
            ax_bands.grid(True, alpha=0.3)

            # Color bars by hits
            max_hits = max(band_hits) if band_hits else 1
            cmap_plasma = plt.get_cmap("plasma")
            for i, bar in enumerate(bars):
                bar.set_color(cmap_plasma(band_hits[i] / max_hits))
        else:
            ax_bands.text(
                0.5,
                0.5,
                "No species\ndetected",
                ha="center",
                va="center",
                transform=ax_bands.transAxes,
                fontsize=14,
                alpha=0.6,
            )
            ax_bands.set_title("Band Hits", fontweight="bold")

        # Summary table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis("off")

        if detection_result.present:
            table_data = []
            headers = ["Species", "Score", "Coefficient", "Bands Hit", "FVE"]

            for detection in detection_result.present:
                row = [
                    detection["species"],
                    f"{detection['score']:.4f}",
                    f"{detection.get('coeff', 0):.4f}",
                    str(detection.get("bands_hit", 0)),
                    f"{detection.get('fve', detection['score']):.4f}",
                ]
                table_data.append(row)

            table = ax_table.table(
                cellText=table_data, colLabels=headers, cellLoc="center", loc="center"
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 1.5)

            # Color table rows
            cmap_set3 = plt.get_cmap("Set3")
            for i in range(len(table_data)):
                for j in range(len(headers)):
                    table[(i + 1, j)].set_facecolor(cmap_set3(i / len(table_data)))

            ax_table.set_title("Detection Summary", fontweight="bold", pad=20)
        else:
            ax_table.text(
                0.5,
                0.5,
                "No species detected above threshold",
                ha="center",
                va="center",
                transform=ax_table.transAxes,
                fontsize=14,
                alpha=0.6,
            )
            ax_table.set_title("Detection Summary", fontweight="bold")

        self._save_and_show(fig, self._get_next_filename("detection_results"), title)

    def plot_analysis_summary(
        self, analysis_result, title: str = "Complete Analysis Summary"
    ):
        """Step 12: Plot complete analysis summary."""
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(4, 3, figure=fig)

        # Overall detections
        ax_overall = fig.add_subplot(gs[0, :])
        if analysis_result.detections:
            species = [d["species"] for d in analysis_result.detections]
            scores = [d["score"] for d in analysis_result.detections]

            bars = ax_overall.barh(range(len(species)), scores)
            ax_overall.set_yticks(range(len(species)))
            ax_overall.set_yticklabels(species)
            ax_overall.set_title(
                "Overall Species Detections (Aggregated)",
                fontweight="bold",
                fontsize=16,
            )
            ax_overall.set_xlabel("Aggregated Score")
            ax_overall.grid(True, alpha=0.3)

            # Color bars
            max_score = max(scores)
            cmap_viridis = plt.get_cmap("viridis")
            for i, bar in enumerate(bars):
                bar.set_color(cmap_viridis(scores[i] / max_score))
                # Add score labels
                ax_overall.text(
                    scores[i] + max_score * 0.01,
                    i,
                    f"{scores[i]:.3f}",
                    va="center",
                    ha="left",
                    fontweight="bold",
                )
        else:
            ax_overall.text(
                0.5,
                0.5,
                "No species detected",
                ha="center",
                va="center",
                transform=ax_overall.transAxes,
                fontsize=16,
                alpha=0.6,
            )
            ax_overall.set_title(
                "Overall Species Detections", fontweight="bold", fontsize=16
            )

        # Single-run R²
        ax_r2 = fig.add_subplot(gs[1, 0])
        r2_value = float(analysis_result.metrics.get("fit_R2", 0.0))
        ax_r2.bar([0], [r2_value], alpha=0.7)
        ax_r2.set_title("Fit Quality", fontweight="bold")
        ax_r2.set_xlabel("")
        ax_r2.set_ylabel("R²")
        ax_r2.set_ylim(0, 1)
        ax_r2.grid(True, alpha=0.3)
        ax_r2.text(
            0,
            r2_value + 0.02,
            f"{r2_value:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

        # Detection statistics
        ax_stats = fig.add_subplot(gs[1, 1])
        if analysis_result.detections:
            scores = [d["score"] for d in analysis_result.detections]
            coeffs = [d.get("coeff", 0) for d in analysis_result.detections]
            bands = [d.get("bands_hit", 0) for d in analysis_result.detections]

            stats_data = {
                "Scores": {
                    "mean": np.mean(scores),
                    "std": np.std(scores),
                    "max": np.max(scores),
                },
                "Coeffs": {
                    "mean": np.mean(coeffs),
                    "std": np.std(coeffs),
                    "max": np.max(coeffs),
                },
                "Bands": {
                    "mean": np.mean(bands),
                    "std": np.std(bands),
                    "max": np.max(bands),
                },
            }

            stats_text = ""
            for metric, values in stats_data.items():
                stats_text += f"{metric}:\n  Mean: {values['mean']:.3f}\n  Std: {values['std']:.3f}\n  Max: {values['max']:.3f}\n\n"

            ax_stats.text(
                0.05,
                0.95,
                stats_text,
                transform=ax_stats.transAxes,
                verticalalignment="top",
                fontfamily="monospace",
                bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
            )
        else:
            ax_stats.text(
                0.5,
                0.5,
                "No detection\nstatistics",
                ha="center",
                va="center",
                transform=ax_stats.transAxes,
                fontsize=12,
                alpha=0.6,
            )

        ax_stats.set_title("Detection Statistics", fontweight="bold")
        ax_stats.axis("off")

        # Processing metrics
        ax_proc = fig.add_subplot(gs[1, 2])
        proc_text = f"R²: {float(analysis_result.metrics.get('fit_R2', 0.0)):.4f}\n"
        proc_text += f"Total Detections: {len(analysis_result.detections)}"

        ax_proc.text(
            0.5,
            0.5,
            proc_text,
            ha="center",
            va="center",
            transform=ax_proc.transAxes,
            fontsize=12,
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )
        ax_proc.set_title("Processing Metrics", fontweight="bold")
        ax_proc.axis("off")

        # Single-run spectrum panel
        if getattr(analysis_result, "detection", None):
            det = analysis_result.detection
            ax_group = fig.add_subplot(gs[2, 0])
            ax_group.plot(
                det.wl_grid, det.y_cr, "b-", alpha=0.7, linewidth=1, label="Observed"
            )
            ax_group.plot(det.wl_grid, det.y_fit, "r-", linewidth=1, label="Fitted")
            ax_group.set_title(f"Detection (R²={det.fit_R2:.3f})", fontweight="bold")
            ax_group.set_xlabel("Wavelength (nm)")
            ax_group.set_ylabel("Intensity")
            ax_group.legend(fontsize=8)
            ax_group.grid(True, alpha=0.3)
            if det.present:
                species_text = ", ".join([d["species"] for d in det.present[:3]])
                if len(det.present) > 3:
                    species_text += "..."
                ax_group.text(
                    0.02,
                    0.98,
                    species_text,
                    transform=ax_group.transAxes,
                    verticalalignment="top",
                    fontsize=8,
                    alpha=0.8,
                    bbox=dict(boxstyle="round", facecolor="yellow", alpha=0.6),
                )

        self._save_and_show(fig, self._get_next_filename("analysis_summary"), title)

    def create_processing_flowchart(self, title: str = "Spectrum Analysis Pipeline"):
        """Create a flowchart showing the complete processing pipeline."""
        fig, ax = plt.subplots(1, 1, figsize=(16, 12))

        # Define processing steps
        steps = [
            "Raw Spectra\n(Measurements + Backgrounds)",
            "Averaging & Grid\nInterpolation",
            "Background\nSubtraction",
            "Continuum\nRemoval",
            "Spectrum\nGrouping",
            "Reference Line\nDatabase",
            "Template\nGeneration",
            "Band Region\nDefinition",
            "Wavelength\nAlignment",
            "NNLS\nFitting",
            "Species\nDetection",
            "Results\nAggregation",
        ]

        # Position steps in a flowchart layout
        positions = [
            (2, 10),
            (2, 8),
            (2, 6),
            (2, 4),
            (2, 2),  # Main preprocessing chain
            (6, 8),
            (6, 6),
            (6, 4),  # Template preparation
            (10, 6),
            (10, 4),
            (10, 2),
            (10, 0),  # Detection chain
        ]

        # Draw boxes and text
        boxes = []
        for i, (step, pos) in enumerate(zip(steps, positions)):
            x, y = pos
            # Color code by processing stage
            if i < 5:  # Preprocessing
                color = "lightblue"
            elif i < 8:  # Template preparation
                color = "lightgreen"
            else:  # Detection
                color = "lightcoral"

            box = patches.FancyBboxPatch(
                (x - 0.8, y - 0.4),
                1.6,
                0.8,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor="black",
                linewidth=2,
            )
            ax.add_patch(box)
            boxes.append(box)

            ax.text(
                x, y, step, ha="center", va="center", fontweight="bold", fontsize=10
            )

        # Draw arrows
        arrows = [
            # Main preprocessing chain
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            # Template chain
            (5, 6),
            (6, 7),
            # Detection chain
            (4, 8),
            (7, 8),
            (8, 9),
            (9, 10),
            (10, 11),
        ]

        for start_idx, end_idx in arrows:
            start_pos = positions[start_idx]
            end_pos = positions[end_idx]

            ax.annotate(
                "",
                xy=end_pos,
                xytext=start_pos,
                arrowprops=dict(arrowstyle="->", lw=2, color="darkblue"),
            )

        # Add stage labels
        ax.text(
            2,
            11,
            "PREPROCESSING",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
        ax.text(
            6,
            9,
            "TEMPLATE\nPREPARATION",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.8),
        )
        ax.text(
            10,
            7,
            "DETECTION",
            ha="center",
            va="center",
            fontsize=14,
            fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.8),
        )

        ax.set_xlim(-1, 12)
        ax.set_ylim(-1, 12)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            "Spectrum Analysis Processing Pipeline",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )

        self._save_and_show(fig, self._get_next_filename("processing_flowchart"), title)

    def plot_full_analysis(
        self,
        *,
        measurements: Sequence[SpectrumLike],
        backgrounds: Optional[Sequence[SpectrumLike]],
        pre,
        ref: RefLines,
        templates: np.ndarray,
        species_names: List[str],
        bands: Dict[str, List[Tuple[float, float]]],
        coeffs: np.ndarray,
        y_fit: np.ndarray,
        R2: float,
        det: DetectionResult,
        result,
        config,
    ) -> None:
        """Render the complete set of analysis visualizations in order.

        This orchestrates all plotting steps that are typically called at the end of the analysis.
        """
        # 01: Pipeline flowchart
        self.create_processing_flowchart("Spectrum Analysis Pipeline Overview")

        # 02: Raw spectra
        self.plot_raw_spectra(measurements, backgrounds, "Raw Input Spectra")

        # 03: Averaged spectra
        if getattr(pre, "avg_meas", None) is not None:
            self.plot_averaged_spectra(
                pre.avg_meas,
                getattr(pre, "avg_bg", None),
                pre.wl_grid,
                "Averaged Spectra",
            )

        # 04: Background subtraction
        self.plot_background_subtraction(
            pre.wl_grid,
            pre.y_meas,
            getattr(pre, "y_bg_interp", None),
            pre.y_sub,
            {"bg_shift_nm": 0.0, "bg_scale_a": 1.0, "bg_offset_b": 0.0},
            "Background Subtraction",
        )

        # 05: Continuum removal
        self.plot_continuum_removal(
            pre.wl_grid, pre.y_sub, pre.baseline, pre.y_cr, "Continuum Removal"
        )

        # 06: Reference lines
        self.plot_reference_lines(
            ref, getattr(config, "species", None), "Reference Line Database"
        )

        # 07: Templates
        self.plot_templates(
            pre.wl_grid, templates, species_names, "Generated Gaussian Templates"
        )

        # 08: Band regions
        self.plot_band_regions(
            pre.wl_grid, bands, templates, species_names, "Species Band Regions"
        )

        # 09: Wavelength alignment optimization
        try:
            # Import locally to avoid any potential import cycles at module import time
            from ._detect import _search_best_shift  # type: ignore

            best_shift, y_shifted = _search_best_shift(
                pre.wl_grid, pre.y_cr, templates, species_names, config
            )
            self.plot_wavelength_shift_optimization(
                pre.wl_grid,
                pre.y_cr,
                y_shifted,
                best_shift,
                "Wavelength Shift Optimization",
            )
        except Exception:
            # If shift computation is unavailable, skip this visualization gracefully
            pass

        # 10: NNLS fitting details
        self.plot_nnls_fitting(
            pre.wl_grid,
            pre.y_cr,
            y_fit,
            coeffs,
            species_names,
            R2,
            templates=templates,
            title="NNLS Fitting Results",
        )

        # 11: Detection results view
        self.plot_detection_results(det, bands=bands, title="Species Detection Results")

        # 12: Overall summary
        self.plot_analysis_summary(result, "Complete Analysis Summary")
