from typing import Optional, Dict

import matplotlib.pyplot as plt
import numpy as np

from ..types import DetectionResult, Templates


def visualize_detection(
    result: DetectionResult,
    templates: Templates,
    title: str = "Detection Result",
    save_path: Optional[str] = None,
    show: bool = True,
    plot_residual: bool = True,
    top_n_components: int = 5,
) -> None:
    """
    Visualize the detection results, showing the original signal, the fitted model,
    and the contributions of individual species.

    Args:
        result: The DetectionResult object containing the signal and coefficients.
        templates: The Templates object used for detection.
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to show the plot.
        plot_residual: Whether to include a subplot for the residual.
        top_n_components: Number of top contributing species to plot individually.
    """
    signal = result.signal
    y_true = signal.intensity
    wl = signal.wavelength

    # Reconstruct the fit
    coeffs_map: Dict[str, float] = result.meta.get("coefficients", {})
    y_fit = np.zeros_like(y_true)
    
    components = []

    for i, species in enumerate(templates.species):
        coeff = coeffs_map.get(species, 0.0)
        if coeff > 0:
            component = templates.matrix[:, i] * coeff
            y_fit += component
            components.append((species, component, coeff))

    # Sort components by max intensity (or coefficient magnitude)
    components.sort(key=lambda x: np.max(x[1]), reverse=True)

    def create_plot(include_fit_line: bool, suffix: str):
        # Setup layout
        # Row 0: Main Detection Plot
        # Row 1: Residuals (optional)
        # Row 2: Bar Chart
        
        nrows = 2 if plot_residual else 1
        # Add one more row for the bar chart
        nrows += 1
        
        height_ratios = [3, 1, 2] if plot_residual else [3, 2]
        
        fig, axes = plt.subplots(
            nrows, 
            1, 
            figsize=(12, 10 if plot_residual else 8), 
            sharex=False, # Bar chart has different x-axis
            gridspec_kw={"height_ratios": height_ratios}
        )
        
        # If axes is a single object (nrows=1 which won't happen here), wrap it
        if nrows == 1:
             axes = [axes]
        
        # --- 1. Main Plot ---
        ax_main = axes[0]
        ax_main.plot(wl, y_true, "k-", label="Original Signal", linewidth=1, alpha=0.7)
        
        if include_fit_line:
            ax_main.plot(wl, y_fit, "r--", label="Total Fit", linewidth=1.5, alpha=0.9)
        
        # Plot top individual components
        colors = plt.cm.tab10(np.linspace(0, 1, top_n_components))
        for idx, (species, comp, coeff) in enumerate(components[:top_n_components]):
            ax_main.plot(
                wl, 
                comp, 
                linestyle="-", 
                linewidth=1, 
                alpha=0.6, 
                color=colors[idx],
                label=f"{species} (coeff={coeff:.2e})"
            )

        ax_main.set_ylabel("Intensity")
        ax_main.set_title(f"{title} (R2={result.meta.get('fit_R2', 0.0):.3f})")
        ax_main.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        ax_main.grid(True, alpha=0.3)
        
        current_row = 1
        
        # --- 2. Residuals (Optional) ---
        if plot_residual:
            residual = y_true - y_fit
            ax_res = axes[current_row]
            ax_res.plot(wl, residual, "b-", linewidth=0.8, label="Residual")
            ax_res.axhline(0, color="k", linewidth=0.5)
            ax_res.set_xlabel("Wavelength (nm)")
            ax_res.set_ylabel("Residual")
            ax_res.grid(True, alpha=0.3)
            ax_res.legend()
            # Share x-axis only between main and residual
            ax_res.sharex(ax_main)
            current_row += 1
        else:
            ax_main.set_xlabel("Wavelength (nm)")

        # --- 3. Bar Chart ---
        ax_bar = axes[current_row]
        if components:
            top_comps = components[:max(10, top_n_components)]
            names = [c[0] for c in top_comps]
            coeffs = [c[2] for c in top_comps]
            
            y_pos = np.arange(len(names))
            ax_bar.barh(y_pos, coeffs, align='center', color='steelblue')
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(names)
            ax_bar.invert_yaxis()
            ax_bar.set_xlabel('Coefficient Value')
            ax_bar.set_title('Species Contributions (Coefficients)')
            ax_bar.grid(True, axis='x', alpha=0.3)
        else:
            ax_bar.text(0.5, 0.5, "No detections", ha='center', va='center')
            ax_bar.axis('off')

        plt.tight_layout()

        if save_path:
            path = save_path
            if suffix:
                path = save_path.replace(".png", f"_{suffix}.png")
                if path == save_path: # fallback if no extension or weird extension
                     path = save_path + f"_{suffix}.png"
            plt.savefig(path, dpi=300)

        if show:
            plt.show()
        else:
            plt.close(fig)

    # Create only the version WITH the fit line
    create_plot(include_fit_line=True, suffix="") # empty suffix = normal filename
