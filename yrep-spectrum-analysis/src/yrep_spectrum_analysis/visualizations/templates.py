from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from ..types import Signal, Templates


def visualize_templates(
    signal: Signal,
    templates: Templates,
    title: str = "Generated Templates",
    save_path: Optional[str] = None,
    show: bool = True,
    max_species: int = 10,
) -> None:
    """
    Visualize the generated templates against the signal.
    Plots the signal and the top N templates (by max intensity or just first N).

    Args:
        signal: The reference signal used to generate templates.
        templates: The generated Templates object.
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to show the plot.
        max_species: Maximum number of species templates to overlay.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot signal
    ax.plot(
        signal.wavelength,
        signal.intensity,
        "k-",
        label="Signal",
        linewidth=1,
        alpha=0.6,
    )

    # Plot templates
    # Templates matrix is (n_points, n_species)
    n_species = len(templates.species)
    limit = min(n_species, max_species)
    
    # Create a color cycle
    cmap = plt.colormaps.get_cmap("tab10")
    colors = cmap(np.linspace(0, 1, limit))

    for i in range(limit):
        species_name = templates.species[i]
        tpl_spectrum = templates.matrix[:, i]
        
        # Scale template for visibility if needed, 
        # but usually they are normalized. 
        # Let's scale it to match the signal's max for visualization purposes
        scale_factor = np.max(signal.intensity) / (np.max(tpl_spectrum) + 1e-12)
        scaled_tpl = tpl_spectrum * scale_factor * 0.8  # 80% of max height

        ax.plot(
            signal.wavelength,
            scaled_tpl,
            linestyle="--",
            linewidth=1,
            color=colors[i],
            label=f"{species_name} (scaled)",
        )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)
