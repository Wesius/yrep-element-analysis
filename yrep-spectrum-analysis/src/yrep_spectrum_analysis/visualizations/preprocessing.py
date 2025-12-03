from typing import Optional

import matplotlib.pyplot as plt

from ..types import Signal


def visualize_preprocessing(
    original: Signal,
    background: Optional[Signal],
    subtracted: Signal,
    arpls_baseline: Optional[Signal],
    rolling_baseline: Optional[Signal],
    final: Signal,
    title: str = "Preprocessing Steps",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Visualize the complete preprocessing pipeline.
    
    Plots:
    1. Original vs Background
    2. Subtracted Signal
    3. ARPLS Continuum Removal (Original, Baseline, Corrected)
    4. Rolling Continuum Removal (Original, Baseline, Corrected)
    5. Final Processed Signal

    Args:
        original: Raw signal.
        background: Background signal (optional).
        subtracted: Signal after background subtraction.
        arpls_baseline: Baseline estimated by ARPLS (optional).
        rolling_baseline: Baseline estimated by rolling ball (optional).
        final: Final processed signal.
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to show the plot.
    """
    rows = 5
    fig, axes = plt.subplots(rows, 1, figsize=(12, 20), sharex=True)

    # 1. Original vs Background
    ax1 = axes[0]
    ax1.plot(original.wavelength, original.intensity, "k-", label="Original Signal", alpha=0.8)
    if background:
        ax1.plot(background.wavelength, background.intensity, "r--", label="Background", alpha=0.7)
    ax1.set_ylabel("Intensity")
    ax1.set_title(f"{title} - 1. Background Subtraction")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Subtracted Signal
    ax2 = axes[1]
    ax2.plot(subtracted.wavelength, subtracted.intensity, "b-", label="Background Subtracted", alpha=0.8)
    ax2.set_ylabel("Intensity")
    ax2.set_title(f"{title} - 2. Signal After Background Removal")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. ARPLS Continuum Removal
    # Line 1: Input to this step (Subtracted)
    # Line 2: Baseline (ARPLS)
    # Line 3: Corrected (Input - Baseline)
    ax3 = axes[2]
    ax3.plot(subtracted.wavelength, subtracted.intensity, "k-", label="Input Signal", alpha=0.6)
    if arpls_baseline:
        ax3.plot(arpls_baseline.wavelength, arpls_baseline.intensity, "r--", label="ARPLS Baseline", linewidth=1.5)
        # Calculate corrected manually for visualization consistency
        corrected_arpls = subtracted.intensity - arpls_baseline.intensity
        ax3.plot(subtracted.wavelength, corrected_arpls, "g-", label="Corrected (Signal - Baseline)", linewidth=1.2, alpha=0.8)
    
    ax3.set_ylabel("Intensity")
    ax3.set_title(f"{title} - 3. ARPLS Continuum Removal")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Rolling Continuum Removal
    # Note: Input to Rolling is typically the output of ARPLS
    # But here we visualize it relative to the signal it was calculated on.
    # Assuming 'final' is the result of Rolling on top of ARPLS.
    
    ax4 = axes[3]
    # Reconstruct the input to rolling: Final + Rolling Baseline
    if rolling_baseline:
        input_to_rolling = final.intensity + rolling_baseline.intensity
        ax4.plot(final.wavelength, input_to_rolling, "k-", label="Input (Post-ARPLS)", alpha=0.6)
        ax4.plot(rolling_baseline.wavelength, rolling_baseline.intensity, "m--", label="Rolling Baseline", linewidth=1.5)
        ax4.plot(final.wavelength, final.intensity, "b-", label="Corrected (Final)", linewidth=1.2)
    
    ax4.set_ylabel("Intensity")
    ax4.set_title(f"{title} - 4. Rolling Continuum Removal")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Final Signal (Redundant but keeps the flow clear)
    ax5 = axes[4]
    ax5.plot(final.wavelength, final.intensity, "b-", label="Final Processed Signal", linewidth=1.2)
    ax5.set_xlabel("Wavelength (nm)")
    ax5.set_ylabel("Intensity")
    ax5.set_title(f"{title} - 5. Final Result")
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()
    else:
        plt.close(fig)
