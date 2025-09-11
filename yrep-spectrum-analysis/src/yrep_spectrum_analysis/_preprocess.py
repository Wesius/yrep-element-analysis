from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np
from scipy.ndimage import percentile_filter, gaussian_filter1d
from pybaselines import Baseline
from skimage.registration import phase_cross_correlation


from .types import AnalysisConfig, Instrument, PreprocessResult, Spectrum, SpectrumLike


def _as_arrays(items: Sequence[SpectrumLike]) -> List[Spectrum]:
    out: List[Spectrum] = []
    for it in items:
        wl = np.asarray(it.wavelength, dtype=float)
        iy = np.asarray(it.intensity, dtype=float)
        out.append(Spectrum(wavelength=wl, intensity=iy))
    return out


def _uniform_grid_from_data(
    wavelength_nm: np.ndarray, instr: Instrument, n_min: int = 1000
) -> np.ndarray:
    wl_min = float(np.min(wavelength_nm))
    wl_max = float(np.max(wavelength_nm))
    if instr.grid_step_nm and instr.grid_step_nm > 0:
        step = float(instr.grid_step_nm)
    else:
        diffs = np.diff(wavelength_nm)
        step = (
            float(np.median(diffs[diffs > 0]))
            if diffs.size
            else (wl_max - wl_min) / float(n_min)
        )
        if not np.isfinite(step) or step <= 0:
            step = (wl_max - wl_min) / float(n_min)
    approx_n = int(np.ceil((wl_max - wl_min) / step))
    if approx_n < n_min:
        step = (wl_max - wl_min) / float(max(1, n_min))
    return np.arange(wl_min, wl_max, step)


def continuum_upper_envelope(
    wl: np.ndarray,
    y: np.ndarray,
    strength: float = 0.6,
    mode: str = "subtract",  # "divide" for multiplicative continuum, "subtract" otherwise
) -> Tuple[np.ndarray, np.ndarray]:
    wl = np.asarray(wl, float)
    y = np.asarray(y, float)

    # grid step
    step = (wl[-1] - wl[0]) / max(1, wl.size - 1)

    # window and quantile scale with "strength"
    win_nm = np.interp(strength, [0.0, 1.0], [18.0, 45.0])
    W = int(max(5, 2 * round(win_nm / step) + 1))

    # pre-smooth so single spikes don't define the envelope
    y_s = gaussian_filter1d(y, sigma=max(1, W // 6), mode="nearest")

    # high local quantile ≈ upper envelope (ignore tallest spikes)
    q = float(np.interp(strength, [0.0, 1.0], [0.82, 0.92])) * 100.0
    base_q = percentile_filter(y_s, percentile=q, size=W, mode="nearest")

    # safety: don't go below a robust lower-envelope spline
    lam = (y.size / 200.0) ** 2 * 1e6 * (0.5 + strength)
    base_lo, _ = Baseline(wl).arpls(y, lam=lam, max_iter=25, tol=1e-2)

    base = np.maximum(base_q, base_lo)

    if mode == "divide":
        # Clamp divisor to >= 1.0 so dividing by <1 never amplifies values
        floor = max(1.0, float(np.percentile(base, 5.0)))
        cr = y / np.maximum(base, floor)
        cr = cr / max(np.percentile(cr, 99.5), 1e-9)  # robust normalize
    else:
        # fully subtract the estimated continuum baseline
        cr = y - base

    return cr.astype(float), base.astype(float)


def _continuum_remove(
    wl: np.ndarray, y: np.ndarray, strength: float, override_fn=None, *, strategy: str = "arpls"
) -> Tuple[np.ndarray, np.ndarray]:
    if override_fn is not None:
        y_cr, baseline = override_fn(wl, y, strength)
        return np.asarray(y_cr, dtype=float), np.asarray(baseline, dtype=float)

    # Strategy selection
    st = (strategy or "arpls").lower().strip()
    if st not in {"arpls", "rolling", "both"}:
        st = "arpls"

    # ARPLS baseline on original signal
    N = int(y.size)
    lam2 = (N / 200.0) ** 2 * 1e6 * (0.5 + strength)
    base_ar, _ = Baseline(wl).arpls(y, lam=lam2, max_iter=25, tol=1e-2)
    y_after_arpls = y - base_ar

    if st == "arpls":
        cr = y_after_arpls
        _continuum_remove._last_div = (None, None)  # type: ignore
        return cr.astype(float), np.asarray(base_ar, dtype=float)

    # Rolling upper-envelope on ARPLS residual
    cr_env, base_upper = continuum_upper_envelope(
        wl, y_after_arpls, strength=strength, mode="subtract"
    )
    _continuum_remove._last_div = (y_after_arpls.astype(float), base_upper.astype(float))  # type: ignore

    if st == "rolling":
        # If only rolling, use the envelope baseline as the baseline output for visualization
        return cr_env.astype(float), np.asarray(base_upper, dtype=float)

    # both: ARPLS then rolling; keep ARPLS baseline for top-row visualization
    return cr_env.astype(float), np.asarray(base_ar, dtype=float)


def _register_and_subtract(
    wl: np.ndarray,
    std_y: np.ndarray,
    bg_y: np.ndarray,
    instr: Instrument,
    override_fn=None,
) -> Tuple[np.ndarray, dict]:
    if override_fn is not None:
        return override_fn(wl, std_y, wl, bg_y, instr)

    # Continuum remove for registration stability
    std_cr, _ = _continuum_remove(wl, std_y, strength=0.4, override_fn=None, strategy="both")
    bg_cr, _ = _continuum_remove(wl, bg_y, strength=0.4, override_fn=None, strategy="both")

    shifts, _, _ = phase_cross_correlation(std_cr, bg_cr, upsample_factor=20)
    shift_samples = float(np.ravel(shifts)[0])
    step_nm = (wl[-1] - wl[0]) / max(1, wl.size - 1)
    shift_nm = shift_samples * step_nm
    bg_shifted = np.interp(
        wl, wl - shift_nm, bg_y, left=float(bg_y[0]), right=float(bg_y[-1])
    )

    # Pure registration: subtract shifted background without scaling/offset
    sub = std_y - bg_shifted
    return sub.astype(float), {
        "bg_shift_nm": shift_nm,
        "bg_scale_a": 1.0,
        "bg_offset_b": 0.0,
    }


def _detect_left_spike_wavelength(probe: Spectrum) -> float:
    dy = np.diff(probe.intensity)
    dw = np.diff(probe.wavelength)
    slope = dy / (dw + 1e-12)
    abs_slope = np.abs(slope)
    thr = np.percentile(abs_slope, 95.0)
    idx = int(np.argmax(abs_slope > thr)) if slope.size > 0 else 0
    wl_min_keep = float(probe.wavelength[max(0, idx)])
    if dw.size > 0:
        wl_min_keep += float(np.median(dw))
    return wl_min_keep


def _detect_right_spike_wavelength(probe: Spectrum) -> float:
    dy = np.diff(probe.intensity)
    dw = np.diff(probe.wavelength)
    slope = dy / (dw + 1e-12)
    abs_slope = np.abs(slope)
    thr = np.percentile(abs_slope, 95.0)
    if slope.size > 0 and np.any(abs_slope > thr):
        idx_r = int(np.where(abs_slope > thr)[0][-1])
    else:
        idx_r = slope.size - 1 if slope.size > 0 else 0
    j = min(idx_r + 1, int(probe.wavelength.size - 1))
    wl_max_keep = float(probe.wavelength[max(0, j)])
    if dw.size > 0:
        wl_max_keep -= float(np.median(dw))
    return wl_max_keep


def _compute_trim_bounds(meas: List[Spectrum], ts) -> Tuple[float, float]:
    wl_min_keep = float("-inf")
    wl_max_keep = float("inf")
    if getattr(ts, "min_wavelength_nm", None) is not None:
        wl_min_keep = float(ts.min_wavelength_nm)
    elif getattr(ts, "auto_trim_left", False):
        try:
            wl_min_keep = _detect_left_spike_wavelength(meas[0])
        except Exception:
            wl_min_keep = float(meas[0].wavelength[0])
    if getattr(ts, "max_wavelength_nm", None) is not None:
        wl_max_keep = float(ts.max_wavelength_nm)
    elif getattr(ts, "auto_trim_right", False):
        try:
            wl_max_keep = _detect_right_spike_wavelength(meas[0])
        except Exception:
            wl_max_keep = float(meas[0].wavelength[-1])
    return wl_min_keep, wl_max_keep


def _trim_spectrum(spec: Spectrum, wl_min_keep: float, wl_max_keep: float) -> Spectrum:
    left_ok = spec.wavelength >= wl_min_keep if np.isfinite(wl_min_keep) else True
    right_ok = spec.wavelength <= wl_max_keep if np.isfinite(wl_max_keep) else True
    mask = np.asarray(left_ok & right_ok, dtype=bool)
    if not np.any(mask):
        return spec
    return Spectrum(wavelength=spec.wavelength[mask], intensity=spec.intensity[mask])


    # Average measurement and background on overlap grid
def _average_spectra(
        specs: List[Spectrum], n_points: Optional[int] = None
) -> Spectrum:
    if len(specs) == 1:
        return specs[0]
    all_wl = [s.wavelength for s in specs]
    wl_min = max(w.min() for w in all_wl)
    wl_max = min(w.max() for w in all_wl)
    if wl_min >= wl_max:
        raise ValueError("spectra have no overlapping wavelength range")
    n_points = n_points or 1000
    wl_common = np.linspace(wl_min, wl_max, n_points)
    intens = [np.interp(wl_common, s.wavelength, s.intensity) for s in specs]
    avg = np.mean(np.stack(intens, axis=0), axis=0)
    return Spectrum(wavelength=wl_common, intensity=avg)

def preprocess(
    measurements: Sequence[SpectrumLike],
    backgrounds: Optional[Sequence[SpectrumLike]],
    config: AnalysisConfig,
) -> PreprocessResult:
    meas = _as_arrays(measurements)
    bg = _as_arrays(backgrounds) if backgrounds else []

    # Optional explicit/automatic wavelength trims (left/right)
    ts = config
    if (
        getattr(ts, "min_wavelength_nm", None) is not None
        or getattr(ts, "max_wavelength_nm", None) is not None
        or getattr(ts, "auto_trim_left", False)
        or getattr(ts, "auto_trim_right", False)
    ):
        wl_min_keep, wl_max_keep = _compute_trim_bounds(meas, ts)
        meas = [_trim_spectrum(s, wl_min_keep, wl_max_keep) for s in meas]
        bg = [_trim_spectrum(s, wl_min_keep, wl_max_keep) for s in bg] if bg else []


    avg_meas = (
        _average_spectra(
            meas,
            n_points=max(
                config.instrument.__dict__.get("average_n_points", 1000), 1000
            ),
        )
        if len(meas) > 1
        else meas[0]
    )
    avg_bg = (
        _average_spectra(
            bg,
            n_points=max(
                config.instrument.__dict__.get("average_n_points", 1000), 1000
            ),
        )
        if bg
        else None
    )

    # Grid: restrict to measurement–background overlap to avoid edge artifacts
    if avg_bg is not None:
        wl_min_common = max(
            float(np.min(avg_meas.wavelength)), float(np.min(avg_bg.wavelength))
        )
        wl_max_common = min(
            float(np.max(avg_meas.wavelength)), float(np.max(avg_bg.wavelength))
        )
        if wl_max_common > wl_min_common:
            meas_wl_for_grid = avg_meas.wavelength[
                (avg_meas.wavelength >= wl_min_common)
                & (avg_meas.wavelength <= wl_max_common)
            ]
            wl_src = (
                meas_wl_for_grid if meas_wl_for_grid.size >= 2 else avg_meas.wavelength
            )
        else:
            wl_src = avg_meas.wavelength
    else:
        wl_src = avg_meas.wavelength
    wl_grid = _uniform_grid_from_data(wl_src, config.instrument, n_min=1000)
    y_meas = np.asarray(
        np.interp(wl_grid, avg_meas.wavelength, avg_meas.intensity), dtype=float
    )
    y_bg = (
        np.asarray(np.interp(wl_grid, avg_bg.wavelength, avg_bg.intensity), dtype=float)
        if avg_bg is not None
        else None
    )
    # Zero-pad tails to avoid large negative edge subtraction
    if y_bg is not None and avg_bg is not None:
        left_mask = wl_grid < float(np.min(avg_bg.wavelength))
        right_mask = wl_grid > float(np.max(avg_bg.wavelength))
        if np.any(left_mask):
            y_bg[left_mask] = float(avg_bg.intensity[0])
        if np.any(right_mask):
            y_bg[right_mask] = float(avg_bg.intensity[-1])

    # Background strategy (single pass)
    if y_bg is not None:
        if getattr(config, "align_background", False):
            # Explicitly align if requested
            y_sub, _ = _register_and_subtract(
                wl_grid,
                y_meas,
                y_bg,
                config.instrument,
                override_fn=config.background_fn,
            )
        else:
            y_sub, _ = (
                (y_meas - y_bg).astype(float),
                {"bg_shift_nm": 0.0, "bg_scale_a": 1.0, "bg_offset_b": 0.0},
            )
    else:
        y_sub, _ = (
            y_meas.astype(float),
            {"bg_shift_nm": 0.0, "bg_scale_a": 0.0, "bg_offset_b": 0.0},
        )

    # Enforce non-negativity on background-subtracted signal BEFORE continuum
    y_sub = np.maximum(y_sub, 0.0)
    # Continuum on non-negative signal
    y_cr, baseline = _continuum_remove(
        wl_grid,
        y_sub,
        strength=float(config.baseline_strength),
        override_fn=config.continuum_fn,
        strategy=str(getattr(config, "continuum_strategy", "both")),
    )

    # Final non-negativity
    y_cr = np.maximum(y_cr, 0.0)

    # Apply optional wavelength mask: zero out y_cr in specified intervals
    mask_intervals = getattr(config, "mask", None)
    if mask_intervals:
        for a, b in mask_intervals:
            try:
                a_f = float(a)
                b_f = float(b)
            except Exception:
                continue
            lo = min(a_f, b_f)
            hi = max(a_f, b_f)
            if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                m = (wl_grid >= lo) & (wl_grid <= hi)
                if np.any(m):
                    y_cr[m] = 0.0

    return PreprocessResult(
        wl_grid=wl_grid,
        y_meas=y_meas,
        y_sub=y_sub,
        y_cr=y_cr,
        baseline=baseline,
        y_div=getattr(_continuum_remove, "_last_div", (None, None))[0],
        baseline_div=getattr(_continuum_remove, "_last_div", (None, None))[1],
        y_bg_interp=y_bg,
        avg_meas=(avg_meas.wavelength, avg_meas.intensity),
        avg_bg=(avg_bg.wavelength, avg_bg.intensity) if avg_bg is not None else None,
    )
