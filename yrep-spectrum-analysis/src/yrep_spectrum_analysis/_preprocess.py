from __future__ import annotations

from typing import List, Optional, Sequence, Tuple
import numpy as np
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


def _uniform_grid_from_data(wavelength_nm: np.ndarray, instr: Instrument, n_min: int = 1000) -> np.ndarray:
    wl_min = float(np.min(wavelength_nm))
    wl_max = float(np.max(wavelength_nm))
    if instr.grid_step_nm and instr.grid_step_nm > 0:
        step = float(instr.grid_step_nm)
    else:
        diffs = np.diff(wavelength_nm)
        step = float(np.median(diffs[diffs > 0])) if diffs.size else (wl_max - wl_min) / float(n_min)
        if not np.isfinite(step) or step <= 0:
            step = (wl_max - wl_min) / float(n_min)
    approx_n = int(np.ceil((wl_max - wl_min) / step))
    if approx_n < n_min:
        step = (wl_max - wl_min) / float(max(1, n_min))
    return np.arange(wl_min, wl_max, step)


def _continuum_remove(wl: np.ndarray, y: np.ndarray, strength: float, override_fn=None) -> Tuple[np.ndarray, np.ndarray]:
    if override_fn is not None:
        y_cr, baseline = override_fn(wl, y, strength)
        return np.asarray(y_cr, dtype=float), np.asarray(baseline, dtype=float)

    # Use pybaselines.ASLS with strength-adjusted lambda
    N = int(y.size)
    lam = max(1e4, (N / 200.0) ** 2 * 1e5)
    lam = lam * (0.5 + strength)  # scale by strength
    base, _ = Baseline(wl).asls(y, lam=lam, p=0.01)
    corrected = y - base

    norm = float(np.linalg.norm(corrected) + 1e-12)
    corrected = corrected / norm
    return corrected.astype(float), np.asarray(base, dtype=float)


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
    std_cr, _ = _continuum_remove(wl, std_y, strength=0.6, override_fn=None)
    bg_cr, _ = _continuum_remove(wl, bg_y, strength=0.6, override_fn=None)

    shifts, _, _ = phase_cross_correlation(std_cr, bg_cr, upsample_factor=20)
    shift_samples = float(np.ravel(shifts)[0])
    step_nm = (wl[-1] - wl[0]) / max(1, wl.size - 1)
    shift_nm = shift_samples * step_nm
    bg_shifted = np.interp(wl, wl - shift_nm, bg_y, left=float(bg_y[0]), right=float(bg_y[-1]))

    # Pure registration: subtract shifted background without scaling/offset
    sub = std_y - bg_shifted
    return sub.astype(float), {"bg_shift_nm": shift_nm, "bg_scale_a": 1.0, "bg_offset_b": 0.0}


def preprocess(
    measurements: Sequence[SpectrumLike],
    backgrounds: Optional[Sequence[SpectrumLike]],
    config: AnalysisConfig,
) -> PreprocessResult:
    meas = _as_arrays(measurements)
    bg = _as_arrays(backgrounds) if backgrounds else []

    # Optional left-trim to remove steep spike region
    if getattr(config, "min_wavelength_nm", None) is not None or getattr(config, "auto_trim_left", False):
        if getattr(config, "min_wavelength_nm", None) is not None:
            wl_min_keep = float(config.min_wavelength_nm)
        else:
            # Heuristic: detect steep left spike by derivative threshold on averaged measurement
            try:
                probe = meas[0]
                dy = np.diff(probe.intensity)
                dw = np.diff(probe.wavelength)
                slope = dy / (dw + 1e-12)
                # find first index where slope exceeds 95th percentile of absolute slope after 5th percentile of range
                abs_slope = np.abs(slope)
                thr = np.percentile(abs_slope, 95.0)
                idx = int(np.argmax(abs_slope > thr)) if slope.size > 0 else 0
                wl_min_keep = float(probe.wavelength[max(0, idx)])
                # Provide a small safety margin of one grid step
                if dw.size > 0:
                    wl_min_keep += float(np.median(dw))
            except Exception:
                wl_min_keep = float(meas[0].wavelength[0])
        def _trim(spec: Spectrum) -> Spectrum:
            mask = np.asarray(spec.wavelength >= wl_min_keep, dtype=bool)
            if not np.any(mask):
                return spec
            return Spectrum(wavelength=spec.wavelength[mask], intensity=spec.intensity[mask])
        meas = [_trim(s) for s in meas]
        bg = [_trim(s) for s in bg] if bg else []

    # Average measurement and background on overlap grid
    def _average_spectra(specs: List[Spectrum], n_points: Optional[int] = None) -> Spectrum:
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

    avg_meas = _average_spectra(meas, n_points=max(config.instrument.__dict__.get("average_n_points", 1000), 1000)) if len(meas) > 1 else meas[0]
    avg_bg = _average_spectra(bg, n_points=max(config.instrument.__dict__.get("average_n_points", 1000), 1000)) if bg else None

    # Grid: restrict to measurement–background overlap to avoid edge artifacts
    if avg_bg is not None:
        wl_min_common = max(float(np.min(avg_meas.wavelength)), float(np.min(avg_bg.wavelength)))
        wl_max_common = min(float(np.max(avg_meas.wavelength)), float(np.max(avg_bg.wavelength)))
        if wl_max_common > wl_min_common:
            meas_wl_for_grid = avg_meas.wavelength[(avg_meas.wavelength >= wl_min_common) & (avg_meas.wavelength <= wl_max_common)]
            wl_src = meas_wl_for_grid if meas_wl_for_grid.size >= 2 else avg_meas.wavelength
        else:
            wl_src = avg_meas.wavelength
    else:
        wl_src = avg_meas.wavelength
    wl_grid = _uniform_grid_from_data(wl_src, config.instrument, n_min=1000)
    y_meas = np.asarray(np.interp(wl_grid, avg_meas.wavelength, avg_meas.intensity), dtype=float)
    y_bg = np.asarray(np.interp(wl_grid, avg_bg.wavelength, avg_bg.intensity), dtype=float) if avg_bg is not None else None
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
            y_sub, _ = _register_and_subtract(wl_grid, y_meas, y_bg, config.instrument, override_fn=config.background_fn)
        else:
            y_sub, _ = (y_meas - y_bg).astype(float), {"bg_shift_nm": 0.0, "bg_scale_a": 1.0, "bg_offset_b": 0.0}
    else:
        y_sub, _ = y_meas.astype(float), {"bg_shift_nm": 0.0, "bg_scale_a": 0.0, "bg_offset_b": 0.0}

    # Enforce non-negativity on background-subtracted signal BEFORE continuum
    y_sub = np.maximum(y_sub, 0.0)
    # Continuum on non-negative signal
    y_cr, baseline = _continuum_remove(wl_grid, y_sub, strength=float(config.baseline_strength), override_fn=config.continuum_fn)

    # Final non-negativity
    y_cr = np.maximum(y_cr, 0.0)

    return PreprocessResult(
        wl_grid=wl_grid,
        y_meas=y_meas,
        y_sub=y_sub,
        y_cr=y_cr,
        baseline=baseline,
        y_bg_interp=y_bg,
        avg_meas=(avg_meas.wavelength, avg_meas.intensity),
        avg_bg=(avg_bg.wavelength, avg_bg.intensity) if avg_bg is not None else None,
    )


