from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple, Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import lsq_linear
from scipy.ndimage import percentile_filter, gaussian_filter1d
from skimage.registration import phase_cross_correlation
from pybaselines import Baseline

from ..types import Detection, DetectionResult, References, Signal, Templates


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def _ensure_signal(signal: Signal) -> Signal:
    if signal.wavelength.ndim != 1 or signal.intensity.ndim != 1:
        raise ValueError("Signal arrays must be 1-D")
    if signal.wavelength.size != signal.intensity.size:
        raise ValueError("Signal wavelength/intensity size mismatch")
    return signal


def _overlap_domain(signals: Sequence[Signal]) -> Tuple[float, float]:
    wl_min = max(float(np.min(sig.wavelength)) for sig in signals)
    wl_max = min(float(np.max(sig.wavelength)) for sig in signals)
    if wl_max <= wl_min:
        raise ValueError("Signals have no overlapping wavelength range")
    return wl_min, wl_max


def _average_signals(signals: Sequence[Signal], n_points: int) -> Signal:
    if len(signals) == 1:
        return signals[0].copy()
    wl_min, wl_max = _overlap_domain(signals)
    grid = np.linspace(wl_min, wl_max, n_points)
    intens = [
        np.interp(grid, sig.wavelength, sig.intensity).astype(float)
        for sig in signals
    ]
    avg_intensity = np.mean(np.stack(intens, axis=0), axis=0)
    meta = {}
    return Signal(wavelength=grid, intensity=avg_intensity, meta=meta)


def average_signals(signals: Sequence[Signal], *, n_points: int = 1000) -> Signal:
    if not signals:
        raise ValueError("No signals provided")
    for sig in signals:
        _ensure_signal(sig)
    return _average_signals(signals, n_points=n_points)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def trim(signal: Signal, *, min_nm: Optional[float] = None, max_nm: Optional[float] = None) -> Signal:
    _ensure_signal(signal)
    wl = signal.wavelength
    mask = np.ones_like(wl, dtype=bool)
    if min_nm is not None:
        mask &= wl >= float(min_nm)
    if max_nm is not None:
        mask &= wl <= float(max_nm)
    if not np.any(mask):
        return signal.copy()
    return Signal(wavelength=wl[mask], intensity=signal.intensity[mask], meta=dict(signal.meta))


def mask(signal: Signal, *, intervals: Sequence[Tuple[float, float]]) -> Signal:
    _ensure_signal(signal)
    if not intervals:
        return signal.copy()
    wl = signal.wavelength
    iy = signal.intensity.copy()
    for a, b in intervals:
        lo = float(min(a, b))
        hi = float(max(a, b))
        mask_idx = (wl >= lo) & (wl <= hi)
        iy[mask_idx] = 0.0
    sig = signal.copy()
    sig.intensity = iy
    return sig


def resample(
    signal: Signal,
    *,
    n_points: Optional[int] = None,
    step_nm: Optional[float] = None,
) -> Signal:
    _ensure_signal(signal)
    wl = signal.wavelength
    if n_points is None and (step_nm is None or step_nm <= 0):
        return signal.copy()
    if step_nm and step_nm > 0:
        new_grid = np.arange(float(wl[0]), float(wl[-1]), float(step_nm))
        if new_grid.size < 2:
            new_grid = np.linspace(float(wl[0]), float(wl[-1]), max(2, n_points or 2))
    else:
        n = int(n_points or max(1000, wl.size))
        new_grid = np.linspace(float(wl[0]), float(wl[-1]), n)
    new_grid = np.asarray(new_grid, dtype=float)
    new_intensity = np.asarray(np.interp(new_grid, wl, signal.intensity), dtype=float)
    return Signal(wavelength=new_grid, intensity=new_intensity, meta=dict(signal.meta))


def subtract_background(
    signal: Signal,
    background: Optional[Signal],
    *,
    align: bool = False,
) -> Signal:
    _ensure_signal(signal)
    if background is None:
        return signal.copy()
    bg = resample(background, n_points=signal.wavelength.size)
    bg_intensity = np.interp(signal.wavelength, bg.wavelength, bg.intensity)
    if align:
        shift_val, _ = _estimate_shift(signal.wavelength, signal.intensity, bg_intensity)
        shifted = _shift_vector(bg_intensity, signal.wavelength, shift_val)
        diff = np.maximum(signal.intensity - shifted, 0.0)
        meta = dict(signal.meta)
        meta.setdefault("background", {})["shift_nm"] = shift_val
        return Signal(wavelength=signal.wavelength.copy(), intensity=diff, meta=meta)
    diff = np.maximum(signal.intensity - bg_intensity, 0.0)
    return Signal(wavelength=signal.wavelength.copy(), intensity=diff, meta=dict(signal.meta))


def continuum_remove_arpls(signal: Signal, *, strength: float) -> Signal:
    _ensure_signal(signal)
    wl = signal.wavelength
    y = signal.intensity
    lam = (y.size / 200.0) ** 2 * 1e6 * (0.5 + strength)
    baseline, _ = Baseline(wl).arpls(y, lam=lam, max_iter=25, tol=1e-2)
    cr = np.maximum(y - baseline, 0.0)
    meta = dict(signal.meta)
    meta.setdefault("continuum", {}).update({"strategy": "arpls", "baseline_lam": lam})
    return Signal(wavelength=wl.copy(), intensity=cr, meta=meta)


def continuum_remove_rolling(
    signal: Signal,
    *,
    strength: float,
    baseline: Optional[np.ndarray] = None,
) -> Signal:
    _ensure_signal(signal)
    wl = signal.wavelength
    y = signal.intensity

    source = y if baseline is None else y - baseline
    wl = np.asarray(wl, float)
    source = np.asarray(source, float)
    step = (wl[-1] - wl[0]) / max(1, wl.size - 1)
    win_nm = np.interp(strength, [0.0, 1.0], [18.0, 45.0])
    W = int(max(5, 2 * round(win_nm / step) + 1))
    y_s = gaussian_filter1d(source, sigma=max(1, W // 6), mode="nearest")
    q = float(np.interp(strength, [0.0, 1.0], [0.82, 0.92])) * 100.0
    base_q = percentile_filter(y_s, percentile=q, size=W, mode="nearest")
    lam = (source.size / 200.0) ** 2 * 1e6 * (0.5 + strength)
    base_lo, _ = Baseline(wl).arpls(source, lam=lam, max_iter=25, tol=1e-2)
    base = np.maximum(base_q, base_lo)

    cr = np.maximum(source - base, 0.0)
    if baseline is not None:
        combined = np.maximum(y - base, 0.0)
    else:
        combined = cr

    meta = dict(signal.meta)
    meta.setdefault("continuum", {}).update({"strategy": "rolling", "baseline_lam": lam})
    return Signal(wavelength=wl.copy(), intensity=combined, meta=meta)


# ---------------------------------------------------------------------------
# Templates
# ---------------------------------------------------------------------------

def build_templates(
    signal: Signal,
    references: References,
    *,
    fwhm_nm: float,
    species_filter: Optional[Iterable[str]] = None,
    bands_kwargs: Optional[Dict[str, Any]] = None,
) -> Templates:
    _ensure_signal(signal)
    if not references.lines:
        raise ValueError("References are empty")
    bands_kwargs = bands_kwargs or {}
    matrix_cols: List[np.ndarray] = []
    species_list: List[str] = []
    for species, (lines_wl, lines_int) in references.lines.items():
        if species_filter and species not in species_filter:
            continue
        tpl = _gaussian_broaden(signal.wavelength, lines_wl, lines_int, fwhm_nm)
        matrix_cols.append(tpl)
        species_list.append(species)
    if not matrix_cols:
        raise ValueError("No templates generated; species filter may be too restrictive")
    matrix = np.stack(matrix_cols, axis=1)
    bands = _build_bands_index(references, species_list, fwhm_nm=fwhm_nm, **bands_kwargs)
    return Templates(matrix=matrix, species=species_list, bands=bands)


# ---------------------------------------------------------------------------
# Alignment + detection
# ---------------------------------------------------------------------------

def shift_search(
    signal: Signal,
    templates: Templates,
    *,
    spread_nm: float,
    iterations: int,
) -> Signal:
    _ensure_signal(signal)
    if spread_nm <= 0 or iterations <= 0:
        return signal.copy()
    y = signal.intensity
    wl = signal.wavelength
    S = templates.matrix
    best_signal = signal.copy()
    current_y = y.copy()
    for i in range(iterations):
        step_factor = 1.0 / float(2 ** i)
        shift, shifted = _search_best_shift(wl, current_y, S, spread_nm=spread_nm, step_factor=step_factor)
        if shift != 0:
            best_y = shifted
            best_signal = Signal(wavelength=wl.copy(), intensity=shifted, meta=dict(signal.meta))
            best_signal.meta.setdefault("alignment", {}).setdefault("shifts_nm", []).append(shift)
        current_y = shifted
    return best_signal


def fwhm_search(
    signal: Signal,
    references: References,
    *,
    initial_fwhm_nm: float,
    spread_nm: float,
    iterations: int,
    species_filter: Optional[Iterable[str]] = None,
) -> Templates:
    """Iteratively re-build templates around the signal to optimize FWHM."""
    _ensure_signal(signal)
    if iterations <= 0 or spread_nm <= 0:
        templates = build_templates(
            signal,
            references,
            fwhm_nm=initial_fwhm_nm,
            species_filter=species_filter,
        )
        templates.meta.setdefault("fwhm_search", {}).update({"best_fwhm_nm": float(initial_fwhm_nm)})
        return templates

    best_fwhm = float(max(initial_fwhm_nm, 1e-6))
    best_templates = build_templates(
        signal,
        references,
        fwhm_nm=best_fwhm,
        species_filter=species_filter,
    )
    best_score = _quick_fit_r2(signal.intensity, best_templates.matrix)

    current_spread = float(spread_nm)
    current_fwhm = best_fwhm

    for _ in range(iterations):
        lo = max(1e-6, current_fwhm - current_spread)
        hi = max(lo + 1e-6, current_fwhm + current_spread)
        candidates = np.linspace(lo, hi, num=5)
        for cand in candidates:
            tpl = build_templates(
                signal,
                references,
                fwhm_nm=float(cand),
                species_filter=species_filter,
            )
            score = _quick_fit_r2(signal.intensity, tpl.matrix)
            if score > best_score:
                best_score = score
                best_templates = tpl
                current_fwhm = float(cand)
        current_spread *= 0.5

    best_templates.meta.setdefault("fwhm_search", {}).update(
        {"best_fwhm_nm": float(current_fwhm), "score": float(best_score)}
    )
    return best_templates


def detect_nnls(
    signal: Signal,
    templates: Templates,
    *,
    presence_threshold: float,
    min_bands: int,
) -> DetectionResult:
    _ensure_signal(signal)
    y = signal.intensity
    S = templates.matrix
    species_names = templates.species

    coeffs = _solve_nnls(S, y)
    y_fit = S @ coeffs

    ss_tot = float(np.sum(y ** 2) + 1e-12)
    base_sse = float(np.sum((y - y_fit) ** 2))

    detections: List[Detection] = []
    per_species_fve = []
    for i, sp in enumerate(species_names):
        if coeffs[i] <= 0:
            per_species_fve.append(0.0)
            continue
        y_without = y_fit - S[:, i] * coeffs[i]
        lift = float(np.sum((y - y_without) ** 2) - base_sse)
        fve = lift / ss_tot
        per_species_fve.append(fve)
        bands_hit = _bands_hit(signal.wavelength, y, y_fit, S[:, i], coeffs[i], templates.bands.get(sp, []))
        if fve >= presence_threshold and (bands_hit >= min_bands or not templates.bands.get(sp)):
            detections.append(
                Detection(
                    species=sp,
                    score=float(fve),
                    meta={
                        "coeff": float(coeffs[i]),
                        "bands_hit": int(bands_hit),
                        "fve": float(fve),
                    },
                )
            )

    detections.sort(key=lambda d: d.score, reverse=True)
    result_meta: Dict[str, Any] = {
        "fit_R2": 1.0 - base_sse / ss_tot,
        "coefficients": {sp: float(coeffs[i]) for i, sp in enumerate(species_names)},
        "per_species_fve": {
            species_names[i]: float(per_species_fve[i]) for i in range(len(species_names))
        },
    }
    return DetectionResult(signal=signal.copy(), detections=detections, meta=result_meta)


# ---------------------------------------------------------------------------
# Internal utilities reused from legacy pipeline
# ---------------------------------------------------------------------------

def _gaussian_broaden(
    grid: NDArray[np.float64],
    lines_wavelength: NDArray[np.float64],
    lines_intensity: NDArray[np.float64],
    fwhm_nm: float,
) -> NDArray[np.float64]:
    if lines_wavelength.size == 0:
        return np.zeros_like(grid, dtype=float)
    sigma = float(fwhm_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    diffs = (grid[:, None] - lines_wavelength[None, :]) / sigma
    tpl = np.exp(-0.5 * (diffs ** 2)) @ lines_intensity
    tpl_area = float(np.trapezoid(tpl, grid) + 1e-12)
    if tpl_area > 0:
        tpl = tpl / tpl_area
    tpl_norm = float(np.linalg.norm(tpl) + 1e-12)
    if tpl_norm > 0:
        tpl = tpl / tpl_norm
    return tpl.astype(float)


def _build_bands_index(
    references: References,
    species_list: Sequence[str],
    *,
    fwhm_nm: float,
    merge_distance_factor: float = 1.5,
    margin_factor: float = 1.25,
    min_width_nm: float = 0.0,
    max_bands_per_species: Optional[int] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    bands: Dict[str, List[Tuple[float, float]]] = {}
    merge_dist = max(0.0, merge_distance_factor * fwhm_nm)
    margin = max(0.0, margin_factor * fwhm_nm)

    for sp in species_list:
        wl_lines, intensity = references.lines.get(sp, (np.array([]), np.array([])))
        wl = np.sort(wl_lines.astype(float))
        inten = intensity.astype(float)
        if wl.size == 0:
            bands[sp] = []
            continue
        clusters: List[Tuple[int, int, float]] = []
        start = 0
        for i in range(1, wl.size):
            if wl[i] - wl[i - 1] > merge_dist:
                score = float(np.sum(inten[start:i]))
                clusters.append((start, i - 1, score))
                start = i
        score = float(np.sum(inten[start: wl.size]))
        clusters.append((start, wl.size - 1, score))
        clusters.sort(key=lambda item: item[2], reverse=True)
        if max_bands_per_species is not None and max_bands_per_species > 0:
            clusters = clusters[:max_bands_per_species]
        intervals: List[Tuple[float, float]] = []
        for s_idx, e_idx, _ in clusters:
            a = float(wl[s_idx]) - margin
            b = float(wl[e_idx]) + margin
            if min_width_nm > 0 and (b - a) < min_width_nm:
                c = 0.5 * (a + b)
                half = 0.5 * min_width_nm
                a, b = c - half, c + half
            intervals.append((a, b))
        if not intervals:
            bands[sp] = []
            continue
        intervals.sort()
        merged: List[Tuple[float, float]] = []
        cur_a, cur_b = intervals[0]
        for a, b in intervals[1:]:
            if a <= cur_b:
                cur_b = max(cur_b, b)
            else:
                merged.append((cur_a, cur_b))
                cur_a, cur_b = a, b
        merged.append((cur_a, cur_b))
        bands[sp] = merged
    return bands


def _shift_vector(x: NDArray[np.float64], wl: NDArray[np.float64], shift_nm: float) -> NDArray[np.float64]:
    return np.interp(wl, wl - shift_nm, x, left=float(x[0]), right=float(x[-1]))


def _estimate_shift(
    wl: NDArray[np.float64],
    y: NDArray[np.float64],
    bg: NDArray[np.float64],
) -> Tuple[float, NDArray[np.float64]]:
    # Reuse continuum removal for stability; operate in sample space for phase correlation
    baseline = Baseline(np.arange(len(y)))
    y_cr, _ = baseline.arpls(y, lam=1e5)
    bg_cr, _ = baseline.arpls(bg, lam=1e5)
    shifts, _, _ = phase_cross_correlation(y_cr, bg_cr, upsample_factor=20)
    shift_samples = float(np.ravel(shifts)[0])
    step_nm = (wl[-1] - wl[0]) / max(1, wl.size - 1)
    shift_nm = shift_samples * step_nm
    return shift_nm, bg


def _search_best_shift(
    wl_grid: NDArray[np.float64],
    y: NDArray[np.float64],
    S: NDArray[np.float64],
    *,
    spread_nm: float,
    step_factor: float,
) -> Tuple[float, NDArray[np.float64]]:
    step_nm = (wl_grid[-1] - wl_grid[0]) / max(1, wl_grid.size - 1)
    step_nm = max(step_nm * step_factor, step_nm)
    max_shift = max(0.0, float(spread_nm))
    shifts = np.arange(-max_shift, max_shift + 1e-12, step_nm)
    best_R2, best_shift, best_y = -np.inf, 0.0, y
    for s in shifts:
        y_shifted = _shift_vector(y, wl_grid, s)
        coeffs = _solve_nnls(S, y_shifted)
        y_fit = S @ coeffs
        ss_res = float(np.sum((y_shifted - y_fit) ** 2))
        ss_tot = float(np.sum(y_shifted ** 2) + 1e-12)
        R2 = 1.0 - ss_res / ss_tot
        if R2 > best_R2:
            best_R2, best_shift, best_y = R2, s, y_shifted
    return float(best_shift), best_y


def _solve_nnls(S: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
    if S.size == 0:
        return np.zeros(S.shape[1], dtype=float)
    sol = lsq_linear(S, y, bounds=(0.0, np.inf), method="trf")
    coeffs = np.asarray(sol.x, dtype=float)
    col_norms = np.linalg.norm(S, axis=0)
    eff = coeffs * col_norms
    thr = 1e-9 * float(np.linalg.norm(y))
    coeffs[eff < thr] = 0.0
    return coeffs


def _quick_fit_r2(y: NDArray[np.float64], S: NDArray[np.float64], lam: float = 1e-3) -> float:
    if S.size == 0:
        return float("-inf")
    n = S.shape[1]
    S_aug = np.vstack([S, np.sqrt(lam) * np.eye(n)])
    y_aug = np.concatenate([y, np.zeros(n)])
    sol = lsq_linear(S_aug, y_aug, bounds=(0.0, np.inf), method="trf")
    y_fit = S @ np.asarray(sol.x, dtype=float)
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum(y ** 2) + 1e-12)
    return 1.0 - ss_res / ss_tot


def _bands_hit(
    wl_grid: NDArray[np.float64],
    y: NDArray[np.float64],
    y_fit: NDArray[np.float64],
    template_column: NDArray[np.float64],
    coeff: float,
    intervals: Sequence[Tuple[float, float]],
) -> int:
    if not intervals:
        return 0
    hits = 0
    for a, b in intervals:
        mask = (wl_grid >= a) & (wl_grid <= b)
        if not np.any(mask):
            continue
        band_tot = float(np.sum(y[mask] ** 2) + 1e-12)
        fit_with = y_fit[mask]
        fit_without = (y_fit - template_column * coeff)[mask]
        lift = (float(np.sum((y[mask] - fit_without) ** 2)) - float(np.sum((y[mask] - fit_with) ** 2))) / band_tot
        if lift >= 0.10:
            hits += 1
    return hits
