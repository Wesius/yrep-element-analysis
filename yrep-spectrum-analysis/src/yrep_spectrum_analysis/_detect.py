from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from scipy.optimize import lsq_linear

from .types import AnalysisConfig


def _shift_vector(x: np.ndarray, wl: np.ndarray, shift_nm: float) -> np.ndarray:
    # positive shift_nm moves features to the right
    return np.asarray(
        np.interp(wl, wl - shift_nm, x, left=float(x[0]), right=float(x[-1])),
        dtype=float,
    )


def _search_best_shift(
    wl_grid_nm: np.ndarray,
    y_cr: np.ndarray,
    S_templates: np.ndarray,
    species_names: Sequence[str],
    config: AnalysisConfig,
    *,
    step_factor: float = 1.0,
) -> Tuple[float, np.ndarray]:
    """Grid-search a small wavelength shift that maximizes R^2 of a quick NNLS fit."""
    if not np.isfinite(config.instrument.max_shift_nm) or config.instrument.max_shift_nm <= 0:
        return 0.0, y_cr
    # step equals grid step × step_factor
    step_nm = (wl_grid_nm[-1] - wl_grid_nm[0]) / max(1, wl_grid_nm.size - 1)
    step_nm = max(step_nm * step_factor, step_nm)
    shifts = np.arange(-config.instrument.max_shift_nm, config.instrument.max_shift_nm + 1e-12, step_nm)
    best_R2, best_shift, best_y = -np.inf, 0.0, y_cr
    # lightweight bounded LS to evaluate candidates
    for s in shifts:
        y_s = _shift_vector(y_cr, wl_grid_nm, s)
        # small ridge to stabilize the quick pass
        lam = 1e-3
        n = S_templates.shape[1]
        S_aug = np.vstack([S_templates, np.sqrt(lam) * np.eye(n)])
        y_aug = np.concatenate([y_s, np.zeros(n)])
        sol = lsq_linear(S_aug, y_aug, bounds=(0.0, np.inf), method="trf")
        y_fit = S_templates @ np.asarray(sol.x, dtype=float)
        ss_res = float(np.sum((y_s - y_fit) ** 2))
        ss_tot = float(np.sum(y_s ** 2) + 1e-12)
        R2 = 1.0 - ss_res / ss_tot
        if R2 > best_R2:
            best_R2, best_shift, best_y = R2, s, y_s
    return float(best_shift), best_y


def _presence_threshold_from_sensitivity(sensitivity: str) -> float:
    s = (sensitivity or "medium").lower()
    if s == "high":
        return 0.01
    if s == "low":
        return 0.05
    return 0.02


def _ridge_lambda(regularization: str) -> float:
    r = (regularization or "none").lower()
    if r == "light":
        return 1e-2
    if r == "strong":
        return 1e-1
    return 0.0


def nnls_detect(
    wl_grid_nm: np.ndarray,
    y_cr: np.ndarray,
    S_templates: np.ndarray,
    species_names: Sequence[str],
    *,
    bands: Optional[Dict[str, List[Tuple[float, float]]]] = None,
    config: Optional[AnalysisConfig] = None,
):
    cfg = config or AnalysisConfig()
    # Ridge via Tikhonov augmentation and bounded least squares
    lam = _ridge_lambda(cfg.regularization)
    S = S_templates
    y = y_cr
    # coarse wavelength alignment to references
    _, y = _search_best_shift(wl_grid_nm, y_cr, S, species_names, cfg)
    if lam and lam > 0:
        n = S.shape[1]
        S_aug = np.vstack([S, np.sqrt(lam) * np.eye(n)])
        y_aug = np.concatenate([y, np.zeros(n)])
    else:
        S_aug, y_aug = S, y
    sol = lsq_linear(S_aug, y_aug, bounds=(0.0, np.inf), method="trf")
    coeffs = np.asarray(sol.x, dtype=float)
    y_fit = S @ coeffs

    # R2
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum(y ** 2) + 1e-12)
    R2 = 1.0 - ss_res / ss_tot

    # Leave-one-out FVE per species
    base_sse = float(np.sum((y - y_fit) ** 2))
    per_species_fve: List[float] = []
    for i in range(len(species_names)):
        if coeffs[i] <= 0:
            per_species_fve.append(0.0)
            continue
        y_fit_wo = y_fit - S[:, i] * coeffs[i]
        lift_num = float(np.sum((y - y_fit_wo) ** 2) - base_sse)
        fve = lift_num / ss_tot
        per_species_fve.append(float(fve))

    # Band corroboration
    def _bands_hit(i: int) -> int:
        if not bands:
            return 0
        sp = species_names[i]
        hit = 0
        for (a, b) in bands.get(sp, []):
            mask = (wl_grid_nm >= a) & (wl_grid_nm <= b)
            if not np.any(mask):
                continue
            band_tot = float(np.sum(y[mask] ** 2) + 1e-12)
            y_fit_with = y_fit[mask]
            y_fit_wo_b = (y_fit - S[:, i] * coeffs[i])[mask]
            band_lift = (float(np.sum((y[mask] - y_fit_wo_b) ** 2)) - float(np.sum((y[mask] - y_fit_with) ** 2))) / band_tot
            if band_lift >= 0.10:
                hit += 1
        return hit

    # Present list
    thr = cfg.presence_threshold if cfg.presence_threshold is not None else _presence_threshold_from_sensitivity(cfg.sensitivity)
    present: List[Dict[str, float | int | str]] = []
    for i, sp in enumerate(species_names):
        if coeffs[i] <= 0:
            continue
        fve = per_species_fve[i]
        score = fve
        hits = _bands_hit(i)
        if score >= thr and (hits >= cfg.min_bands_required or not bands):
            present.append({
                "species": sp,
                "score": float(score),
                "fve": float(fve),
                "coeff": float(coeffs[i]),
                "bands_hit": int(hits),
            })

    present.sort(key=lambda d: float(d["score"]), reverse=True)

    per_species_scores = {species_names[i]: float(per_species_fve[i]) for i in range(len(species_names))}
    return coeffs, y_fit, present, per_species_scores, float(R2)


