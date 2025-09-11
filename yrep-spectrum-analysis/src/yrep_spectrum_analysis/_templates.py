from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd

from .types import RefLines


 


def _base(sym: str) -> str:
    return str(sym).strip().split()[0].upper() if sym else ""


def normalize_reference(
    ref: Union[
        "pd.DataFrame",
        Tuple[Sequence[str], np.ndarray, np.ndarray],
        Dict[str, Iterable],
    ],
) -> RefLines:
    if isinstance(ref, pd.DataFrame):
        df = ref
        if not {"wavelength_nm", "species", "intensity"}.issubset(df.columns):
            raise ValueError(
                "reference DataFrame must include columns: wavelength_nm, species, intensity"
            )
        species = df["species"].astype(str).tolist()
        wl = np.asarray(df["wavelength_nm"].to_numpy(), dtype=float)
        inten = np.asarray(df["intensity"].to_numpy(), dtype=float)
        return RefLines(species=species, wavelength_nm=wl, intensity=inten)

    if isinstance(ref, tuple) and len(ref) == 3:
        species, wl, inten = ref
        return RefLines(
            species=list(species),
            wavelength_nm=np.asarray(wl, dtype=float),
            intensity=np.asarray(inten, dtype=float),
        )

    if isinstance(ref, dict):
        species = list(ref.get("species", []))
        wl = np.asarray(ref.get("wavelength_nm", []), dtype=float)
        inten = np.asarray(ref.get("intensity", []), dtype=float)
        return RefLines(species=species, wavelength_nm=wl, intensity=inten)

    raise TypeError("Unsupported reference format for normalize_reference")


def build_templates(
    ref: RefLines,
    wl_grid_nm: np.ndarray,
    fwhm_nm: float,
    *,
    species_filter: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, List[str]]:
    # Build per-species templates via Gaussian broadening
    sigma = float(fwhm_nm) / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    names_all = sorted(set(_base(s) for s in ref.species))
    if species_filter is not None and len(species_filter) > 0:
        filt = {_base(s) for s in species_filter}
        names = [n for n in names_all if n in filt]
    else:
        names = names_all
    S_cols: List[np.ndarray] = []
    for sp in names:
        mask = np.asarray([_base(s) == sp for s in ref.species], dtype=bool)
        lines_wl = ref.wavelength_nm[mask]
        lines_w = ref.intensity[mask]
        if lines_wl.size == 0:
            S_cols.append(np.zeros_like(wl_grid_nm, dtype=float))
            continue
        # Fast broadening by summing normalized Gaussians
        tpl = np.zeros_like(wl_grid_nm, dtype=float)
        # Use vectorized broadcasting for efficiency
        # tpl += sum_i w_i * exp(-0.5*((x - mu_i)/sigma)^2)
        diffs = (wl_grid_nm[:, None] - lines_wl[None, :]) / sigma
        tpl = np.exp(-0.5 * (diffs**2)) @ lines_w
        area = float(np.trapezoid(tpl, wl_grid_nm) + 1e-12)
        if area > 0:
            tpl = tpl / area
        # Ensure templates have unit L2 norm for stable NNLS scaling
        norm2 = float(np.linalg.norm(tpl) + 1e-12)
        if norm2 > 0:
            tpl = tpl / norm2
        S_cols.append(tpl)
    S = (
        np.stack(S_cols, axis=1)
        if S_cols
        else np.zeros((wl_grid_nm.shape[0], 0), dtype=float)
    )
    return S, names


def build_bands_index(
    ref: RefLines,
    species_list: Sequence[str],
    fwhm_nm: float,
    *,
    merge_distance_factor: float = 1.5,
    margin_factor: float = 1.25,
    min_width_nm: float = 0.0,
    max_bands_per_species: Optional[int] = None,
) -> Dict[str, List[Tuple[float, float]]]:
    # Cluster lines per species using distance threshold (DBSCAN-like without dependency)
    bands: Dict[str, List[Tuple[float, float]]] = {}
    fwhm = float(fwhm_nm)
    merge_dist = max(0.0, merge_distance_factor * fwhm)
    margin = max(0.0, margin_factor * fwhm)

    for sp in species_list:
        spb = _base(sp)
        mask = np.asarray([_base(s) == spb for s in ref.species], dtype=bool)
        wl = np.sort(ref.wavelength_nm[mask].astype(float))
        inten = ref.intensity[mask].astype(float)
        if wl.size == 0:
            bands[sp] = []
            continue
        # cluster by gaps
        clusters: List[Tuple[int, int, float]] = []  # (start_idx, end_idx, score)
        start = 0
        for i in range(1, wl.size):
            if wl[i] - wl[i - 1] > merge_dist:
                score = float(np.sum(inten[start:i]))
                clusters.append((start, i - 1, score))
                start = i
        score = float(np.sum(inten[start : wl.size]))
        clusters.append((start, wl.size - 1, score))
        # rank and keep top-K
        clusters.sort(key=lambda t: t[2], reverse=True)
        if max_bands_per_species is not None and max_bands_per_species > 0:
            clusters = clusters[:max_bands_per_species]
        # build intervals with margin and min width
        intervals: List[Tuple[float, float]] = []
        for s_idx, e_idx, _ in clusters:
            a = float(wl[s_idx]) - margin
            b = float(wl[e_idx]) + margin
            if min_width_nm > 0 and (b - a) < min_width_nm:
                c = 0.5 * (a + b)
                half = 0.5 * min_width_nm
                a, b = c - half, c + half
            intervals.append((a, b))
        # merge overlaps only
        if not intervals:
            bands[sp] = []
        else:
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
