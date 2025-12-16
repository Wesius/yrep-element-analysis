from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import glob
import numpy as np
import pandas as pd

from .types import References, Signal


def average_signals(signals: Sequence[Signal], *, n_points: int = 1000) -> Signal:
    if not signals:
        raise ValueError("No signals provided")
    if len(signals) == 1:
        return signals[0]
    wl_min = max(float(np.min(sig.wavelength)) for sig in signals)
    wl_max = min(float(np.max(sig.wavelength)) for sig in signals)
    if wl_max <= wl_min:
        raise ValueError("signals have no overlapping wavelength range")
    grid = np.linspace(wl_min, wl_max, n_points)
    intens = [np.interp(grid, sig.wavelength, sig.intensity) for sig in signals]
    avg_intensity = np.mean(np.stack(intens, axis=0), axis=0)
    return Signal(wavelength=grid, intensity=avg_intensity, meta={})


# ---------------------------------------------------------------------------
# Loading utilities
# ---------------------------------------------------------------------------

def load_txt_spectrum(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    wl: List[float] = []
    iy: List[float] = []
    data = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if not data:
                if line.startswith(">>>>>Begin Spectral Data<<<<<"):
                    data = True
                continue
            if "\t" in line:
                a, b = line.split("\t", 1)
                try:
                    wl.append(float(a))
                    iy.append(float(b))
                except Exception:
                    continue
    if not wl:
        raise ValueError(f"No data in {path}")
    return np.asarray(wl, dtype=float), np.asarray(iy, dtype=float)


def load_signals_from_dir(root: Path) -> List[Signal]:
    if not root.exists():
        return []
    signals: List[Signal] = []
    for fp in sorted(glob.glob(str(root / "*.txt"))):
        path = Path(fp)
        try:
            wl, iy = load_txt_spectrum(path)
        except ValueError:
            continue
        signals.append(Signal(wavelength=wl, intensity=iy, meta={"file": path.name}))
    return signals


def load_references(lists_dir: Path, *, element_only: bool = True) -> References:
    csvs = sorted(lists_dir.glob("*.csv"))
    lines: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for csv in csvs:
        df = pd.read_csv(csv)
        wl_col = None
        for cand in [
            "Wavelength (Å)",
            "Wavelength (A)",
            "Wavelength_A",
            "Wavelength_Angstrom",
            "Wavelength (nm)",
            "Wavelength_nm",
        ]:
            if cand in df.columns:
                wl_col = cand
                break
        spec_col = None
        for cand in ["Spectrum", "Species", "Element"]:
            if cand in df.columns:
                spec_col = cand
                break
        inten_col = None
        for cand in ["Intensity", "Relative Intensity", "rel_intensity", "Strength"]:
            if cand in df.columns:
                inten_col = cand
                break
        if not (wl_col and spec_col and inten_col):
            continue
        wl_nm = (
            (pd.to_numeric(df[wl_col], errors="coerce") / 10.0)
            if "nm" not in wl_col.lower()
            else pd.to_numeric(df[wl_col], errors="coerce")
        )
        species = df[spec_col].astype(str).str.strip().str.upper()
        if element_only:
            species = species.str.split().str[0]
        intensity = (
            df[inten_col]
            .astype(str)
            .str.extract(r"^\s*([0-9]+(?:\.[0-9]+)?)")[0]
            .astype(float)
        )
        frame = pd.DataFrame(
            {
                "wavelength_nm": wl_nm,
                "species": species,
                "intensity": intensity,
            }
        ).dropna()
        for sp, group in frame.groupby("species"):
            species_key = str(sp)
            wl_vals = group["wavelength_nm"].to_numpy(dtype=float)
            int_vals = group["intensity"].to_numpy(dtype=float)
            if species_key in lines:
                wl_prev, int_prev = lines[species_key]
                wl_vals = np.concatenate([wl_prev, wl_vals])
                int_vals = np.concatenate([int_prev, int_vals])
            sorter = np.argsort(wl_vals)
            lines[species_key] = (wl_vals[sorter], int_vals[sorter])
    if not lines:
        raise RuntimeError("No reference lines parsed")
    return References(lines=lines, meta={"element_only": element_only})


# ---------------------------------------------------------------------------
# Grouping & quality metrics
# ---------------------------------------------------------------------------

def group_signals(signals: Sequence[Signal], *, grid_points: int = 1000) -> List[List[Signal]]:
    if not signals:
        return []
    if len(signals) == 1:
        return [[signals[0]]]
    wl_min = max(float(np.min(sig.wavelength)) for sig in signals)
    wl_max = min(float(np.max(sig.wavelength)) for sig in signals)
    if not np.isfinite(wl_min) or not np.isfinite(wl_max) or wl_max <= wl_min:
        return [list(signals)]
    grid = np.linspace(wl_min, wl_max, grid_points)
    vectors = [np.interp(grid, s.wavelength, s.intensity).astype(float) for s in signals]
    avg = np.mean(np.stack(vectors, axis=0), axis=0)

    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        num = float(np.dot(u, v))
        den = float(np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        s = num / den
        return max(0.0, min(1.0, s))

    sims = np.asarray([_cosine(vec, avg) for vec in vectors], dtype=float)
    counts, edges = np.histogram(sims, bins=50)
    positive = counts > 0
    labels = np.zeros(sims.size, dtype=int)
    gid = 0
    for idx, active in enumerate(positive):
        if not active:
            continue
        a = float(edges[idx])
        b = float(edges[idx + 1])
        mask = (sims >= a) & (sims < b if idx < positive.size - 1 else sims <= b)
        if not np.any(mask):
            continue
        labels[mask] = gid
        if idx + 1 < positive.size and not positive[idx + 1]:
            gid += 1
    groups: List[List[Signal]] = []
    for g in sorted(set(labels.tolist())):
        idxs = [i for i, lbl in enumerate(labels) if lbl == g]
        groups.append([signals[i] for i in idxs])
    return groups


def signal_quality(signal: Signal) -> float:
    y = np.asarray(signal.intensity, dtype=float)
    if y.size < 10 or not np.all(np.isfinite(y)):
        return 0.0
    base, sigma, residual = _robust_baseline_and_sigma(y)
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    k = 6.0
    thr = base + k * sigma
    runs = _find_runs_above(y, thr)
    peaks = []
    n = y.size
    for a, b in runs:
        width = b - a + 1
        center = (a + b) // 2
        if 2 <= width <= 25 and 2 <= center < n - 2:
            height = y[center] - base[center]
            peaks.append((center, width, float(height)))
    n_peaks = len(peaks)
    max_peak = max((h for (_, _, h) in peaks), default=0.0)
    smooth = _moving_median(y, _odd(max(21, int(0.03 * y.size))))
    dyn = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    dyn_snr = dyn / (sigma + 1e-12)
    maxpk_snr = max_peak / (sigma + 1e-12)
    dyn_score = 1.0 - np.exp(-dyn_snr / 10.0)
    maxpk_score = 1.0 - np.exp(-maxpk_snr / 12.0)
    count_score = 1.0 - np.exp(-n_peaks / 3.0)
    score = 0.45 * dyn_score + 0.35 * maxpk_score + 0.20 * count_score
    return float(np.clip(score, 0.0, 1.0))


def is_junk_group(signals: Sequence[Signal], *, debug: bool = False) -> bool:
    if not signals:
        if debug:
            print("[junk] reason=empty_group")
        return True
    try:
        avg = average_signals(signals, n_points=1000)
    except Exception as exc:
        if debug:
            print(f"[junk] reason=average_failure error={exc}")
        return True
    y = np.asarray(avg.intensity, dtype=float)
    if y.size < 10 or not np.all(np.isfinite(y)):
        if debug:
            print("[junk] reason=invalid_average")
        return True
    base, sigma, residual = _robust_baseline_and_sigma(y)
    if not np.isfinite(sigma) or sigma <= 0:
        if debug:
            print("[junk] reason=noise_estimation_failed")
        return True
    k = 6.0
    thr = base + k * sigma
    runs = _find_runs_above(y, thr)
    peaks = []
    n = y.size
    for a, b in runs:
        width = b - a + 1
        center = (a + b) // 2
        if 2 <= width <= 25 and 2 <= center < n - 2:
            height = y[center] - base[center]
            peaks.append((center, width, float(height)))
    n_peaks = len(peaks)
    max_peak = max((h for (_, _, h) in peaks), default=0.0)
    smooth = _moving_median(y, _odd(max(21, int(0.03 * y.size))))
    dyn = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    dyn_snr = dyn / (sigma + 1e-12)
    good_by_lines = (n_peaks >= 2 and max_peak >= 6.0 * sigma)
    good_by_structure = (dyn_snr >= 10.0)
    if good_by_lines or good_by_structure:
        return False
    if debug:
        print(
            "[junk] reason=weak_signal "
            f"peaks={n_peaks} maxpk_snr={max_peak/(sigma+1e-12):.2f} "
            f"dyn_snr={dyn_snr:.2f}"
        )
    return True


# ---------------------------------------------------------------------------
# Species helpers
# ---------------------------------------------------------------------------

def expand_species_filter(ref_species: Iterable[str], requested: Optional[Iterable[str]]) -> Optional[List[str]]:
    if not requested:
        return None
    labels = {str(s).strip().upper() for s in ref_species}
    requested_norm = [str(s).strip().upper() for s in requested]
    refs_have_ions = any(" " in lab for lab in labels)
    if not refs_have_ions:
        return [lab for lab in requested_norm if lab in labels]
    base_to_labels: Dict[str, List[str]] = {}
    for lab in labels:
        base = lab.split()[0]
        base_to_labels.setdefault(base, []).append(lab)
    expanded: List[str] = []
    for lab in requested_norm:
        if lab in labels:
            expanded.append(lab)
        else:
            expanded.extend(base_to_labels.get(lab, []))
    seen = set()
    ordered: List[str] = []
    for lab in expanded:
        if lab not in seen:
            seen.add(lab)
            ordered.append(lab)
    return ordered or requested_norm


# ---------------------------------------------------------------------------
# Internal helpers reused across metrics
# ---------------------------------------------------------------------------

def _odd(n: int) -> int:
    n = max(3, int(n))
    return n if (n % 2) == 1 else n + 1


def _moving_median(y: np.ndarray, win: int) -> np.ndarray:
    win = _odd(win)
    n = y.size
    if n < win + 2:
        med = float(np.median(y)) if n else 0.0
        return np.full_like(y, med, dtype=float)
    pad = win // 2
    yp = np.pad(y.astype(float), (pad, pad), mode="edge")
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(yp, window_shape=win), axis=-1)


def _robust_baseline_and_sigma(y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    win = _odd(max(11, min(101, int(round(0.02 * max(50, y.size))))))
    base = _moving_median(y, win)
    residual = y.astype(float) - base
    mad = np.median(np.abs(residual - np.median(residual))) + 1e-12
    sigma = 1.4826 * float(mad)
    return base, sigma, residual


def _find_runs_above(y: np.ndarray, thr: np.ndarray) -> List[Tuple[int, int]]:
    above = y > thr
    if not np.any(above):
        return []
    idx = np.flatnonzero(above)
    runs: List[Tuple[int, int]] = []
    i = 0
    m = idx.size
    while i < m:
        j = i + 1
        while j < m and (idx[j] == idx[j - 1] + 1):
            j += 1
        runs.append((int(idx[i]), int(idx[j - 1])))
        i = j
    return runs
