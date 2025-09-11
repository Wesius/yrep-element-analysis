from __future__ import annotations

from typing import List, Sequence, Tuple, Optional
import numpy as np
import pandas as pd
from pathlib import Path
import glob

from .types import Spectrum, SpectrumLike


    # Average measurement and background on overlap grid
def average_spectra(
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


def group_spectra(measurements: Sequence[SpectrumLike]) -> List[List[Spectrum]]:
    """
    Group spectra by cosine similarity to the average spectrum.

    Returns a list of groups, where each group is a list of Spectrum objects.
    """
    if not measurements:
        return []

    specs: List[Spectrum] = []
    for it in measurements:
        wl = np.asarray(it.wavelength, dtype=float)
        iy = np.asarray(it.intensity, dtype=float)
        specs.append(Spectrum(wavelength=wl, intensity=iy))

    if len(specs) == 1:
        return [specs]

    # Build a common grid over the overlap of all spectra
    wl_min = max(float(np.min(s.wavelength)) for s in specs)
    wl_max = min(float(np.max(s.wavelength)) for s in specs)
    if not np.isfinite(wl_min) or not np.isfinite(wl_max) or wl_max <= wl_min:
        return [specs]
    grid = np.linspace(wl_min, wl_max, 1000)

    # Interpolate to common grid
    vectors: List[np.ndarray] = [
        np.interp(grid, s.wavelength, s.intensity).astype(float) for s in specs
    ]
    avg = np.mean(np.stack(vectors, axis=0), axis=0)

    def _cosine(u: np.ndarray, v: np.ndarray) -> float:
        num = float(np.dot(u, v))
        den = float(np.linalg.norm(u) * np.linalg.norm(v) + 1e-12)
        s = num / den
        if s < 0:
            s = 0.0
        if s > 1:
            s = 1.0
        return s

    sims = np.asarray([_cosine(vec, avg) for vec in vectors], dtype=float)

    # Simple histogram-based grouping: contiguous non-empty bins form a group
    counts, edges = np.histogram(sims, bins=50)
    positive = counts > 0
    labels = np.zeros(sims.size, dtype=int)
    gid = 0
    for bi in range(positive.size):
        if not positive[bi]:
            continue
        a = float(edges[bi])
        b = float(edges[bi + 1])
        mask = (sims >= a) & (sims < b if bi < positive.size - 1 else sims <= b)
        if not np.any(mask):
            continue
        labels[mask] = gid
        if bi + 1 < positive.size and not positive[bi + 1]:
            gid += 1

    groups: List[List[Spectrum]] = []
    label_list = labels.tolist()
    for g in sorted(set(label_list)):
        idxs = [i for i, lab in enumerate(label_list) if lab == g]
        if not idxs:
            continue
        groups.append([specs[i] for i in idxs])

    return groups


# -------------------------
# Quality assessment
# -------------------------

# ---------- helpers ----------

def _odd(n: int) -> int:
    """Return the nearest odd integer ≥ 3."""
    n = max(3, int(n))
    return n if (n % 2) == 1 else n + 1

def _moving_median(y: np.ndarray, win: int) -> np.ndarray:
    """
    Robust local baseline using a centered moving median.
    For short arrays, falls back to the global median.
    """
    win = _odd(win)
    n = y.size
    if n < win + 2:
        med = float(np.median(y)) if n else 0.0
        return np.full_like(y, med, dtype=float)
    pad = win // 2
    yp = np.pad(y.astype(float), (pad, pad), mode="edge")
    # sliding window view -> median along last axis
    from numpy.lib.stride_tricks import sliding_window_view
    return np.median(sliding_window_view(yp, window_shape=win), axis=-1)

def _robust_baseline_and_sigma(y: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Return (baseline, sigma, residual).
    sigma is MAD of residuals (Gaussian-consistent).
    """
    # window ≈ 2% of length, capped to reasonable bounds
    win = _odd(max(11, min(101, int(round(0.02 * max(50, y.size))))))
    base = _moving_median(y, win)
    r = y.astype(float) - base
    # robust sigma on residuals
    mad = np.median(np.abs(r - np.median(r))) + 1e-12
    sigma = 1.4826 * float(mad)
    return base, sigma, r

def _find_runs_above(y: np.ndarray, thr: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of (start, end) indices (inclusive) for contiguous runs where y > thr.
    """
    above = y > thr
    if not np.any(above):
        return []
    idx = np.flatnonzero(above)
    runs: List[Tuple[int,int]] = []
    i = 0
    m = idx.size
    while i < m:
        j = i + 1
        while j < m and (idx[j] == idx[j-1] + 1):
            j += 1
        runs.append((int(idx[i]), int(idx[j-1])))
        i = j
    return runs


def is_junk_group(measurements: Sequence[SpectrumLike], *, debug: bool = True) -> bool:
    """
    Simpler junk detector:
      - Average spectra on a common grid.
      - Remove a robust baseline; estimate noise from residuals (MAD).
      - Define significant peaks as runs above (baseline + k*sigma) with 2–25 bins width.
      - Flag as GOOD if:
          (A) at least 2 significant peaks (typical emission lines), OR
          (B) smoothed dynamic range is high relative to noise.
        Otherwise -> JUNK.

    Parameters
    ----------
    measurements : Sequence[SpectrumLike]
        Group of spectra (same source/condition).
    debug : bool
        Print short reason codes when classifying as junk.

    Returns
    -------
    bool
        True if the group is junk; False if usable.
    """
    if not measurements:
        if debug:
            print("[junk] reason=empty_group")
        return True

    # Normalize inputs and average on an overlap grid
    specs: List[Spectrum] = []
    for it in measurements:
        specs.append(
            Spectrum(
                wavelength=np.asarray(it.wavelength, dtype=float),
                intensity=np.asarray(it.intensity, dtype=float),
            )
        )
    try:
        avg = average_spectra(specs, n_points=1000)
    except Exception as e:
        if debug:
            print(f"[junk] reason=average_failure error={e}")
        return True

    y = np.asarray(avg.intensity, dtype=float)
    if y.size < 10 or not np.all(np.isfinite(y)):
        if debug:
            print("[junk] reason=invalid_average")
        return True

    base, sigma, r = _robust_baseline_and_sigma(y)
    if not np.isfinite(sigma) or sigma <= 0:
        if debug:
            print("[junk] reason=noise_estimation_failed")
        return True

    # Significant peaks (narrow, tall relative to noise)
    k = 6.0                             # peak SNR threshold ~6σ
    thr = base + k * sigma
    runs = _find_runs_above(y, thr)
    # keep only runs with width within [2, 25] bins and away from edges
    peaks = []
    n = y.size
    for a, b in runs:
        w = b - a + 1
        c = (a + b) // 2
        if 2 <= w <= 25 and 2 <= c < n - 2:   # ignore single-bin spikes and edge artifacts
            height = y[c] - base[c]
            peaks.append((c, w, float(height)))

    n_peaks = len(peaks)
    max_peak = max((h for (_, _, h) in peaks), default=0.0)

    # Dynamic range of the smoothed signal relative to noise
    # (captures the "good continuum + lines" cases)
    smooth = _moving_median(y, _odd(max(21, int(0.03 * y.size))))
    dyn = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))  # structured amplitude
    dyn_snr = dyn / (sigma + 1e-12)

    # Decision: GOOD if enough lines OR strong structured signal
    good_by_lines = (n_peaks >= 2 and max_peak >= 6.0 * sigma)
    good_by_structure = (dyn_snr >= 10.0)

    if good_by_lines or good_by_structure:
        return False  # not junk

    if debug:
        print(f"[junk] reason=weak_signal peaks={n_peaks} maxpk_snr={max_peak/(sigma+1e-12):.2f} dyn_snr={dyn_snr:.2f}")
    return True


def spectrum_quality(spec: SpectrumLike) -> float:
    """
    Quality score in [0, 1] for a single spectrum.
    Higher when there are a few strong, narrow peaks and/or strong structured signal.

    Implementation:
      - robust baseline removal + MAD sigma on residuals
      - peak runs above baseline + 6σ with width 2–25 bins
      - score = 0.45*dyn_score + 0.35*max_peak_score + 0.20*count_score

    The weights and saturations are chosen to be conservative and stable.
    """
    y = np.asarray(spec.intensity, dtype=float)
    if y.size < 10 or not np.all(np.isfinite(y)):
        return 0.0

    base, sigma, r = _robust_baseline_and_sigma(y)
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0

    k = 6.0
    thr = base + k * sigma
    runs = _find_runs_above(y, thr)
    peaks = []
    n = y.size
    for a, b in runs:
        w = b - a + 1
        c = (a + b) // 2
        if 2 <= w <= 25 and 2 <= c < n - 2:
            height = y[c] - base[c]
            peaks.append((c, w, float(height)))

    n_peaks = len(peaks)
    max_peak = max((h for (_, _, h) in peaks), default=0.0)

    smooth = _moving_median(y, _odd(max(21, int(0.03 * y.size))))
    dyn = float(np.percentile(smooth, 95) - np.percentile(smooth, 5))
    dyn_snr = dyn / (sigma + 1e-12)
    maxpk_snr = max_peak / (sigma + 1e-12)

    # Convert features -> [0,1] via simple saturating maps (no hard cliffs)
    dyn_score   = 1.0 - np.exp(-dyn_snr / 10.0)         # ~0.63 at 10σ, ~0.86 at 20σ
    maxpk_score = 1.0 - np.exp(-maxpk_snr / 12.0)       # ~0.56 at 10σ, ~0.80 at 20σ
    count_score = 1.0 - np.exp(-n_peaks / 3.0)          # saturates near 6+ peaks

    score = 0.45 * dyn_score + 0.35 * maxpk_score + 0.20 * count_score
    return float(np.clip(score, 0.0, 1.0))

# -------------------------
# Data loading utilities
# -------------------------


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


def load_batch(meas_root: Path, bg_root: Path) -> Tuple[List[Spectrum], List[Spectrum]]:
    """
    Load measurement and background spectra from the provided directories.

    Parameters
    ----------
    meas_root: Path
        Directory containing measurement .txt spectra files.
    bg_root: Path
        Directory containing background .txt spectra files.
    """

    meas_specs: List[Spectrum] = []
    for fp in sorted(glob.glob(str(meas_root / "*.txt"))):
        wl, iy = load_txt_spectrum(Path(fp))
        meas_specs.append(Spectrum(wavelength=wl, intensity=iy))

    bg_specs: List[Spectrum] = []
    for fp in sorted(glob.glob(str(bg_root / "*.txt"))):
        wl, iy = load_txt_spectrum(Path(fp))
        bg_specs.append(Spectrum(wavelength=wl, intensity=iy))

    return meas_specs, bg_specs


def load_references(lists_dir: Path, *, element_only: bool = True) -> pd.DataFrame:
    """
    Load reference line lists from the provided directory containing CSV files.

    Parameters
    ----------
    lists_dir: Path
        Directory containing reference line list CSVs.
    element_only: bool (default True)
        If True, normalize species labels to base chemical symbols (e.g.,
        "Fe I"/"Fe II" -> "FE"). If False, keep ionization states or other
        suffixes intact (e.g., "FE I", "FE II").
    """
    csvs = sorted(lists_dir.glob("*.csv"))
    frames: List[pd.DataFrame] = []
    for c in csvs:
        df = pd.read_csv(c)
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
            if "nm" not in wl_col
            else pd.to_numeric(df[wl_col], errors="coerce")
        )
        frame = pd.DataFrame(
            {
                "wavelength_nm": wl_nm,
                "species": df[spec_col].astype(str),
                "intensity": df[inten_col]
                .astype(str)
                .str.extract(r"^\s*([0-9]+(?:\.[0-9]+)?)")[0]
                .astype(float),
            }
        ).dropna()
        # Normalize species labeling here so downstream code can use it directly.
        # - Always uppercase and strip whitespace.
        # - If element_only, collapse to base element token before the first space.
        species_norm = (
            frame["species"].astype(str).str.strip().str.upper()
        )
        if element_only:
            species_norm = species_norm.str.split().str[0]
        frame["species"] = species_norm
        frames.append(frame)
    if not frames:
        raise RuntimeError("No reference lines parsed")
    return pd.concat(frames, ignore_index=True)


def expand_species_filter(ref_species: Sequence[str], requested_species: Optional[Sequence[str]]) -> Optional[List[str]]:
    """Expand base species filters to exact labels present in references.

    - Input labels and output labels are compared case-insensitively and trimmed.
    - If references contain ionized labels (e.g., "FE I", "FE II") and the user
      requested base symbols (e.g., "FE"), expand to all matching exact labels.
    - If the user already provided exact labels, they are preserved if present.
    - If requested_species is None or empty, return it unchanged.
    """
    if not requested_species:
        return requested_species  # None or []
    labels = {str(s).strip().upper() for s in ref_species}
    requested = [str(s).strip().upper() for s in requested_species]
    # Detect if refs include ionization states (presence of a space)
    refs_have_ions = any(" " in lab for lab in labels)
    if not refs_have_ions:
        # No ions in refs; just keep intersection to be safe
        return [s for s in requested if s in labels]
    # Build base -> exact map
    base_to_labels: dict[str, List[str]] = {}
    for lab in labels:
        base = lab.split()[0]
        base_to_labels.setdefault(base, []).append(lab)
    expanded: List[str] = []
    for s in requested:
        if s in labels:
            expanded.append(s)
        elif s in base_to_labels:
            expanded.extend(base_to_labels[s])
        # else: keep s as-is so downstream can attempt exact match (no harm)
    # Deduplicate while keeping a stable order roughly aligned with requested
    seen = set()
    result: List[str] = []
    for s in expanded:
        if s not in seen:
            seen.add(s)
            result.append(s)
    return result or requested
