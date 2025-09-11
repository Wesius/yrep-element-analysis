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


def is_junk_group(measurements: Sequence[SpectrumLike], *, debug: bool = True) -> bool:
    """
    Minimal junk detector with two simple rules:
      1) Junk if the averaged spectrum has no significant peaks above a robust threshold
         (median + 6 * MAD), where a peak is at least 2 consecutive bins above threshold.
      2) Junk if "too many" bins are above threshold (not a few sharp spikes):
         either >3% of bins above threshold or >15 distinct peaks.
    """
    if not measurements:
        if debug:
            print("[junk] reason=empty_group")
        return True

    # Convert inputs
    specs: List[Spectrum] = []
    for it in measurements:
        specs.append(
            Spectrum(
                wavelength=np.asarray(it.wavelength, dtype=float),
                intensity=np.asarray(it.intensity, dtype=float),
            )
        )

    # Average on overlap grid; if it fails, treat as junk
    try:
        avg = average_spectra(specs, n_points=1000)
    except Exception as e:
        if debug:
            print(f"[junk] reason=average_failure error={e}")
        return True

    y = np.asarray(avg.intensity, dtype=float)
    if y.size == 0 or not np.all(np.isfinite(y)):
        if debug:
            print("[junk] reason=invalid_average")
        return True

    # Robust threshold: median + 6 * MAD (match quality function)
    med = float(np.median(y))
    mad = 1.4826 * float(np.median(np.abs(y - med)) + 1e-12)
    thr = med + 6.0 * mad
    above = y > thr
    if not np.any(above):
        if debug:
            print(f"[junk] reason=no_significant_peaks thr={thr:.6g}")
        return True
    # Count contiguous runs (peak widths)
    idx = np.where(above)[0]
    widths: List[int] = []
    i = 0
    while i < idx.size:
        j = i + 1
        while j < idx.size and (idx[j] - idx[j - 1]) == 1:
            j += 1
        widths.append(int(idx[j - 1] - idx[i] + 1))
        i = j
    peaks_ge2 = int(np.sum(np.asarray(widths, dtype=int) >= 2))
    # print(f"[junk] widths: ", str(len(widths)))
    if peaks_ge2 <= 0:
        if debug:
            print("[junk] reason=only_single_bin_spikes (no 2-bin peaks)")
        return True

    # Density rule: too many bins/peaks above threshold → junk (no longer "few spikes")
    frac_bins = float(np.mean(above))
    if frac_bins > 0.03 or len(widths) > 19:
        if debug:
            print(f"[junk] reason=excess_spike_density frac_bins={frac_bins:.4f} peaks={len(widths)}")
        return True

    return False


# ---------------------------------
# Data quality scoring (single spec)
# ---------------------------------


def spectrum_quality(spec: SpectrumLike) -> float:
    """
    Return a quality score in [0, 1] for a single spectrum.

    High score when the baseline is flat and there are a few sharp spikes.
    Implementation is intentionally simple and robust.
    """
    wl = np.asarray(spec.wavelength, dtype=float)
    y = np.asarray(spec.intensity, dtype=float)
    if y.size == 0 or not np.all(np.isfinite(y)) or not np.all(np.isfinite(wl)):
        return 0.0

    med = float(np.median(y))
    mad = 1.4826 * float(np.median(np.abs(y - med)) + 1e-12)
    thr = med + 6.0 * mad
    above = y > thr
    if not np.any(above):
        return 0.0

    # Sparsity of spikes: fewer bins above threshold is better
    frac_bins = float(np.mean(above))
    sparsity_score = float(np.exp(-frac_bins / 0.01))  # ~1 if <1% bins, falls quickly

    # Peak prominence: taller spikes vs noise are better
    height_ratio = float((float(np.max(y)) - thr) / (6.0 * mad + 1e-12))
    prominence_score = float(np.tanh(max(0.0, height_ratio)))  # maps [0, inf) -> [0, 1)

    # Width penalty: single-bin spikes should not score as perfect quality
    # approximate spike width using run-length of consecutive bins above threshold
    idx = np.where(above)[0]
    widths: List[int] = []
    i = 0
    while i < idx.size:
        j = i + 1
        while j < idx.size and (idx[j] - idx[j - 1]) == 1:
            j += 1
        widths.append(int(idx[j - 1] - idx[i] + 1))
        i = j
    median_width = float(np.median(widths)) if widths else 1.0
    # prefer narrow spikes but not 1-bin noise; ideal width around 2–5 bins
    width_penalty = float(np.exp(-abs(median_width - 3.0) / 3.0))

    score = 0.4 * sparsity_score + 0.4 * prominence_score + 0.2 * width_penalty
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


def load_references(lists_dir: Path) -> pd.DataFrame:
    """
    Load reference line lists from the provided directory containing CSV files.

    Parameters
    ----------
    lists_dir: Path
        Directory containing reference line list CSVs.
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
        frames.append(frame)
    if not frames:
        raise RuntimeError("No reference lines parsed")
    return pd.concat(frames, ignore_index=True)
