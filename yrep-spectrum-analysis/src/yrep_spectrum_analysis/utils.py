from __future__ import annotations

from typing import List, Sequence, Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import glob

from .types import Spectrum, SpectrumLike


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
    vectors: List[np.ndarray] = [np.interp(grid, s.wavelength, s.intensity).astype(float) for s in specs]
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
        for cand in ["Wavelength (Å)", "Wavelength (A)", "Wavelength_A", "Wavelength_Angstrom", "Wavelength (nm)", "Wavelength_nm"]:
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
        wl_nm = (pd.to_numeric(df[wl_col], errors="coerce") / 10.0) if "nm" not in wl_col else pd.to_numeric(df[wl_col], errors="coerce")
        frame = pd.DataFrame({
            "wavelength_nm": wl_nm,
            "species": df[spec_col].astype(str),
            "intensity": df[inten_col].astype(str).str.extract(r"^\s*([0-9]+(?:\.[0-9]+)?)")[0].astype(float),
        }).dropna()
        frames.append(frame)
    if not frames:
        raise RuntimeError("No reference lines parsed")
    return pd.concat(frames, ignore_index=True)


