"""Reference library API routes."""

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from backend.data import REFERENCES_DIR, list_bundled_references

router = APIRouter()


@router.get("/bundled")
async def list_bundled() -> Dict[str, Any]:
    """List bundled reference libraries.

    Returns the sample NIST reference libraries included with
    the package for immediate use without external data.
    """
    references = []

    for ref_file in sorted(REFERENCES_DIR.glob("*.csv")):
        try:
            # Read file to get line count and species info
            df = pd.read_csv(ref_file, nrows=1000)
            spectrum_col = None
            for col in ["Spectrum", "Species", "Element"]:
                if col in df.columns:
                    spectrum_col = col
                    break

            species = []
            if spectrum_col:
                species = sorted(df[spectrum_col].unique().tolist())

            references.append({
                "id": ref_file.stem,
                "name": ref_file.stem.title(),
                "file": ref_file.name,
                "line_count": len(df),
                "species": species,
                "path": str(ref_file),
            })
        except Exception:
            # Skip files that can't be read
            continue

    return {
        "count": len(references),
        "references": references,
        "path": str(REFERENCES_DIR),
        "description": "Sample NIST reference libraries for common LIBS elements",
    }


@router.get("/bundled/{reference_id}")
async def get_bundled_reference(reference_id: str) -> Dict[str, Any]:
    """Get details for a specific bundled reference.

    Returns metadata and sample data from the reference library.
    """
    ref_file = REFERENCES_DIR / f"{reference_id}.csv"

    if not ref_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Bundled reference not found: {reference_id}"
        )

    try:
        df = pd.read_csv(ref_file)

        # Find column names
        wavelength_col = None
        intensity_col = None
        spectrum_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "wavelength" in col_lower:
                wavelength_col = col
            elif "intensity" in col_lower or "strength" in col_lower:
                intensity_col = col
            elif col in ["Spectrum", "Species", "Element"]:
                spectrum_col = col

        species = []
        if spectrum_col:
            species = sorted(df[spectrum_col].unique().tolist())

        # Get wavelength range
        wl_min = wl_max = None
        if wavelength_col:
            # Convert Angstroms to nm if needed
            wl_values = df[wavelength_col].values
            if wl_values.mean() > 1000:  # Likely Angstroms
                wl_values = wl_values / 10.0
            wl_min = float(wl_values.min())
            wl_max = float(wl_values.max())

        return {
            "id": reference_id,
            "name": reference_id.title(),
            "file": ref_file.name,
            "path": str(ref_file),
            "line_count": len(df),
            "species": species,
            "wavelength_range_nm": [wl_min, wl_max] if wl_min else None,
            "columns": list(df.columns),
            "sample_lines": df.head(10).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading reference: {e}"
        )


@router.get("/bundled/{reference_id}/lines")
async def get_reference_lines(
    reference_id: str,
    species: Optional[str] = Query(None, description="Filter by species"),
    min_wavelength: Optional[float] = Query(None, description="Min wavelength (nm)"),
    max_wavelength: Optional[float] = Query(None, description="Max wavelength (nm)"),
    min_intensity: Optional[float] = Query(None, description="Min intensity"),
    limit: int = Query(100, description="Max lines to return"),
) -> Dict[str, Any]:
    """Get spectral lines from a bundled reference.

    Supports filtering by species, wavelength range, and intensity.
    """
    ref_file = REFERENCES_DIR / f"{reference_id}.csv"

    if not ref_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Bundled reference not found: {reference_id}"
        )

    try:
        df = pd.read_csv(ref_file)

        # Find column names
        wavelength_col = None
        intensity_col = None
        spectrum_col = None

        for col in df.columns:
            col_lower = col.lower()
            if "wavelength" in col_lower:
                wavelength_col = col
            elif "intensity" in col_lower or "strength" in col_lower:
                intensity_col = col
            elif col in ["Spectrum", "Species", "Element"]:
                spectrum_col = col

        # Convert wavelength to nm if in Angstroms
        if wavelength_col and df[wavelength_col].mean() > 1000:
            df["wavelength_nm"] = df[wavelength_col] / 10.0
        elif wavelength_col:
            df["wavelength_nm"] = df[wavelength_col]

        # Apply filters
        if species and spectrum_col:
            df = df[df[spectrum_col].str.contains(species, case=False, na=False)]

        if min_wavelength and "wavelength_nm" in df.columns:
            df = df[df["wavelength_nm"] >= min_wavelength]

        if max_wavelength and "wavelength_nm" in df.columns:
            df = df[df["wavelength_nm"] <= max_wavelength]

        if min_intensity and intensity_col:
            # Handle intensity values that may have qualifiers like "1000 P"
            try:
                intensities = df[intensity_col].astype(str).str.extract(r'(\d+)')[0].astype(float)
                df = df[intensities >= min_intensity]
            except Exception:
                pass

        # Limit results
        df = df.head(limit)

        lines = []
        for _, row in df.iterrows():
            line = {
                "wavelength_nm": row.get("wavelength_nm"),
                "species": row.get(spectrum_col) if spectrum_col else None,
            }
            if intensity_col:
                line["intensity"] = row[intensity_col]
            if wavelength_col:
                line["wavelength_original"] = row[wavelength_col]
            lines.append(line)

        return {
            "reference_id": reference_id,
            "line_count": len(lines),
            "total_in_file": len(pd.read_csv(ref_file)),
            "filters": {
                "species": species,
                "min_wavelength": min_wavelength,
                "max_wavelength": max_wavelength,
                "min_intensity": min_intensity,
            },
            "lines": lines,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading reference lines: {e}"
        )
