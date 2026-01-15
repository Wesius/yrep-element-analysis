"""Bundled data for YREP backend.

This package contains sample reference libraries for common elements,
allowing students to start immediately without finding external data.
"""

from pathlib import Path

# Path to bundled data directory
DATA_DIR = Path(__file__).parent
REFERENCES_DIR = DATA_DIR / "references"


def get_bundled_references_path() -> Path:
    """Get path to bundled reference libraries."""
    return REFERENCES_DIR


def list_bundled_references() -> list[str]:
    """List available bundled reference files."""
    if not REFERENCES_DIR.exists():
        return []
    return sorted([f.stem for f in REFERENCES_DIR.glob("*.csv")])
