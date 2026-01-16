"""File browser API routes."""

import os
import re
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.models.core import FileInfo, DirectoryListing

router = APIRouter()


# Allowed base directories for file browsing
# In production, configure via environment variable
_ALLOWED_BASE_DIRS: List[Path] = []


def _get_allowed_dirs() -> List[Path]:
    """Get list of allowed base directories."""
    if _ALLOWED_BASE_DIRS:
        return _ALLOWED_BASE_DIRS

    # Default: user's home directory and common data locations
    allowed = [
        Path.home(),
        Path("/tmp"),
    ]

    # Allow override via environment variable (colon-separated paths)
    env_dirs = os.environ.get("YREP_ALLOWED_DIRS", "")
    if env_dirs:
        allowed = [Path(p).resolve() for p in env_dirs.split(":") if p]

    return [d.resolve() for d in allowed if d.exists()]


def _validate_path(path: Path) -> None:
    """Validate that a path is within allowed directories.

    Raises HTTPException if path is outside allowed boundaries.
    """
    resolved = path.resolve()
    allowed_dirs = _get_allowed_dirs()

    for allowed in allowed_dirs:
        try:
            if resolved == allowed or resolved.is_relative_to(allowed):
                return
        except ValueError:
            continue

    raise HTTPException(
        status_code=403,
        detail=f"Access denied: path is outside allowed directories",
    )


def _sanitize_glob_pattern(pattern: str) -> str:
    """Sanitize a glob pattern to prevent path traversal.

    Raises HTTPException if pattern is dangerous.
    """
    # Reject patterns with path traversal
    if ".." in pattern:
        raise HTTPException(
            status_code=400,
            detail="Invalid pattern: path traversal not allowed",
        )

    # Reject absolute paths in pattern
    if pattern.startswith("/") or pattern.startswith("\\"):
        raise HTTPException(
            status_code=400,
            detail="Invalid pattern: absolute paths not allowed",
        )

    # Reject overly broad recursive patterns at the start
    if pattern.startswith("**"):
        raise HTTPException(
            status_code=400,
            detail="Invalid pattern: recursive wildcard at start not allowed",
        )

    # Only allow simple glob patterns (alphanumeric, *, ?, [], .)
    if not re.match(r'^[\w\s.*?\[\]\-_]+$', pattern):
        raise HTTPException(
            status_code=400,
            detail="Invalid pattern: contains disallowed characters",
        )

    return pattern


def _get_file_info(path: Path) -> FileInfo:
    """Convert a Path to FileInfo."""
    try:
        stat = path.stat()
        return FileInfo(
            name=path.name,
            path=str(path.resolve()),
            size=stat.st_size if path.is_file() else 0,
            is_dir=path.is_dir(),
            extension=path.suffix[1:] if path.suffix else None,
        )
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Error reading path: {e}")


@router.get("/list", response_model=DirectoryListing)
async def list_directory(
    path: str = Query(..., description="Directory path to list"),
    pattern: Optional[str] = Query(None, description="Glob pattern filter (e.g., '*.txt')"),
):
    """List contents of a directory.

    Returns files and subdirectories in the specified path.
    Optionally filter by glob pattern.
    """
    dir_path = Path(path).expanduser().resolve()

    # Validate path is within allowed directories
    _validate_path(dir_path)

    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")

    try:
        if pattern:
            # Sanitize glob pattern before use
            safe_pattern = _sanitize_glob_pattern(pattern)
            items = list(dir_path.glob(safe_pattern))
        else:
            items = list(dir_path.iterdir())

        # Sort: directories first, then files, alphabetically
        items.sort(key=lambda p: (not p.is_dir(), p.name.lower()))

        files = [_get_file_info(item) for item in items]

        parent = str(dir_path.parent) if dir_path.parent != dir_path else None

        return DirectoryListing(
            path=str(dir_path),
            files=files,
            parent=parent,
        )
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {path}")
    except OSError as e:
        raise HTTPException(status_code=500, detail=f"Error listing directory: {e}")


@router.get("/info", response_model=FileInfo)
async def get_file_info(
    path: str = Query(..., description="File path"),
):
    """Get information about a file or directory."""
    file_path = Path(path).expanduser().resolve()

    # Validate path is within allowed directories
    _validate_path(file_path)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    return _get_file_info(file_path)


@router.get("/exists")
async def check_exists(
    path: str = Query(..., description="Path to check"),
):
    """Check if a path exists."""
    file_path = Path(path).expanduser().resolve()

    # Validate path is within allowed directories
    _validate_path(file_path)

    return {
        "path": str(file_path),
        "exists": file_path.exists(),
        "is_file": file_path.is_file() if file_path.exists() else False,
        "is_dir": file_path.is_dir() if file_path.exists() else False,
    }


@router.get("/spectrum-dirs")
async def find_spectrum_directories(
    root: str = Query(..., description="Root directory to search"),
    max_depth: int = Query(3, ge=1, le=5, description="Maximum search depth (1-5)"),
):
    """Find directories containing spectrum files.

    Searches for directories containing .txt files that look like spectra.
    Useful for discovering data directories.
    """
    root_path = Path(root).expanduser().resolve()

    # Validate path is within allowed directories
    _validate_path(root_path)

    if not root_path.exists() or not root_path.is_dir():
        raise HTTPException(status_code=404, detail=f"Root directory not found: {root}")

    spectrum_dirs = []

    def search(path: Path, depth: int):
        if depth > max_depth:
            return

        try:
            txt_files = list(path.glob("*.txt"))
            if txt_files:
                spectrum_dirs.append({
                    "path": str(path),
                    "name": path.name,
                    "file_count": len(txt_files),
                })

            for subdir in path.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    search(subdir, depth + 1)
        except PermissionError:
            pass

    search(root_path, 0)

    return {"root": str(root_path), "directories": spectrum_dirs}
