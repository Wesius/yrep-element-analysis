"""File browser API routes."""

from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query

from backend.models.core import FileInfo, DirectoryListing

router = APIRouter()


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

    if not dir_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if not dir_path.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {path}")

    try:
        if pattern:
            items = list(dir_path.glob(pattern))
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

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    return _get_file_info(file_path)


@router.get("/exists")
async def check_exists(
    path: str = Query(..., description="Path to check"),
):
    """Check if a path exists."""
    file_path = Path(path).expanduser().resolve()
    return {
        "path": str(file_path),
        "exists": file_path.exists(),
        "is_file": file_path.is_file() if file_path.exists() else False,
        "is_dir": file_path.is_dir() if file_path.exists() else False,
    }


@router.get("/spectrum-dirs")
async def find_spectrum_directories(
    root: str = Query(..., description="Root directory to search"),
    max_depth: int = Query(3, description="Maximum search depth"),
):
    """Find directories containing spectrum files.

    Searches for directories containing .txt files that look like spectra.
    Useful for discovering data directories.
    """
    root_path = Path(root).expanduser().resolve()

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
