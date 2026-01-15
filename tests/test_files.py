"""Tests for the files API endpoints."""

from pathlib import Path

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_directory_success(client: AsyncClient, temp_dir: Path):
    """Test listing directory contents."""
    # Create some files
    (temp_dir / "file1.txt").write_text("content")
    (temp_dir / "file2.py").write_text("content")
    subdir = temp_dir / "subdir"
    subdir.mkdir()

    response = await client.get("/api/files/list", params={"path": str(temp_dir)})

    assert response.status_code == 200
    data = response.json()
    # Resolve paths to handle macOS /private/var symlinks
    assert Path(data["path"]).resolve() == temp_dir.resolve()
    assert len(data["files"]) == 3
    # Directories should be first
    assert data["files"][0]["is_dir"] is True
    assert data["files"][0]["name"] == "subdir"


@pytest.mark.asyncio
async def test_list_directory_with_pattern(client: AsyncClient, temp_dir: Path):
    """Test listing directory with glob pattern filter."""
    (temp_dir / "file1.txt").write_text("content")
    (temp_dir / "file2.txt").write_text("content")
    (temp_dir / "file3.py").write_text("content")

    response = await client.get(
        "/api/files/list",
        params={"path": str(temp_dir), "pattern": "*.txt"}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["files"]) == 2
    assert all(f["name"].endswith(".txt") for f in data["files"])


@pytest.mark.asyncio
async def test_list_directory_not_found(client: AsyncClient):
    """Test listing non-existent directory returns 404."""
    response = await client.get(
        "/api/files/list",
        params={"path": "/nonexistent/path/that/does/not/exist"}
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_list_directory_not_a_dir(client: AsyncClient, sample_spectrum_file: Path):
    """Test listing a file (not directory) returns 400."""
    response = await client.get(
        "/api/files/list",
        params={"path": str(sample_spectrum_file)}
    )

    assert response.status_code == 400
    assert "not a directory" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_file_info_file(client: AsyncClient, sample_spectrum_file: Path):
    """Test getting info for a file."""
    response = await client.get(
        "/api/files/info",
        params={"path": str(sample_spectrum_file)}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "sample.txt"
    assert data["is_dir"] is False
    assert data["extension"] == "txt"
    assert data["size"] > 0


@pytest.mark.asyncio
async def test_get_file_info_directory(client: AsyncClient, temp_dir: Path):
    """Test getting info for a directory."""
    response = await client.get(
        "/api/files/info",
        params={"path": str(temp_dir)}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["is_dir"] is True


@pytest.mark.asyncio
async def test_get_file_info_not_found(client: AsyncClient):
    """Test getting info for non-existent path returns 404."""
    response = await client.get(
        "/api/files/info",
        params={"path": "/nonexistent/file.txt"}
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_check_exists_file(client: AsyncClient, sample_spectrum_file: Path):
    """Test checking existence of a file."""
    response = await client.get(
        "/api/files/exists",
        params={"path": str(sample_spectrum_file)}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is True
    assert data["is_file"] is True
    assert data["is_dir"] is False


@pytest.mark.asyncio
async def test_check_exists_directory(client: AsyncClient, temp_dir: Path):
    """Test checking existence of a directory."""
    response = await client.get(
        "/api/files/exists",
        params={"path": str(temp_dir)}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is True
    assert data["is_file"] is False
    assert data["is_dir"] is True


@pytest.mark.asyncio
async def test_check_exists_not_found(client: AsyncClient):
    """Test checking existence of non-existent path."""
    response = await client.get(
        "/api/files/exists",
        params={"path": "/nonexistent/file.txt"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["exists"] is False
    assert data["is_file"] is False
    assert data["is_dir"] is False


@pytest.mark.asyncio
async def test_find_spectrum_directories(client: AsyncClient, sample_spectrum_dir: Path):
    """Test finding directories with spectrum files."""
    response = await client.get(
        "/api/files/spectrum-dirs",
        params={"root": str(sample_spectrum_dir.parent)}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["directories"]) >= 1
    # Find the spectra directory
    spectra = next(
        (d for d in data["directories"] if d["name"] == "spectra"),
        None
    )
    assert spectra is not None
    assert spectra["file_count"] == 3


@pytest.mark.asyncio
async def test_find_spectrum_directories_not_found(client: AsyncClient):
    """Test finding spectrum dirs with invalid root returns 404."""
    response = await client.get(
        "/api/files/spectrum-dirs",
        params={"root": "/nonexistent/directory"}
    )

    assert response.status_code == 404
