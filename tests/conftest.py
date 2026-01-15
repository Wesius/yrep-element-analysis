"""Pytest configuration and fixtures for API tests."""

import tempfile
from pathlib import Path
from typing import Generator

import pytest
from httpx import ASGITransport, AsyncClient

from backend.main import app


@pytest.fixture
def anyio_backend():
    """Use asyncio backend for pytest-asyncio."""
    return "asyncio"


@pytest.fixture
async def client() -> Generator[AsyncClient, None, None]:
    """Create an async test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for file tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_spectrum_file(temp_dir: Path) -> Path:
    """Create a sample spectrum file for testing."""
    spectrum_file = temp_dir / "sample.txt"
    # Create a simple two-column spectrum file
    content = """# Sample spectrum
300.0\t0.1
350.0\t0.5
400.0\t1.0
450.0\t0.8
500.0\t0.3
550.0\t0.2
600.0\t0.1
"""
    spectrum_file.write_text(content)
    return spectrum_file


@pytest.fixture
def sample_spectrum_dir(temp_dir: Path) -> Path:
    """Create a directory with sample spectrum files."""
    spectra_dir = temp_dir / "spectra"
    spectra_dir.mkdir()

    # Create a few spectrum files
    for i in range(3):
        spectrum_file = spectra_dir / f"sample_{i}.txt"
        content = f"""# Sample spectrum {i}
300.0\t{0.1 + i * 0.1}
400.0\t{0.5 + i * 0.1}
500.0\t{0.3 + i * 0.1}
"""
        spectrum_file.write_text(content)

    return spectra_dir
