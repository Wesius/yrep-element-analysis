"""Tests for the references API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_bundled_references(client: AsyncClient):
    """Test listing bundled reference libraries."""
    response = await client.get("/api/references/bundled")

    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "references" in data
    assert data["count"] >= 8  # We have 8 bundled references
    # Check structure
    ref = data["references"][0]
    assert "id" in ref
    assert "name" in ref
    assert "file" in ref
    assert "line_count" in ref
    assert "species" in ref


@pytest.mark.asyncio
async def test_list_bundled_has_expected_elements(client: AsyncClient):
    """Test that bundled references include expected elements."""
    response = await client.get("/api/references/bundled")

    assert response.status_code == 200
    data = response.json()
    ref_ids = [r["id"] for r in data["references"]]
    # Check for expected elements
    assert "copper" in ref_ids
    assert "iron" in ref_ids
    assert "lead" in ref_ids
    assert "aluminum" in ref_ids
    assert "calcium" in ref_ids
    assert "magnesium" in ref_ids
    assert "sodium" in ref_ids
    assert "zinc" in ref_ids


@pytest.mark.asyncio
async def test_get_bundled_reference(client: AsyncClient):
    """Test getting details for a specific bundled reference."""
    response = await client.get("/api/references/bundled/copper")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "copper"
    assert data["name"] == "Copper"
    assert "line_count" in data
    assert data["line_count"] > 0
    assert "species" in data
    assert any("Cu" in s for s in data["species"])
    assert "wavelength_range_nm" in data
    assert "columns" in data
    assert "sample_lines" in data


@pytest.mark.asyncio
async def test_get_bundled_reference_not_found(client: AsyncClient):
    """Test getting non-existent bundled reference returns 404."""
    response = await client.get("/api/references/bundled/nonexistent")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_reference_lines(client: AsyncClient):
    """Test getting spectral lines from a bundled reference."""
    response = await client.get("/api/references/bundled/copper/lines")

    assert response.status_code == 200
    data = response.json()
    assert data["reference_id"] == "copper"
    assert "line_count" in data
    assert "lines" in data
    assert len(data["lines"]) > 0
    # Check line structure
    line = data["lines"][0]
    assert "wavelength_nm" in line
    assert "species" in line


@pytest.mark.asyncio
async def test_get_reference_lines_with_filters(client: AsyncClient):
    """Test filtering spectral lines."""
    response = await client.get(
        "/api/references/bundled/copper/lines",
        params={
            "min_wavelength": 300,
            "max_wavelength": 400,
            "limit": 10
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["line_count"] <= 10
    # All returned lines should be in range
    for line in data["lines"]:
        if line["wavelength_nm"]:
            assert 300 <= line["wavelength_nm"] <= 400


@pytest.mark.asyncio
async def test_get_reference_lines_species_filter(client: AsyncClient):
    """Test filtering lines by species."""
    response = await client.get(
        "/api/references/bundled/copper/lines",
        params={"species": "Cu I"}
    )

    assert response.status_code == 200
    data = response.json()
    # All returned lines should be Cu I
    for line in data["lines"]:
        if line["species"]:
            assert "Cu" in line["species"]


@pytest.mark.asyncio
async def test_get_reference_lines_not_found(client: AsyncClient):
    """Test getting lines from non-existent reference returns 404."""
    response = await client.get("/api/references/bundled/nonexistent/lines")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_bundled_references_have_valid_data(client: AsyncClient):
    """Test that all bundled references have valid line data."""
    list_response = await client.get("/api/references/bundled")
    refs = list_response.json()["references"]

    for ref in refs:
        ref_id = ref["id"]
        response = await client.get(f"/api/references/bundled/{ref_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["line_count"] > 0, f"{ref_id} has no lines"
        assert len(data["species"]) > 0, f"{ref_id} has no species"
        assert data["wavelength_range_nm"] is not None, f"{ref_id} has no wavelength range"
