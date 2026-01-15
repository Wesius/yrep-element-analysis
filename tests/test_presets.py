"""Tests for the presets API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_presets(client: AsyncClient):
    """Test listing all available presets."""
    response = await client.get("/api/presets/")

    assert response.status_code == 200
    data = response.json()
    assert "presets" in data
    assert "categories" in data
    assert len(data["presets"]) >= 3  # We have 3 starter presets
    # Check preset structure
    preset = data["presets"][0]
    assert "id" in preset
    assert "name" in preset
    assert "description" in preset
    assert "parameters" in preset


@pytest.mark.asyncio
async def test_list_preset_categories(client: AsyncClient):
    """Test listing preset categories."""
    response = await client.get("/api/presets/categories")

    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert len(data["categories"]) > 0
    cat = data["categories"][0]
    assert "name" in cat
    assert "count" in cat


@pytest.mark.asyncio
async def test_get_preset_basic_detection(client: AsyncClient):
    """Test getting the basic_detection preset."""
    response = await client.get("/api/presets/basic_detection")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "basic_detection"
    assert data["name"] == "Basic Element Detection"
    assert "parameters" in data
    assert len(data["parameters"]) > 0
    # Check for required parameters
    param_names = [p["name"] for p in data["parameters"]]
    assert "signal_path" in param_names
    assert "references_path" in param_names


@pytest.mark.asyncio
async def test_get_preset_batch_analysis(client: AsyncClient):
    """Test getting the batch_analysis preset."""
    response = await client.get("/api/presets/batch_analysis")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "batch_analysis"
    assert "parameters" in data


@pytest.mark.asyncio
async def test_get_preset_full_pipeline(client: AsyncClient):
    """Test getting the full_pipeline preset."""
    response = await client.get("/api/presets/full_pipeline")

    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "full_pipeline"
    # Full pipeline has background subtraction
    param_names = [p["name"] for p in data["parameters"]]
    assert "background_dir" in param_names


@pytest.mark.asyncio
async def test_get_preset_not_found(client: AsyncClient):
    """Test getting non-existent preset returns 404."""
    response = await client.get("/api/presets/nonexistent_preset")

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_get_preset_parameters(client: AsyncClient):
    """Test getting parameters for a preset."""
    response = await client.get("/api/presets/basic_detection/parameters")

    assert response.status_code == 200
    data = response.json()
    assert data["preset_id"] == "basic_detection"
    assert "parameters" in data
    assert "grouped" in data
    assert "required_count" in data
    assert data["required_count"] >= 2  # signal_path and references_path are required


@pytest.mark.asyncio
async def test_get_preset_parameters_not_found(client: AsyncClient):
    """Test getting parameters for non-existent preset."""
    response = await client.get("/api/presets/nonexistent/parameters")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_validate_preset_parameters_valid(client: AsyncClient):
    """Test validating valid preset parameters."""
    response = await client.post(
        "/api/presets/basic_detection/validate",
        json={
            "signal_path": "/path/to/signal.txt",
            "references_path": "/path/to/refs",
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["errors"] == []


@pytest.mark.asyncio
async def test_validate_preset_parameters_missing_required(client: AsyncClient):
    """Test validating preset parameters with missing required fields."""
    response = await client.post(
        "/api/presets/basic_detection/validate",
        json={
            # Missing signal_path and references_path
            "species_filter": "Cu,Fe"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert len(data["errors"]) >= 2  # Both required params missing


@pytest.mark.asyncio
async def test_validate_preset_parameters_invalid_type(client: AsyncClient):
    """Test validating preset parameters with invalid types."""
    response = await client.post(
        "/api/presets/basic_detection/validate",
        json={
            "signal_path": "/path/to/signal.txt",
            "references_path": "/path/to/refs",
            "detection_threshold": "not a number"  # Should be number
        }
    )

    assert response.status_code == 200
    data = response.json()
    # Should have error for invalid type
    assert len(data["errors"]) > 0


@pytest.mark.asyncio
async def test_validate_preset_not_found(client: AsyncClient):
    """Test validating parameters for non-existent preset."""
    response = await client.post(
        "/api/presets/nonexistent/validate",
        json={}
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_build_pipeline_from_preset(client: AsyncClient):
    """Test building a pipeline from a preset."""
    response = await client.post(
        "/api/presets/basic_detection/build-pipeline",
        json={
            "signal_path": "/path/to/signal.txt",
            "references_path": "/path/to/refs"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0
    # Should have meta with preset info
    assert data.get("meta", {}).get("preset_id") == "basic_detection"


@pytest.mark.asyncio
async def test_build_pipeline_from_preset_not_found(client: AsyncClient):
    """Test building pipeline from non-existent preset."""
    response = await client.post(
        "/api/presets/nonexistent/build-pipeline",
        json={}
    )

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_preset_has_educational_content(client: AsyncClient):
    """Test that presets have educational content."""
    for preset_id in ["basic_detection", "batch_analysis", "full_pipeline"]:
        response = await client.get(f"/api/presets/{preset_id}")
        assert response.status_code == 200
        data = response.json()
        assert len(data.get("explanation", "")) > 0, f"{preset_id} missing explanation"
        assert len(data.get("use_cases", [])) > 0, f"{preset_id} missing use_cases"


@pytest.mark.asyncio
async def test_preset_parameters_have_descriptions(client: AsyncClient):
    """Test that preset parameters have descriptions."""
    response = await client.get("/api/presets/basic_detection")
    assert response.status_code == 200
    data = response.json()

    for param in data["parameters"]:
        assert "name" in param
        assert "label" in param
        assert "type" in param
        # Required params should have description
        if param.get("required"):
            assert param.get("description"), f"Parameter {param['name']} missing description"
