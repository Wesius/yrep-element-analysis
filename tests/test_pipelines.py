"""Tests for the pipelines API endpoints."""

import pytest
from httpx import AsyncClient


# Sample valid pipeline graph
VALID_SIMPLE_PIPELINE = {
    "version": 1,
    "name": "Test Pipeline",
    "nodes": [
        {"id": "1", "identifier": "load_signal", "config": {"path": "/tmp/test.txt"}, "position": {"x": 0, "y": 0}},
        {"id": "2", "identifier": "trim", "config": {"min_nm": 300, "max_nm": 600}, "position": {"x": 200, "y": 0}},
    ],
    "edges": [
        {"id": "e1", "source_node": "1", "source_port": 0, "target_node": "2", "target_port": 0}
    ],
}

# Pipeline with cycle
CYCLIC_PIPELINE = {
    "version": 1,
    "nodes": [
        {"id": "1", "identifier": "trim", "config": {}, "position": {"x": 0, "y": 0}},
        {"id": "2", "identifier": "trim", "config": {}, "position": {"x": 200, "y": 0}},
    ],
    "edges": [
        {"id": "e1", "source_node": "1", "source_port": 0, "target_node": "2", "target_port": 0},
        {"id": "e2", "source_node": "2", "source_port": 0, "target_node": "1", "target_port": 0},
    ],
}

# Pipeline with invalid node type
INVALID_NODE_PIPELINE = {
    "version": 1,
    "nodes": [
        {"id": "1", "identifier": "nonexistent_node_type", "config": {}, "position": {"x": 0, "y": 0}},
    ],
    "edges": [],
}

# Pipeline with disconnected required input
DISCONNECTED_INPUT_PIPELINE = {
    "version": 1,
    "nodes": [
        {"id": "1", "identifier": "trim", "config": {}, "position": {"x": 0, "y": 0}},
    ],
    "edges": [],
}

# More complex valid pipeline
COMPLEX_PIPELINE = {
    "version": 1,
    "name": "Detection Pipeline",
    "nodes": [
        {"id": "1", "identifier": "load_signal", "config": {"path": "/test.txt"}, "position": {"x": 0, "y": 0}},
        {"id": "2", "identifier": "load_references", "config": {"directory": "/refs"}, "position": {"x": 0, "y": 100}},
        {"id": "3", "identifier": "trim", "config": {"min_nm": 300, "max_nm": 600}, "position": {"x": 200, "y": 0}},
        {"id": "4", "identifier": "build_templates", "config": {"fwhm_nm": 0.75}, "position": {"x": 400, "y": 50}},
        {"id": "5", "identifier": "detect_nnls", "config": {"presence_threshold": 0.02}, "position": {"x": 600, "y": 50}},
    ],
    "edges": [
        {"id": "e1", "source_node": "1", "source_port": 0, "target_node": "3", "target_port": 0},
        {"id": "e2", "source_node": "3", "source_port": 0, "target_node": "4", "target_port": 0},
        {"id": "e3", "source_node": "2", "source_port": 0, "target_node": "4", "target_port": 1},
        {"id": "e4", "source_node": "3", "source_port": 0, "target_node": "5", "target_port": 0},
        {"id": "e5", "source_node": "4", "source_port": 0, "target_node": "5", "target_port": 1},
    ],
}


@pytest.mark.asyncio
async def test_validate_valid_pipeline(client: AsyncClient):
    """Test validation of a valid pipeline."""
    response = await client.post("/api/pipelines/validate", json=VALID_SIMPLE_PIPELINE)

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["errors"] == []
    assert data["node_count"] == 2
    assert data["edge_count"] == 1


@pytest.mark.asyncio
async def test_validate_cyclic_pipeline(client: AsyncClient):
    """Test validation detects cycles."""
    response = await client.post("/api/pipelines/validate", json=CYCLIC_PIPELINE)

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert "cycle" in str(data["errors"]).lower()


@pytest.mark.asyncio
async def test_validate_invalid_node_type(client: AsyncClient):
    """Test validation detects invalid node types."""
    response = await client.post("/api/pipelines/validate", json=INVALID_NODE_PIPELINE)

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert any("unknown" in e.lower() for e in data["errors"])


@pytest.mark.asyncio
async def test_validate_disconnected_required_input(client: AsyncClient):
    """Test validation detects disconnected required inputs."""
    response = await client.post("/api/pipelines/validate", json=DISCONNECTED_INPUT_PIPELINE)

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert any("not connected" in e.lower() for e in data["errors"])


@pytest.mark.asyncio
async def test_validate_empty_pipeline(client: AsyncClient):
    """Test validation of empty pipeline."""
    empty = {"version": 1, "nodes": [], "edges": []}
    response = await client.post("/api/pipelines/validate", json=empty)

    assert response.status_code == 200
    data = response.json()
    # Empty pipeline is technically valid (no errors)
    assert data["node_count"] == 0


@pytest.mark.asyncio
async def test_analyze_valid_pipeline(client: AsyncClient):
    """Test analysis of a valid pipeline."""
    response = await client.post("/api/pipelines/analyze", json=COMPLEX_PIPELINE)

    assert response.status_code == 200
    data = response.json()
    assert "execution_order" in data
    assert "source_nodes" in data
    assert "sink_nodes" in data
    assert "dependencies" in data
    # Check execution order length
    assert len(data["execution_order"]) == 5
    # Source nodes should be load nodes
    assert "1" in data["source_nodes"]  # load_signal
    assert "2" in data["source_nodes"]  # load_references
    # Sink should be detect_nnls
    assert "5" in data["sink_nodes"]


@pytest.mark.asyncio
async def test_analyze_cyclic_pipeline_error(client: AsyncClient):
    """Test analysis of cyclic pipeline returns error."""
    response = await client.post("/api/pipelines/analyze", json=CYCLIC_PIPELINE)

    assert response.status_code == 400
    data = response.json()
    assert "errors" in data["detail"]


@pytest.mark.asyncio
async def test_analyze_invalid_pipeline_error(client: AsyncClient):
    """Test analysis of invalid pipeline returns error."""
    response = await client.post("/api/pipelines/analyze", json=INVALID_NODE_PIPELINE)

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_execute_valid_pipeline(client: AsyncClient):
    """Test execution of valid pipeline."""
    request = {"graph": VALID_SIMPLE_PIPELINE}
    response = await client.post("/api/pipelines/execute", json=request)

    assert response.status_code == 200
    data = response.json()
    # Should return partial status (placeholder implementation)
    assert data["status"] in ["partial", "success"]
    assert "node_results" in data
    assert "execution_order" in data
    assert len(data["node_results"]) == 2


@pytest.mark.asyncio
async def test_execute_invalid_pipeline(client: AsyncClient):
    """Test execution of invalid pipeline returns error status."""
    request = {"graph": INVALID_NODE_PIPELINE}
    response = await client.post("/api/pipelines/execute", json=request)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert data["error"] is not None


@pytest.mark.asyncio
async def test_execute_cyclic_pipeline(client: AsyncClient):
    """Test execution of cyclic pipeline returns error status."""
    request = {"graph": CYCLIC_PIPELINE}
    response = await client.post("/api/pipelines/execute", json=request)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "error"
    assert "cycle" in data["error"].lower()


@pytest.mark.asyncio
async def test_create_from_template_basic(client: AsyncClient):
    """Test creating pipeline from basic_detection template."""
    response = await client.post(
        "/api/pipelines/from-template",
        params={"template_name": "basic_detection"},
        json={"signal_path": "/test/signal.txt", "references_path": "/test/refs"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) > 0


@pytest.mark.asyncio
async def test_create_from_template_batch(client: AsyncClient):
    """Test creating pipeline from batch_analysis template."""
    response = await client.post(
        "/api/pipelines/from-template",
        params={"template_name": "batch_analysis"},
        json={}
    )

    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    # Batch template should have group and select nodes
    identifiers = [n["identifier"] for n in data["nodes"]]
    assert "group_signals" in identifiers or "load_signal_batch" in identifiers


@pytest.mark.asyncio
async def test_create_from_template_full(client: AsyncClient):
    """Test creating pipeline from full_pipeline template."""
    response = await client.post(
        "/api/pipelines/from-template",
        params={"template_name": "full_pipeline"},
        json={}
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["nodes"]) > 5  # Full pipeline should have many nodes


@pytest.mark.asyncio
async def test_create_from_template_invalid(client: AsyncClient):
    """Test creating pipeline from invalid template returns 404."""
    response = await client.post(
        "/api/pipelines/from-template",
        params={"template_name": "nonexistent_template"},
        json={}
    )

    assert response.status_code == 404
    assert "not found" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_validate_duplicate_node_ids(client: AsyncClient):
    """Test validation detects duplicate node IDs."""
    pipeline = {
        "version": 1,
        "nodes": [
            {"id": "1", "identifier": "load_signal", "config": {}, "position": {"x": 0, "y": 0}},
            {"id": "1", "identifier": "trim", "config": {}, "position": {"x": 100, "y": 0}},
        ],
        "edges": [],
    }
    response = await client.post("/api/pipelines/validate", json=pipeline)

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert any("duplicate" in e.lower() for e in data["errors"])


@pytest.mark.asyncio
async def test_validate_invalid_edge_references(client: AsyncClient):
    """Test validation detects edges referencing non-existent nodes."""
    pipeline = {
        "version": 1,
        "nodes": [
            {"id": "1", "identifier": "load_signal", "config": {}, "position": {"x": 0, "y": 0}},
        ],
        "edges": [
            {"id": "e1", "source_node": "1", "source_port": 0, "target_node": "999", "target_port": 0}
        ],
    }
    response = await client.post("/api/pipelines/validate", json=pipeline)

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert any("unknown" in e.lower() for e in data["errors"])
