"""Tests for the nodes API endpoints."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_list_all_nodes(client: AsyncClient):
    """Test listing all available nodes."""
    response = await client.get("/api/nodes/")

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Check structure of first node
    node = data[0]
    assert "identifier" in node
    assert "title" in node
    assert "category" in node


@pytest.mark.asyncio
async def test_list_nodes_by_category(client: AsyncClient):
    """Test filtering nodes by category."""
    response = await client.get("/api/nodes/", params={"category": "Preprocess"})

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # All nodes should be in Preprocess category
    assert all(n["category"] == "Preprocess" for n in data)


@pytest.mark.asyncio
async def test_list_nodes_invalid_category(client: AsyncClient):
    """Test filtering by invalid category returns 404."""
    response = await client.get("/api/nodes/", params={"category": "InvalidCategory"})

    assert response.status_code == 404
    assert "Category not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_list_categories(client: AsyncClient):
    """Test listing node categories."""
    response = await client.get("/api/nodes/categories")

    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "order" in data
    assert len(data["categories"]) > 0
    # Check structure
    cat = data["categories"][0]
    assert "name" in cat
    assert "count" in cat
    # Known categories should exist
    cat_names = [c["name"] for c in data["categories"]]
    assert "I/O" in cat_names
    assert "Preprocess" in cat_names
    assert "Detection" in cat_names


@pytest.mark.asyncio
async def test_get_grouped_nodes(client: AsyncClient):
    """Test getting nodes grouped by category."""
    response = await client.get("/api/nodes/grouped")

    assert response.status_code == 200
    data = response.json()
    assert "categories" in data
    assert "groups" in data
    # Groups should have same keys as categories
    for cat in data["categories"]:
        assert cat in data["groups"]


@pytest.mark.asyncio
async def test_get_node_by_identifier(client: AsyncClient):
    """Test getting a specific node by identifier."""
    response = await client.get("/api/nodes/trim")

    assert response.status_code == 200
    data = response.json()
    assert data["identifier"] == "trim"
    assert data["title"] == "Trim"
    assert data["category"] == "Preprocess"
    assert "inputs" in data
    assert "outputs" in data
    assert "config_fields" in data


@pytest.mark.asyncio
async def test_get_node_not_found(client: AsyncClient):
    """Test getting non-existent node returns 404."""
    response = await client.get("/api/nodes/nonexistent_node")

    assert response.status_code == 404
    assert "Node not found" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_node_help(client: AsyncClient):
    """Test getting educational content for a node."""
    response = await client.get("/api/nodes/detect_nnls/help")

    assert response.status_code == 200
    data = response.json()
    assert data["identifier"] == "detect_nnls"
    assert "title" in data
    assert "description" in data
    assert "explanation" in data
    assert "tips" in data
    assert "related_nodes" in data
    # Should have educational content
    assert len(data["explanation"]) > 0
    assert isinstance(data["tips"], list)


@pytest.mark.asyncio
async def test_get_node_help_not_found(client: AsyncClient):
    """Test getting help for non-existent node returns 404."""
    response = await client.get("/api/nodes/nonexistent/help")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_search_nodes_by_title(client: AsyncClient):
    """Test searching nodes by title."""
    response = await client.get("/api/nodes/search/trim")

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "trim"
    assert data["count"] > 0
    # Trim node should be in results
    identifiers = [n["identifier"] for n in data["results"]]
    assert "trim" in identifiers


@pytest.mark.asyncio
async def test_search_nodes_by_category(client: AsyncClient):
    """Test searching nodes matches category."""
    response = await client.get("/api/nodes/search/detection")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] > 0
    # detect_nnls should be found via category
    identifiers = [n["identifier"] for n in data["results"]]
    assert "detect_nnls" in identifiers


@pytest.mark.asyncio
async def test_search_nodes_no_results(client: AsyncClient):
    """Test search with no matches returns empty results."""
    response = await client.get("/api/nodes/search/xyznonexistent123")

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 0
    assert data["results"] == []


@pytest.mark.asyncio
async def test_node_has_educational_content(client: AsyncClient):
    """Test that nodes have educational content populated."""
    # Check a few key nodes
    for identifier in ["trim", "continuum_remove_arpls", "detect_nnls", "build_templates"]:
        response = await client.get(f"/api/nodes/{identifier}")
        assert response.status_code == 200
        data = response.json()
        assert len(data["description"]) > 0, f"{identifier} missing description"
        assert len(data["explanation"]) > 0, f"{identifier} missing explanation"
        assert len(data["tips"]) > 0, f"{identifier} missing tips"
