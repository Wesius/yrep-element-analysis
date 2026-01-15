"""Tests for the visualizations API endpoints."""

import pytest
from httpx import AsyncClient


# Sample signal data
SAMPLE_SIGNAL = {
    "wavelength": [300.0, 350.0, 400.0, 450.0, 500.0, 550.0, 600.0],
    "intensity": [0.1, 0.3, 0.8, 1.0, 0.7, 0.4, 0.2],
    "meta": {"source": "test"}
}

# Sample detection result
SAMPLE_DETECTION = {
    "signal": SAMPLE_SIGNAL,
    "detections": [
        {"species": "Cu", "score": 0.92, "meta": {"bands_hit": 12, "primary_wavelength": 324.7}},
        {"species": "Fe", "score": 0.75, "meta": {"bands_hit": 8, "primary_wavelength": 358.1}},
        {"species": "Pb", "score": 0.45, "meta": {"bands_hit": 5}},
    ],
    "meta": {"fit_R2": 0.95}
}


@pytest.mark.asyncio
async def test_visualize_signal(client: AsyncClient):
    """Test generating visualization for a signal."""
    response = await client.post(
        "/api/visualizations/signal",
        json=SAMPLE_SIGNAL,
        params={"title": "Test Signal"}
    )

    assert response.status_code == 200
    data = response.json()
    assert "plot" in data
    plot = data["plot"]
    assert plot["title"] == "Test Signal"
    assert len(plot["series"]) > 0
    series = plot["series"][0]
    assert len(series["x"]) == len(SAMPLE_SIGNAL["wavelength"])
    assert len(series["y"]) == len(SAMPLE_SIGNAL["intensity"])


@pytest.mark.asyncio
async def test_visualize_signal_normalized(client: AsyncClient):
    """Test generating normalized signal visualization."""
    response = await client.post(
        "/api/visualizations/signal",
        json=SAMPLE_SIGNAL,
        params={"normalize": "true"}
    )

    assert response.status_code == 200
    data = response.json()
    plot = data["plot"]
    assert "Normalized" in plot["y_label"]


@pytest.mark.asyncio
async def test_visualize_signal_summary(client: AsyncClient):
    """Test signal visualization includes summary."""
    response = await client.post(
        "/api/visualizations/signal",
        json=SAMPLE_SIGNAL
    )

    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "7 points" in data["summary"] or "7" in data["summary"]


@pytest.mark.asyncio
async def test_visualize_detection(client: AsyncClient):
    """Test generating visualization for detection results."""
    response = await client.post(
        "/api/visualizations/detection",
        json=SAMPLE_DETECTION
    )

    assert response.status_code == 200
    data = response.json()
    assert "plot" in data
    assert "summary" in data
    # Summary should mention detections
    assert "3" in data["summary"] or "species" in data["summary"].lower()


@pytest.mark.asyncio
async def test_visualize_comparison(client: AsyncClient):
    """Test generating comparison visualization of multiple signals via from-data."""
    signal2 = {
        "wavelength": [300.0, 400.0, 500.0, 600.0],
        "intensity": [0.2, 0.6, 0.9, 0.3],
        "meta": {}
    }

    response = await client.post(
        "/api/visualizations/from-data",
        json={
            "type": "comparison",
            "data": {"signals": [SAMPLE_SIGNAL, signal2]},
            "options": {"title": "Signal Comparison"}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "plot" in data
    plot = data["plot"]
    assert len(plot["series"]) == 2  # Two signals
    assert plot["title"] == "Signal Comparison"


@pytest.mark.asyncio
async def test_visualize_comparison_with_labels(client: AsyncClient):
    """Test comparison visualization with custom labels via from-data."""
    signal2 = {
        "wavelength": [300.0, 400.0, 500.0],
        "intensity": [0.1, 0.5, 0.2],
        "meta": {}
    }

    response = await client.post(
        "/api/visualizations/from-data",
        json={
            "type": "comparison",
            "data": {"signals": [SAMPLE_SIGNAL, signal2], "labels": ["Sample A", "Sample B"]},
            "options": {}
        }
    )

    assert response.status_code == 200
    data = response.json()
    series_names = [s["name"] for s in data["plot"]["series"]]
    assert "Sample A" in series_names
    assert "Sample B" in series_names


@pytest.mark.asyncio
async def test_visualize_comparison_empty(client: AsyncClient):
    """Test comparison with empty signal list returns error."""
    response = await client.post(
        "/api/visualizations/from-data",
        json={
            "type": "comparison",
            "data": {"signals": []},
            "options": {}
        }
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_visualize_from_data_signal(client: AsyncClient):
    """Test flexible visualization endpoint with signal type."""
    response = await client.post(
        "/api/visualizations/from-data",
        json={
            "type": "signal",
            "data": SAMPLE_SIGNAL,
            "options": {"title": "From Data Test", "normalize": False}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "plot" in data


@pytest.mark.asyncio
async def test_visualize_from_data_detection(client: AsyncClient):
    """Test flexible visualization endpoint with detection type."""
    response = await client.post(
        "/api/visualizations/from-data",
        json={
            "type": "detection",
            "data": SAMPLE_DETECTION,
            "options": {}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "plot" in data


@pytest.mark.asyncio
async def test_visualize_from_data_comparison(client: AsyncClient):
    """Test flexible visualization endpoint with comparison type."""
    response = await client.post(
        "/api/visualizations/from-data",
        json={
            "type": "comparison",
            "data": {"signals": [SAMPLE_SIGNAL, SAMPLE_SIGNAL]},
            "options": {"title": "Comparison"}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert "plot" in data


@pytest.mark.asyncio
async def test_visualize_from_data_preprocessing_stages(client: AsyncClient):
    """Test visualization of preprocessing stages."""
    stages = [
        {"name": "Raw", "wavelength": [300, 400, 500], "intensity": [1.0, 1.2, 0.8]},
        {"name": "Trimmed", "wavelength": [300, 400, 500], "intensity": [1.0, 1.2, 0.8]},
        {"name": "Continuum Removed", "wavelength": [300, 400, 500], "intensity": [0.1, 0.3, -0.1]},
    ]

    response = await client.post(
        "/api/visualizations/from-data",
        json={
            "type": "preprocessing_stages",
            "data": {"stages": stages},
            "options": {"title": "Preprocessing Steps"}
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["plot"]["series"]) == 3
    assert "3" in data["summary"]


@pytest.mark.asyncio
async def test_visualize_from_data_invalid_type(client: AsyncClient):
    """Test visualization with invalid type returns validation error."""
    response = await client.post(
        "/api/visualizations/from-data",
        json={
            "type": "invalid_type",
            "data": {},
            "options": {}
        }
    )

    # Pydantic validates Literal types before the route runs, returning 422
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_visualize_from_data_missing_data(client: AsyncClient):
    """Test visualization with missing required data returns error."""
    response = await client.post(
        "/api/visualizations/from-data",
        json={
            "type": "signal",
            "data": {},  # Missing wavelength and intensity
            "options": {}
        }
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_get_plot_options(client: AsyncClient):
    """Test getting available plot options."""
    response = await client.get("/api/visualizations/plot-options")

    assert response.status_code == 200
    data = response.json()
    assert "visualization_types" in data
    assert "plot_options" in data
    assert "series_options" in data
    # Check known visualization types
    viz_types = [v["type"] for v in data["visualization_types"]]
    assert "signal" in viz_types
    assert "detection" in viz_types
    assert "comparison" in viz_types


@pytest.mark.asyncio
async def test_plot_has_default_labels(client: AsyncClient):
    """Test that plots have default axis labels."""
    response = await client.post(
        "/api/visualizations/signal",
        json=SAMPLE_SIGNAL
    )

    assert response.status_code == 200
    plot = response.json()["plot"]
    assert "Wavelength" in plot["x_label"]
    assert "Intensity" in plot["y_label"]


@pytest.mark.asyncio
async def test_plot_series_has_properties(client: AsyncClient):
    """Test that plot series have required properties."""
    response = await client.post(
        "/api/visualizations/signal",
        json=SAMPLE_SIGNAL
    )

    assert response.status_code == 200
    series = response.json()["plot"]["series"][0]
    assert "name" in series
    assert "x" in series
    assert "y" in series
    assert "type" in series
