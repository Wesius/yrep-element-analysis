"""Export API routes for detection results."""

import csv
import io
import json
from datetime import datetime
from typing import List

from fastapi import APIRouter
from fastapi.responses import Response

from backend.models.core import (
    DetectionResult,
    ExportFormat,
    ExportRequest,
    ExportResponse,
)

router = APIRouter()


def _generate_filename(base: str | None, format: ExportFormat) -> str:
    """Generate a filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = base or "detection_results"
    return f"{base}_{timestamp}.{format.value}"


def _result_to_csv(result: DetectionResult, include_signal: bool) -> str:
    """Convert detection result to CSV format."""
    output = io.StringIO()
    writer = csv.writer(output)

    # Write detections table
    writer.writerow(["# Detections"])
    writer.writerow(["species", "score", "metadata"])
    for detection in result.detections:
        meta_str = json.dumps(detection.meta) if detection.meta else ""
        writer.writerow([detection.species, detection.score, meta_str])

    # Write analysis metadata
    writer.writerow([])
    writer.writerow(["# Analysis Metadata"])
    writer.writerow(["key", "value"])
    for key, value in result.meta.items():
        writer.writerow([key, json.dumps(value) if not isinstance(value, (str, int, float)) else value])

    # Write signal metadata
    writer.writerow([])
    writer.writerow(["# Signal Metadata"])
    writer.writerow(["key", "value"])
    for key, value in result.signal.meta.items():
        writer.writerow([key, json.dumps(value) if not isinstance(value, (str, int, float)) else value])

    # Optionally include signal data
    if include_signal:
        writer.writerow([])
        writer.writerow(["# Signal Data"])
        writer.writerow(["wavelength_nm", "intensity"])
        for wl, intensity in zip(result.signal.wavelength, result.signal.intensity):
            writer.writerow([wl, intensity])

    return output.getvalue()


def _result_to_json(result: DetectionResult, include_signal: bool) -> str:
    """Convert detection result to JSON format."""
    export_data = {
        "detections": [
            {
                "species": d.species,
                "score": d.score,
                "metadata": d.meta,
            }
            for d in result.detections
        ],
        "analysis_metadata": result.meta,
        "signal_metadata": result.signal.meta,
    }

    if include_signal:
        export_data["signal"] = {
            "wavelength": result.signal.wavelength,
            "intensity": result.signal.intensity,
        }

    return json.dumps(export_data, indent=2)


@router.post("/", response_model=ExportResponse)
async def export_result(request: ExportRequest) -> ExportResponse:
    """Export a detection result to CSV or JSON format.

    Returns the exported content as a string along with metadata.
    The client can then save this to a file or process it further.
    """
    filename = _generate_filename(request.filename, request.format)

    if request.format == ExportFormat.CSV:
        content = _result_to_csv(request.result, request.include_signal)
    else:
        content = _result_to_json(request.result, request.include_signal)

    return ExportResponse(
        filename=filename,
        format=request.format,
        content=content,
        detection_count=len(request.result.detections),
    )


@router.post("/download")
async def download_result(request: ExportRequest) -> Response:
    """Export a detection result as a downloadable file.

    Returns the file directly with appropriate headers for download.
    """
    filename = _generate_filename(request.filename, request.format)

    if request.format == ExportFormat.CSV:
        content = _result_to_csv(request.result, request.include_signal)
        media_type = "text/csv"
    else:
        content = _result_to_json(request.result, request.include_signal)
        media_type = "application/json"

    return Response(
        content=content,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        },
    )


@router.post("/batch", response_model=List[ExportResponse])
async def export_batch(
    results: List[DetectionResult],
    format: ExportFormat = ExportFormat.JSON,
    include_signal: bool = False,
) -> List[ExportResponse]:
    """Export multiple detection results at once.

    Useful for batch processing pipelines that produce multiple results.
    """
    exports = []
    for i, result in enumerate(results):
        filename = _generate_filename(f"batch_{i+1:03d}", format)

        if format == ExportFormat.CSV:
            content = _result_to_csv(result, include_signal)
        else:
            content = _result_to_json(result, include_signal)

        exports.append(ExportResponse(
            filename=filename,
            format=format,
            content=content,
            detection_count=len(result.detections),
        ))

    return exports
