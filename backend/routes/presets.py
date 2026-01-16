"""Preset workflow API routes."""

from pathlib import Path
from typing import Dict, List, Any
from fastapi import APIRouter, HTTPException

from backend.models.presets import Preset, PresetParameter, PresetList

router = APIRouter()


# =============================================================================
# PRESET DEFINITIONS
# =============================================================================

_PRESETS: Dict[str, Preset] = {}


def _register_preset(preset: Preset) -> Preset:
    """Register a preset."""
    _PRESETS[preset.id] = preset
    return preset


# Preset 1: Basic Element Detection
_register_preset(Preset(
    id="basic_detection",
    name="Basic Element Detection",
    description="Detect elements in a single spectrum file with standard preprocessing",
    category="Detection",
    icon="search",
    explanation="""
This preset performs a complete element detection analysis on a single spectrum file.

**Workflow Steps:**
1. Load the spectrum file
2. Trim to visible range (300-600 nm)
3. Resample to uniform grid (1500 points)
4. Remove baseline using arPLS algorithm
5. Build detection templates from reference database
6. Detect elements using NNLS fitting

**Best For:**
- Quick element identification in unknown samples
- Single-file analysis
- Learning spectroscopy basics
    """.strip(),
    use_cases=[
        "Quick element identification",
        "Single sample screening",
        "Educational demonstrations",
    ],
    parameters=[
        PresetParameter(
            name="signal_path",
            label="Spectrum File",
            type="file",
            default="",
            description="Path to spectrum file (.txt with wavelength, intensity columns)",
            required=True,
            group="Input",
        ),
        PresetParameter(
            name="references_path",
            label="Reference Database",
            type="directory",
            default="",
            description="Directory containing spectral line reference files",
            required=True,
            group="Input",
        ),
        PresetParameter(
            name="species_filter",
            label="Target Elements",
            type="string",
            default="",
            description="Comma-separated elements to search for (e.g., 'Cu,Fe,Pb'). Leave empty for all.",
            required=False,
            group="Detection",
        ),
        PresetParameter(
            name="min_wavelength",
            label="Min Wavelength (nm)",
            type="number",
            default=300,
            description="Lower wavelength bound for analysis",
            required=False,
            group="Processing",
        ),
        PresetParameter(
            name="max_wavelength",
            label="Max Wavelength (nm)",
            type="number",
            default=600,
            description="Upper wavelength bound for analysis",
            required=False,
            group="Processing",
        ),
        PresetParameter(
            name="detection_threshold",
            label="Detection Threshold",
            type="number",
            default=0.02,
            description="Minimum score to report a detection (0-1)",
            required=False,
            group="Detection",
        ),
    ],
    pipeline_template={
        "template": "basic_detection",
        "parameter_mapping": {
            "signal_path": "nodes.load_signal.config.path",
            "references_path": "nodes.load_references.config.directory",
            "species_filter": "nodes.build_templates.config.species_filter",
            "min_wavelength": "nodes.trim.config.min_nm",
            "max_wavelength": "nodes.trim.config.max_nm",
            "detection_threshold": "nodes.detect_nnls.config.presence_threshold",
        },
    },
))

# Preset 2: Batch Analysis
_register_preset(Preset(
    id="batch_analysis",
    name="Batch Spectrum Analysis",
    description="Analyze multiple spectra from a measurement session with automatic grouping",
    category="Detection",
    icon="layers",
    explanation="""
This preset processes multiple spectra from a directory, automatically groups similar
measurements, and produces a clean averaged result for detection.

**Workflow Steps:**
1. Load all spectra from a directory
2. Group similar spectra using cosine similarity
3. Select the highest-quality group
4. Average signals within the group for noise reduction
5. Apply standard preprocessing
6. Detect elements

**Best For:**
- Processing measurement sessions with multiple acquisitions
- Reducing noise through averaging
- Handling samples with occasional bad measurements
    """.strip(),
    use_cases=[
        "Multi-acquisition measurement sessions",
        "Noisy sample analysis",
        "Quality-controlled batch processing",
    ],
    parameters=[
        PresetParameter(
            name="signal_dir",
            label="Spectra Directory",
            type="directory",
            default="",
            description="Directory containing spectrum files",
            required=True,
            group="Input",
        ),
        PresetParameter(
            name="references_path",
            label="Reference Database",
            type="directory",
            default="",
            description="Directory containing reference files",
            required=True,
            group="Input",
        ),
        PresetParameter(
            name="min_quality",
            label="Minimum Quality",
            type="number",
            default=0.0,
            description="Reject groups below this quality score (0-1)",
            required=False,
            group="Processing",
        ),
        PresetParameter(
            name="average_points",
            label="Average Resolution",
            type="number",
            default=1500,
            description="Number of points in averaged signal",
            required=False,
            group="Processing",
        ),
    ],
    pipeline_template={
        "template": "batch_analysis",
        "parameter_mapping": {
            "signal_dir": "nodes.load_signal_batch.config.directory",
            "references_path": "nodes.load_references.config.directory",
            "min_quality": "nodes.select_best_group.config.min_quality",
            "average_points": "nodes.average_signals.config.n_points",
        },
    },
))

# Preset 3: Full Pipeline with Background Subtraction
_register_preset(Preset(
    id="full_pipeline",
    name="Full Analysis Pipeline",
    description="Complete analysis with background subtraction, alignment, and thorough preprocessing",
    category="Detection",
    icon="activity",
    explanation="""
This preset provides the most thorough analysis workflow, including background
subtraction and wavelength alignment optimization.

**Workflow Steps:**
1. Load sample spectra and background measurements
2. Group and average both signal and background
3. Subtract background from signal
4. Apply dual continuum removal (arPLS + rolling)
5. Build templates and optimize wavelength alignment
6. Detect elements with refined templates

**Best For:**
- High-accuracy analysis
- Samples with significant background
- Research-grade measurements
    """.strip(),
    use_cases=[
        "Research-grade analysis",
        "Samples with fluorescence background",
        "Maximum detection accuracy",
    ],
    parameters=[
        PresetParameter(
            name="signal_dir",
            label="Sample Spectra Directory",
            type="directory",
            default="",
            description="Directory containing sample spectrum files",
            required=True,
            group="Input",
        ),
        PresetParameter(
            name="background_dir",
            label="Background Directory",
            type="directory",
            default="",
            description="Directory containing background measurements",
            required=True,
            group="Input",
        ),
        PresetParameter(
            name="references_path",
            label="Reference Database",
            type="directory",
            default="",
            description="Directory containing reference files",
            required=True,
            group="Input",
        ),
        PresetParameter(
            name="continuum_strength",
            label="Continuum Removal Strength",
            type="number",
            default=0.5,
            description="Baseline removal aggressiveness (0-1)",
            required=False,
            group="Processing",
        ),
        PresetParameter(
            name="shift_spread",
            label="Alignment Search Range (nm)",
            type="number",
            default=0.5,
            description="Wavelength range to search for optimal alignment",
            required=False,
            group="Alignment",
        ),
        PresetParameter(
            name="fwhm",
            label="Template FWHM (nm)",
            type="number",
            default=0.75,
            description="Gaussian peak width for templates",
            required=False,
            group="Templates",
        ),
    ],
    pipeline_template={
        "template": "full_pipeline",
        "parameter_mapping": {
            "signal_dir": "nodes.load_signal_batch_1.config.directory",
            "background_dir": "nodes.load_signal_batch_2.config.directory",
            "references_path": "nodes.load_references.config.directory",
            "continuum_strength": "nodes.continuum_remove_arpls.config.strength",
            "shift_spread": "nodes.shift_search.config.spread_nm",
            "fwhm": "nodes.build_templates.config.fwhm_nm",
        },
    },
))


# =============================================================================
# API ROUTES
# =============================================================================

@router.get("/", response_model=PresetList)
async def list_presets():
    """List all available presets.

    Returns presets with their parameters and educational content.
    """
    presets = list(_PRESETS.values())
    categories = sorted(set(p.category for p in presets))

    return PresetList(
        presets=presets,
        categories=categories,
    )


@router.get("/categories")
async def list_preset_categories():
    """List preset categories with counts."""
    by_category: Dict[str, List[Preset]] = {}
    for preset in _PRESETS.values():
        by_category.setdefault(preset.category, []).append(preset)

    return {
        "categories": [
            {"name": cat, "count": len(presets)}
            for cat, presets in sorted(by_category.items())
        ]
    }


@router.get("/{preset_id}", response_model=Preset)
async def get_preset(preset_id: str):
    """Get a specific preset by ID."""
    if preset_id not in _PRESETS:
        raise HTTPException(
            status_code=404,
            detail=f"Preset not found: {preset_id}. Available: {list(_PRESETS.keys())}"
        )
    return _PRESETS[preset_id]


@router.get("/{preset_id}/parameters")
async def get_preset_parameters(preset_id: str):
    """Get just the parameters for a preset.

    Useful for rendering parameter forms.
    """
    if preset_id not in _PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset not found: {preset_id}")

    preset = _PRESETS[preset_id]

    # Group parameters
    grouped: Dict[str, List[PresetParameter]] = {}
    for param in preset.parameters:
        group = param.group or "General"
        grouped.setdefault(group, []).append(param)

    return {
        "preset_id": preset_id,
        "preset_name": preset.name,
        "parameters": preset.parameters,
        "grouped": grouped,
        "required_count": sum(1 for p in preset.parameters if p.required),
    }


@router.post("/{preset_id}/validate")
async def validate_preset_parameters(
    preset_id: str,
    parameters: Dict[str, Any],
):
    """Validate parameters for a preset.

    Checks required parameters and type compatibility.
    """
    if preset_id not in _PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset not found: {preset_id}")

    preset = _PRESETS[preset_id]
    errors = []
    warnings = []

    for param in preset.parameters:
        value = parameters.get(param.name)

        # Check required
        if param.required and (value is None or value == ""):
            errors.append(f"Required parameter '{param.label}' is missing")
            continue

        # Type validation
        if value is not None and value != "":
            if param.type == "number":
                try:
                    float(value)
                except (TypeError, ValueError):
                    errors.append(f"Parameter '{param.label}' must be a number")
            elif param.type == "boolean":
                if not isinstance(value, bool):
                    warnings.append(f"Parameter '{param.label}' should be boolean")
            elif param.type == "file":
                # Validate file path exists
                path = Path(value).expanduser()
                if not path.exists():
                    errors.append(f"File not found: '{param.label}' ({value})")
                elif not path.is_file():
                    errors.append(f"Path is not a file: '{param.label}' ({value})")
            elif param.type == "directory":
                # Validate directory path exists
                path = Path(value).expanduser()
                if not path.exists():
                    errors.append(f"Directory not found: '{param.label}' ({value})")
                elif not path.is_dir():
                    errors.append(f"Path is not a directory: '{param.label}' ({value})")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


@router.post("/{preset_id}/build-pipeline")
async def build_pipeline_from_preset(
    preset_id: str,
    parameters: Dict[str, Any],
):
    """Build a pipeline graph from a preset with user parameters.

    Returns a PipelineGraph that can be executed or edited.
    """
    if preset_id not in _PRESETS:
        raise HTTPException(status_code=404, detail=f"Preset not found: {preset_id}")

    preset = _PRESETS[preset_id]

    # Import pipeline template builder
    from backend.routes.pipelines import (
        _template_basic_detection,
        _template_batch_analysis,
        _template_full_pipeline,
    )

    template_name = preset.pipeline_template.get("template", preset_id)
    templates = {
        "basic_detection": _template_basic_detection,
        "batch_analysis": _template_batch_analysis,
        "full_pipeline": _template_full_pipeline,
    }

    if template_name not in templates:
        raise HTTPException(
            status_code=500,
            detail=f"Preset references unknown template: {template_name}"
        )

    # Validate parameter mappings before building
    mapping = preset.pipeline_template.get("parameter_mapping", {})
    invalid_mappings = []
    for param_name, mapping_path in mapping.items():
        # Mapping format: "nodes.<node_id>.config.<field>"
        parts = mapping_path.split(".")
        if len(parts) < 4 or parts[0] != "nodes" or parts[2] != "config":
            invalid_mappings.append(f"{param_name}: invalid mapping format '{mapping_path}'")

    if invalid_mappings:
        raise HTTPException(
            status_code=500,
            detail=f"Invalid parameter mappings: {'; '.join(invalid_mappings)}"
        )

    # Build pipeline with parameters
    pipeline = templates[template_name](parameters)
    pipeline.name = f"{preset.name} (from preset)"
    pipeline.meta["preset_id"] = preset_id
    pipeline.meta["preset_parameters"] = parameters

    return pipeline
