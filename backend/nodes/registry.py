"""Node registry with educational content for spectral analysis pipeline.

This module defines all available pipeline nodes with detailed descriptions,
explanations, and best-practice tips to help users understand spectroscopy
analysis workflows.
"""

from typing import Dict, List, Optional
from backend.models.nodes import NodeDefinition, PortDefinition, ConfigField


CATEGORY_ORDER: List[str] = [
    "I/O",
    "Aggregate",
    "Preprocess",
    "Templates",
    "Alignment",
    "Detection",
    "Visualization",
]


def _port(name: str, type_: str, optional: bool = False, multi: bool = False) -> PortDefinition:
    """Helper to create port definitions."""
    return PortDefinition(name=name, type=type_, optional=optional, multi=multi)


def _field(
    name: str,
    type_: str,
    default,
    label: str,
    description: str = "",
    **kwargs
) -> ConfigField:
    """Helper to create config field definitions."""
    return ConfigField(
        name=name,
        type=type_,
        default=default,
        label=label,
        description=description,
        **kwargs
    )


# =============================================================================
# NODE DEFINITIONS WITH EDUCATIONAL CONTENT
# =============================================================================

_REGISTRY: Dict[str, NodeDefinition] = {}


def _register(node: NodeDefinition) -> NodeDefinition:
    """Register a node definition."""
    _REGISTRY[node.identifier] = node
    return node


# -----------------------------------------------------------------------------
# I/O Nodes
# -----------------------------------------------------------------------------

_register(NodeDefinition(
    identifier="load_signal",
    title="Load Signal",
    category="I/O",
    description="Load a single spectrum file",
    explanation="""
Loads spectral data from a text file containing wavelength and intensity columns.

**Supported Formats:**
- Two-column text files (wavelength, intensity)
- Tab or space-delimited
- Optional header lines (auto-detected)

The loaded signal becomes the starting point for your analysis pipeline.
Wavelengths should be in nanometers (nm) and intensity in arbitrary units.
    """.strip(),
    tips=[
        "Verify your file has consistent column delimiters",
        "Check wavelength units - YREP expects nanometers",
        "Use Load Signal Batch for multiple files from the same measurement session",
    ],
    related_nodes=["load_signal_batch", "load_references"],
    inputs=[],
    outputs=[_port("Signal", "Signal")],
    config_fields=[
        _field("path", "file", "", "Spectrum File",
               "Path to spectrum text file (.txt)")
    ],
    default_config={"path": ""},
))

_register(NodeDefinition(
    identifier="load_signal_batch",
    title="Load Signal Batch",
    category="I/O",
    description="Load all spectrum files from a directory",
    explanation="""
Loads multiple spectrum files from a directory for batch processing.

This is useful when you have:
- Multiple measurements of the same sample
- A measurement session with multiple spectra to average
- Samples to compare or process identically

All .txt files in the directory will be loaded as Signal objects.
    """.strip(),
    tips=[
        "Organize measurement sessions into separate directories",
        "Use Group Signals to cluster similar spectra before averaging",
        "Background measurements should be in a separate directory",
    ],
    related_nodes=["load_signal", "group_signals", "average_signals"],
    inputs=[],
    outputs=[_port("SignalBatch", "SignalBatch")],
    config_fields=[
        _field("directory", "directory", "", "Spectra Directory",
               "Directory containing spectrum files")
    ],
    default_config={"directory": ""},
))

_register(NodeDefinition(
    identifier="load_references",
    title="Load References",
    category="I/O",
    description="Load spectral line reference database",
    explanation="""
Loads reference spectral lines from a database directory (e.g., NIST atomic lines).

**Reference Database Structure:**
Each species has a file containing known emission/absorption lines with
wavelength positions and relative intensities. These are used to build
detection templates.

**Element Only Mode:**
When enabled, loads only base element entries (e.g., "Cu", "Fe").
When disabled, includes ionization states (e.g., "Cu I", "Cu II", "Fe I").
    """.strip(),
    tips=[
        "Use element_only=True for simpler detection of major elements",
        "Disable element_only when analyzing plasma/high-temperature samples",
        "Ensure your reference database covers your wavelength range",
    ],
    related_nodes=["build_templates"],
    inputs=[],
    outputs=[_port("References", "References")],
    config_fields=[
        _field("directory", "directory", "", "Reference Database",
               "Directory containing reference line files"),
        _field("element_only", "boolean", True, "Element Only",
               "Load only base elements, not ionization states"),
    ],
    default_config={"directory": "", "element_only": True},
))

# -----------------------------------------------------------------------------
# Aggregate Nodes
# -----------------------------------------------------------------------------

_register(NodeDefinition(
    identifier="group_signals",
    title="Group Signals",
    category="Aggregate",
    description="Cluster similar signals using cosine similarity",
    explanation="""
Groups signals by spectral similarity using cosine similarity clustering.

**Why Group Signals?**
When loading multiple spectra, some may be outliers (cosmic rays, failed
acquisitions, different samples). Grouping identifies clusters of similar
spectra so you can:
- Average within groups for noise reduction
- Identify and exclude bad measurements
- Process different sample types separately

**Quality Metrics:**
Each group reports quality scores to help identify the best cluster.
    """.strip(),
    tips=[
        "Follow with Select Best Group to automatically pick the cleanest cluster",
        "Groups marked as 'junk' likely contain failed measurements",
        "Adjust grid_points if grouping is too aggressive or too loose",
    ],
    related_nodes=["select_best_group", "average_signals", "load_signal_batch"],
    inputs=[_port("SignalBatch", "SignalBatch")],
    outputs=[_port("SignalGroupBatch", "SignalGroupBatch")],
    config_fields=[
        _field("grid_points", "number", 1000, "Grid Points",
               "Resolution for similarity comparison", min=100, max=10000, step=100),
    ],
    default_config={"grid_points": 1000},
))

_register(NodeDefinition(
    identifier="select_best_group",
    title="Select Best Group",
    category="Aggregate",
    description="Automatically select the highest-quality signal group",
    explanation="""
Selects the best signal group based on quality metrics.

**Quality Metrics:**
- **avg_quality**: Quality of the averaged signal (recommended)
- **quality_mean**: Mean quality of individual signals in the group

Groups marked as "junk" are deprioritized unless no valid groups exist.
    """.strip(),
    tips=[
        "Use avg_quality for most cases - it evaluates the final averaged result",
        "Set min_quality > 0 to reject all low-quality groups",
        "Connect to Average Signals to get a single clean signal",
    ],
    related_nodes=["group_signals", "average_signals"],
    inputs=[_port("SignalGroupBatch", "SignalGroupBatch")],
    outputs=[_port("SignalBatch", "SignalBatch")],
    config_fields=[
        _field("quality_metric", "select", "avg_quality", "Quality Metric",
               "Metric for ranking groups", options=["avg_quality", "quality_mean"]),
        _field("min_quality", "number", 0.0, "Minimum Quality",
               "Reject groups below this quality", min=0.0, max=1.0, step=0.05),
    ],
    default_config={"quality_metric": "avg_quality", "min_quality": 0.0},
))

_register(NodeDefinition(
    identifier="average_signals",
    title="Average Signals",
    category="Aggregate",
    description="Average multiple signals to reduce noise",
    explanation="""
Combines multiple spectra into a single averaged signal.

**Why Average?**
Averaging N similar spectra reduces random noise by a factor of √N.
This is fundamental to improving signal-to-noise ratio (SNR).

**Process:**
1. Signals are interpolated to a common wavelength grid
2. Intensity values are averaged at each wavelength
3. Result is a cleaner signal with reduced noise

**Multi-Input:**
This node accepts multiple input connections - all connected signals
will be flattened and averaged together.
    """.strip(),
    tips=[
        "More signals = cleaner average (diminishing returns after ~10-20)",
        "Only average signals from the same sample/measurement session",
        "Use higher n_points for spectra with sharp features",
    ],
    related_nodes=["group_signals", "select_best_group", "load_signal_batch"],
    inputs=[_port("SignalBatch", "SignalBatch", multi=True)],
    outputs=[_port("Signal", "Signal")],
    config_fields=[
        _field("n_points", "number", 1000, "Output Points",
               "Number of points in averaged signal", min=100, max=10000, step=100),
    ],
    default_config={"n_points": 1000},
))

# -----------------------------------------------------------------------------
# Preprocess Nodes
# -----------------------------------------------------------------------------

_register(NodeDefinition(
    identifier="trim",
    title="Trim",
    category="Preprocess",
    description="Restrict signal to a wavelength range",
    explanation="""
Removes data outside a specified wavelength range.

**Why Trim?**
- Remove noisy edge regions where detector sensitivity drops
- Focus on wavelength ranges containing features of interest
- Reduce data size for faster processing

**Common Ranges:**
- UV-Vis: 200-800 nm
- Visible: 380-700 nm
- Near-IR: 700-2500 nm
    """.strip(),
    tips=[
        "Check your spectrometer's valid range before setting trim bounds",
        "Leave some margin around features of interest",
        "Trim early in the pipeline to reduce computation",
    ],
    related_nodes=["mask", "resample"],
    inputs=[_port("Signal", "Signal")],
    outputs=[_port("Signal", "Signal")],
    config_fields=[
        _field("min_nm", "number", 300.0, "Min Wavelength (nm)",
               "Lower wavelength bound", min=100, max=3000, step=10),
        _field("max_nm", "number", 600.0, "Max Wavelength (nm)",
               "Upper wavelength bound", min=100, max=3000, step=10),
    ],
    default_config={"min_nm": 300.0, "max_nm": 600.0},
))

_register(NodeDefinition(
    identifier="mask",
    title="Mask Interval",
    category="Preprocess",
    description="Zero out specific wavelength intervals",
    explanation="""
Sets intensity to zero within specified wavelength intervals.

**Use Cases:**
- Remove known interference lines (e.g., atmospheric lines)
- Exclude regions with detector artifacts
- Block strong lines that overwhelm weaker features

Unlike trim, masking preserves the wavelength grid while nullifying
specific regions.
    """.strip(),
    tips=[
        "Use for removing known contaminants or interference",
        "Multiple intervals can be masked in one operation",
        "Format: [[start1, end1], [start2, end2], ...]",
    ],
    related_nodes=["trim"],
    inputs=[_port("Signal", "Signal")],
    outputs=[_port("Signal", "Signal")],
    config_fields=[
        _field("intervals", "json", [], "Mask Intervals",
               "Wavelength intervals to mask: [[min, max], ...]"),
    ],
    default_config={"intervals": []},
))

_register(NodeDefinition(
    identifier="resample",
    title="Resample",
    category="Preprocess",
    description="Resample signal to uniform wavelength grid",
    explanation="""
Interpolates the signal to a new wavelength grid.

**Why Resample?**
- Standardize grid spacing for template matching
- Match resolution between signal and templates
- Reduce data volume while preserving features

**Methods:**
- **n_points**: Specify total number of output points
- **step_nm**: Specify wavelength step size

Only one method should be specified.
    """.strip(),
    tips=[
        "Use n_points=1500-2000 for typical visible-range spectra",
        "Higher resolution preserves narrow features but increases computation",
        "Resample before template building for consistent grids",
    ],
    related_nodes=["trim", "build_templates"],
    inputs=[_port("Signal", "Signal")],
    outputs=[_port("Signal", "Signal")],
    config_fields=[
        _field("n_points", "number", 1500, "Number of Points",
               "Output grid size (0 to use step_nm instead)", min=0, max=10000, step=100),
        _field("step_nm", "number", 0.0, "Step Size (nm)",
               "Wavelength step (0 to use n_points instead)", min=0, max=10, step=0.1),
    ],
    default_config={"n_points": 1500, "step_nm": 0.0},
))

_register(NodeDefinition(
    identifier="subtract_background",
    title="Subtract Background",
    category="Preprocess",
    description="Subtract background signal from sample",
    explanation="""
Removes background contribution from the sample signal.

**Background Sources:**
- Dark current (detector noise without light)
- Ambient light contamination
- Substrate/matrix emission

**Process:**
The background signal is subtracted point-by-point from the sample.
Both signals should cover the same wavelength range.

**Optional Input:**
The background input is optional - if not connected, the node
passes the signal through unchanged.
    """.strip(),
    tips=[
        "Acquire background under identical conditions as sample",
        "Ensure wavelength ranges match between signal and background",
        "Use align=True if grids differ slightly",
    ],
    related_nodes=["load_signal", "continuum_remove_arpls"],
    inputs=[
        _port("Signal", "Signal"),
        _port("Background", "Signal", optional=True),
    ],
    outputs=[_port("Signal", "Signal")],
    config_fields=[
        _field("align", "boolean", False, "Align Grids",
               "Interpolate background to match signal grid"),
    ],
    default_config={"align": False},
))

_register(NodeDefinition(
    identifier="continuum_remove_arpls",
    title="Continuum Remove (arPLS)",
    category="Preprocess",
    description="Remove baseline using arPLS algorithm",
    explanation="""
Removes smooth baseline/continuum using asymmetrically reweighted
penalized least squares (arPLS).

**What is Continuum?**
The continuum is the smooth background curve underlying spectral
features. It comes from:
- Blackbody radiation
- Broad fluorescence
- Scattering effects

**arPLS Algorithm:**
Iteratively fits a smooth curve that stays below the data, effectively
estimating and removing the baseline while preserving peaks.

**Strength Parameter:**
- Lower values (0.1-0.3): Gentle removal, preserves broad features
- Higher values (0.7-1.0): Aggressive removal, flattens baseline
    """.strip(),
    tips=[
        "Start with strength=0.5 and adjust based on results",
        "Use lower strength for spectra with broad emission bands",
        "Can be combined with rolling continuum removal for better results",
    ],
    related_nodes=["continuum_remove_rolling", "subtract_background"],
    inputs=[_port("Signal", "Signal")],
    outputs=[_port("Signal", "Signal")],
    config_fields=[
        _field("strength", "number", 0.5, "Strength",
               "Removal aggressiveness (0-1)", min=0.0, max=1.0, step=0.1),
    ],
    default_config={"strength": 0.5},
))

_register(NodeDefinition(
    identifier="continuum_remove_rolling",
    title="Continuum Remove (Rolling)",
    category="Preprocess",
    description="Remove baseline using rolling percentile filter",
    explanation="""
Removes baseline using a rolling minimum/percentile filter.

**How It Works:**
A window slides across the spectrum, computing a low percentile at
each position. This traces the baseline which is then subtracted.

**Comparison with arPLS:**
- Rolling: Faster, good for simple baselines
- arPLS: More sophisticated, better for complex shapes
- Often used together for best results

**Strength Parameter:**
Controls the percentile used (lower = more aggressive removal).
    """.strip(),
    tips=[
        "Apply after arPLS for thorough baseline removal",
        "Effective for removing residual slope after arPLS",
        "strength=0.5 is a good starting point",
    ],
    related_nodes=["continuum_remove_arpls"],
    inputs=[_port("Signal", "Signal")],
    outputs=[_port("Signal", "Signal")],
    config_fields=[
        _field("strength", "number", 0.5, "Strength",
               "Removal aggressiveness (0-1)", min=0.0, max=1.0, step=0.1),
    ],
    default_config={"strength": 0.5},
))

# -----------------------------------------------------------------------------
# Templates Nodes
# -----------------------------------------------------------------------------

_register(NodeDefinition(
    identifier="build_templates",
    title="Build Templates",
    category="Templates",
    description="Build detection templates from reference spectra",
    explanation="""
Creates Gaussian-broadened templates from reference line data.

**Template Building Process:**
1. Reference lines are loaded for each species
2. Lines are broadened with Gaussian profiles (FWHM parameter)
3. Broadened lines are summed and aligned to the signal's grid
4. Result: A matrix where each column is a species template

**FWHM (Full Width at Half Maximum):**
Controls peak width in nanometers. Should match your instrument's
spectral resolution. Typical values: 0.5-1.5 nm for portable spectrometers.

**Species Filter:**
Optionally limit which species to include in templates.
Format: ["Cu", "Fe", "Pb"] or leave empty for all.
    """.strip(),
    tips=[
        "Match FWHM to your spectrometer's resolution",
        "Use species_filter to focus on elements of interest",
        "Templates must cover your signal's wavelength range",
    ],
    related_nodes=["load_references", "shift_search", "detect_nnls"],
    inputs=[
        _port("Signal", "Signal"),
        _port("References", "References"),
    ],
    outputs=[_port("Templates", "Templates")],
    config_fields=[
        _field("fwhm_nm", "number", 0.75, "FWHM (nm)",
               "Gaussian peak width", min=0.1, max=5.0, step=0.05),
        _field("species_filter", "json", [], "Species Filter",
               "Limit to specific species: [\"Cu\", \"Fe\", ...]"),
        _field("bands_kwargs", "json", {}, "Band Options",
               "Advanced band-building options (JSON)"),
    ],
    default_config={"fwhm_nm": 0.75, "species_filter": [], "bands_kwargs": {}},
))

# -----------------------------------------------------------------------------
# Alignment Nodes
# -----------------------------------------------------------------------------

_register(NodeDefinition(
    identifier="shift_search",
    title="Shift Search",
    category="Alignment",
    description="Optimize wavelength alignment between signal and templates",
    explanation="""
Finds the optimal wavelength shift to align signal with templates.

**Why Shift?**
Wavelength calibration can drift between measurements or differ between
instruments. Even small shifts (0.1-1 nm) can significantly affect
detection accuracy.

**Process:**
1. Signal is shifted by small increments
2. Template fit quality is computed at each shift
3. Best shift is selected and applied

**Parameters:**
- **spread_nm**: Search range (±spread from initial position)
- **iterations**: Number of refinement passes
    """.strip(),
    tips=[
        "Use spread_nm=0.5 for well-calibrated instruments",
        "Increase spread_nm (1-2) if calibration is uncertain",
        "More iterations improve accuracy but take longer",
    ],
    related_nodes=["build_templates", "detect_nnls"],
    inputs=[
        _port("Signal", "Signal"),
        _port("Templates", "Templates"),
    ],
    outputs=[
        _port("Signal", "Signal"),
        _port("Templates", "Templates"),
    ],
    config_fields=[
        _field("spread_nm", "number", 0.5, "Search Spread (nm)",
               "Wavelength range to search", min=0.1, max=5.0, step=0.1),
        _field("iterations", "number", 3, "Iterations",
               "Number of refinement passes", min=1, max=10, step=1),
    ],
    default_config={"spread_nm": 0.5, "iterations": 3},
))

# -----------------------------------------------------------------------------
# Detection Nodes
# -----------------------------------------------------------------------------

_register(NodeDefinition(
    identifier="detect_nnls",
    title="NNLS Detect",
    category="Detection",
    description="Detect species using non-negative least squares fitting",
    explanation="""
Identifies species by fitting templates to the signal using NNLS.

**NNLS (Non-Negative Least Squares):**
Finds the best combination of template spectra that reconstructs
the observed signal, with the constraint that coefficients must
be non-negative (species can't have negative abundance).

**Detection Scoring:**
Each species receives a score (0-1) based on:
- Coefficient magnitude (contribution to fit)
- Number of spectral bands matched
- Fit quality at band positions

**Thresholds:**
- **presence_threshold**: Minimum score to report a detection
- **min_bands**: Minimum matched bands for valid detection
    """.strip(),
    tips=[
        "Lower threshold (0.01-0.02) catches weak signals but may have false positives",
        "Higher threshold (0.05-0.1) gives confident detections only",
        "min_bands=5 is good for rejecting spurious single-line matches",
    ],
    related_nodes=["build_templates", "shift_search"],
    inputs=[
        _port("Signal", "Signal"),
        _port("Templates", "Templates"),
    ],
    outputs=[_port("DetectionResult", "DetectionResult")],
    config_fields=[
        _field("presence_threshold", "number", 0.02, "Presence Threshold",
               "Minimum score to report detection", min=0.0, max=1.0, step=0.01),
        _field("min_bands", "number", 5, "Minimum Bands",
               "Minimum matched spectral bands", min=1, max=50, step=1),
    ],
    default_config={"presence_threshold": 0.02, "min_bands": 5},
))

# -----------------------------------------------------------------------------
# Visualization Nodes
# -----------------------------------------------------------------------------

_register(NodeDefinition(
    identifier="plot_signal",
    title="Plot Signal",
    category="Visualization",
    description="Display signal as a plot",
    explanation="""
Renders the signal as an interactive wavelength vs. intensity plot.

**Options:**
- **title**: Custom plot title
- **normalize**: Scale intensity to 0-1 range

The signal passes through unchanged, allowing plot nodes to be
inserted anywhere in the pipeline for debugging.
    """.strip(),
    tips=[
        "Insert plot nodes between processing steps to visualize changes",
        "Use normalize for comparing signals with different scales",
        "Multiple plot nodes can be active simultaneously",
    ],
    related_nodes=["trim", "continuum_remove_arpls"],
    inputs=[_port("Signal", "Signal")],
    outputs=[_port("Signal", "Signal")],
    config_fields=[
        _field("title", "string", "", "Plot Title",
               "Custom title for the plot"),
        _field("normalize", "boolean", False, "Normalize",
               "Scale intensity to 0-1 range"),
    ],
    default_config={"title": "", "normalize": False},
))


# =============================================================================
# REGISTRY ACCESS FUNCTIONS
# =============================================================================

def get_node_definition(identifier: str) -> Optional[NodeDefinition]:
    """Get a node definition by identifier."""
    return _REGISTRY.get(identifier)


def get_all_definitions() -> List[NodeDefinition]:
    """Get all registered node definitions."""
    return list(_REGISTRY.values())


def get_definitions_by_category() -> Dict[str, List[NodeDefinition]]:
    """Get node definitions grouped by category."""
    grouped: Dict[str, List[NodeDefinition]] = {cat: [] for cat in CATEGORY_ORDER}
    for node in _REGISTRY.values():
        grouped.setdefault(node.category, []).append(node)
    # Sort within categories
    for nodes in grouped.values():
        nodes.sort(key=lambda n: n.title.lower())
    return grouped


def get_category_order() -> List[str]:
    """Get the display order for categories."""
    return list(CATEGORY_ORDER)
