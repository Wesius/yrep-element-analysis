# YREP Spectral Node Editor - System Architecture

## Overview
This is a PyQt6-based node editor for building spectral analysis pipelines. Users create visual graphs by connecting nodes that process spectroscopy signals through a series of transformations (preprocessing, template building, detection, etc.).

## Architecture Layers

### 1. **Main Window** (`main_window.py` - `MainWindow`)
The top-level container with three dock panels:
- **Palette Dock (left)**: Drag-and-drop node library organized by category (I/O, Preprocess, Templates, Detection, etc.)
- **Inspector Dock (right)**: Configuration editor for selected nodes with auto-generated forms
- **Log Dock (bottom)**: Pipeline execution output

Menu bar provides: File operations (new/open/save `.yrep.json` graphs), View controls (zoom/pan/fit), Run commands (execute pipeline).

### 2. **Node Editor** (`node_editor.py` - `NodeEditor`, `NodeScene`, `NodeView`)

**NodeEditor**: Composite widget managing graph state
- Maintains lists of `_nodes` and `_edges`
- Handles node spawning, connection logic, selection
- Exports/imports graph as JSON with nodes (id, identifier, config, position) and edges (source/target node+port)
- Dynamically expands scene bounds as nodes are added (via `update_scene_bounds()`)

**NodeScene**: QGraphicsScene with dark grid background
- Initial bounds: -2000,-1500 to 4000x3000, expands as needed
- Delegates mouse/keyboard events to editor
- Handles drag-and-drop from palette (via MIME type `application/x-yrep-node`)

**NodeView**: Interactive QGraphicsView
- Pan: Middle-click drag or Space+drag
- Zoom: Mouse wheel with limits (0.1x to 3.0x scale)
- Navigation: Arrow keys, Home (fit all), 0 (reset zoom)
- Delete: Del/Backspace removes selected nodes

### 3. **Node Graphics** (`node_item.py`)

**NodeItem**: Visual node representation (QGraphicsRectItem)
- Rounded rectangle with header (title, category, instance ID) and config preview
- Dynamic height based on config text
- Ports positioned at fixed vertical intervals (24px spacing)
- Movable, selectable, sends geometry changes to update edges

**NodePort**: Circular connection anchor (QGraphicsEllipseItem)
- Types: "input" (left side, blue) or "output" (right side, orange)
- Flags: `allow_multiple` (multi-input), `optional` (nullable input)
- Click behavior: Outputs start connections, inputs finish or detach them

**NodeConnection**: Cubic Bezier edge (QGraphicsPathItem)
- Rendered with horizontal control points for smooth curves
- Active drag state uses accent color, finalized connections use default

### 4. **Node Registry** (`nodes/registry.py`)

**NodeDefinition**: Declarative node spec
- `identifier`: Unique key (e.g., "load_signal")
- `title`/`category`: Display name and grouping
- `inputs`/`outputs`: Port name tuples
- `default_config`: Dict of parameter defaults
- `multi_input_ports`: Indices accepting multiple connections
- `optional_input_ports`: Indices that may be unconnected

Example nodes:
- **I/O**: Load Signal (file), Load References (directory)
- **Preprocess**: Trim, Mask, Resample, Subtract Background, Continuum Remove
- **Templates**: Build Templates (from signal + references)
- **Alignment**: Shift Search
- **Detection**: NNLS Detect
- **Visualization**: Plot Signal

### 5. **Inspector Panel** (`inspector.py`)

Auto-generates config forms based on value types:
- `bool` → QCheckBox
- `int` → QSpinBox
- `float` → QDoubleSpinBox
- `str` → QLineEdit + "Browse" button if key matches "path"/"directory"
- Complex types (list/dict) → JSON QPlainTextEdit with Apply button

Changes immediately update `node.config` and refresh node display.

### 6. **Pipeline Execution** (`main_window.py` - `_action_run_graph`)

**Execution Flow**:
1. **Prepare**: Topologically sort nodes via Kahn's algorithm (detect cycles)
2. **Build dependency map**: Track incoming edges per input port (allows multi-input aggregation)
3. **Execute in order**: For each node:
   - Collect inputs from `results` dict (indexed by node ID)
   - Dispatch to `_execute_node` (routes by identifier to specialized handlers or generic callable)
   - Store output in `results[node_id]`
4. **Report**: Log terminal nodes (those with no dependents) showing detection counts, signal stats, etc.

**Node Execution Dispatch**:
- Custom handlers for I/O nodes (load files/directories via `_resolve_path` relative to workspace)
- Pipeline functions fetched via `PipelineRunner.get_callable(identifier)`
- Config coercion for intervals, trim ranges, template kwargs

**Error Handling**: `GraphExecutionError` for validation failures (missing inputs, cycles, file not found)

## Key Features

### Graph Persistence
- **Format**: JSON with `version: 1`, `nodes` array (id, identifier, config, position), `edges` array (source/target node/port)
- **Save/Load**: `export_graph_data()` / `load_graph_data()` with node lookup reconstruction
- Scene bounds automatically updated after loading graphs

### Connection Rules
- Input ports typically single-connection (last wins), unless `allow_multiple=True`
- Cannot connect port to itself or same node
- Multi-input ports aggregate values as lists
- Optional inputs pass `None` if unconnected

### Interaction Model
- **Add node**: Double-click palette item or drag onto canvas
- **Connect**: Click output port → drag → click input port (Esc cancels)
- **Configure**: Select node → edit in inspector (changes live-update)
- **Delete**: Select + Del/Backspace removes node and all attached edges

### Visual Design
- Dark theme: #212129 background, #353B48 nodes, #4AA0E0 selection highlight
- Grid: 25px fine lines, 125px coarse lines
- Antialiased rendering with cosmetic pens (constant pixel width)

## Data Model

**Core Types** (from `yrep_spectrum_analysis.types`):
- `Signal`: `{wavelength: ndarray, intensity: ndarray, meta: dict}`
- `References`: Collection of reference spectra
- `Templates`: Processed templates for detection
- `DetectionResult`: `{detections: [Detection], ...}` where `Detection` has `species`, `score`, `meta`

**Execution Context**:
- Nodes are stateless; config is immutable during run
- Each node produces a single output value (may be list for batches)
- Multi-input ports flatten nested lists recursively

## Extension Points

To add a new node:
1. Add `NodeDefinition` to `_REGISTRY` in `registry.py`
2. Implement execution logic in `_execute_node` (main_window.py:374+)
3. If needed, add custom input collection in `_collect_inputs` (main_window.py:337+)

Node categories auto-populate palette from `CATEGORY_ORDER` list.

---

**Usage Context**: This superprompt enables an LLM to understand the codebase structure for tasks like: debugging execution flows, adding new node types, modifying UI interactions, explaining graph serialization, or implementing new analysis features.
