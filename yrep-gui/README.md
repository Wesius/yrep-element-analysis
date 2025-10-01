# YREP GUI

PyQt6 desktop interface for composing spectral analysis pipelines via a node-based canvas.

## Setup

```bash
uv sync
uv run yrep-gui
```

Requires a Python environment with the workspace (see root `pyproject.toml`).

## Current Status

- Node editor shell using `QGraphicsScene` with grid background
- Palette dock fed by the node definition registry (double-click or drag entries onto the canvas to add nodes)
- Inspector panel exposes every node parameter with inline editing widgets and file/directory pickers for paths
- Load Signal Batch node ingests directories of spectra for aggregation workflows
- Plot Signal node renders quick Matplotlib previews without breaking the pipeline
- Run menu compiles the current graph and executes it through `yrep-spectrum-analysis`, logging terminal node summaries
- Connectors let you drag from orange outputs to blue inputs to assemble execution graphs; edges stay attached as you move nodes
- Log dock records pipeline execution summaries and detections
- Pipeline runner service executes yrep-spectrum-analysis stages directly from the current graph

## Next Steps

- Store node metadata within the scene and serialize graphs to disk
- Wire node output previews to embedded plotting widgets
- Add threaded execution of the compiled pipeline graph
