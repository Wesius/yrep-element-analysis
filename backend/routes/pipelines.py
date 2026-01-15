"""Pipeline graph API routes."""

from pathlib import Path
from typing import Dict, List, Optional, Set, Any
from fastapi import APIRouter, HTTPException

from backend.models.pipeline import (
    PipelineGraph,
    PipelineNode,
    PipelineEdge,
    PipelineExecutionRequest,
    PipelineExecutionResult,
    NodeExecutionResult,
)
from backend.nodes import get_node_definition
from backend.services.executor import PipelineExecutor, ExecutionError

router = APIRouter()


def _validate_graph(graph: PipelineGraph) -> List[str]:
    """Validate a pipeline graph and return any errors."""
    errors = []
    node_ids: Set[str] = set()

    # Check for duplicate node IDs
    for node in graph.nodes:
        if node.id in node_ids:
            errors.append(f"Duplicate node ID: {node.id}")
        node_ids.add(node.id)

        # Check node type exists
        node_def = get_node_definition(node.identifier)
        if not node_def:
            errors.append(f"Unknown node type '{node.identifier}' in node {node.id}")

    # Check edges reference valid nodes and ports
    for edge in graph.edges:
        if edge.source_node not in node_ids:
            errors.append(f"Edge {edge.id} references unknown source node: {edge.source_node}")
        if edge.target_node not in node_ids:
            errors.append(f"Edge {edge.id} references unknown target node: {edge.target_node}")

        # Validate port indices
        if edge.source_node in node_ids:
            source_node = next(n for n in graph.nodes if n.id == edge.source_node)
            source_def = get_node_definition(source_node.identifier)
            if source_def and edge.source_port >= len(source_def.outputs):
                errors.append(
                    f"Edge {edge.id} references invalid output port {edge.source_port} "
                    f"on node {edge.source_node} (has {len(source_def.outputs)} outputs)"
                )

        if edge.target_node in node_ids:
            target_node = next(n for n in graph.nodes if n.id == edge.target_node)
            target_def = get_node_definition(target_node.identifier)
            if target_def and edge.target_port >= len(target_def.inputs):
                errors.append(
                    f"Edge {edge.id} references invalid input port {edge.target_port} "
                    f"on node {edge.target_node} (has {len(target_def.inputs)} inputs)"
                )

    return errors


def _check_cycles(graph: PipelineGraph) -> bool:
    """Check if graph has cycles using topological sort."""
    # Build adjacency list
    adj: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}
    in_degree: Dict[str, int] = {node.id: 0 for node in graph.nodes}

    for edge in graph.edges:
        if edge.source_node in adj and edge.target_node in in_degree:
            adj[edge.source_node].append(edge.target_node)
            in_degree[edge.target_node] += 1

    # Kahn's algorithm
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    count = 0

    while queue:
        node = queue.pop(0)
        count += 1
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return count != len(graph.nodes)


def _get_execution_order(graph: PipelineGraph) -> List[str]:
    """Get topologically sorted execution order."""
    adj: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}
    in_degree: Dict[str, int] = {node.id: 0 for node in graph.nodes}

    for edge in graph.edges:
        if edge.source_node in adj and edge.target_node in in_degree:
            adj[edge.source_node].append(edge.target_node)
            in_degree[edge.target_node] += 1

    queue = sorted([nid for nid, deg in in_degree.items() if deg == 0])
    order = []

    while queue:
        node = queue.pop(0)
        order.append(node)
        for neighbor in sorted(adj[node]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
        queue.sort()

    return order


@router.post("/validate")
async def validate_pipeline(graph: PipelineGraph):
    """Validate a pipeline graph.

    Checks for:
    - Valid node types
    - Valid port connections
    - No cycles
    - Required inputs connected
    """
    errors = _validate_graph(graph)

    if not errors and _check_cycles(graph):
        errors.append("Graph contains cycles")

    # Check required inputs
    if not errors:
        for node in graph.nodes:
            node_def = get_node_definition(node.identifier)
            if not node_def:
                continue

            # Find connected inputs
            connected_inputs: Set[int] = set()
            for edge in graph.edges:
                if edge.target_node == node.id:
                    connected_inputs.add(edge.target_port)

            # Check required inputs
            for idx, port in enumerate(node_def.inputs):
                if not port.optional and idx not in connected_inputs:
                    errors.append(
                        f"Node {node.id} ({node_def.title}): "
                        f"Required input '{port.name}' is not connected"
                    )

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
    }


@router.post("/analyze")
async def analyze_pipeline(graph: PipelineGraph):
    """Analyze a pipeline graph structure.

    Returns execution order, input/output nodes, and data flow information.
    """
    validation = _validate_graph(graph)
    if validation:
        raise HTTPException(status_code=400, detail={"errors": validation})

    if _check_cycles(graph):
        raise HTTPException(status_code=400, detail={"errors": ["Graph contains cycles"]})

    order = _get_execution_order(graph)

    # Find source nodes (no inputs)
    source_nodes = []
    for node in graph.nodes:
        node_def = get_node_definition(node.identifier)
        if node_def and len(node_def.inputs) == 0:
            source_nodes.append(node.id)

    # Find sink nodes (no outgoing edges)
    has_outgoing = set(edge.source_node for edge in graph.edges)
    sink_nodes = [node.id for node in graph.nodes if node.id not in has_outgoing]

    # Build dependency graph
    dependencies: Dict[str, List[str]] = {node.id: [] for node in graph.nodes}
    for edge in graph.edges:
        if edge.target_node in dependencies:
            dependencies[edge.target_node].append(edge.source_node)

    return {
        "execution_order": order,
        "source_nodes": source_nodes,
        "sink_nodes": sink_nodes,
        "dependencies": dependencies,
        "node_count": len(graph.nodes),
        "edge_count": len(graph.edges),
    }


@router.post("/execute", response_model=PipelineExecutionResult)
async def execute_pipeline(request: PipelineExecutionRequest):
    """Execute a pipeline graph.

    Runs the pipeline using the yrep-spectrum-analysis library and returns
    results from all nodes including terminal outputs.

    The workspace_root parameter is used to resolve relative file paths
    in load nodes.
    """
    graph = request.graph

    # Validate first
    errors = _validate_graph(graph)
    if errors:
        return PipelineExecutionResult(
            status="error",
            error=f"Validation failed: {'; '.join(errors)}",
            node_results=[],
            terminal_outputs={},
            execution_order=[],
        )

    if _check_cycles(graph):
        return PipelineExecutionResult(
            status="error",
            error="Graph contains cycles",
            node_results=[],
            terminal_outputs={},
            execution_order=[],
        )

    # Determine workspace root
    workspace_root = None
    if request.workspace_root:
        workspace_root = Path(request.workspace_root).expanduser().resolve()

    # Execute the pipeline
    executor = PipelineExecutor(workspace_root=workspace_root)
    result = executor.execute(graph)

    return result


@router.post("/from-template")
async def create_from_template(
    template_name: str,
    parameters: Dict[str, Any] = {},
):
    """Create a pipeline graph from a template.

    Available templates:
    - basic_detection: Load -> Preprocess -> Detect
    - batch_analysis: Batch Load -> Group -> Average -> Detect
    - full_pipeline: Complete analysis with all preprocessing steps
    """
    templates = {
        "basic_detection": _template_basic_detection,
        "batch_analysis": _template_batch_analysis,
        "full_pipeline": _template_full_pipeline,
    }

    if template_name not in templates:
        raise HTTPException(
            status_code=404,
            detail=f"Template not found: {template_name}. Available: {list(templates.keys())}"
        )

    return templates[template_name](parameters)


def _template_basic_detection(params: Dict[str, Any]) -> PipelineGraph:
    """Basic detection pipeline template."""
    return PipelineGraph(
        name="Basic Detection",
        description="Simple pipeline: Load signal, preprocess, detect elements",
        nodes=[
            PipelineNode(id="1", identifier="load_signal",
                        config={"path": params.get("signal_path", "")},
                        position={"x": 100, "y": 200}),
            PipelineNode(id="2", identifier="load_references",
                        config={"directory": params.get("references_path", ""), "element_only": True},
                        position={"x": 100, "y": 350}),
            PipelineNode(id="3", identifier="trim",
                        config={"min_nm": 300, "max_nm": 600},
                        position={"x": 300, "y": 200}),
            PipelineNode(id="4", identifier="resample",
                        config={"n_points": 1500},
                        position={"x": 500, "y": 200}),
            PipelineNode(id="5", identifier="continuum_remove_arpls",
                        config={"strength": 0.5},
                        position={"x": 700, "y": 200}),
            PipelineNode(id="6", identifier="build_templates",
                        config={"fwhm_nm": 0.75},
                        position={"x": 900, "y": 275}),
            PipelineNode(id="7", identifier="detect_nnls",
                        config={"presence_threshold": 0.02, "min_bands": 5},
                        position={"x": 1100, "y": 275}),
        ],
        edges=[
            PipelineEdge(id="e1", source_node="1", source_port=0, target_node="3", target_port=0),
            PipelineEdge(id="e2", source_node="3", source_port=0, target_node="4", target_port=0),
            PipelineEdge(id="e3", source_node="4", source_port=0, target_node="5", target_port=0),
            PipelineEdge(id="e4", source_node="5", source_port=0, target_node="6", target_port=0),
            PipelineEdge(id="e5", source_node="2", source_port=0, target_node="6", target_port=1),
            PipelineEdge(id="e6", source_node="5", source_port=0, target_node="7", target_port=0),
            PipelineEdge(id="e7", source_node="6", source_port=0, target_node="7", target_port=1),
        ],
    )


def _template_batch_analysis(params: Dict[str, Any]) -> PipelineGraph:
    """Batch analysis pipeline template."""
    return PipelineGraph(
        name="Batch Analysis",
        description="Process multiple spectra: Load batch, group, select best, average, detect",
        nodes=[
            PipelineNode(id="1", identifier="load_signal_batch",
                        config={"directory": params.get("signal_dir", "")},
                        position={"x": 100, "y": 200}),
            PipelineNode(id="2", identifier="load_references",
                        config={"directory": params.get("references_path", "")},
                        position={"x": 100, "y": 400}),
            PipelineNode(id="3", identifier="group_signals",
                        config={"grid_points": 1000},
                        position={"x": 300, "y": 200}),
            PipelineNode(id="4", identifier="select_best_group",
                        config={"quality_metric": "avg_quality"},
                        position={"x": 500, "y": 200}),
            PipelineNode(id="5", identifier="average_signals",
                        config={"n_points": 1500},
                        position={"x": 700, "y": 200}),
            PipelineNode(id="6", identifier="trim",
                        config={"min_nm": 300, "max_nm": 600},
                        position={"x": 900, "y": 200}),
            PipelineNode(id="7", identifier="continuum_remove_arpls",
                        config={"strength": 0.5},
                        position={"x": 1100, "y": 200}),
            PipelineNode(id="8", identifier="build_templates",
                        config={"fwhm_nm": 0.75},
                        position={"x": 1300, "y": 300}),
            PipelineNode(id="9", identifier="detect_nnls",
                        config={"presence_threshold": 0.02},
                        position={"x": 1500, "y": 300}),
        ],
        edges=[
            PipelineEdge(id="e1", source_node="1", source_port=0, target_node="3", target_port=0),
            PipelineEdge(id="e2", source_node="3", source_port=0, target_node="4", target_port=0),
            PipelineEdge(id="e3", source_node="4", source_port=0, target_node="5", target_port=0),
            PipelineEdge(id="e4", source_node="5", source_port=0, target_node="6", target_port=0),
            PipelineEdge(id="e5", source_node="6", source_port=0, target_node="7", target_port=0),
            PipelineEdge(id="e6", source_node="7", source_port=0, target_node="8", target_port=0),
            PipelineEdge(id="e7", source_node="2", source_port=0, target_node="8", target_port=1),
            PipelineEdge(id="e8", source_node="7", source_port=0, target_node="9", target_port=0),
            PipelineEdge(id="e9", source_node="8", source_port=0, target_node="9", target_port=1),
        ],
    )


def _template_full_pipeline(params: Dict[str, Any]) -> PipelineGraph:
    """Full analysis pipeline with all preprocessing steps."""
    return PipelineGraph(
        name="Full Analysis Pipeline",
        description="Complete pipeline with background subtraction, dual continuum removal, and alignment",
        nodes=[
            PipelineNode(id="1", identifier="load_signal_batch",
                        config={"directory": params.get("signal_dir", "")},
                        position={"x": 100, "y": 150}),
            PipelineNode(id="2", identifier="load_signal_batch",
                        config={"directory": params.get("background_dir", "")},
                        position={"x": 100, "y": 350}),
            PipelineNode(id="3", identifier="load_references",
                        config={"directory": params.get("references_path", "")},
                        position={"x": 100, "y": 550}),
            PipelineNode(id="4", identifier="group_signals",
                        config={"grid_points": 1000},
                        position={"x": 300, "y": 150}),
            PipelineNode(id="5", identifier="select_best_group",
                        config={},
                        position={"x": 500, "y": 150}),
            PipelineNode(id="6", identifier="average_signals",
                        config={"n_points": 1200},
                        position={"x": 700, "y": 150}),
            PipelineNode(id="7", identifier="average_signals",
                        config={"n_points": 1200},
                        position={"x": 300, "y": 350}),
            PipelineNode(id="8", identifier="trim",
                        config={"min_nm": 300, "max_nm": 600},
                        position={"x": 900, "y": 150}),
            PipelineNode(id="9", identifier="resample",
                        config={"n_points": 1500},
                        position={"x": 1100, "y": 150}),
            PipelineNode(id="10", identifier="subtract_background",
                         config={"align": False},
                         position={"x": 1100, "y": 250}),
            PipelineNode(id="11", identifier="continuum_remove_arpls",
                         config={"strength": 0.5},
                         position={"x": 1300, "y": 250}),
            PipelineNode(id="12", identifier="continuum_remove_rolling",
                         config={"strength": 0.5},
                         position={"x": 1500, "y": 250}),
            PipelineNode(id="13", identifier="build_templates",
                         config={"fwhm_nm": 0.75},
                         position={"x": 1700, "y": 350}),
            PipelineNode(id="14", identifier="shift_search",
                         config={"spread_nm": 0.5, "iterations": 3},
                         position={"x": 1900, "y": 350}),
            PipelineNode(id="15", identifier="detect_nnls",
                         config={"presence_threshold": 0.02, "min_bands": 5},
                         position={"x": 2100, "y": 350}),
        ],
        edges=[
            PipelineEdge(id="e1", source_node="1", source_port=0, target_node="4", target_port=0),
            PipelineEdge(id="e2", source_node="4", source_port=0, target_node="5", target_port=0),
            PipelineEdge(id="e3", source_node="5", source_port=0, target_node="6", target_port=0),
            PipelineEdge(id="e4", source_node="2", source_port=0, target_node="7", target_port=0),
            PipelineEdge(id="e5", source_node="6", source_port=0, target_node="8", target_port=0),
            PipelineEdge(id="e6", source_node="8", source_port=0, target_node="9", target_port=0),
            PipelineEdge(id="e7", source_node="9", source_port=0, target_node="10", target_port=0),
            PipelineEdge(id="e8", source_node="7", source_port=0, target_node="10", target_port=1),
            PipelineEdge(id="e9", source_node="10", source_port=0, target_node="11", target_port=0),
            PipelineEdge(id="e10", source_node="11", source_port=0, target_node="12", target_port=0),
            PipelineEdge(id="e11", source_node="12", source_port=0, target_node="13", target_port=0),
            PipelineEdge(id="e12", source_node="3", source_port=0, target_node="13", target_port=1),
            PipelineEdge(id="e13", source_node="12", source_port=0, target_node="14", target_port=0),
            PipelineEdge(id="e14", source_node="13", source_port=0, target_node="14", target_port=1),
            PipelineEdge(id="e15", source_node="14", source_port=0, target_node="15", target_port=0),
            PipelineEdge(id="e16", source_node="14", source_port=1, target_node="15", target_port=1),
        ],
    )
