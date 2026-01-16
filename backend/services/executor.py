"""Pipeline execution service.

Ported from yrep-gui/ui/main_window.py to provide headless pipeline execution.
Handles topological sort, input collection, and node dispatch.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

from yrep_spectrum_analysis import (
    average_signals,
    build_templates,
    continuum_remove_arpls,
    continuum_remove_rolling,
    detect_nnls,
    mask,
    resample,
    shift_search,
    subtract_background,
    trim,
)
from yrep_spectrum_analysis.types import DetectionResult, Signal, Templates, References
from yrep_spectrum_analysis.utils import (
    group_signals,
    is_junk_group,
    load_references,
    load_signals_from_dir,
    load_txt_spectrum,
    signal_quality,
)

from backend.models.pipeline import (
    PipelineGraph,
    PipelineNode,
    PipelineEdge,
    PipelineExecutionResult,
    NodeExecutionResult,
)
from backend.nodes import get_node_definition


# Default timeout for pipeline execution (5 minutes)
DEFAULT_EXECUTION_TIMEOUT = 300.0

# Maximum number of items to serialize in lists (to prevent memory issues)
MAX_SERIALIZED_LIST_ITEMS = 1000


class ExecutionError(Exception):
    """Raised when pipeline execution fails."""
    pass


class ExecutionTimeoutError(ExecutionError):
    """Raised when pipeline execution exceeds timeout."""
    pass


@dataclass
class ExecutionContext:
    """Context for pipeline execution."""
    workspace_root: Path = field(default_factory=Path.cwd)
    results: Dict[str, Any] = field(default_factory=dict)
    node_results: List[NodeExecutionResult] = field(default_factory=list)
    logs: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        """Add a log message."""
        self.logs.append(message)

    def resolve_path(self, path_str: str) -> Path:
        """Resolve a path relative to workspace root.

        All paths are resolved relative to workspace_root. Absolute paths
        and path traversal attempts are rejected for security.
        """
        # Reject empty paths
        if not path_str or not path_str.strip():
            raise ExecutionError("Empty path provided")

        path = Path(path_str).expanduser()

        # Always resolve relative to workspace_root for security
        if path.is_absolute():
            resolved = path.resolve()
        else:
            resolved = (self.workspace_root / path).resolve()

        # Validate the resolved path stays within workspace_root
        workspace_resolved = self.workspace_root.resolve()
        try:
            if not (resolved == workspace_resolved or resolved.is_relative_to(workspace_resolved)):
                raise ExecutionError(
                    f"Path escapes workspace boundary: {path_str}"
                )
        except ValueError:
            raise ExecutionError(
                f"Path escapes workspace boundary: {path_str}"
            )

        return resolved


def _safe_float(value: Any, name: str, default: float) -> float:
    """Safely convert config value to float with clear error message."""
    if value is None:
        return default
    try:
        return float(value)
    except (ValueError, TypeError) as e:
        raise ExecutionError(f"Invalid value for '{name}': expected number, got {type(value).__name__} ({value!r})")


def _safe_int(value: Any, name: str, default: int) -> int:
    """Safely convert config value to int with clear error message."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError) as e:
        raise ExecutionError(f"Invalid value for '{name}': expected integer, got {type(value).__name__} ({value!r})")


class PipelineExecutor:
    """Executes pipeline graphs using yrep-spectrum-analysis functions.

    This class ports the execution logic from the yrep-gui MainWindow
    to provide headless pipeline execution for the API backend.
    """

    def __init__(
        self,
        workspace_root: Optional[Path] = None,
        timeout: Optional[float] = None,
    ):
        self.workspace_root = workspace_root or Path.cwd()
        self.timeout = timeout if timeout is not None else DEFAULT_EXECUTION_TIMEOUT

        # Map of callable functions for pipeline stages
        self._callables: Dict[str, Callable[..., Any]] = {
            "average_signals": average_signals,
            "trim": trim,
            "mask": mask,
            "resample": resample,
            "subtract_background": subtract_background,
            "continuum_remove_arpls": continuum_remove_arpls,
            "continuum_remove_rolling": continuum_remove_rolling,
            "build_templates": self._build_templates_wrapper,
            "shift_search": self._shift_search_wrapper,
            "detect_nnls": detect_nnls,
        }

    def execute(self, graph: PipelineGraph) -> PipelineExecutionResult:
        """Execute a pipeline graph.

        Args:
            graph: The pipeline graph to execute

        Returns:
            PipelineExecutionResult with per-node results and terminal outputs
        """
        start_time = time.time()
        ctx = ExecutionContext(workspace_root=self.workspace_root)

        try:
            # Prepare execution (validate and topologically sort)
            order, node_map, incoming, dependents = self._prepare_execution(graph)

            # Execute nodes in order
            for node_id in order:
                # Check timeout before starting each node
                elapsed = time.time() - start_time
                if self.timeout > 0 and elapsed >= self.timeout:
                    raise ExecutionTimeoutError(
                        f"Pipeline execution timed out after {elapsed:.1f}s "
                        f"(limit: {self.timeout:.1f}s)"
                    )

                node = node_map[node_id]
                node_start = time.time()

                try:
                    inputs = self._collect_inputs(node, incoming[node_id], ctx)
                    output = self._execute_node(node, inputs, ctx)
                    ctx.results[node_id] = output

                    node_result = NodeExecutionResult(
                        node_id=node_id,
                        status="success",
                        output_type=self._get_output_type(output),
                        output_summary=self._summarize_output(output),
                        duration_ms=(time.time() - node_start) * 1000,
                    )
                except ExecutionError as e:
                    node_result = NodeExecutionResult(
                        node_id=node_id,
                        status="error",
                        error=str(e),
                        duration_ms=(time.time() - node_start) * 1000,
                    )
                    ctx.node_results.append(node_result)
                    raise

                ctx.node_results.append(node_result)

            # Collect terminal outputs
            terminal_outputs = self._collect_terminal_outputs(
                ctx.results, node_map, dependents
            )

            return PipelineExecutionResult(
                status="success",
                node_results=ctx.node_results,
                terminal_outputs=terminal_outputs,
                execution_order=order,
                total_duration_ms=(time.time() - start_time) * 1000,
            )

        except ExecutionError as e:
            return PipelineExecutionResult(
                status="error",
                node_results=ctx.node_results,
                terminal_outputs={},
                execution_order=[],
                total_duration_ms=(time.time() - start_time) * 1000,
                error=str(e),
            )
        except Exception as e:
            return PipelineExecutionResult(
                status="error",
                node_results=ctx.node_results,
                terminal_outputs={},
                execution_order=[],
                total_duration_ms=(time.time() - start_time) * 1000,
                error=f"Unexpected error: {e}",
            )

    # -------------------------------------------------------------------------
    # Execution Preparation
    # -------------------------------------------------------------------------

    def _prepare_execution(
        self,
        graph: PipelineGraph,
    ) -> Tuple[
        List[str],
        Dict[str, PipelineNode],
        Dict[str, Dict[int, List[Tuple[str, int]]]],
        Dict[str, List[str]],
    ]:
        """Prepare graph for execution with topological sort.

        Returns:
            Tuple of (execution_order, node_map, incoming_edges, dependents)
        """
        if not graph.nodes:
            # Empty pipeline is valid - return empty execution order
            return [], {}, {}, {}

        node_map = {node.id: node for node in graph.nodes}
        incoming: Dict[str, Dict[int, List[Tuple[str, int]]]] = {}
        dependencies: Dict[str, Set[str]] = {node_id: set() for node_id in node_map}
        dependents: Dict[str, List[str]] = {node_id: [] for node_id in node_map}

        # Initialize incoming port maps
        for node in graph.nodes:
            node_def = get_node_definition(node.identifier)
            port_count = len(node_def.inputs) if node_def else 0
            incoming[node.id] = {idx: [] for idx in range(port_count)}

        # Build edge maps
        for edge in graph.edges:
            src_id = edge.source_node
            dst_id = edge.target_node
            src_port = edge.source_port
            dst_port = edge.target_port

            if src_id not in node_map:
                raise ExecutionError(f"Edge '{edge.id}' references unknown source node: {src_id}")
            if dst_id not in node_map:
                raise ExecutionError(f"Edge '{edge.id}' references unknown target node: {dst_id}")

            incoming[dst_id].setdefault(dst_port, []).append((src_id, src_port))
            dependencies[dst_id].add(src_id)
            dependents[src_id].append(dst_id)

        # Topological sort (Kahn's algorithm)
        queue = sorted([nid for nid, deps in dependencies.items() if not deps])
        order: List[str] = []

        while queue:
            current = queue.pop(0)
            order.append(current)
            for child in sorted(set(dependents[current])):
                deps = dependencies[child]
                if current in deps:
                    deps.remove(current)
                if not deps and child not in order and child not in queue:
                    queue.append(child)
                    queue.sort()

        if len(order) != len(node_map):
            raise ExecutionError("Graph contains cycles or unresolved dependencies.")

        return order, node_map, incoming, dependents

    # -------------------------------------------------------------------------
    # Input Collection
    # -------------------------------------------------------------------------

    def _collect_inputs(
        self,
        node: PipelineNode,
        port_map: Dict[int, List[Tuple[str, int]]],
        ctx: ExecutionContext,
    ) -> List[Any]:
        """Collect inputs for a node from upstream results."""
        node_def = get_node_definition(node.identifier)
        if not node_def:
            raise ExecutionError(f"Unknown node type: {node.identifier}")

        inputs: List[Any] = []

        # Determine optional and multi-input ports from node definition
        optional_ports: Set[int] = set()
        multi_ports: Set[int] = set()
        for idx, port in enumerate(node_def.inputs):
            if port.optional:
                optional_ports.add(idx)
            if port.multi:
                multi_ports.add(idx)

        for idx in range(len(node_def.inputs)):
            edges = port_map.get(idx, [])

            if idx in multi_ports:
                # Multi-input port: collect all connected values as list
                if not edges:
                    raise ExecutionError(
                        f"Input {idx + 1} of '{node_def.title}' expects one or more connections."
                    )
                collected = []
                for src_id, src_port in sorted(edges):
                    value = self._extract_output(ctx.results, src_id, src_port, node_def.title, idx)
                    collected.append(value)
                inputs.append(collected)
            else:
                # Single-input port
                if not edges:
                    if idx in optional_ports:
                        inputs.append(None)
                        continue
                    raise ExecutionError(
                        f"Input {idx + 1} of '{node_def.title}' must be connected."
                    )
                if len(edges) != 1:
                    raise ExecutionError(
                        f"Input {idx + 1} of '{node_def.title}' must have exactly one connection."
                    )
                src_id, src_port = edges[0]
                inputs.append(self._extract_output(ctx.results, src_id, src_port, node_def.title, idx))

        return inputs

    def _extract_output(
        self,
        results: Dict[str, Any],
        src_id: str,
        src_port: int,
        target_title: str,
        target_port: int,
    ) -> Any:
        """Extract a specific output port value from a node's result."""
        if src_id not in results:
            raise ExecutionError(
                f"Upstream node {src_id} has no result for input {target_port + 1} of '{target_title}'."
            )

        value = results[src_id]
        return self._select_output_port(value, src_port)

    def _select_output_port(self, node_value: Any, port_index: int) -> Any:
        """Select a specific output port from a node's result."""
        if isinstance(node_value, tuple):
            if port_index < len(node_value):
                return node_value[port_index]
            raise ExecutionError(f"Output port {port_index} not available (tuple has {len(node_value)} elements)")

        if port_index == 0:
            return node_value

        if isinstance(node_value, dict):
            if port_index in node_value:
                return node_value[port_index]
            if str(port_index) in node_value:
                return node_value[str(port_index)]
            raise ExecutionError(f"Output port {port_index} not in dict result")

        raise ExecutionError(f"Cannot extract port {port_index} from non-tuple result")

    # -------------------------------------------------------------------------
    # Node Execution
    # -------------------------------------------------------------------------

    def _execute_node(
        self,
        node: PipelineNode,
        inputs: List[Any],
        ctx: ExecutionContext,
    ) -> Any:
        """Execute a single node with collected inputs."""
        identifier = node.identifier
        config = dict(node.config)
        node_def = get_node_definition(identifier)

        def normalize(value: Any) -> Any:
            return self._normalize_outputs(node_def, value)

        # I/O nodes
        if identifier == "load_signal":
            return normalize(self._exec_load_signal(config, ctx))

        if identifier == "load_signal_batch":
            return normalize(self._exec_load_signal_batch(config, ctx))

        if identifier == "load_references":
            return normalize(self._exec_load_references(config, ctx))

        # Aggregate nodes
        if identifier == "group_signals":
            return normalize(self._exec_group_signals(inputs, config, ctx))

        if identifier == "select_best_group":
            return normalize(self._exec_select_best_group(inputs, config, ctx))

        if identifier == "average_signals":
            return normalize(self._exec_average_signals(inputs, config, ctx))

        # Visualization nodes
        if identifier == "plot_signal":
            return normalize(self._exec_plot_signal(inputs, config, ctx))

        # Processing nodes using library functions
        func = self._callables.get(identifier)
        if func is None:
            raise ExecutionError(f"No handler for node type: {identifier}")

        if identifier == "subtract_background":
            signal = inputs[0]
            background = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
            return normalize(func(signal, background, align=bool(config.get("align", False))))

        if identifier == "build_templates":
            signal, references = inputs
            kwargs = self._build_templates_kwargs(config)
            return normalize(func(signal, references, **kwargs))

        if identifier == "shift_search":
            signal, templates = inputs
            return normalize(func(
                signal,
                templates,
                spread_nm=_safe_float(config.get("spread_nm"), "spread_nm", 0.5),
                iterations=_safe_int(config.get("iterations"), "iterations", 3),
            ))

        if identifier == "detect_nnls":
            signal, templates = inputs
            return normalize(func(
                signal,
                templates,
                presence_threshold=_safe_float(config.get("presence_threshold"), "presence_threshold", 0.02),
                min_bands=_safe_int(config.get("min_bands"), "min_bands", 5),
            ))

        if identifier == "mask":
            signal = inputs[0]
            intervals = self._coerce_intervals(config.get("intervals", []))
            return normalize(func(signal, intervals=intervals))

        if identifier == "resample":
            signal = inputs[0]
            kwargs: Dict[str, Any] = {}
            n_points = config.get("n_points")
            step_nm = config.get("step_nm")
            if n_points:
                kwargs["n_points"] = _safe_int(n_points, "n_points", 0)
            if step_nm:
                kwargs["step_nm"] = _safe_float(step_nm, "step_nm", 0.0)
            return normalize(func(signal, **kwargs))

        if identifier == "trim":
            signal = inputs[0]
            kwargs = self._coerce_trim_kwargs(config)
            return normalize(func(signal, **kwargs))

        if identifier in {"continuum_remove_arpls", "continuum_remove_rolling"}:
            signal = inputs[0]
            strength = _safe_float(config.get("strength"), "strength", 0.5)
            return normalize(func(signal, strength=strength))

        # Fallback for unary operations
        if inputs:
            return normalize(func(inputs[0], **config))

        raise ExecutionError(f"Node '{identifier}' has no inputs to execute.")

    def _normalize_outputs(self, node_def, value: Any) -> Any:
        """Normalize outputs to match expected port count."""
        if node_def is None:
            return value

        expected = len(node_def.outputs)
        if expected <= 1:
            return value

        if isinstance(value, tuple):
            if len(value) == expected:
                return value
            raise ExecutionError(
                f"Node '{node_def.title}' returned {len(value)} outputs but {expected} expected."
            )

        if isinstance(value, list):
            if len(value) == expected:
                return tuple(value)
            raise ExecutionError(
                f"Node '{node_def.title}' returned {len(value)} outputs but {expected} expected."
            )

        if isinstance(value, dict):
            ordered = []
            for idx, port in enumerate(node_def.outputs):
                if idx in value:
                    ordered.append(value[idx])
                elif port.name in value:
                    ordered.append(value[port.name])
                elif str(idx) in value:
                    ordered.append(value[str(idx)])
                else:
                    raise ExecutionError(
                        f"Node '{node_def.title}' missing output for port {idx + 1}."
                    )
            return tuple(ordered)

        raise ExecutionError(
            f"Node '{node_def.title}' must return {expected} values."
        )

    # -------------------------------------------------------------------------
    # I/O Node Handlers
    # -------------------------------------------------------------------------

    def _exec_load_signal(self, config: Dict[str, Any], ctx: ExecutionContext) -> Signal:
        """Load a single signal file."""
        path_str = str(config.get("path", "")).strip()
        if not path_str:
            raise ExecutionError("Load Signal requires a file path.")

        path = ctx.resolve_path(path_str)
        if not path.exists():
            raise ExecutionError(f"Signal file not found: {path}")

        wavelength, intensity = load_txt_spectrum(path)
        ctx.log(f"Loaded signal from {path.name} ({len(wavelength)} points)")
        return Signal(wavelength=wavelength, intensity=intensity, meta={"path": str(path)})

    def _exec_load_signal_batch(self, config: Dict[str, Any], ctx: ExecutionContext) -> List[Signal]:
        """Load signals from a directory."""
        directory = str(config.get("directory", "")).strip()
        if not directory:
            raise ExecutionError("Load Signal Batch requires a directory path.")

        path = ctx.resolve_path(directory)
        if not path.exists() or not path.is_dir():
            raise ExecutionError(f"Signal directory not found: {path}")

        signals = load_signals_from_dir(path)
        if not signals:
            raise ExecutionError(f"No spectra found in directory: {path}")

        ctx.log(f"Loaded {len(signals)} signals from {path.name}")
        return signals

    def _exec_load_references(self, config: Dict[str, Any], ctx: ExecutionContext) -> References:
        """Load reference spectral lines."""
        directory = str(config.get("directory", "")).strip()
        if not directory:
            raise ExecutionError("Load References requires a directory path.")

        path = ctx.resolve_path(directory)
        if not path.exists() or not path.is_dir():
            raise ExecutionError(f"Reference directory not found: {path}")

        element_only = bool(config.get("element_only", True))
        refs = load_references(path, element_only=element_only)
        ctx.log(f"Loaded references for {len(refs.species())} species")
        return refs

    # -------------------------------------------------------------------------
    # Aggregate Node Handlers
    # -------------------------------------------------------------------------

    def _exec_group_signals(
        self,
        inputs: List[Any],
        config: Dict[str, Any],
        ctx: ExecutionContext,
    ) -> List[Dict[str, Any]]:
        """Group signals by similarity."""
        if not inputs:
            raise ExecutionError("Group Signals requires a batch input.")

        batch = self._flatten_signal_inputs(inputs[0])
        if not batch:
            raise ExecutionError("Group Signals received an empty batch.")

        grid_points = _safe_int(config.get("grid_points"), "grid_points", 1000)
        groups_raw = group_signals(batch, grid_points=grid_points)

        payload = []
        for group_idx, group in enumerate(groups_raw):
            qualities = [signal_quality(sig) for sig in group]
            mean_quality = float(np.mean(qualities)) if qualities else 0.0
            junk = bool(is_junk_group(group))

            avg_quality = 0.0
            avg_error = None
            try:
                averaged = average_signals(group, n_points=grid_points)
                avg_quality = float(signal_quality(averaged))
            except Exception as e:
                avg_quality = 0.0
                avg_error = str(e)
                ctx.log(f"Warning: Failed to compute average quality for group {group_idx + 1}: {e}")

            payload.append({
                "signals": group,
                "junk": junk,
                "quality_mean": mean_quality,
                "avg_quality": avg_quality,
                "avg_error": avg_error,
                "size": len(group),
            })

        ctx.log(f"Grouped batch into {len(payload)} group(s)")
        return payload

    def _exec_select_best_group(
        self,
        inputs: List[Any],
        config: Dict[str, Any],
        ctx: ExecutionContext,
    ) -> List[Signal]:
        """Select the best signal group."""
        if not inputs:
            raise ExecutionError("Select Best Group requires grouped signals.")

        groups = inputs[0]
        if not isinstance(groups, list) or not groups:
            raise ExecutionError("Select Best Group received no groups.")

        metric = str(config.get("quality_metric", "avg_quality")).lower()
        min_quality = _safe_float(config.get("min_quality"), "min_quality", 0.0)
        metric_key = "avg_quality" if metric not in {"avg_quality", "quality_mean"} else metric

        def score(group_info: Dict[str, Any]) -> float:
            return float(group_info.get(metric_key, 0.0))

        best = None
        best_score = float("-inf")

        # First pass: find best non-junk group above min_quality
        for idx, info in enumerate(groups):
            signals = info.get("signals")
            if not signals:
                continue
            current_score = score(info)
            if info.get("junk") or current_score < min_quality:
                continue
            if current_score > best_score:
                best = (idx, info, signals, current_score)
                best_score = current_score

        # Fallback: take any highest-scoring group
        if best is None:
            for idx, info in enumerate(groups):
                signals = info.get("signals")
                if not signals:
                    continue
                current_score = score(info)
                if current_score > best_score:
                    best = (idx, info, signals, current_score)
                    best_score = current_score

        if best is None:
            raise ExecutionError("No viable groups found.")

        idx, info, signals, score_val = best
        ctx.log(f"Selected group {idx + 1} (size={info.get('size')}, score={score_val:.3f})")
        return signals

    def _exec_average_signals(
        self,
        inputs: List[Any],
        config: Dict[str, Any],
        ctx: ExecutionContext,
    ) -> Signal:
        """Average multiple signals."""
        if not inputs:
            raise ExecutionError("Average Signals requires upstream signals.")

        aggregated = inputs[0]
        flattened = self._flatten_signal_inputs(aggregated)
        if not flattened:
            raise ExecutionError("Average Signals requires at least one input signal.")

        n_points = _safe_int(config.get("n_points"), "n_points", 1000)
        result = average_signals(flattened, n_points=n_points)
        ctx.log(f"Averaged {len(flattened)} signals to {n_points} points")
        return result

    def _exec_plot_signal(
        self,
        inputs: List[Any],
        config: Dict[str, Any],
        ctx: ExecutionContext,
    ) -> Signal:
        """Handle plot node (passthrough in headless mode)."""
        if not inputs:
            raise ExecutionError("Plot node requires a connected signal.")

        signal = inputs[0]
        if signal is None:
            raise ExecutionError("Plot node received no signal input.")

        # In headless mode, just pass through the signal
        # The visualization will be generated via the visualization API
        ctx.log(f"Plot node: {len(signal.wavelength)} samples (headless passthrough)")
        return signal

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _flatten_signal_inputs(self, value: Any) -> List[Signal]:
        """Recursively flatten signal inputs into a list."""
        def _iter_items(item: Any) -> Iterable[Signal]:
            if item is None:
                return
            if isinstance(item, (list, tuple)):
                for sub in item:
                    yield from _iter_items(sub)
            elif isinstance(item, Signal):
                yield item
            elif isinstance(item, dict) and "signals" in item:
                yield from _iter_items(item.get("signals"))

        return list(_iter_items(value))

    def _build_templates_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build kwargs for template building."""
        species_filter = config.get("species_filter")
        if isinstance(species_filter, list) and not species_filter:
            species_filter = None
        elif isinstance(species_filter, str) and not species_filter.strip():
            species_filter = None

        bands_kwargs = config.get("bands_kwargs") or {}
        if isinstance(bands_kwargs, str):
            bands_kwargs = {}

        return {
            "fwhm_nm": _safe_float(config.get("fwhm_nm"), "fwhm_nm", 0.75),
            "species_filter": species_filter,
            "bands_kwargs": dict(bands_kwargs),
        }

    def _coerce_trim_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Coerce trim configuration to kwargs."""
        kwargs: Dict[str, Any] = {}
        min_nm = config.get("min_nm")
        max_nm = config.get("max_nm")
        if min_nm not in (None, ""):
            kwargs["min_nm"] = _safe_float(min_nm, "min_nm", 0.0)
        if max_nm not in (None, ""):
            kwargs["max_nm"] = _safe_float(max_nm, "max_nm", 0.0)
        return kwargs

    def _coerce_intervals(self, intervals: Any) -> List[Tuple[float, float]]:
        """Coerce interval configuration to list of tuples."""
        if not intervals:
            return []
        parsed: List[Tuple[float, float]] = []
        for idx, entry in enumerate(intervals):
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                a, b = entry
                parsed.append((
                    _safe_float(a, f"intervals[{idx}][0]", 0.0),
                    _safe_float(b, f"intervals[{idx}][1]", 0.0),
                ))
        return parsed

    def _build_templates_wrapper(
        self,
        signal: Signal,
        references: References,
        *,
        fwhm_nm: float,
        species_filter: Optional[Iterable[str]] = None,
        bands_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Templates:
        """Wrapper for build_templates with default args."""
        return build_templates(
            signal,
            references,
            fwhm_nm=fwhm_nm,
            species_filter=species_filter,
            bands_kwargs=bands_kwargs or {},
            **kwargs,
        )

    def _shift_search_wrapper(
        self,
        signal: Signal,
        templates: Templates,
        *,
        spread_nm: float,
        iterations: int,
        **kwargs: Any,
    ) -> Tuple[Signal, Templates]:
        """Wrapper for shift_search that returns both signal and templates."""
        aligned = shift_search(signal, templates, spread_nm=spread_nm, iterations=iterations, **kwargs)
        return aligned, templates

    # -------------------------------------------------------------------------
    # Result Processing
    # -------------------------------------------------------------------------

    def _collect_terminal_outputs(
        self,
        results: Dict[str, Any],
        node_map: Dict[str, PipelineNode],
        dependents: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Collect outputs from terminal (sink) nodes."""
        terminals = {}

        for node_id, deps in dependents.items():
            if not deps:  # No dependents = terminal node
                value = results.get(node_id)
                if value is not None:
                    terminals[node_id] = self._serialize_output(value)

        return terminals

    def _serialize_output(self, value: Any) -> Any:
        """Serialize output for JSON response."""
        if isinstance(value, Signal):
            wl = value.wavelength
            wl_range = [float(wl.min()), float(wl.max())] if len(wl) > 0 else [0.0, 0.0]
            return {
                "type": "Signal",
                "wavelength_count": len(wl),
                "wavelength_range": wl_range,
                "meta": value.meta,
            }

        if isinstance(value, Templates):
            species = value.species
            truncated = len(species) > MAX_SERIALIZED_LIST_ITEMS
            return {
                "type": "Templates",
                "species": species[:MAX_SERIALIZED_LIST_ITEMS],
                "species_count": len(species),
                "truncated": truncated,
            }

        if isinstance(value, DetectionResult):
            detections = value.detections
            truncated = len(detections) > MAX_SERIALIZED_LIST_ITEMS
            return {
                "type": "DetectionResult",
                "detection_count": len(detections),
                "detections": [
                    {
                        "species": d.species,
                        "score": d.score,
                        "meta": d.meta,
                    }
                    for d in detections[:MAX_SERIALIZED_LIST_ITEMS]
                ],
                "meta": value.meta,
                "truncated": truncated,
            }

        if isinstance(value, References):
            species = list(value.lines.keys())
            truncated = len(species) > MAX_SERIALIZED_LIST_ITEMS
            return {
                "type": "References",
                "species": species[:MAX_SERIALIZED_LIST_ITEMS],
                "species_count": len(species),
                "truncated": truncated,
            }

        if isinstance(value, tuple):
            truncated = len(value) > MAX_SERIALIZED_LIST_ITEMS
            items = [self._serialize_output(v) for v in value[:MAX_SERIALIZED_LIST_ITEMS]]
            if truncated:
                return {"items": items, "count": len(value), "truncated": True}
            return items

        if isinstance(value, list):
            if value and isinstance(value[0], Signal):
                return {
                    "type": "SignalBatch",
                    "count": len(value),
                }
            truncated = len(value) > MAX_SERIALIZED_LIST_ITEMS
            items = [self._serialize_output(v) for v in value[:MAX_SERIALIZED_LIST_ITEMS]]
            if truncated:
                return {"items": items, "count": len(value), "truncated": True}
            return items

        if isinstance(value, dict):
            keys = list(value.keys())
            truncated = len(keys) > MAX_SERIALIZED_LIST_ITEMS
            result = {k: self._serialize_output(value[k]) for k in keys[:MAX_SERIALIZED_LIST_ITEMS]}
            if truncated:
                result["_truncated"] = True
                result["_total_keys"] = len(keys)
            return result

        return value

    def _get_output_type(self, value: Any) -> str:
        """Get the type name of an output value."""
        if isinstance(value, Signal):
            return "Signal"
        if isinstance(value, Templates):
            return "Templates"
        if isinstance(value, DetectionResult):
            return "DetectionResult"
        if isinstance(value, References):
            return "References"
        if isinstance(value, tuple):
            return f"Tuple[{len(value)}]"
        if isinstance(value, list):
            if value and isinstance(value[0], Signal):
                return "SignalBatch"
            return "List"
        return type(value).__name__

    def _summarize_output(self, value: Any) -> str:
        """Generate a human-readable summary of output."""
        if isinstance(value, Signal):
            wl = value.wavelength
            if len(wl) == 0:
                return "Signal: 0 points (empty)"
            return f"Signal: {len(wl)} points, {wl.min():.1f}-{wl.max():.1f} nm"

        if isinstance(value, Templates):
            return f"Templates: {len(value.species)} species"

        if isinstance(value, DetectionResult):
            top = sorted(value.detections, key=lambda d: d.score, reverse=True)[:3]
            species_str = ", ".join(f"{d.species}({d.score:.2f})" for d in top)
            return f"Detection: {len(value.detections)} species - {species_str}"

        if isinstance(value, References):
            return f"References: {len(value.species())} species"

        if isinstance(value, tuple):
            return f"Tuple with {len(value)} outputs"

        if isinstance(value, list):
            if value and isinstance(value[0], Signal):
                return f"SignalBatch: {len(value)} signals"
            return f"List: {len(value)} items"

        return str(type(value).__name__)
