"""High-level window framing the node editor workspace."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from PyQt6.QtCore import Qt, QMimeData
from PyQt6.QtGui import QDrag
from PyQt6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMenuBar,
    QMessageBox,
    QTextEdit,
)

from yrep_spectrum_analysis.types import DetectionResult, References, Signal, Templates
from yrep_spectrum_analysis.utils import load_references, load_signals_from_dir, load_txt_spectrum

from yrep_gui.nodes import registry
from yrep_gui.services.pipeline_runner import PipelineRunner
from yrep_gui.ui.inspector import InspectorPanel
from yrep_gui.ui.node_editor import NODE_MIME_TYPE, NodeEditor
from yrep_gui.ui.node_item import NodeConnection, NodeItem, NodePort


class GraphExecutionError(RuntimeError):
    """Raised when a pipeline graph cannot be executed."""


class PaletteList(QListWidget):
    """List widget that supports dragging node identifiers onto the canvas."""

    def startDrag(self, supportedActions) -> None:  # noqa: N802
        item = self.currentItem()
        if item is None:
            return
        identifier = item.data(Qt.ItemDataRole.UserRole)
        if not identifier:
            return
        mime = QMimeData()
        mime.setData(NODE_MIME_TYPE, str(identifier).encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.DropAction.CopyAction)


class MainWindow(QMainWindow):
    """Top-level window containing the node editor and supporting docks."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YREP Spectral Node Editor")
        self.resize(1360, 840)

        self._workspace_root = Path.cwd()
        self._pipeline_runner = PipelineRunner()

        self._node_editor = NodeEditor(self)
        self.setCentralWidget(self._node_editor)

        self._palette_list: PaletteList | None = None
        self._inspector_panel: InspectorPanel | None = None
        self._log_view: QTextEdit | None = None

        self._palette_dock = self._create_palette_dock()
        self._inspector_dock = self._create_inspector_dock()
        self._log_dock = self._create_log_dock()

        self._node_editor.nodeSelected.connect(self._update_inspector)
        self._install_menu_bar()
        self.statusBar().showMessage("Ready")

    # ------------------------------------------------------------------
    # Dock setup
    # ------------------------------------------------------------------
    def _create_palette_dock(self) -> QDockWidget:
        dock = QDockWidget("Nodes", self)
        dock.setObjectName("node-palette")
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        list_widget = PaletteList(dock)
        groups = registry.grouped_definitions()
        for category in registry.category_order():
            definitions = groups.get(category) or []
            if not definitions:
                continue
            header = QListWidgetItem(category)
            header_font = header.font()
            header_font.setBold(True)
            header.setFont(header_font)
            header.setFlags(Qt.ItemFlag.ItemIsEnabled)
            header.setData(Qt.ItemDataRole.UserRole, None)
            list_widget.addItem(header)
            for definition in definitions:
                item = QListWidgetItem(f"  {definition.title}")
                item.setData(Qt.ItemDataRole.UserRole, definition.identifier)
                list_widget.addItem(item)
        list_widget.itemActivated.connect(self._spawn_node_from_palette)
        dock.setWidget(list_widget)
        self._palette_list = list_widget
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)
        return dock

    def _create_inspector_dock(self) -> QDockWidget:
        dock = QDockWidget("Inspector", self)
        dock.setObjectName("inspector-dock")
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        panel = InspectorPanel(dock)
        dock.setWidget(panel)
        self._inspector_panel = panel
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)
        return dock

    def _create_log_dock(self) -> QDockWidget:
        dock = QDockWidget("Log", self)
        dock.setObjectName("log-dock")
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        log = QTextEdit(dock)
        log.setReadOnly(True)
        log.setPlaceholderText("Pipeline output will appear here.")
        dock.setWidget(log)
        self._log_view = log
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        return dock

    # ------------------------------------------------------------------
    def _install_menu_bar(self) -> None:
        menu_bar = QMenuBar(self)
        self.setMenuBar(menu_bar)

        file_menu = menu_bar.addMenu("File")
        assert isinstance(file_menu, QMenu)
        file_menu.addAction("New Graph", self._action_new_graph)
        file_menu.addAction("Open Graph…", self._action_open_graph)
        file_menu.addAction("Save Graph", self._action_save_graph)
        file_menu.addSeparator()
        file_menu.addAction("Quit", self.close)

        run_menu = menu_bar.addMenu("Run")
        assert isinstance(run_menu, QMenu)
        run_menu.addAction("Run Graph", self._action_run_graph)
        run_menu.addAction("Stop", self._action_unimplemented)

        help_menu = menu_bar.addMenu("Help")
        assert isinstance(help_menu, QMenu)
        help_menu.addAction("About", self._show_about_dialog)
        run_menu = menu_bar.addMenu("Run")
        run_menu.addAction("Run Graph", self._action_run_graph)
        run_menu.addAction("Stop", self._action_unimplemented)

        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("About", self._show_about_dialog)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _action_new_graph(self) -> None:
        self._node_editor.clear_graph()
        if self._inspector_panel is not None:
            self._inspector_panel.set_node(None)
        if self._log_view is not None:
            self._log_view.clear()
        self.statusBar().showMessage("Started a new graph", 2000)

    def _action_unimplemented(self) -> None:
        QMessageBox.information(self, "Not Implemented", "This feature is not implemented yet.")

    def _action_open_graph(self) -> None:
        start_dir = str(self._workspace_root)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Graph",
            start_dir,
            "YREP Graph (*.yrep.json);;JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            QMessageBox.critical(self, "Load Failed", f"Could not load graph:\n{exc}")
            return
        try:
            self._node_editor.load_graph_data(payload)
        except Exception as exc:  # pragma: no cover - defensive
            QMessageBox.critical(self, "Load Failed", f"Graph is invalid:\n{exc}")
            return
        if self._inspector_panel is not None:
            self._inspector_panel.set_node(self._node_editor.selected_node())
        self.statusBar().showMessage(f"Loaded graph: {Path(path).name}", 3000)

    def _action_save_graph(self) -> None:
        start_dir = str(self._workspace_root)
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Graph",
            start_dir,
            "YREP Graph (*.yrep.json);;JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        file_path = Path(path)
        if file_path.suffix == "":
            file_path = file_path.with_suffix(".yrep.json")
        payload = self._node_editor.export_graph_data()
        try:
            with open(file_path, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, indent=2)
        except OSError as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not save graph:\n{exc}")
            return
        self.statusBar().showMessage(f"Saved graph: {file_path.name}", 3000)

    def _action_run_graph(self) -> None:
        try:
            order, node_map, incoming, dependents = self._prepare_execution()
            results: Dict[int, Any] = {}
            for node_id in order:
                node = node_map[node_id]
                inputs = self._collect_inputs(node, incoming[node_id], results)
                outputs = self._execute_node(node, inputs)
                results[node_id] = outputs
            self._report_results(results, node_map, dependents)
            self.statusBar().showMessage("Pipeline run completed", 3000)
        except GraphExecutionError as exc:
            self._append_log(f"ERROR: {exc}")
            QMessageBox.critical(self, "Pipeline Error", str(exc))
        except Exception as exc:  # pragma: no cover - defensive
            self._append_log(f"Unexpected error: {exc}")
            QMessageBox.critical(self, "Pipeline Error", f"Unexpected error: {exc}")

    def _show_about_dialog(self) -> None:
        QMessageBox.about(
            self,
            "About YREP Node Editor",
            "A PyQt6 interface for composing spectral analysis pipelines.",
        )

    # ------------------------------------------------------------------
    # Palette / inspector plumbing
    # ------------------------------------------------------------------
    def _spawn_node_from_palette(self, item: QListWidgetItem) -> None:
        identifier = item.data(Qt.ItemDataRole.UserRole)
        if not identifier:
            return
        definition = registry.get(str(identifier))
        node = self._node_editor.add_node(definition)
        self.statusBar().showMessage(f"Added node: {definition.title}", 2000)
        self._update_inspector(node)

    def _update_inspector(self, node: NodeItem | None) -> None:
        if self._inspector_panel is not None:
            self._inspector_panel.set_node(node)

    # ------------------------------------------------------------------
    # Graph execution helpers
    # ------------------------------------------------------------------
    def _prepare_execution(
        self,
    ) -> Tuple[List[int], Dict[int, NodeItem], Dict[int, Dict[int, List[Tuple[int, int]]]], Dict[int, List[int]]]:
        nodes, edges = self._node_editor.export_graph()
        if not nodes:
            raise GraphExecutionError("The graph is empty.")
        node_map = {node.instance_id: node for node in nodes}
        incoming: Dict[int, Dict[int, List[Tuple[int, int]]]] = {}
        dependencies: Dict[int, set[int]] = {node_id: set() for node_id in node_map}
        dependents: Dict[int, List[int]] = {node_id: [] for node_id in node_map}

        for node in nodes:
            port_map: Dict[int, List[Tuple[int, int]]] = {}
            for idx in range(len(node.definition.inputs)):
                port_map[idx] = []
            incoming[node.instance_id] = port_map

        for edge in edges:
            src_node = edge.start_port.owner
            dst_node = edge.end_port.owner
            src_id = src_node.instance_id
            dst_id = dst_node.instance_id
            src_port = edge.start_port.index
            dst_port = edge.end_port.index
            incoming[dst_id].setdefault(dst_port, []).append((src_id, src_port))
            if dst_id not in dependencies:
                dependencies[dst_id] = set()
            dependencies[dst_id].add(src_id)
            dependents[src_id].append(dst_id)

        queue = sorted([node_id for node_id, parents in dependencies.items() if not parents])
        order: List[int] = []
        while queue:
            current = queue.pop(0)
            order.append(current)
            for child in sorted(set(dependents[current])):
                deps = dependencies.setdefault(child, set())
                if current in deps:
                    deps.remove(current)
                if not deps:
                    if child not in order and child not in queue:
                        queue.append(child)
                        queue.sort()

        if len(order) != len(node_map):
            raise GraphExecutionError("Graph contains cycles or unresolved inputs.")

        return order, node_map, incoming, dependents

    def _collect_inputs(
        self,
        node: NodeItem,
        port_map: Dict[int, List[Tuple[int, int]]],
        results: Dict[int, Any],
    ) -> List[Any]:
        inputs: List[Any] = []
        optional = set(node.definition.optional_input_ports)
        for idx, _ in enumerate(node.definition.inputs):
            edges = port_map.get(idx, [])
            if idx in node.definition.multi_input_ports:
                if not edges:
                    raise GraphExecutionError(
                        f"Input {idx + 1} of '{node.definition.title}' expects one or more signals."
                    )
                collected = [results[src_id] for src_id, _ in sorted(edges, key=lambda item: item[0])]
                inputs.append(collected)
            else:
                if not edges:
                    if idx in optional:
                        inputs.append(None)
                        continue
                    raise GraphExecutionError(
                        f"Input {idx + 1} of '{node.definition.title}' must be connected."
                    )
                if len(edges) != 1:
                    raise GraphExecutionError(
                        f"Input {idx + 1} of '{node.definition.title}' must be connected exactly once."
                    )
                src_id, _ = edges[0]
                if src_id not in results:
                    raise GraphExecutionError(
                        f"Upstream node for input {idx + 1} of '{node.definition.title}' has no value."
                    )
                inputs.append(results[src_id])
        return inputs

    def _execute_node(self, node: NodeItem, inputs: List[Any]) -> Any:
        identifier = node.definition.identifier
        config = dict(node.config)

        if identifier == "load_signal":
            return self._exec_load_signal(config)
        if identifier == "load_references":
            return self._exec_load_references(config)
        if identifier == "load_signal_batch":
            return self._exec_load_signal_batch(config)
        if identifier == "average_signals":
            return self._exec_average_signals(inputs, config)
        if identifier == "plot_signal":
            return self._exec_plot_signal(inputs, config)

        func = self._pipeline_runner.get_callable(identifier)

        if identifier == "subtract_background":
            signal = inputs[0]
            background = inputs[1] if len(inputs) > 1 and inputs[1] is not None else None
            return func(signal, background, align=bool(config.get("align", False)))

        if identifier in {"build_templates"}:
            signal, references = inputs
            kwargs = self._build_templates_kwargs(config)
            return func(signal, references, **kwargs)

        if identifier == "shift_search":
            signal, templates = inputs
            return func(
                signal,
                templates,
                spread_nm=float(config.get("spread_nm", 0.5)),
                iterations=int(config.get("iterations", 3)),
            )

        if identifier == "detect_nnls":
            signal, templates = inputs
            return func(
                signal,
                templates,
                presence_threshold=float(config.get("presence_threshold", 0.02)),
                min_bands=int(config.get("min_bands", 5)),
            )

        if identifier == "mask":
            signal = inputs[0]
            intervals = self._coerce_intervals(config.get("intervals", []))
            return func(signal, intervals=intervals)

        if identifier == "resample":
            signal = inputs[0]
            kwargs: Dict[str, Any] = {}
            n_points = config.get("n_points")
            step_nm = config.get("step_nm")
            if n_points:
                kwargs["n_points"] = int(n_points)
            if step_nm:
                kwargs["step_nm"] = float(step_nm)
            return func(signal, **kwargs)

        if identifier == "trim":
            signal = inputs[0]
            kwargs = self._coerce_trim_kwargs(config)
            return func(signal, **kwargs)

        if identifier in {"continuum_remove_arpls", "continuum_remove_rolling"}:
            signal = inputs[0]
            strength = float(config.get("strength", 0.5))
            return func(signal, strength=strength)

        # Unary fall-through (trim-like stages)
        if inputs:
            return func(inputs[0], **config)
        raise GraphExecutionError(f"Node '{node.definition.title}' has no inputs to execute.")

    def _exec_load_signal(self, config: Dict[str, Any]) -> Signal:
        path_str = str(config.get("path", "")).strip()
        if not path_str:
            raise GraphExecutionError("Load Signal requires a file path.")
        path = self._resolve_path(path_str)
        if not path.exists():
            raise GraphExecutionError(f"Signal path does not exist: {path}")
        wavelength, intensity = load_txt_spectrum(path)
        return Signal(wavelength=wavelength, intensity=intensity, meta={"path": str(path)})

    def _exec_load_signal_batch(self, config: Dict[str, Any]):
        directory = str(config.get("directory", "")).strip()
        if not directory:
            raise GraphExecutionError("Load Signal Batch requires a directory path.")
        path = self._resolve_path(directory)
        if not path.exists() or not path.is_dir():
            raise GraphExecutionError(f"Signal directory not found: {path}")
        signals = load_signals_from_dir(path)
        if not signals:
            raise GraphExecutionError(f"No spectra found in directory: {path}")
        return signals

    def _exec_load_references(self, config: Dict[str, Any]):
        directory = str(config.get("directory", "")).strip()
        if not directory:
            raise GraphExecutionError("Load References requires a directory path.")
        path = self._resolve_path(directory)
        if not path.exists() or not path.is_dir():
            raise GraphExecutionError(f"Reference directory not found: {path}")
        element_only = bool(config.get("element_only", True))
        return load_references(path, element_only=element_only)

    def _exec_average_signals(self, inputs: List[Any], config: Dict[str, Any]) -> Signal:
        if not inputs:
            raise GraphExecutionError("Average Signals requires upstream signals.")
        aggregated = inputs[0]
        flattened = self._flatten_signal_inputs(aggregated)
        if not flattened:
            raise GraphExecutionError("Average Signals requires at least one input signal.")
        n_points = int(config.get("n_points", 1000))
        func = self._pipeline_runner.get_callable("average_signals")
        return func(flattened, n_points=n_points)

    def _exec_plot_signal(self, inputs: List[Any], config: Dict[str, Any]) -> Signal:
        if not inputs:
            raise GraphExecutionError("Plot node requires a connected signal.")
        signal = inputs[0]
        if signal is None:
            raise GraphExecutionError("Plot node received no signal input.")
        title = str(config.get("title", "")).strip()
        normalize = bool(config.get("normalize", False))
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError as exc:
            raise GraphExecutionError("Matplotlib is required for plotting.") from exc
        x = signal.wavelength
        y = signal.intensity
        if normalize:
            max_val = float(np.max(np.abs(y))) if y.size else 0.0
            if max_val > 0:
                y = y / max_val
        plt.figure(figsize=(6, 3.5))
        plt.plot(x, y, linewidth=1.2)
        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Intensity (a.u.)")
        if title:
            plt.title(title)
        plt.tight_layout()
        plt.show(block=False)
        self._append_log(f"Plotted signal with {x.size} samples")
        return signal

    # ------------------------------------------------------------------
    def _flatten_signal_inputs(self, value: Any) -> List[Signal]:
        def _iter_items(item: Any) -> Iterable[Signal]:
            if item is None:
                return
            if isinstance(item, list) or isinstance(item, tuple):
                for sub in item:
                    yield from _iter_items(sub)
            elif isinstance(item, Signal):
                yield item
        return list(_iter_items(value))

    def _build_templates_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        species_filter = config.get("species_filter")
        if isinstance(species_filter, list) and not species_filter:
            species_filter = None
        elif isinstance(species_filter, str) and not species_filter.strip():
            species_filter = None
        bands_kwargs = config.get("bands_kwargs") or {}
        if isinstance(bands_kwargs, str):
            bands_kwargs = {}
        return {
            "fwhm_nm": float(config.get("fwhm_nm", 0.75)),
            "species_filter": species_filter,
            "bands_kwargs": dict(bands_kwargs),
        }

    def _coerce_trim_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {}
        min_nm = config.get("min_nm")
        max_nm = config.get("max_nm")
        if min_nm not in (None, ""):
            kwargs["min_nm"] = float(min_nm)
        if max_nm not in (None, ""):
            kwargs["max_nm"] = float(max_nm)
        return kwargs

    def _coerce_intervals(self, intervals: Any) -> List[Tuple[float, float]]:
        if not intervals:
            return []
        parsed: List[Tuple[float, float]] = []
        for entry in intervals:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                a, b = entry
                parsed.append((float(a), float(b)))
        return parsed

    def _resolve_path(self, path_str: str) -> Path:
        path = Path(path_str).expanduser()
        if not path.is_absolute():
            path = (self._workspace_root / path).resolve()
        return path

    def _report_results(
        self,
        results: Dict[int, Any],
        node_map: Dict[int, NodeItem],
        dependents: Dict[int, List[int]],
    ) -> None:
        sinks = [node_map[nid] for nid, children in dependents.items() if not children]
        if not sinks:
            self._append_log("Run completed (no terminal nodes to report).")
            return
        self._append_log("Run completed. Terminal node summaries:")
        for sink in sinks:
            value = results.get(sink.instance_id)
            title = sink.definition.title
            if isinstance(value, DetectionResult):
                self._append_log(f"- {title}: {len(value.detections)} detections")
                for det in value.detections[:10]:
                    self._append_log(
                        f"    • {det.species}: score={det.score:.4f}, bands={det.meta.get('bands_hit', '?')}"
                    )
            elif isinstance(value, Signal):
                self._append_log(
                    f"- {title}: Signal with {value.wavelength.size} samples, meta keys={list(value.meta.keys())}"
                )
            elif isinstance(value, Templates):
                self._append_log(
                    f"- {title}: Templates for {len(value.species)} species"
                )
            elif isinstance(value, list):
                self._append_log(
                    f"- {title}: Signal batch with {len(value)} entries"
                )
            else:
                self._append_log(f"- {title}: {value}")

    def _append_log(self, message: str) -> None:
        if self._log_view is not None:
            self._log_view.append(message)

    # ------------------------------------------------------------------
    @property
    def node_editor(self) -> NodeEditor:
        return self._node_editor
