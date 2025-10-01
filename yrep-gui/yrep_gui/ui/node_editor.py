"""Node editor primitives built on QGraphicsScene/QGraphicsView."""

from __future__ import annotations

from typing import Any, Iterable, TYPE_CHECKING

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtWidgets import QGraphicsSceneMouseEvent

from yrep_gui.nodes import registry
from yrep_gui.nodes.registry import NodeDefinition
from yrep_gui.ui.node_item import NodeConnection, NodeItem, NodePort

if TYPE_CHECKING:  # pragma: no cover - hinting only
    from PyQt6.QtGui import QKeyEvent, QMouseEvent

GRID_SIZE = 25
FINE_GRID_COLOR = QColor(60, 60, 60)
COARSE_GRID_COLOR = QColor(80, 80, 80)
BACKGROUND_COLOR = QColor(33, 33, 33)

NODE_MIME_TYPE = 'application/x-yrep-node'


class NodeScene(QGraphicsScene):
    """Scene that renders a dark grid background and mediates editor events."""

    def __init__(self, editor: "NodeEditor") -> None:
        super().__init__(editor)
        self._editor = editor
        # Start with reasonable bounds, will expand as needed
        self.setSceneRect(-2000, -1500, 4000, 3000)

    @property
    def editor(self) -> "NodeEditor":
        return self._editor

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        painter.fillRect(rect, BACKGROUND_COLOR)

        left = int(rect.left()) - (int(rect.left()) % GRID_SIZE)
        top = int(rect.top()) - (int(rect.top()) % GRID_SIZE)

        lines_fine: list[tuple[QPointF, QPointF]] = []
        lines_coarse: list[tuple[QPointF, QPointF]] = []
        grid_step = GRID_SIZE

        for x in range(left, int(rect.right()), grid_step):
            line = (QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            target = lines_coarse if x % (grid_step * 5) == 0 else lines_fine
            target.append(line)

        for y in range(top, int(rect.bottom()), grid_step):
            line = (QPointF(rect.left(), y), QPointF(rect.right(), y))
            target = lines_coarse if y % (grid_step * 5) == 0 else lines_fine
            target.append(line)

        painter.setPen(QPen(FINE_GRID_COLOR, 0))
        _draw_lines(painter, lines_fine)

        painter.setPen(QPen(COARSE_GRID_COLOR, 0))
        _draw_lines(painter, lines_coarse)

    # Event surfaces -----------------------------------------------------
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # pragma: no cover - UI only
        if self._editor.update_drag_edge(event.scenePos()):
            event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # pragma: no cover - UI only
        if event.button() == Qt.MouseButton.LeftButton and self._editor.finish_connection(event.scenePos()):
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: "QKeyEvent") -> None:  # pragma: no cover - UI only
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self._editor.delete_selection()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._editor.cancel_connection()
            event.accept()
            return
        super().keyPressEvent(event)


def _draw_lines(painter: QPainter, lines: Iterable[tuple[QPointF, QPointF]]) -> None:
    for start, end in lines:
        painter.drawLine(start, end)


class NodeView(QGraphicsView):
    """Interactive view configured for panning/zooming."""

    def __init__(self, scene: NodeScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setBackgroundBrush(BACKGROUND_COLOR)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setAcceptDrops(True)
        self.viewport().setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        self._panning = False
        self._pan_start = QPointF()
        self._space_pan_shortcut = False

    def wheelEvent(self, event):  # noqa: ANN001 - UI callback
        # Zoom with mouse wheel, with limits to prevent extreme zoom
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        # Get current scale
        current_scale = self.transform().m11()

        # Apply zoom limits (0.1x to 3.0x)
        if event.angleDelta().y() > 0:
            # Zoom in
            if current_scale < 3.0:
                self.scale(zoom_in_factor, zoom_in_factor)
        else:
            # Zoom out
            if current_scale > 0.1:
                self.scale(zoom_out_factor, zoom_out_factor)

        event.accept()

    def mousePressEvent(self, event):  # noqa: ANN001 - UI callback
        if event.button() == Qt.MouseButton.MiddleButton or (
            event.button() == Qt.MouseButton.LeftButton and self._space_pan_shortcut
        ):
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):  # noqa: ANN001 - UI callback
        if self._panning:
            delta = event.position() - self._pan_start
            self._pan_start = event.position()
            # Smooth panning using scrollbars
            hbar = self.horizontalScrollBar()
            vbar = self.verticalScrollBar()
            hbar.setValue(int(hbar.value() - delta.x()))
            vbar.setValue(int(vbar.value() - delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):  # noqa: ANN001 - UI callback
        if self._panning and event.button() in {
            Qt.MouseButton.MiddleButton,
            Qt.MouseButton.LeftButton,
        }:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event):  # noqa: ANN001 - UI callback
        if event.key() == Qt.Key.Key_Space:
            if not self._space_pan_shortcut and not self._panning:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            self._space_pan_shortcut = True
            event.accept()
            return

        # Arrow key navigation
        pan_amount = 50
        if event.key() == Qt.Key.Key_Left:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - pan_amount)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Right:
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + pan_amount)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Up:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - pan_amount)
            event.accept()
            return
        if event.key() == Qt.Key.Key_Down:
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() + pan_amount)
            event.accept()
            return

        # Home key - fit all nodes
        if event.key() == Qt.Key.Key_Home:
            self.fit_all_nodes()
            event.accept()
            return

        # Zero key - reset zoom
        if event.key() == Qt.Key.Key_0 and event.modifiers() == Qt.KeyboardModifier.NoModifier:
            self.reset_zoom()
            event.accept()
            return

        super().keyPressEvent(event)

    def keyReleaseEvent(self, event):  # noqa: ANN001 - UI callback
        if event.key() == Qt.Key.Key_Space:
            self._space_pan_shortcut = False
            if not self._panning:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def dragEnterEvent(self, event):  # noqa: ANN001 - UI callback
        if event.mimeData().hasFormat(NODE_MIME_TYPE):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event):  # noqa: ANN001 - UI callback
        if event.mimeData().hasFormat(NODE_MIME_TYPE):
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):  # noqa: ANN001 - UI callback
        if event.mimeData().hasFormat(NODE_MIME_TYPE):
            identifier = bytes(event.mimeData().data(NODE_MIME_TYPE)).decode("utf-8").strip()
            if identifier:
                scene_pos = self.mapToScene(event.position().toPoint())
                self.scene().editor.add_node_by_identifier(identifier, scene_pos)
                event.acceptProposedAction()
                return
        super().dropEvent(event)

    # View navigation helpers ---------------------------------------------
    def fit_all_nodes(self) -> None:
        """Zoom and pan to fit all nodes in the viewport."""
        items = [item for item in self.scene().items() if hasattr(item, "definition")]
        if not items:
            # No nodes, center on origin
            self.centerOn(0, 0)
            self.resetTransform()
            return

        # Get bounding rect of all nodes
        rect = items[0].sceneBoundingRect()
        for item in items[1:]:
            rect = rect.united(item.sceneBoundingRect())

        # Add padding
        padding = 100
        rect.adjust(-padding, -padding, padding, padding)

        # Fit the rect in view
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)

        # Limit zoom to reasonable range
        transform = self.transform()
        scale = transform.m11()  # horizontal scale factor
        if scale > 2.0:
            self.resetTransform()
            self.scale(2.0, 2.0)
            self.centerOn(rect.center())
        elif scale < 0.1:
            self.resetTransform()
            self.scale(0.1, 0.1)
            self.centerOn(rect.center())

    def reset_zoom(self) -> None:
        """Reset zoom to 100% (1:1 scale)."""
        center = self.mapToScene(self.viewport().rect().center())
        self.resetTransform()
        self.centerOn(center)

    def center_view(self) -> None:
        """Center view on the origin (0, 0)."""
        self.centerOn(0, 0)



class NodeEditor(QWidget):
    """Composite widget hosting the node scene and managing graph state."""

    nodeSelected = pyqtSignal(object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = NodeScene(self)
        self._view = NodeView(self._scene, self)
        self._view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._node_counter = 0
        self._nodes: list[NodeItem] = []
        self._edges: list[NodeConnection] = []
        self._drag_edge: NodeConnection | None = None

        self._scene.selectionChanged.connect(self._handle_selection_changed)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def clear_graph(self) -> None:
        self.cancel_connection()
        for edge in list(self._edges):
            self._remove_edge(edge)
        self._scene.clear()
        self._edges.clear()
        self._node_counter = 0
        self._nodes.clear()
        self.nodeSelected.emit(None)

    def add_node(
        self,
        definition: NodeDefinition,
        position: QPointF | None = None,
        *,
        instance_id: int | None = None,
        config: dict[str, Any] | None = None,
    ) -> NodeItem:
        if instance_id is None:
            self._node_counter += 1
            node_id = self._node_counter
        else:
            node_id = int(instance_id)
            self._node_counter = max(self._node_counter, node_id)
        node = NodeItem(
            self,
            definition,
            instance_id=node_id,
            config=dict(config or definition.default_config),
        )
        target = position if isinstance(position, QPointF) else self._default_spawn_position()
        node.center_on(target)
        self._scene.addItem(node)
        self._scene.clearSelection()
        node.setSelected(True)
        self._nodes.append(node)
        self.update_scene_bounds()
        return node

    def add_node_by_identifier(self, identifier: str, position: QPointF | None = None) -> NodeItem:
        definition = registry.get(identifier)
        return self.add_node(definition, position)

    def selected_node(self) -> NodeItem | None:
        for item in self._scene.selectedItems():
            if isinstance(item, NodeItem):
                return item
        return None

    def nodes(self) -> tuple[NodeItem, ...]:
        return tuple(self._nodes)

    def handle_port_pressed(self, port: NodePort) -> None:
        if port.port_type == "output":
            if self._drag_edge and self._drag_edge.start_port is port:
                self.cancel_connection()
                return
            self.begin_connection(port)
            return

        # Input ports: finish an in-flight connection or detach existing link.
        if self._drag_edge:
            self._complete_connection(port)
            return
        if port.edges:
            self._remove_edge(port.edges[0])

    def begin_connection(self, port: NodePort) -> None:
        self.cancel_connection()
        self._drag_edge = NodeConnection(port)
        self._scene.addItem(self._drag_edge)
        self._drag_edge.set_end_pos(port.scene_anchor())

    def update_drag_edge(self, scene_pos: QPointF) -> bool:
        if not self._drag_edge:
            return False
        self._drag_edge.set_end_pos(scene_pos)
        return True

    def finish_connection(self, scene_pos: QPointF) -> bool:
        if not self._drag_edge:
            return False
        target = self._port_at(scene_pos)
        if target and target.port_type == "input":
            self._complete_connection(target)
        else:
            self.cancel_connection()
        return True

    def cancel_connection(self) -> None:
        if self._drag_edge is not None:
            self._scene.removeItem(self._drag_edge)
            self._drag_edge = None

    def delete_selection(self) -> None:
        to_delete = [item for item in self._scene.selectedItems() if isinstance(item, NodeItem)]
        for node in to_delete:
            self._remove_node(node)
        if to_delete:
            self.nodeSelected.emit(self.selected_node())

    def update_edges_for(self, node: NodeItem) -> None:
        for port in (*node.inputs, *node.outputs):
            for edge in list(port.edges):
                edge.update_path()

    def update_scene_bounds(self) -> None:
        """Expand scene rect to include all nodes with padding."""
        if not self._nodes:
            return

        # Get bounding rect of all nodes
        rect = self._nodes[0].sceneBoundingRect()
        for node in self._nodes[1:]:
            rect = rect.united(node.sceneBoundingRect())

        # Add generous padding
        padding = 1000
        rect.adjust(-padding, -padding, padding, padding)

        # Expand scene rect if needed (never shrink it to avoid jarring jumps)
        current = self._scene.sceneRect()
        new_rect = current.united(rect)
        if new_rect != current:
            self._scene.setSceneRect(new_rect)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _complete_connection(self, input_port: NodePort) -> None:
        if not self._drag_edge:
            return
        output_port = self._drag_edge.start_port
        if input_port is output_port or input_port.owner is output_port.owner:
            self.cancel_connection()
            return
        if any(edge.end_port is input_port for edge in output_port.edges):
            self.cancel_connection()
            return
        if input_port.edges and not input_port.allow_multiple:
            self._remove_edge(input_port.edges[0])
        if input_port.allow_multiple:
            if any(edge.start_port.owner is output_port.owner for edge in input_port.edges):
                self.cancel_connection()
                return

        self._drag_edge.finalize(input_port)
        self._edges.append(self._drag_edge)
        self._drag_edge = None
        self.update_edges_for(input_port.owner)
        self.update_edges_for(output_port.owner)

    def _remove_node(self, node: NodeItem) -> None:
        for port in (*node.inputs, *node.outputs):
            for edge in list(port.edges):
                self._remove_edge(edge)
        self._scene.removeItem(node)
        if node in self._nodes:
            self._nodes.remove(node)

    def _remove_edge(self, edge: NodeConnection) -> None:
        edge.disconnect()
        if edge in self._edges:
            self._edges.remove(edge)
        if edge is self._drag_edge:
            self._drag_edge = None
        self._scene.removeItem(edge)

    def _port_at(self, scene_pos: QPointF) -> NodePort | None:
        for item in self._scene.items(scene_pos):
            if isinstance(item, NodePort):
                return item
        return None

    def _connect_nodes(
        self,
        source: NodeItem,
        source_port_index: int,
        target: NodeItem,
        target_port_index: int,
    ) -> None:
        if not (0 <= source_port_index < len(source.outputs)):
            return
        if not (0 <= target_port_index < len(target.inputs)):
            return
        source_port = source.outputs[source_port_index]
        target_port = target.inputs[target_port_index]
        if target_port.edges and not target_port.allow_multiple:
            self._remove_edge(target_port.edges[0])
        if target_port.allow_multiple and any(edge.start_port.owner is source for edge in target_port.edges):
            return
        edge = NodeConnection(source_port)
        self._scene.addItem(edge)
        edge.finalize(target_port)
        self._edges.append(edge)
        self.update_edges_for(source)
        self.update_edges_for(target)

    def _default_spawn_position(self) -> QPointF:
        viewport_rect = self._view.viewport().rect()
        center = self._view.mapToScene(viewport_rect.center())
        return QPointF(center)

    def _handle_selection_changed(self) -> None:
        self.nodeSelected.emit(self.selected_node())

    def export_graph(self) -> tuple[tuple[NodeItem, ...], tuple[NodeConnection, ...]]:
        return tuple(self._nodes), tuple(self._edges)

    def export_graph_data(self) -> dict[str, Any]:
        nodes_payload = []
        for node in self._nodes:
            pos = node.pos()
            nodes_payload.append(
                {
                    "id": node.instance_id,
                    "identifier": node.definition.identifier,
                    "config": node.config,
                    "position": [float(pos.x()), float(pos.y())],
                }
            )

        edges_payload = []
        for edge in self._edges:
            if edge.start_port is None or edge.end_port is None:
                continue
            edges_payload.append(
                {
                    "source": edge.start_port.owner.instance_id,
                    "source_port": edge.start_port.index,
                    "target": edge.end_port.owner.instance_id,
                    "target_port": edge.end_port.index,
                }
            )

        return {
            "version": 1,
            "nodes": nodes_payload,
            "edges": edges_payload,
        }

    def load_graph_data(self, payload: dict[str, Any]) -> None:
        self.clear_graph()
        node_lookup: dict[int, NodeItem] = {}

        for node_info in payload.get("nodes", []):
            try:
                identifier = node_info["identifier"]
                node_id = int(node_info["id"])
                config = dict(node_info.get("config", {}))
                position = node_info.get("position", [0.0, 0.0])
            except (KeyError, TypeError, ValueError):
                continue
            definition = registry.get(identifier)
            pos = QPointF(float(position[0]), float(position[1])) if isinstance(position, (list, tuple)) and len(position) == 2 else None
            node = self.add_node(definition, position=pos, instance_id=node_id, config=config)
            node_lookup[node_id] = node

        for edge_info in payload.get("edges", []):
            try:
                src_id = int(edge_info["source"])
                src_port = int(edge_info["source_port"])
                dst_id = int(edge_info["target"])
                dst_port = int(edge_info["target_port"])
            except (KeyError, TypeError, ValueError):
                continue
            source_node = node_lookup.get(src_id)
            target_node = node_lookup.get(dst_id)
            if source_node is None or target_node is None:
                continue
            self._connect_nodes(source_node, src_port, target_node, dst_port)

        self.update_scene_bounds()
        self.nodeSelected.emit(self.selected_node())

    @property
    def scene(self) -> NodeScene:
        return self._scene

    @property
    def view(self) -> NodeView:
        return self._view
