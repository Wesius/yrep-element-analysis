"""Graphics primitives for node items and their connection ports."""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

from PyQt6.QtCore import QPointF, Qt
from PyQt6.QtGui import QBrush, QColor, QFont, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsSimpleTextItem,
    QGraphicsTextItem,
    QStyleOptionGraphicsItem,
    QWidget,
)

from yrep_gui.nodes.registry import NodeDefinition

if TYPE_CHECKING:  # pragma: no cover - hinting only
    from yrep_gui.ui.node_editor import NodeEditor


# Layout metrics --------------------------------------------------------------
BASE_NODE_WIDTH = 240
BASE_NODE_HEIGHT = 140
NODE_RADIUS = 8
HEADER_HEIGHT = 64
PORT_SPACING = 24
PORT_RADIUS = 6
PORT_DIAMETER = PORT_RADIUS * 2

TEXT_PADDING = 14
CONFIG_SPACING = 12


# Palette ---------------------------------------------------------------------
COLOR_BACKGROUND = QColor(53, 59, 72)
COLOR_BACKGROUND_SELECTED = QColor(74, 160, 224)
COLOR_BORDER = QColor(93, 109, 126)
COLOR_TEXT = QColor(234, 237, 240)
COLOR_CATEGORY = QColor(176, 190, 197)
COLOR_PORT_FILL = QColor(232, 124, 56)
COLOR_PORT_FILL_INPUT = QColor(124, 192, 248)
COLOR_PORT_BORDER = QColor(35, 40, 48)
COLOR_PORT_LABEL = QColor(204, 214, 223)
COLOR_EDGE = QColor(170, 197, 255)
COLOR_EDGE_ACTIVE = QColor(255, 214, 102)


class NodeConnection(QGraphicsPathItem):
    """Cubic-bezier curve linking two ports."""

    def __init__(self, start_port: NodePort) -> None:
        super().__init__()
        self.setZValue(-1)
        self.setPen(_edge_pen(COLOR_EDGE))

        self.start_port = start_port
        self.end_port: NodePort | None = None
        self._end_pos = self.start_port.scene_anchor()

        self.update_path()

    def update_path(self) -> None:
        start = self.start_port.scene_anchor()
        end = self.end_port.scene_anchor() if self.end_port else self._end_pos
        dx = max(abs(end.x() - start.x()) * 0.45, 60.0)
        c1 = QPointF(start.x() + dx, start.y())
        c2 = QPointF(end.x() - dx, end.y())
        path = QPainterPath(start)
        path.cubicTo(c1, c2, end)
        self.setPath(path)

    def set_end_pos(self, pos: QPointF) -> None:
        self._end_pos = pos
        self.end_port = None
        self.setPen(_edge_pen(COLOR_EDGE_ACTIVE))
        self.update_path()

    def finalize(self, end_port: NodePort) -> None:
        self.end_port = end_port
        self._end_pos = end_port.scene_anchor()
        self.setPen(_edge_pen(COLOR_EDGE))
        self.update_path()
        self.start_port.add_edge(self)
        end_port.add_edge(self)

    def disconnect(self) -> None:
        self.start_port.remove_edge(self)
        if self.end_port is not None:
            self.end_port.remove_edge(self)
        self.end_port = None


class NodePort(QGraphicsEllipseItem):
    """Input or output anchor rendered on the side of a node."""

    def __init__(
        self,
        editor: NodeEditor,
        owner: NodeItem,
        *,
        port_type: str,
        name: str,
        index: int,
        allow_multiple: bool = False,
        optional: bool = False,
    ) -> None:
        super().__init__(-PORT_RADIUS, -PORT_RADIUS, PORT_DIAMETER, PORT_DIAMETER, owner)
        self._editor = editor
        self.owner = owner
        self.port_type = port_type  # "input" or "output"
        self.name = name
        self.index = index
        self.edges: list[NodeConnection] = []
        self.allow_multiple = allow_multiple
        self.optional = optional

        base_color = COLOR_PORT_FILL_INPUT if port_type == "input" else COLOR_PORT_FILL
        if optional and port_type == "input":
            base_color = base_color.lighter(140)
        self._default_brush = QBrush(base_color)
        self.setBrush(self._default_brush)
        self.setPen(QPen(COLOR_PORT_BORDER, 1.1))
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations)
        self.setAcceptHoverEvents(True)
        tip = f"{port_type.capitalize()} port: {name}"
        if optional and port_type == "input":
            tip += " (optional)"
        if allow_multiple and port_type == "input":
            tip += " (multi-input)"
        self.setToolTip(tip)

    # Convenience ---------------------------------------------------------
    def scene_anchor(self) -> QPointF:
        return self.mapToScene(self.boundingRect().center())

    def add_edge(self, edge: NodeConnection) -> None:
        if edge not in self.edges:
            self.edges.append(edge)

    def remove_edge(self, edge: NodeConnection) -> None:
        if edge in self.edges:
            self.edges.remove(edge)

    # Event hooks ---------------------------------------------------------
    def mousePressEvent(self, event) -> None:  # noqa: ANN001
        if event.button() == Qt.MouseButton.LeftButton:
            self._editor.handle_port_pressed(self)
            event.accept()
            return
        super().mousePressEvent(event)

    def hoverEnterEvent(self, event) -> None:  # noqa: ANN001
        self.setBrush(QBrush(self._default_brush.color().lighter(120)))
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: ANN001
        self.setBrush(self._default_brush)
        super().hoverLeaveEvent(event)


class NodeItem(QGraphicsRectItem):
    """Visual representation of a node definition placed on the canvas."""

    def __init__(
        self,
        editor: NodeEditor,
        definition: NodeDefinition,
        *,
        instance_id: int,
        config: dict[str, Any],
    ) -> None:
        super().__init__()
        self._editor = editor
        self.definition = definition
        self.instance_id = instance_id
        self.config = dict(config)

        self._max_ports = max(len(definition.inputs), len(definition.outputs), 1)
        self._port_labels: list[QGraphicsSimpleTextItem] = []
        self._output_labels: list[QGraphicsSimpleTextItem] = []

        self.setRect(0, 0, BASE_NODE_WIDTH, BASE_NODE_HEIGHT)
        self.setPen(QPen(COLOR_BORDER, 1.2))
        self.setBrush(QBrush(COLOR_BACKGROUND))
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

        self.inputs: list[NodePort] = []
        self.outputs: list[NodePort] = []

        self._title_item = QGraphicsSimpleTextItem(definition.title, self)
        title_font = QFont()
        title_font.setPointSizeF(12.0)
        title_font.setBold(True)
        self._title_item.setFont(title_font)
        self._title_item.setBrush(QBrush(COLOR_TEXT))
        self._title_item.setPos(14, 14)

        category_text = f"{definition.category} · #{instance_id}".strip()
        self._category_item = QGraphicsSimpleTextItem(category_text, self)
        category_font = QFont()
        category_font.setPointSizeF(9.5)
        self._category_item.setFont(category_font)
        self._category_item.setBrush(QBrush(COLOR_CATEGORY))
        self._category_item.setPos(14, 38)

        self._config_item = QGraphicsTextItem(self)
        config_font = QFont()
        config_font.setPointSizeF(9.3)
        self._config_item.setFont(config_font)
        self._config_item.setDefaultTextColor(COLOR_TEXT)
        self._config_item.setTextWidth(BASE_NODE_WIDTH - 2 * TEXT_PADDING)

        self._build_ports()
        self.update_config(self.config)

        self.setToolTip(self._build_tooltip())

    # ------------------------------------------------------------------
    # QGraphicsItem overrides
    # ------------------------------------------------------------------
    def paint(
        self,
        painter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(painter.RenderHint.Antialiasing)
        painter.setPen(self.pen())
        painter.setBrush(QBrush(COLOR_BACKGROUND_SELECTED if self.isSelected() else COLOR_BACKGROUND))
        rect = self.rect()
        painter.drawRoundedRect(rect, NODE_RADIUS, NODE_RADIUS, Qt.SizeMode.AbsoluteSize)

    def itemChange(self, change, value):  # noqa: ANN001
        if change == self.GraphicsItemChange.ItemSelectedHasChanged:
            self.update()
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            self._editor.update_edges_for(self)
        return super().itemChange(change, value)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _build_ports(self) -> None:
        label_font = QFont()
        label_font.setPointSizeF(9.0)

        for index, name in enumerate(self.definition.inputs):
            allow_multi = index in self.definition.multi_input_ports
            optional = index in self.definition.optional_input_ports
            display_name = name
            if self.definition.identifier == "subtract_background":
                display_name = "Foreground" if index == 0 else "Background"
            port = NodePort(
                self._editor,
                self,
                port_type="input",
                name=display_name,
                index=index,
                allow_multiple=allow_multi,
                optional=optional,
            )
            y = HEADER_HEIGHT + index * PORT_SPACING
            port.setPos(self.rect().left() + 8, y)
            label = QGraphicsSimpleTextItem(display_name, self)
            label.setFont(label_font)
            label.setBrush(QBrush(COLOR_PORT_LABEL))
            label.setPos(20, y - label.boundingRect().height() / 2)
            self.inputs.append(port)
            self._port_labels.append(label)

        for index, name in enumerate(self.definition.outputs):
            display_name = name
            port = NodePort(self._editor, self, port_type="output", name=display_name, index=index)
            y = HEADER_HEIGHT + index * PORT_SPACING
            port.setPos(self.rect().right() - 8, y)
            label = QGraphicsSimpleTextItem(display_name, self)
            label.setFont(label_font)
            label.setBrush(QBrush(COLOR_PORT_LABEL))
            label.setPos(self.rect().right() - 20 - label.boundingRect().width(), y - label.boundingRect().height() / 2)
            self.outputs.append(port)
            self._output_labels.append(label)

        self._update_output_positions()
        self._update_layout()

    def _update_output_positions(self) -> None:
        right = self.rect().right()
        for port in self.outputs:
            port.setPos(right - 8, port.pos().y())
        for label in self._output_labels:
            y = label.pos().y()
            width = label.boundingRect().width()
            label.setPos(right - 20 - width, y)

    def _update_layout(self) -> None:
        config_text = self._format_config_preview()
        self._config_item.setPlainText(config_text)
        self._config_item.setTextWidth(self.rect().width() - 2 * TEXT_PADDING)
        config_top = HEADER_HEIGHT + self._max_ports * PORT_SPACING + CONFIG_SPACING
        self._config_item.setPos(TEXT_PADDING, config_top)
        text_height = self._config_item.boundingRect().height()
        min_height = HEADER_HEIGHT + self._max_ports * PORT_SPACING + CONFIG_SPACING * 2
        required_height = max(min_height, config_top + text_height + CONFIG_SPACING)
        if abs(required_height - self.rect().height()) > 0.1:
            self.prepareGeometryChange()
            self.setRect(0, 0, BASE_NODE_WIDTH, required_height)
            self._update_output_positions()
        else:
            self._update_output_positions()

    def _format_config_preview(self) -> str:
        if not self.config:
            return "(no parameters)"
        lines = []
        for key, value in self.config.items():
            if isinstance(value, float):
                value_str = f"{value:.6g}"
            elif isinstance(value, (list, tuple, dict)):
                value_str = str(value)
            else:
                value_str = value
            lines.append(f"{key}: {value_str}")
        return "\n".join(lines)

    def refresh_config(self) -> None:
        self._update_layout()
        self.setToolTip(self._build_tooltip())

    def set_config_value(self, key: str, value: Any) -> None:
        self.config[key] = value
        self.refresh_config()

    def update_config(self, new_config: dict[str, Any]) -> None:
        self.config = dict(new_config)
        self.refresh_config()

    def _build_tooltip(self) -> str:
        lines = [
            f"{self.definition.title} (id: {self.definition.identifier})",
            f"Category: {self.definition.category}",
            "",
        ]
        if not self.config:
            lines.append("No configuration parameters")
        else:
            lines.append("Configuration:")
            for key, value in self.config.items():
                lines.append(f"  • {key} = {value}")
        return "\n".join(lines)

    # Convenience API -------------------------------------------------
    def center_on(self, point: QPointF) -> None:
        half = self.rect().center()
        self.setPos(point - half)


def _edge_pen(color: QColor) -> QPen:
    pen = QPen(color, 2.2)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    pen.setCosmetic(True)
    return pen


__all__ = ["NodeItem", "NodePort", "NodeConnection"]
