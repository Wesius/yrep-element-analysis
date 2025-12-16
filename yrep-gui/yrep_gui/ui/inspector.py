"""Inspector panel for editing node configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from yrep_gui.ui.node_item import NodeItem


_JSON_INDENT = 2


class InspectorPanel(QWidget):
    """Widget that renders and edits a node's configuration."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._current_node: NodeItem | None = None
        self._updating = False
        self._field_widgets: Dict[str, QWidget] = {}
        self._field_status: Dict[str, QLabel] = {}
        self._line_widgets: Dict[str, QLineEdit] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self._header_label = QLabel("Select a node to configure parameters.", self)
        self._header_label.setWordWrap(True)
        layout.addWidget(self._header_label)

        self._form_container = QWidget(self)
        self._form_layout = QFormLayout(self._form_container)
        self._form_layout.setContentsMargins(0, 0, 0, 0)
        self._form_layout.setSpacing(6)
        layout.addWidget(self._form_container, 1)

        layout.addStretch(1)
        self._form_container.hide()

    # ------------------------------------------------------------------
    def set_node(self, node: NodeItem | None) -> None:
        self._current_node = node
        self._clear_form()
        if node is None:
            self._header_label.setText(
                "Select a node to configure parameters.\n\n"
                "Hint: drag palette entries onto the canvas, then adjust their parameters here."
            )
            self._form_container.hide()
            return

        definition = node.definition
        self._header_label.setText(
            f"<b>{definition.title}</b><br/>"
            f"Identifier: <code>{definition.identifier}</code><br/>"
            f"Category: {definition.category}"
        )

        keys = list(dict.fromkeys(list(definition.default_config.keys()) + list(node.config.keys())))
        for key in keys:
            default_value = definition.default_config.get(key)
            current_value = node.config.get(key, default_value)
            widget = self._create_editor_widget(key, current_value, default_value)
            self._form_layout.addRow(f"{key}", widget)
        self._form_container.show()

    # ------------------------------------------------------------------
    def _clear_form(self) -> None:
        self._updating = True
        while self._form_layout.count():
            item = self._form_layout.takeAt(0)
            if item is None:
                continue
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._field_widgets.clear()
        self._field_status.clear()
        self._line_widgets.clear()
        self._updating = False

    def _create_editor_widget(self, key: str, value: Any, default: Any) -> QWidget:
        self._updating = True
        widget: QWidget

        # Special handling for agent configuration fields
        if key == "model":
            combo = QComboBox(self)
            models = ["gpt-5", "gpt-5-pro", "gpt-4", "gpt-4-turbo", "gpt-3.5-turbo", "gpt-4o", "gpt-4o-mini"]
            combo.addItems(models)
            current_val = str(value) if value else "gpt-5"
            if current_val in models:
                combo.setCurrentText(current_val)
            combo.currentTextChanged.connect(lambda val, k=key: self._on_str_changed(k, val))
            widget = combo
        elif key == "task":
            combo = QComboBox(self)
            tasks = ["general", "quality_control", "report_generator", "parameter_optimizer"]
            combo.addItems(tasks)
            current_val = str(value) if value else "general"
            if current_val in tasks:
                combo.setCurrentText(current_val)
            combo.currentTextChanged.connect(lambda val, k=key: self._on_str_changed(k, val))
            widget = combo
        elif key in {"custom_prompt"}:
            # Multi-line text editor for prompts
            editor = QPlainTextEdit(self)
            editor.setPlainText(str(value) if value else "")
            editor.setMaximumHeight(120)
            editor.textChanged.connect(lambda k=key, e=editor: self._on_text_changed(k, e.toPlainText()))
            widget = editor
        elif isinstance(value, bool) or isinstance(default, bool):
            checkbox = QCheckBox(self)
            checkbox.setChecked(bool(value))
            checkbox.stateChanged.connect(lambda state, k=key: self._on_bool_changed(k, state))
            widget = checkbox
        elif isinstance(value, int) or isinstance(default, int):
            spin = QSpinBox(self)
            spin.setRange(-10_000_000, 10_000_000)
            spin.setValue(int(value))
            spin.valueChanged.connect(lambda val, k=key: self._on_int_changed(k, val))
            widget = spin
        elif isinstance(value, float) or isinstance(default, float):
            spin = QDoubleSpinBox(self)
            spin.setDecimals(6)
            spin.setRange(-1_000_000.0, 1_000_000.0)
            spin.setSingleStep(0.1)
            spin.setValue(float(value))
            spin.valueChanged.connect(lambda val, k=key: self._on_float_changed(k, val))
            widget = spin
        elif isinstance(value, str) or isinstance(default, str):
            container = QWidget(self)
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(4)

            line = QLineEdit(container)
            line.setText(str(value) if value is not None else "")
            line.setClearButtonEnabled(True)
            if key == "api_key":
                line.setEchoMode(QLineEdit.EchoMode.Password)
                line.setPlaceholderText("Leave empty to use OPENAI_API_KEY env var")
            line.editingFinished.connect(lambda k=key, w=line: self._on_str_changed(k, w.text()))
            layout.addWidget(line, 1)

            browse_mode = None
            lowered = key.lower()
            if lowered in {"path", "file", "filepath"}:
                browse_mode = "file"
            elif lowered in {"directory", "folder", "dir"}:
                browse_mode = "directory"

            if browse_mode is not None:
                button = QPushButton("Browse…", container)
                button.setAutoDefault(False)
                button.clicked.connect(lambda _=False, k=key, m=browse_mode, line=line: self._browse_for_path(k, line, m))
                layout.addWidget(button)

            widget = container
            self._line_widgets[key] = line
        else:
            widget = self._create_json_editor(key, value)

        self._field_widgets[key] = widget
        self._updating = False
        return widget

    def _create_json_editor(self, key: str, value: Any) -> QWidget:
        container = QWidget(self)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        editor = QPlainTextEdit(container)
        editor.setTabChangesFocus(True)
        try:
            text = json.dumps(value, indent=_JSON_INDENT)
        except TypeError:
            text = json.dumps(str(value))
        editor.setPlainText(text)
        layout.addWidget(editor)

        controls = QHBoxLayout()
        apply_btn = QPushButton("Apply", container)
        status = QLabel("", container)
        status.setStyleSheet("color: #888;")
        controls.addWidget(apply_btn)
        controls.addWidget(status, 1)
        layout.addLayout(controls)

        apply_btn.clicked.connect(lambda k=key, e=editor, s=status: self._on_json_apply(k, e, s))
        self._field_status[key] = status
        self._field_widgets[key] = editor
        return container

    # ------------------------------------------------------------------
    def _on_bool_changed(self, key: str, state: int) -> None:
        if self._updating or not self._current_node:
            return
        self._current_node.set_config_value(key, bool(state))

    def _on_int_changed(self, key: str, value: int) -> None:
        if self._updating or not self._current_node:
            return
        self._current_node.set_config_value(key, int(value))

    def _on_float_changed(self, key: str, value: float) -> None:
        if self._updating or not self._current_node:
            return
        self._current_node.set_config_value(key, float(value))

    def _on_str_changed(self, key: str, value: str) -> None:
        if self._updating or not self._current_node:
            return
        self._current_node.set_config_value(key, value)

    def _on_text_changed(self, key: str, value: str) -> None:
        if self._updating or not self._current_node:
            return
        self._current_node.set_config_value(key, value)

    def _browse_for_path(self, key: str, line: QLineEdit, mode: str) -> None:
        start_path = line.text().strip()
        if mode == "file":
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select Signal",
                start_path or str(Path.cwd()),
                "Text files (*.txt);;All files (*)",
            )
            if not file_path:
                return
            new_value = file_path
        else:
            directory = QFileDialog.getExistingDirectory(
                self,
                "Select Directory",
                start_path or str(Path.cwd()),
            )
            if not directory:
                return
            new_value = directory
        self._updating = True
        line.setText(new_value)
        self._updating = False
        self._on_str_changed(key, new_value)

    def _on_json_apply(self, key: str, editor: QPlainTextEdit, status: QLabel) -> None:
        if not self._current_node:
            return
        text = editor.toPlainText().strip() or "null"
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            status.setText(f"Parse error: {exc.msg}")
            status.setStyleSheet("color: #d9534f;")
            return
        self._current_node.set_config_value(key, parsed)
        status.setText("Applied")
        status.setStyleSheet("color: #5cb85c;")
