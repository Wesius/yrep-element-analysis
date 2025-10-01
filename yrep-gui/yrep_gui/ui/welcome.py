"""Welcome screen for YREP GUI."""

from __future__ import annotations

from typing import Dict, Mapping

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QVBoxLayout,
)

from yrep_gui.templates import TemplateDetails


class WelcomeDialog(QDialog):
    """Initial landing dialog offering blank or template projects."""

    def __init__(
        self,
        templates: Dict[str, dict],
        template_details: Mapping[str, TemplateDetails] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("YREP Node Editor")
        self.setModal(True)
        self._templates = templates
        self._template_details: Dict[str, TemplateDetails] = dict(template_details or {})
        self.selected_template: str | None = None

        layout = QVBoxLayout(self)

        header = QLabel("Choose how to get started")
        header.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        layout.addWidget(header)

        self._list = QListWidget(self)
        blank_item = QListWidgetItem("Start a blank graph")
        blank_item.setData(Qt.ItemDataRole.UserRole, None)
        self._list.addItem(blank_item)

        if templates:
            template_header = QListWidgetItem("Templates")
            font = template_header.font()
            font.setBold(True)
            template_header.setFont(font)
            template_header.setFlags(Qt.ItemFlag.ItemIsEnabled)
            template_header.setData(Qt.ItemDataRole.UserRole, "header")
            self._list.addItem(template_header)
            for name in templates.keys():
                item = QListWidgetItem(f"  {name}")
                item.setData(Qt.ItemDataRole.UserRole, name)
                self._list.addItem(item)

        self._list.setCurrentRow(0)
        self._list.itemActivated.connect(self._handle_item_activated)
        self._list.currentItemChanged.connect(self._update_details)
        layout.addWidget(self._list)

        self._details = QLabel(self)
        self._details.setWordWrap(True)
        self._details.setAlignment(Qt.AlignmentFlag.AlignTop)
        self._details.setObjectName("template-details")
        layout.addWidget(self._details)
        self._update_details(self._list.currentItem())

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _handle_item_activated(self, item: QListWidgetItem) -> None:
        self._select_item(item)
        self._update_details(item)
        self.accept()

    def exec_and_get_template(self) -> tuple[str | None, bool]:
        result = self.exec()
        accepted = result == QDialog.DialogCode.Accepted
        return (self.selected_template if accepted else None, accepted)

    def accept(self) -> None:  # noqa: D401
        item = self._list.currentItem()
        self._select_item(item)
        super().accept()

    def _select_item(self, item: QListWidgetItem | None) -> None:
        if item is None:
            self.selected_template = None
            return
        value = item.data(Qt.ItemDataRole.UserRole)
        if value == "header":
            self.selected_template = None
            return
        if value is None:
            self.selected_template = None
        else:
            self.selected_template = str(value)

    def _update_details(self, current: QListWidgetItem | None, _previous: QListWidgetItem | None = None) -> None:
        self._details.setText(self._details_for_item(current))

    def _details_for_item(self, item: QListWidgetItem | None) -> str:
        if item is None:
            return "Select an option to continue."
        value = item.data(Qt.ItemDataRole.UserRole)
        if value == "header":
            return "Choose a template below to preview what it loads."
        if value is None:
            return "Start from an empty canvas with no pre-configured nodes."
        details = self._template_details.get(str(value))
        if details is None:
            return f"Create the '{value}' template."
        return details.description
