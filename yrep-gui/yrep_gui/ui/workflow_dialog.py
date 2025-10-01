"""Dialog for AI-powered workflow generation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Dict, Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QVBoxLayout,
    QMessageBox,
    QComboBox,
)

from yrep_gui.services.agent_runner import WorkflowBuilderAgent, AgentExecutionError

if TYPE_CHECKING:
    from yrep_gui.ui.main_window import MainWindow


class WorkflowBuilderDialog(QDialog):
    """Dialog for generating workflows using AI."""

    def __init__(self, parent: Optional[MainWindow] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI Workflow Builder")
        self.resize(600, 500)

        self._main_window = parent
        self._generated_graph: Optional[Dict[str, Any]] = None

        layout = QVBoxLayout(self)

        # Header
        header = QLabel(
            "<b>Generate Workflow with AI</b><br/>"
            "Describe what you want to accomplish and the AI will build the workflow graph."
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        # Configuration form
        form_layout = QFormLayout()

        self._model_combo = QComboBox(self)
        self._model_combo.addItems(["gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-3.5-turbo"])
        self._model_combo.setCurrentText("gpt-4o")
        form_layout.addRow("Model:", self._model_combo)

        self._api_key_input = QLineEdit(self)
        self._api_key_input.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_input.setPlaceholderText("Leave empty to use .env file")
        form_layout.addRow("API Key:", self._api_key_input)

        layout.addLayout(form_layout)

        # Description input
        desc_label = QLabel("Describe your workflow:")
        layout.addWidget(desc_label)

        self._description_edit = QPlainTextEdit(self)
        self._description_edit.setPlaceholderText(
            "Example:\n"
            "Load a signal from data/sample.txt, trim to 350-550nm, "
            "apply continuum removal, load references from refs/, "
            "build templates, and detect species"
        )
        self._description_edit.setMinimumHeight(150)
        layout.addWidget(self._description_edit)

        # Context input (optional)
        context_label = QLabel("Additional context (optional):")
        layout.addWidget(context_label)

        self._context_edit = QPlainTextEdit(self)
        self._context_edit.setPlaceholderText(
            "JSON format:\n"
            '{\n'
            '  "signal_path": "data/sample.txt",\n'
            '  "references_path": "refs/"\n'
            '}'
        )
        self._context_edit.setMaximumHeight(100)
        layout.addWidget(self._context_edit)

        # Buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self
        )
        button_box.accepted.connect(self._generate_workflow)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _generate_workflow(self) -> None:
        """Generate workflow using AI."""
        description = self._description_edit.toPlainText().strip()
        if not description:
            QMessageBox.warning(self, "Input Required", "Please describe the workflow you want to create.")
            return

        # Parse context if provided
        context = None
        context_text = self._context_edit.toPlainText().strip()
        if context_text:
            try:
                import json
                context = json.loads(context_text)
            except json.JSONDecodeError:
                QMessageBox.warning(
                    self,
                    "Invalid JSON",
                    "Context must be valid JSON or empty."
                )
                return

        # Get API key
        api_key = self._api_key_input.text().strip() or None
        model = self._model_combo.currentText()

        # Show progress
        QMessageBox.information(
            self,
            "Generating...",
            "Generating workflow with AI. This may take a few seconds..."
        )

        try:
            # Create agent and generate workflow
            agent = WorkflowBuilderAgent(api_key=api_key, model=model)
            graph_data = agent.build_workflow(description, context)

            # Validate structure
            if not isinstance(graph_data, dict) or "nodes" not in graph_data:
                raise AgentExecutionError("Generated graph is missing required structure")

            self._generated_graph = graph_data

            # Show success and close
            QMessageBox.information(
                self,
                "Success",
                f"Workflow generated successfully!\n"
                f"Nodes: {len(graph_data.get('nodes', []))}\n"
                f"Edges: {len(graph_data.get('edges', []))}"
            )
            self.accept()

        except AgentExecutionError as exc:
            QMessageBox.critical(
                self,
                "Generation Failed",
                f"Failed to generate workflow:\n\n{exc}"
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "Error",
                f"Unexpected error:\n\n{exc}"
            )

    def get_generated_graph(self) -> Optional[Dict[str, Any]]:
        """Get the generated graph data."""
        return self._generated_graph
