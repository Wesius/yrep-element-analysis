"""Application bootstrap for the YREP GUI."""

from __future__ import annotations

import sys
from typing import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from yrep_gui.ui.main_window import MainWindow
from yrep_gui.ui.welcome import WelcomeDialog
from yrep_gui.templates import TEMPLATE_CATALOG, TEMPLATES


def create_application(argv: Sequence[str] | None = None) -> QApplication:
    """Configure and return the QApplication instance."""
    for attr_name in ("AA_EnableHighDpiScaling", "AA_UseHighDpiPixmaps"):
        attr = getattr(Qt.ApplicationAttribute, attr_name, None)
        if attr is not None:
            QApplication.setAttribute(attr, True)
    app = QApplication(list(argv) if argv is not None else sys.argv)
    app.setApplicationName("YREP Node Editor")
    app.setOrganizationName("YREP")
    return app


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point used by uv scripts and python -m yrep_gui."""
    app = create_application(argv)
    dialog = WelcomeDialog(TEMPLATES, TEMPLATE_CATALOG)
    selected_key, accepted = dialog.exec_and_get_template()
    if not accepted:
        return 0
    window = MainWindow()
    if selected_key is not None:
        payload = TEMPLATES.get(selected_key)
        if payload is not None:
            try:
                window.load_graph_payload(payload, label=selected_key)
            except Exception as exc:  # pragma: no cover - defensive path
                window.statusBar().showMessage(f"Failed to load template: {exc}", 5000)
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
