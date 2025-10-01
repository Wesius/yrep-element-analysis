"""Application bootstrap for the YREP GUI."""

from __future__ import annotations

import sys
from typing import Sequence

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from yrep_gui.ui.main_window import MainWindow


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
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
