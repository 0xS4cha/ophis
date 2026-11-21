from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ophis.gui.main_window import MainWindow


def run(dataset_root: Path | None = None) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ophis")
    window = MainWindow(dataset_root)
    window.show()
    return app.exec()
