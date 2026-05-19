"""Console view — lightweight event log panel."""
from __future__ import annotations

from datetime import datetime

from PyQt6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class ConsoleWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setStyleSheet(
            "background:#0f0f0f;color:#b5b5b5;font-family:Menlo, monospace;font-size:11px;"
        )
        layout.addWidget(self._text)
        self.log("phy-remote console ready")

    def log(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._text.append(f"[{ts}] {message}")
