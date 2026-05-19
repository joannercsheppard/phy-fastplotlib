"""Probe view — channel geometry with selected cluster home channels."""
from __future__ import annotations

import io
import logging

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class ProbeWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._positions: np.ndarray | None = None
        self._selected_channels: list[tuple[list[int], tuple]] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel("Probe view")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._label.setStyleSheet("background:#1e1e1e; color:#aaaaaa; font-size:12px;")
        layout.addWidget(self._label)

    def set_channel_positions(self, positions: np.ndarray) -> None:
        self._positions = np.asarray(positions, dtype=np.float32)
        self._redraw()

    def set_selected_channels(self, selected_channels: list[tuple[list[int], tuple]]) -> None:
        self._selected_channels = list(selected_channels)
        self._redraw()

    def _redraw(self) -> None:
        if self._positions is None:
            self._label.setText("No channel positions")
            return
        try:
            pm = self._render()
            self._label.setPixmap(pm.scaled(
                self._label.width(), self._label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation))
        except Exception as exc:
            logger.exception("Probe render failed")
            self._label.setText(f"Probe error:\n{exc}")

    def _render(self) -> QPixmap:
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        pos = self._positions
        fig = Figure(figsize=(2.8, 3.2), facecolor="#1e1e1e")
        fig.subplots_adjust(left=0.12, right=0.96, top=0.92, bottom=0.08)
        ax = fig.add_subplot(1, 1, 1)
        ax.set_facecolor("#2a2a2a")
        ax.set_title("Probe", color="#ccc", fontsize=9, pad=4)
        ax.tick_params(colors="#888", labelsize=7)
        for spine in ax.spines.values():
            spine.set_color("#555")
        ax.scatter(pos[:, 0], pos[:, 1], s=8, color="#7b7b7b", alpha=0.7, linewidths=0)
        for channels, color in self._selected_channels:
            rgba = color if len(color) == 4 else (*color, 1.0)
            for k, ch in enumerate(channels):
                if 0 <= ch < len(pos):
                    size = 46 if k == 0 else 24
                    alpha = rgba[3] if len(rgba) == 4 else 1.0
                    c = (rgba[0], rgba[1], rgba[2], alpha if k == 0 else alpha * 0.65)
                    ax.scatter([pos[ch, 0]], [pos[ch, 1]], s=size, color=[c], edgecolor="white", linewidths=0.35)

        buf = io.BytesIO()
        FigureCanvasAgg(fig).print_png(buf)
        import matplotlib.pyplot as plt
        plt.close(fig)
        buf.seek(0)
        return QPixmap.fromImage(QImage.fromData(buf.read()))
