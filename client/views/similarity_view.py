"""Similarity widget — Phy-style ranked similar-cluster table."""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)

class SimilarityWidget(QWidget):
    """Render a ranked table of similar clusters."""

    # Emitted when the user clicks a row — carries the cluster id
    cluster_add_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cluster_info: dict[int, dict] = {}
        self._similarity_rows: list[dict] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._status = QLabel("Select 2+ clusters to compute similarity")
        self._status.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self._status.setStyleSheet("background:#1e1e1e; color:#aaaaaa; font-size:11px; padding:4px;")
        layout.addWidget(self._status)

        self._table = QTableWidget(0, 4)
        self._table.setHorizontalHeaderLabels(["Cluster", "Similarity", "Spikes", "Label"])
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._table.setCursor(Qt.CursorShape.PointingHandCursor)
        self._table.cellClicked.connect(self._on_cell_clicked)
        self._table.setShowGrid(False)
        self._table.setStyleSheet(
            "QTableWidget{background:#151515;color:#cfcfcf;font-size:11px;border:none;}"
            "QHeaderView::section{background:#202020;color:#b0b0b0;border:none;padding:3px;}"
        )
        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        layout.addWidget(self._table)

    def set_cluster_info(self, clusters: list[dict]) -> None:
        self._cluster_info = {int(c["id"]): c for c in clusters}

    def set_templates(self, data_per_cluster: dict) -> None:
        """Accept {cluster_id: (template_or_waveforms, ch_ids)} or {cluster_id: arr}."""
        if not data_per_cluster or len(data_per_cluster) < 2:
            self._status.setText("Select 2+ clusters to compute similarity")
            self._table.setRowCount(0)
            return
        try:
            primary_id, rows = self._compute_ranked_similarity(data_per_cluster)
            if not rows:
                self._status.setText("Need at least 2 valid templates")
                self._table.setRowCount(0)
                return
            self._status.setText(
                f"Primary cluster: {primary_id}  |  Similar clusters shown: {len(rows)}"
            )
            self._populate_table(rows)
        except Exception as exc:
            logger.exception("Similarity render failed")
            self._status.setText(f"Similarity error: {exc}")
            self._table.setRowCount(0)

    def current_rows(self) -> list[dict]:
        """Return the currently displayed similarity rows."""
        return list(self._similarity_rows)

    def set_similarity_rows(self, primary_cluster_id: int, rows: list[dict]) -> None:
        """Set ranked similar clusters from server-side similarity data."""
        self._similarity_rows = list(rows)
        if primary_cluster_id is None:
            self._status.setText("Select a cluster to compute similarity")
            self._table.setRowCount(0)
            return
        self._status.setText(
            f"Primary cluster: {primary_cluster_id}  |  Similar clusters shown: {len(rows)}"
        )
        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            cid = int(row.get("id", -1))
            sim = float(row.get("similarity", 0.0))
            n_spikes = row.get("n_spikes", "")
            label = row.get("label", "")
            values = [str(cid), f"{sim:.3f}", str(n_spikes), str(label)]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col in (0, 1, 2):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(i, col, item)

    def _compute_ranked_similarity(self, data_per_cluster: dict) -> tuple[int, list[tuple[int, float]]]:
        ids = list(data_per_cluster.keys())
        primary_id = int(ids[0])

        vectors: dict[int, np.ndarray] = {}
        for cid in ids:
            val = data_per_cluster[cid]
            arr = val[0] if isinstance(val, tuple) else val
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 3:
                arr = arr.mean(axis=0)
            if arr.ndim != 2:
                continue
            vec = arr.reshape(-1).astype(np.float32)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-12:
                continue
            vectors[int(cid)] = vec / norm

        if primary_id not in vectors:
            return primary_id, []

        ref = vectors[primary_id]
        out: list[tuple[int, float]] = []
        for cid, vec in vectors.items():
            if cid == primary_id:
                continue
            sim = float(np.clip(np.dot(ref, vec), -1.0, 1.0))
            out.append((cid, sim))
        out.sort(key=lambda x: x[1], reverse=True)
        return primary_id, out

    def _on_cell_clicked(self, row: int, _col: int) -> None:
        item = self._table.item(row, 0)
        if item is None:
            return
        try:
            cid = int(item.text())
        except ValueError:
            return
        self.cluster_add_requested.emit(cid)

    def _populate_table(self, rows: list[tuple[int, float]]) -> None:
        self._table.setRowCount(len(rows))
        for i, (cid, sim) in enumerate(rows):
            info = self._cluster_info.get(cid, {})
            n_spikes = info.get("n_spikes", "")
            label = info.get("label", "")
            values = [str(cid), f"{sim:.3f}", str(n_spikes), str(label)]
            for col, text in enumerate(values):
                item = QTableWidgetItem(text)
                if col in (0, 1, 2):
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self._table.setItem(i, col, item)
