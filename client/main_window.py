"""
MainWindow — PyQt6 main window for phy-remote.

Layout
------
+--------------------------------------------------+
|  Menu bar                                        |
+--------------------+-----------------------------+
|  Cluster table     |  Waveform view (central)    |
|  (left dock)       |                             |
|  id | label | fr   |                             |
|  id | label | fr   |                             |
+--------------------+-----------------------------+
|  Status bar   [label buttons: g  m  n  u]        |
+--------------------------------------------------+

Keyboard shortcuts:
  g            →  merge selected clusters
  Alt+g/m/n/u  →  label selected clusters good/mua/noise/unsorted
  Ctrl+g/m/n/u →  label similar clusters good/mua/noise/unsorted
  Ctrl+Alt+g/m/n/u → label ALL clusters good/mua/noise/unsorted
  Space        →  select next cluster
  Shift+Space  →  select previous cluster
  Up/Down      →  move selection up/down
  Home/End     →  first/last cluster
  Ctrl+Z       →  undo last merge
  Ctrl+Y / Ctrl+Shift+Z → redo
  Ctrl+S       →  save to disk
  Ctrl+Q       →  quit
  Ctrl+Up/Down →  increase/decrease waveform amplitude
"""

from __future__ import annotations

import logging
import threading
from collections import OrderedDict
from typing import Sequence

import numpy as np

from PyQt6.QtCore import (
    Qt, QObject,
    QAbstractTableModel, QModelIndex, QItemSelectionModel,
    pyqtSignal, pyqtSlot, QTimer,
)
from PyQt6.QtGui import QColor, QKeySequence, QShortcut, QFont
from PyQt6.QtWidgets import (
    QMainWindow, QDockWidget, QSplitter, QWidget, QVBoxLayout, QHBoxLayout,
    QTableView, QHeaderView, QStatusBar, QLabel, QPushButton,
    QAbstractItemView, QSizePolicy, QComboBox, QToolBar,
)

from phy_remote.client.transport import PhyTransport, TransportError, ServerLoadingError
from phy_remote.client.views.waveform_view import WaveformWidget
from phy_remote.client.views.isi_view import ISIWidget
from phy_remote.client.views.amplitude_view import AmplitudeWidget
from phy_remote.client.views.similarity_view import SimilarityWidget
from phy_remote.client.views.template_features_view import TemplateFeaturesWidget
from phy_remote.client.views.feature_cloud_view import FeatureViewWidget
from phy_remote.client.views.probe_view import ProbeWidget
from phy_remote.client.views.raster_view import RasterWidget
from phy_remote.client.views.correlogram_view import CorrelogramWidget
from phy_remote.client.views.console_view import ConsoleWidget
from phy_remote.client.views.trace_view import TraceWidget

logger = logging.getLogger(__name__)
_WAVEFORM_SPIKE_CAP = 100

# ---------------------------------------------------------------------------
# Label colours  (match phy conventions)
# ---------------------------------------------------------------------------

LABEL_COLORS: dict[str, QColor] = {
    "good":      QColor(100, 200, 100),   # green
    "mua":       QColor(220, 160,  60),   # orange
    "noise":     QColor(160,  60,  60),   # red
    "unsorted":  QColor(180, 180, 180),   # grey
}
TEXT_COLOR = QColor(230, 230, 230)


# ---------------------------------------------------------------------------
# Cluster table model
# ---------------------------------------------------------------------------

_COLUMNS = ["ID", "Label", "KS", "SUA", "SUA conf", "Noise", "Noise conf", "Spikes", "Ampl (µV)", "FR (Hz)"]
_COL_ID, _COL_LABEL, _COL_KS, _COL_SUA, _COL_SUA_CONF, _COL_NOISE, _COL_NOISE_CONF, _COL_SPIKES, _COL_AMPL, _COL_FR = range(10)


class ClusterTableModel(QAbstractTableModel):
    """Qt model backed by a list of cluster-info dicts."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._rows: list[dict] = []

    def load(self, clusters: list[dict]) -> None:
        self.beginResetModel()
        self._rows = clusters
        self.endResetModel()

    def update_label(self, cluster_id: int, label: str) -> None:
        for i, row in enumerate(self._rows):
            if row["id"] == cluster_id:
                row["label"] = label
                idx = self.index(i, _COL_LABEL)
                self.dataChanged.emit(idx, idx, [Qt.ItemDataRole.DisplayRole,
                                                  Qt.ItemDataRole.BackgroundRole])
                return

    def rowCount(self, parent=QModelIndex()) -> int:
        return len(self._rows)

    def columnCount(self, parent=QModelIndex()) -> int:
        return len(_COLUMNS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return _COLUMNS[section]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        row = self._rows[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.DisplayRole:
            if col == _COL_ID:         return str(row["id"])
            if col == _COL_LABEL:      return row["label"]
            if col == _COL_KS:         return row.get("ks_label", "")
            if col == _COL_SUA:        return row.get("sua_pred", "")
            if col == _COL_SUA_CONF:
                v = row.get("sua_conf", 0.0)
                return f"{v:.2f}" if v else ""
            if col == _COL_NOISE:      return row.get("noise_pred", "")
            if col == _COL_NOISE_CONF:
                v = row.get("noise_conf", 0.0)
                return f"{v:.2f}" if v else ""
            if col == _COL_SPIKES:     return str(row["n_spikes"])
            if col == _COL_AMPL:       return f'{row["amplitude"]:.1f}'
            if col == _COL_FR:         return f'{row["fr"]:.2f}'

        if role == Qt.ItemDataRole.BackgroundRole:
            if col == _COL_LABEL:
                return LABEL_COLORS.get(row["label"], LABEL_COLORS["unsorted"])
            for pred_col, key in ((_COL_KS, "ks_label"), (_COL_SUA, "sua_pred"), (_COL_NOISE, "noise_pred")):
                if col == pred_col:
                    val = row.get(key, "")
                    if val in LABEL_COLORS:
                        return LABEL_COLORS[val]

        if role == Qt.ItemDataRole.ForegroundRole and col in (_COL_LABEL, _COL_KS, _COL_SUA, _COL_NOISE):
            return TEXT_COLOR

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (_COL_LABEL, _COL_KS, _COL_SUA, _COL_NOISE):
                return Qt.AlignmentFlag.AlignCenter
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter

        if role == Qt.ItemDataRole.UserRole:
            return row  # entire dict for sorting

        return None

    def sort(self, column: int, order=Qt.SortOrder.AscendingOrder) -> None:
        self.layoutAboutToBeChanged.emit()
        reverse = (order == Qt.SortOrder.DescendingOrder)
        key = {
            _COL_ID:         lambda r: r["id"],
            _COL_LABEL:      lambda r: r["label"],
            _COL_KS:         lambda r: r.get("ks_label", ""),
            _COL_SUA:        lambda r: r.get("sua_pred", ""),
            _COL_SUA_CONF:   lambda r: r.get("sua_conf", 0.0),
            _COL_NOISE:      lambda r: r.get("noise_pred", ""),
            _COL_NOISE_CONF: lambda r: r.get("noise_conf", 0.0),
            _COL_SPIKES:     lambda r: r["n_spikes"],
            _COL_AMPL:       lambda r: r["amplitude"],
            _COL_FR:         lambda r: r["fr"],
        }[column]
        self._rows.sort(key=key, reverse=reverse)
        self.layoutChanged.emit()

    def cluster_id_at_row(self, row: int) -> int:
        return self._rows[row]["id"]


# ---------------------------------------------------------------------------
# Cluster table dock
# ---------------------------------------------------------------------------

class _ClusterTableDock(QDockWidget):
    clusters_selected = pyqtSignal(list)   # list[int]

    def __init__(self, parent=None):
        super().__init__("Clusters", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self._primary_cluster_id: int | None = None

        self._model = ClusterTableModel()
        self._view = QTableView()
        self._view.setModel(self._model)
        self._view.setSortingEnabled(True)
        self._view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._view.horizontalHeader().setSectionResizeMode(
            _COL_LABEL, QHeaderView.ResizeMode.Stretch
        )
        self._view.horizontalHeader().setSectionResizeMode(
            _COL_ID, QHeaderView.ResizeMode.ResizeToContents
        )
        self._view.setAlternatingRowColors(True)
        self._view.verticalHeader().setVisible(False)
        self._view.setShowGrid(False)
        mono = QFont("Menlo", 11)
        self._view.setFont(mono)

        self._view.selectionModel().selectionChanged.connect(self._on_selection_changed)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)
        self.setWidget(container)

    def populate(self, clusters: list[dict]) -> None:
        self._model.load(clusters)
        self._view.sortByColumn(_COL_ID, Qt.SortOrder.AscendingOrder)

    def update_label(self, cluster_id: int, label: str) -> None:
        self._model.update_label(cluster_id, label)

    def selected_cluster_ids(self) -> list[int]:
        rows = {idx.row() for idx in self._view.selectionModel().selectedRows()}
        return [self._model.cluster_id_at_row(r) for r in sorted(rows)]

    def all_cluster_ids(self) -> list[int]:
        return [self._model.cluster_id_at_row(r) for r in range(self._model.rowCount())]

    def select_cluster_by_id(self, cluster_id: int) -> None:
        """Select the row whose cluster id matches *cluster_id*, if present."""
        for row in range(self._model.rowCount()):
            if self._model.cluster_id_at_row(row) == cluster_id:
                self._jump_to_row(row)
                return

    def add_cluster_to_selection(self, cluster_id: int) -> None:
        """Add cluster_id to the current selection without clearing existing rows."""
        for row in range(self._model.rowCount()):
            if self._model.cluster_id_at_row(row) == cluster_id:
                idx = self._model.index(row, 0)
                self._view.selectionModel().select(
                    idx,
                    QItemSelectionModel.SelectionFlag.Select |
                    QItemSelectionModel.SelectionFlag.Rows,
                )
                self._view.scrollTo(idx)
                return

    def adjacent_cluster_ids(self, cluster_ids: list[int], n: int = 3) -> list[int]:
        """Return up to n cluster ids before and n after the current selection."""
        rows = sorted({idx.row() for idx in self._view.selectionModel().selectedRows()})
        if not rows:
            return []
        total = self._model.rowCount()
        result = []
        for delta in range(1, n + 1):
            if rows[0] - delta >= 0:
                result.append(self._model.cluster_id_at_row(rows[0] - delta))
            if rows[-1] + delta < total:
                result.append(self._model.cluster_id_at_row(rows[-1] + delta))
        return result

    def move_selection(self, delta: int) -> None:
        """Move selection by delta rows (negative = up, positive = down)."""
        current = self._view.selectionModel().currentIndex()
        if not current.isValid():
            new_row = 0 if delta > 0 else self._model.rowCount() - 1
        else:
            new_row = max(0, min(self._model.rowCount() - 1, current.row() + delta))
        self._jump_to_row(new_row)

    def jump_to_row(self, row: int) -> None:
        """Jump to absolute row; row=-1 means last row."""
        n = self._model.rowCount()
        if n == 0:
            return
        if row < 0:
            row = n + row
        row = max(0, min(n - 1, row))
        self._jump_to_row(row)

    def _jump_to_row(self, row: int) -> None:
        new_index = self._model.index(row, 0)
        self._primary_cluster_id = self._model.cluster_id_at_row(row)
        self._view.selectionModel().setCurrentIndex(
            new_index,
            QItemSelectionModel.SelectionFlag.ClearAndSelect |
            QItemSelectionModel.SelectionFlag.Rows,
        )
        self._view.scrollTo(new_index)

    def _on_selection_changed(self, *_) -> None:
        ids = self.selected_cluster_ids()
        if not ids:
            return
        # A single-row selection always means the user clicked one cluster directly
        # (mouse click or keyboard nav) — that cluster becomes the new primary.
        if len(ids) == 1:
            self._primary_cluster_id = ids[0]
        elif self._primary_cluster_id in ids and ids[0] != self._primary_cluster_id:
            ids.remove(self._primary_cluster_id)
            ids.insert(0, self._primary_cluster_id)
        self.clusters_selected.emit(ids)


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------

class _LRUCache:
    """Simple LRU cache keyed by cluster id."""

    def __init__(self, maxsize: int = 20):
        self._data: OrderedDict = OrderedDict()
        self._maxsize = maxsize

    def get(self, key):
        if key not in self._data:
            return None
        self._data.move_to_end(key)
        return self._data[key]

    def put(self, key, value) -> None:
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self._maxsize:
            self._data.popitem(last=False)

    def __contains__(self, key) -> bool:
        return key in self._data

    def clear(self) -> None:
        self._data.clear()


# ---------------------------------------------------------------------------
# Async fetch workers
# ---------------------------------------------------------------------------

class _FetchWorker(QObject):
    """
    Fetches waveforms in two stages using a plain Python thread.
    PyQt6 queues cross-thread signal emissions to the main event loop
    automatically, so no QThread / moveToThread needed.
    """
    template_ready           = pyqtSignal(dict)   # {cid: (n_samples, n_channels)}
    features_ready           = pyqtSignal(dict)   # {cid: features_array}
    template_features_ready  = pyqtSignal(dict)   # {cid: template_features_array}
    spike_data_ready         = pyqtSignal(dict)   # {cid: (n_spikes, 2) [time, template_amp]}
    spike_times_ready        = pyqtSignal(dict)   # {cid: (n_spikes,) float64} full spike times for ISI
    feat_spike_data_ready    = pyqtSignal(dict)   # {cid: (n_spikes, 2) [time, PC0_feat]}
    background_features_ready = pyqtSignal(object) # (n_spikes, n_ch, n_pc) float32 or None
    finished                 = pyqtSignal(dict)   # {cid: (n_spikes, n_samples, n_channels)}
    error                    = pyqtSignal(str)

    def __init__(
        self,
        transport: PhyTransport,
        cluster_ids: list[int],
        n_spikes: int,
        skip_templates: bool = False,
    ):
        super().__init__()
        # Store connection params — create a private transport in run() so this
        # worker's ZMQ socket never races with the main-thread transport.
        self._host = transport.host
        self._port = transport.port
        self.cluster_ids = cluster_ids
        self.n_spikes = n_spikes
        self.skip_templates = skip_templates
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        transport = PhyTransport(host=self._host, port=self._port)
        try:
            self._run(transport)
        finally:
            transport.close()

    def _run(self, transport: PhyTransport):
        # Fetch order chosen to maximise progressive visual feedback:
        #   templates   → waveform view shows template shape immediately
        #   spike_data  → ISI + amplitude views update
        #   features    → feature cloud updates (before the slow waveform read)
        #   waveforms   → real waveform traces (may be slow on first fetch)
        #   feat_spike  → feature amplitude view
        #   bg_features → grey backdrop in feature cloud
        #   tmpl_feats  → template features view
        #
        # Putting features before waveforms is important: if the user navigates
        # away during the slow waveform fetch the worker is cancelled and the
        # feature view would otherwise stay empty.

        if not self.skip_templates:
            templates = {}
            for cid in self.cluster_ids:
                if self._cancelled:
                    return
                try:
                    hdr, tmpl = transport.get_templates(cid)
                    ch_ids  = hdr.get("channel_ids", list(range(tmpl.shape[-1])))
                    best_ch = hdr.get("best_ch", ch_ids[0] if ch_ids else None)
                    templates[cid] = (tmpl, ch_ids, best_ch)
                except Exception as exc:
                    self.error.emit(f"Template fetch failed for cluster {cid}: {exc}")
                    return
            if not self._cancelled:
                self.template_ready.emit(templates)

        # Spike data (fast — mmap reads)
        spike_data = {}
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                spike_data[cid] = transport.get_spike_data(cid)
            except Exception as exc:
                self.error.emit(f"Spike data fetch failed for cluster {cid}: {exc}")
                return
        if not self._cancelled:
            self.spike_data_ready.emit(spike_data)

        # Full spike times for ISI — must use all spikes, not the subsampled spike_data
        full_spike_times = {}
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                _, times = transport.get_spike_times(cid)
                full_spike_times[cid] = times
            except Exception as exc:
                logger.debug("Full spike times fetch skipped for cluster %d: %s", cid, exc)
        if not self._cancelled and full_spike_times:
            self.spike_times_ready.emit(full_spike_times)

        # PC features — before waveforms so feature view is never blocked by waveform I/O
        features = {}
        bg_channel_ids: list[int] = []
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                hdr, f = transport.get_features(cid)
                features[cid] = f
                if not bg_channel_ids:
                    bg_channel_ids = hdr.get("channel_ids", [])
            except Exception as exc:
                logger.debug("Feature fetch skipped for cluster %d: %s", cid, exc)
        if not self._cancelled and features:
            self.features_ready.emit(features)

        # Real waveforms (may be slow on first fetch if not pre-extracted)
        waveforms = {}
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                hdr, w = transport.get_waveforms(cid, self.n_spikes)
                ch_ids = hdr.get("channel_ids", list(range(w.shape[-1])))
                waveforms[cid] = (w, ch_ids)
            except Exception as exc:
                self.error.emit(f"Waveform fetch failed for cluster {cid}: {exc}")
                return
        if not self._cancelled:
            self.finished.emit(waveforms)

        # Feature amplitude (PC-0 on best channel)
        feat_spike_data = {}
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                result = transport.get_feature_spike_data(cid)
                if result is not None:
                    feat_spike_data[cid] = result
            except Exception as exc:
                logger.debug("Feature spike data skipped for cluster %d: %s", cid, exc)
        if not self._cancelled and feat_spike_data:
            self.feat_spike_data_ready.emit(feat_spike_data)

        # Background features (grey scatter in feature cloud)
        if not self._cancelled and bg_channel_ids:
            try:
                bg_feat = transport.get_background_features(bg_channel_ids, n_spikes=500)
                if not self._cancelled:
                    self.background_features_ready.emit(bg_feat)
            except Exception as exc:
                logger.debug("Background features fetch skipped: %s", exc)

        # Template features
        template_features = {}
        for cid in self.cluster_ids:
            if self._cancelled:
                return
            try:
                result = transport.get_template_features(cid)
                if result is not None:
                    _, tf = result
                    template_features[cid] = tf
            except Exception as exc:
                logger.debug("Template features fetch failed for cluster %d: %s", cid, exc)
        if not self._cancelled and template_features:
            self.template_features_ready.emit(template_features)


class _PrefetchWorker(QObject):
    """
    Background worker that silently pre-warms the template cache for
    clusters adjacent to the current selection.  Creates its own
    transport so it never shares a ZMQ socket with _FetchWorker.
    """
    template_ready = pyqtSignal(int, object)   # (cluster_id, ndarray)
    finished       = pyqtSignal()

    def __init__(self, host: str, port: int, cluster_ids: list[int]):
        super().__init__()
        self._host = host
        self._port = port
        self.cluster_ids = list(cluster_ids)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self):
        try:
            transport = PhyTransport(host=self._host, port=self._port)
        except Exception as exc:
            logger.debug("Prefetch transport failed: %s", exc)
            self.finished.emit()
            return
        try:
            for cid in self.cluster_ids:
                if self._cancelled:
                    break
                try:
                    _, tmpl = transport.get_templates(cid)
                    if not self._cancelled:
                        self.template_ready.emit(cid, tmpl)
                except Exception as exc:
                    logger.debug("Prefetch skipped cluster %d: %s", cid, exc)
        finally:
            transport.close()
            self.finished.emit()



# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """
    Parameters
    ----------
    transport : PhyTransport
    n_spikes  : int   max spikes to fetch per cluster for display
    """
    _best_channels_loaded  = pyqtSignal(dict)   # {cluster_id: best_ch}
    _amp_bg_ready          = pyqtSignal(object) # np.ndarray (n,2) background spike data
    _raster_ready          = pyqtSignal(object) # dict: {data, cluster_ids, duration}
    _similarity_ready      = pyqtSignal(int, object)  # (cluster_id, rows list)

    def __init__(self, transport: PhyTransport, n_spikes: int = 100):
        super().__init__()
        self.transport = transport
        self.n_spikes = max(1, min(int(n_spikes), _WAVEFORM_SPIKE_CAP))
        self._fetch_worker:    _FetchWorker    | None = None
        self._prefetch_worker: _PrefetchWorker | None = None
        self._pending_cluster_ids: list[int] = []
        self._selection_debounce = QTimer(self)
        self._selection_debounce.setSingleShot(True)
        self._selection_debounce.setInterval(80)
        self._selection_debounce.timeout.connect(self._apply_cluster_selection)

        # Caches: templates are cheap (fast to fetch, ~KB each),
        # waveforms are expensive (slow to fetch, ~MB each).
        self._template_cache  = _LRUCache(maxsize=30)
        self._waveform_cache  = _LRUCache(maxsize=10)
        self._similarity_cache: dict[int, list] = {}  # cluster_id → similarity rows
        self._current_cluster_ids: list[int] = []
        # home channel for each cluster (populated from template headers + bulk load at startup)
        self._home_channels: dict[int, int] = {}
        self._template_channels: dict[int, list[int]] = {}
        # best channel for ALL clusters — loaded once at startup for trace-view tick overlay
        self._all_best_channels: dict[int, int] = {}
        # spike times for each cluster (populated from spike_data)
        self._spike_times: dict[int, np.ndarray] = {}
        self._spike_data: dict[int, np.ndarray] = {}      # (n_spikes,2) template amplitude
        self._feat_spike_data: dict[int, np.ndarray] = {} # (n_spikes,2) PC-0 feature amplitude
        self._features: dict[int, np.ndarray] = {}
        self._template_features: dict[int, np.ndarray] = {}

        self.setWindowTitle("phy-remote")
        self.resize(1700, 1150)
        self.setDockNestingEnabled(True)

        # Central widget: two waveform panels side-by-side (top) + amplitude (bottom)
        self._waveform_view_0 = WaveformWidget(max_spikes=self.n_spikes, cluster_slot=0, parent=self)
        self._waveform_view_1 = WaveformWidget(max_spikes=self.n_spikes, cluster_slot=1, parent=self)
        self._waveform_view_0.render_debug.connect(self._console_view_log)
        self._waveform_view_1.render_debug.connect(self._console_view_log)
        _wf_splitter = QSplitter(Qt.Orientation.Horizontal)
        _wf_splitter.addWidget(self._waveform_view_0)
        _wf_splitter.addWidget(self._waveform_view_1)
        # Two amplitude views side-by-side: template (left) and feature PC-0 (right)
        self._amp_view      = AmplitudeWidget(title="Template amplitude", parent=self)
        self._feat_amp_view = AmplitudeWidget(title="Feature amplitude (PC0)", parent=self)
        _amp_splitter = QSplitter(Qt.Orientation.Horizontal)
        _amp_splitter.addWidget(self._amp_view)
        _amp_splitter.addWidget(self._feat_amp_view)
        _central = QSplitter(Qt.Orientation.Vertical)
        _central.addWidget(_wf_splitter)
        _central.addWidget(_amp_splitter)
        _central.setSizes([600, 280])
        self.setCentralWidget(_central)

        # Left dock — cluster table
        self._cluster_dock = _ClusterTableDock(self)
        self._cluster_dock.clusters_selected.connect(self._on_clusters_selected)
        self._best_channels_loaded.connect(self._on_best_channels_loaded)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._cluster_dock)
        self._cluster_dock.setMinimumWidth(320)

        self._similarity_dock = QDockWidget("Similarity", self)
        self._similarity_view = SimilarityWidget(self)
        self._similarity_view.cluster_add_requested.connect(
            self._set_comparison_cluster
        )
        self._similarity_dock.setWidget(self._similarity_view)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._similarity_dock)

        # Right dock — feature cloud (large, always visible) + template features below
        self._feature_cloud_dock = QDockWidget("Feature view", self)
        self._feature_cloud_view = FeatureViewWidget(self)
        self._feature_cloud_dock.setWidget(self._feature_cloud_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._feature_cloud_dock)

        self._template_feat_dock = QDockWidget("Template features", self)
        self._template_feat_view = TemplateFeaturesWidget(self)
        self._template_feat_dock.setWidget(self._template_feat_view)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._template_feat_dock)

        # Trace dock
        _TRACE_ENABLED = True
        self._trace_dock = QDockWidget("Traces", self)
        if _TRACE_ENABLED:
            self._trace_view = TraceWidget(
                host=self.transport.host,
                port=self.transport.port,
                parent=self,
            )
            self._trace_dock.setWidget(self._trace_view)
        else:
            self._trace_view = None
            self._trace_dock.setWidget(QWidget(self))
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._trace_dock)

        # Bottom docks — ISI, Raster, Correlogram, Console
        self._isi_dock = QDockWidget("ISI", self)
        self._isi_view = ISIWidget(self)
        self._isi_dock.setWidget(self._isi_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._isi_dock)

        self._raster_dock = QDockWidget("Raster", self)
        self._raster_view = RasterWidget(self)
        self._raster_dock.setWidget(self._raster_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._raster_dock)

        self._correlogram_dock = QDockWidget("Correlogram", self)
        self._correlogram_view = CorrelogramWidget(parent=self)
        self._correlogram_dock.setWidget(self._correlogram_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._correlogram_dock)

        self._probe_dock = QDockWidget("Probe", self)
        self._probe_view = ProbeWidget(self)
        self._probe_dock.setWidget(self._probe_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._probe_dock)

        self._console_dock = QDockWidget("Console", self)
        self._console_view = ConsoleWidget(self)
        self._console_dock.setWidget(self._console_view)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._console_dock)

        # Layout splits
        self.splitDockWidget(self._cluster_dock, self._similarity_dock, Qt.Orientation.Vertical)
        self.splitDockWidget(self._feature_cloud_dock, self._template_feat_dock, Qt.Orientation.Vertical)
        # Trace dock on its own row; ISI etc. stacked below it
        self.splitDockWidget(self._trace_dock, self._isi_dock, Qt.Orientation.Vertical)
        self.splitDockWidget(self._isi_dock, self._raster_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._raster_dock, self._correlogram_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._correlogram_dock, self._probe_dock, Qt.Orientation.Horizontal)
        self.splitDockWidget(self._probe_dock, self._console_dock, Qt.Orientation.Horizontal)

        self.resizeDocks(
            [self._cluster_dock, self._similarity_dock],
            [620, 280],
            Qt.Orientation.Vertical,
        )
        self.resizeDocks(
            [self._feature_cloud_dock, self._template_feat_dock],
            [580, 240],
            Qt.Orientation.Vertical,
        )
        # Left dock wider, right column wider than central widget
        self.resizeDocks(
            [self._cluster_dock],
            [520],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self._feature_cloud_dock],
            [900],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self._isi_dock, self._raster_dock, self._correlogram_dock,
             self._probe_dock, self._console_dock],
            [220, 220, 220, 220, 220],
            Qt.Orientation.Horizontal,
        )
        self.resizeDocks(
            [self._trace_dock, self._isi_dock],
            [380, 320],
            Qt.Orientation.Vertical,
        )

        # Status bar + label buttons
        self._status_label = QLabel("Connecting…")
        status_bar = QStatusBar()
        status_bar.addWidget(self._status_label, stretch=1)
        status_bar.addPermanentWidget(self._make_label_buttons())

        # Dataset / shank switcher (hidden until server reports >1 dataset)
        self._dataset_combo = QComboBox()
        self._dataset_combo.setMinimumWidth(120)
        self._dataset_combo.setToolTip("Switch shank / dataset")
        self._dataset_combo.setVisible(False)
        self._dataset_combo.currentTextChanged.connect(self._on_dataset_selected)
        status_bar.addPermanentWidget(self._dataset_combo)

        self.setStatusBar(status_bar)

        # Menu
        view_menu = self.menuBar().addMenu("View")
        view_menu.addAction(self._cluster_dock.toggleViewAction())
        view_menu.addAction(self._similarity_dock.toggleViewAction())
        view_menu.addAction(self._feature_cloud_dock.toggleViewAction())
        view_menu.addAction(self._template_feat_dock.toggleViewAction())
        view_menu.addAction(self._trace_dock.toggleViewAction())
        view_menu.addAction(self._isi_dock.toggleViewAction())
        view_menu.addAction(self._raster_dock.toggleViewAction())
        view_menu.addAction(self._correlogram_dock.toggleViewAction())
        view_menu.addAction(self._probe_dock.toggleViewAction())
        view_menu.addAction(self._console_dock.toggleViewAction())

        # ------------------------------------------------------------------
        # Keyboard shortcuts
        # ------------------------------------------------------------------

        # g = smart: label good (1 cluster selected) OR merge (2+ selected)
        self._add_shortcut("g", self._smart_g)

        # Label selected: bare letters (m/n/u) — same feel as original phy-remote
        self._add_shortcut("m", lambda: self._apply_label("mua"))
        self._add_shortcut("n", lambda: self._apply_label("noise"))
        self._add_shortcut("u", lambda: self._apply_label("unsorted"))

        # Label similar clusters: Ctrl+letter
        self._add_shortcut("Ctrl+G", lambda: self._label_similar("good"))
        self._add_shortcut("Ctrl+M", lambda: self._label_similar("mua"))
        self._add_shortcut("Ctrl+N", lambda: self._label_similar("noise"))
        self._add_shortcut("Ctrl+U", lambda: self._label_similar("unsorted"))

        # Label all clusters: Ctrl+Alt+letter
        self._add_shortcut("Ctrl+Alt+G", lambda: self._label_all("good"))
        self._add_shortcut("Ctrl+Alt+M", lambda: self._label_all("mua"))
        self._add_shortcut("Ctrl+Alt+N", lambda: self._label_all("noise"))
        self._add_shortcut("Ctrl+Alt+U", lambda: self._label_all("unsorted"))

        # Navigation
        self._add_shortcut("Space",       lambda: self._navigate_cluster(+1))
        self._add_shortcut("Shift+Space", lambda: self._navigate_cluster(-1))
        self._add_shortcut("Down",        lambda: self._navigate_cluster(+1))
        self._add_shortcut("Up",          lambda: self._navigate_cluster(-1))
        self._add_shortcut("Home",        lambda: self._navigate_cluster_abs(0))
        self._add_shortcut("End",         lambda: self._navigate_cluster_abs(-1))

        # Undo / Redo
        self._add_shortcut("Ctrl+Z",       self._undo)
        self._add_shortcut("Ctrl+Shift+Z", self._redo)
        self._add_shortcut("Ctrl+Y",       self._redo)

        # Save / Quit
        self._add_shortcut("Ctrl+S", self._save)
        self._add_shortcut("Ctrl+Q", self.close)

        # Waveform amplitude zoom (Ctrl+Up/Down; scroll zoom handled in WaveformWidget)
        self._add_shortcut("Ctrl+Up",   lambda: self._waveform_amp_zoom(1.2))
        self._add_shortcut("Ctrl+Down", lambda: self._waveform_amp_zoom(1.0 / 1.2))

        # Cross-view linking: trace window position ↔ both amplitude time bars + Alt+click
        if self._trace_view is not None:
            self._trace_view.time_range_changed.connect(self._amp_view.show_time_range)
            self._trace_view.time_range_changed.connect(self._feat_amp_view.show_time_range)
            self._amp_view.time_clicked.connect(self._trace_view.go_to_time)
            self._feat_amp_view.time_clicked.connect(self._trace_view.go_to_time)

        # Background spike data → amplitude grey scatter
        self._amp_bg_ready.connect(self._amp_view.set_background_spike_data)
        self._amp_bg_seq = 0   # used to discard stale background fetches

        # Similarity: fetched in background, cached per cluster
        self._similarity_ready.connect(self._on_similarity_ready)

        # Raster: global data fetched once at startup
        self._raster_ready.connect(self._on_raster_ready)

        # Load channel positions then cluster table (dataset list loaded after success)
        self._load_channel_positions()
        self._load_cluster_info()

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _make_label_buttons(self) -> QWidget:
        w = QWidget()
        layout = QHBoxLayout(w)
        layout.setContentsMargins(4, 0, 4, 0)
        layout.setSpacing(4)
        for label, shortcut in [("Good [g]", "good"), ("MUA [m]", "mua"),
                                  ("Noise [n]", "noise"), ("Unsorted [u]", "unsorted")]:
            btn = QPushButton(label)
            btn.setFixedHeight(22)
            color = LABEL_COLORS[shortcut].name()
            btn.setStyleSheet(
                f"QPushButton {{ background: {color}; color: white; "
                f"border: none; border-radius: 3px; padding: 0 6px; }}"
                f"QPushButton:hover {{ opacity: 0.8; }}"
            )
            btn.clicked.connect(lambda checked, lbl=shortcut: self._apply_label(lbl))
            layout.addWidget(btn)

        # Separator + Save button
        sep = QWidget()
        sep.setFixedWidth(8)
        layout.addWidget(sep)
        save_btn = QPushButton("Save [Ctrl+S]")
        save_btn.setFixedHeight(22)
        save_btn.setStyleSheet(
            "QPushButton { background: #444; color: white; border: none; "
            "border-radius: 3px; padding: 0 8px; }"
            "QPushButton:hover { background: #666; }"
        )
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)
        return w

    def _add_shortcut(self, key: str, slot) -> None:
        sc = QShortcut(QKeySequence(key), self)
        sc.activated.connect(slot)

    def _navigate_cluster(self, delta: int) -> None:
        """Move the cluster table selection by delta rows."""
        self._cluster_dock.move_selection(delta)

    def _navigate_cluster_abs(self, row: int) -> None:
        """Jump to an absolute row (0 = first, -1 = last)."""
        self._cluster_dock.jump_to_row(row)

    @staticmethod
    def _fmt_ids(ids: list[int]) -> str:
        s = ", ".join(str(c) for c in ids[:4])
        if len(ids) > 4:
            s += f" (+{len(ids) - 4})"
        return s

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _load_channel_positions(self) -> None:
        try:
            positions = self.transport.get_channel_positions()
            self._waveform_view_0.set_channel_positions(positions)
            self._waveform_view_1.set_channel_positions(positions)
            self._probe_view.set_channel_positions(positions)
            if self._trace_view is not None:
                self._trace_view.set_channel_positions(positions)
            logger.info("Loaded %d channel positions", len(positions))
        except Exception as exc:
            logger.warning("Could not load channel positions: %s", exc)

    def _load_all_best_channels(self) -> None:
        try:
            transport = PhyTransport(host=self.transport.host, port=self.transport.port)
            try:
                best = transport.get_cluster_best_channels()
            finally:
                transport.close()
            self._all_best_channels.update(best)
            self._home_channels.update(best)   # fill gaps before per-cluster template fetches
            logger.info("Loaded best channels for %d clusters", len(best))
            self._best_channels_loaded.emit(best)
        except Exception as exc:
            logger.warning("Could not load cluster best channels: %s", exc)

    @pyqtSlot(dict)
    def _on_best_channels_loaded(self, best: dict) -> None:
        if self._trace_view is not None:
            self._trace_view.set_all_best_channels(best)

    def _load_cluster_info(self, _retries: int = 0) -> None:
        self._set_status("Loading clusters…")
        try:
            clusters = self.transport.get_cluster_info()
            self._cluster_dock.populate(clusters)
            self._similarity_view.set_cluster_info(clusters)
            n_good = sum(1 for c in clusters if c["label"] == "good")
            self._set_status(f"{len(clusters)} clusters  —  {n_good} good")
            self._console_view.log(f"Loaded {len(clusters)} clusters")
        except ServerLoadingError:
            self._set_status(f"Server loading model… ({_retries + 1})")
            delay = min(500 * (2 ** _retries), 3000)  # 500ms → 1s → 2s → 3s
            QTimer.singleShot(delay, lambda: self._load_cluster_info(_retries + 1))
            return
        except Exception as exc:
            logger.error("Could not load cluster info: %s", exc)
            self._set_status(f"Error: {exc}")
            return
        # Fetch raster data and best channels in background — server is confirmed ready now
        threading.Thread(target=self._fetch_raster_data_bg, daemon=True).start()
        threading.Thread(target=self._load_all_best_channels, daemon=True).start()
        self._load_dataset_list()

    def _load_dataset_list(self) -> None:
        """Fetch dataset list from server and populate the shank switcher combo."""
        try:
            datasets, current = self.transport.get_dataset_list()
        except Exception as exc:
            logger.debug("Dataset list not available: %s", exc)
            return
        if len(datasets) <= 1:
            return  # single dataset — no switcher needed

        self._dataset_combo.blockSignals(True)
        self._dataset_combo.clear()
        for d in datasets:
            self._dataset_combo.addItem(d["label"])
        if current:
            idx = self._dataset_combo.findText(current)
            if idx >= 0:
                self._dataset_combo.setCurrentIndex(idx)
        self._dataset_combo.blockSignals(False)
        self._dataset_combo.setVisible(True)

    def _on_dataset_selected(self, label: str) -> None:
        """Called when the user picks a different shank in the combo."""
        if not label:
            return
        self._set_status(f"Switching to {label}…")
        try:
            self.transport.switch_dataset(label)
        except Exception as exc:
            logger.error("Switch dataset failed: %s", exc)
            self._set_status(f"Switch failed: {exc}")
            return
        # Clear all caches
        self._template_cache.clear()
        self._waveform_cache.clear()
        self._similarity_cache.clear()
        self._spike_data.clear()
        self._spike_times.clear()
        self._features.clear()
        self._template_features.clear()
        self._feat_spike_data.clear()
        self._current_cluster_ids = []
        if self._fetch_worker:
            self._fetch_worker.cancel()
            self._fetch_worker = None
        self._waveform_view_0.clear()
        self._waveform_view_1.clear()
        self._load_channel_positions()
        self._load_cluster_info()  # retries automatically on ServerLoadingError

    # ------------------------------------------------------------------
    # Cluster selection → async waveform fetch with cache + prefetch
    # ------------------------------------------------------------------

    def _on_clusters_selected(self, cluster_ids: list[int]) -> None:
        # Debounce rapid table navigation to avoid heavy redraw/refetch on every row step.
        self._pending_cluster_ids = list(cluster_ids)
        self._selection_debounce.start()

    def _apply_cluster_selection(self) -> None:
        cluster_ids = self._pending_cluster_ids
        self._current_cluster_ids = cluster_ids
        self._amp_bg_seq += 1   # cancel any in-flight background fetch
        primary_cluster_id = cluster_ids[0] if cluster_ids else None
        # Reset feature-based panes; they will repopulate from model-backed RPCs.
        self._feature_cloud_view.set_feature_data({})
        self._template_feat_view.set_feature_data({})

        if primary_cluster_id is not None:
            self._console_view.log(f"Selected clusters: {self._fmt_ids(cluster_ids)}")
            # Show cached similarity immediately; fetch in background on cache miss.
            cached = self._similarity_cache.get(primary_cluster_id)
            if cached is not None:
                self._similarity_view.set_similarity_rows(primary_cluster_id, cached)
            else:
                self._fetch_similarity_bg(primary_cluster_id)

        # Cancel any in-flight workers (they check _cancelled before emitting)
        if self._prefetch_worker is not None:
            self._prefetch_worker.cancel()
        if self._fetch_worker is not None:
            self._fetch_worker.cancel()

        # --- Check template cache ---
        cached_templates = {
            cid: self._template_cache.get(cid)
            for cid in cluster_ids
            if cid in self._template_cache
        }

        if len(cached_templates) == len(cluster_ids):
            # All templates in cache → show immediately
            self._update_probe_view()

            # Check if real waveforms are also cached
            cached_waveforms = {
                cid: self._waveform_cache.get(cid)
                for cid in cluster_ids
                if cid in self._waveform_cache
            }
            if len(cached_waveforms) == len(cluster_ids):
                # Dispatch real waveforms (not just templates) so the view shows
                # actual spike traces, not just the template shape.
                self._dispatch_waveforms(cached_waveforms)
                self._set_status(f"Cluster(s) {self._fmt_ids(cluster_ids)}  (cached)")
                self._update_secondary_views_from_cache()
                self._start_prefetch(cluster_ids)
                return

            # Real waveforms not yet cached — show template shape while fetching
            self._dispatch_waveforms(cached_templates)

            # Show templates now; fetch only real waveforms in background
            # But if spike data is already cached, update secondary views immediately
            self._update_secondary_views_from_cache()
            self._set_status(
                f"Cluster(s) {self._fmt_ids(cluster_ids)}  — loading waveforms…"
            )
            self._start_fetch(cluster_ids, skip_templates=True)
            return

        # Not cached → full two-stage fetch
        self._update_probe_view()
        self._update_secondary_views_from_cache()
        self._set_status(f"Fetching: {self._fmt_ids(cluster_ids)}…")
        self._start_fetch(cluster_ids, skip_templates=False)

    def _start_fetch(self, cluster_ids: list[int], skip_templates: bool) -> None:
        worker = _FetchWorker(
            self.transport, cluster_ids, self.n_spikes,
            skip_templates=skip_templates,
        )
        if not skip_templates:
            worker.template_ready.connect(self._on_templates_ready)
        worker.features_ready.connect(self._on_features_ready)
        worker.template_features_ready.connect(self._on_template_features_ready)
        worker.spike_data_ready.connect(self._on_spike_data_ready)
        worker.spike_times_ready.connect(self._on_spike_times_ready)
        worker.feat_spike_data_ready.connect(self._on_feat_spike_data_ready)
        worker.background_features_ready.connect(self._on_background_features_ready)
        worker.finished.connect(self._on_waveforms_ready)
        worker.error.connect(self._on_fetch_error)
        self._fetch_worker = worker
        threading.Thread(target=worker.run, daemon=True).start()

    def _start_prefetch(self, current_ids: list[int]) -> None:
        """Silently warm the template cache for adjacent clusters."""
        adjacent = self._cluster_dock.adjacent_cluster_ids(current_ids, n=3)
        to_fetch = [cid for cid in adjacent if cid not in self._template_cache]
        if not to_fetch:
            return

        worker = _PrefetchWorker(self.transport.host, self.transport.port, to_fetch)
        worker.template_ready.connect(self._on_prefetch_template)
        self._prefetch_worker = worker
        threading.Thread(target=worker.run, daemon=True).start()

    # ------------------------------------------------------------------
    # Signal handlers
    # ------------------------------------------------------------------

    @pyqtSlot(dict)
    def _on_templates_ready(self, templates: dict) -> None:
        # templates: {cid: (array, ch_ids)}
        logger.info("Templates ready for clusters %s", list(templates.keys()))
        debug_parts = []
        for cid, (arr, ch_ids, best_ch) in templates.items():
            shape = tuple(np.asarray(arr).shape)
            debug_parts.append(f"{cid}:shape={shape},ch={len(ch_ids)},best={best_ch}")
        if debug_parts:
            msg = "Templates debug: " + " | ".join(debug_parts)
            logger.info(msg)
            self._console_view.log(msg)
        for cid, (arr, ch_ids, best_ch) in templates.items():
            self._template_cache.put(cid, (arr, ch_ids))
            # best_ch = highest-amplitude channel (not just ch_ids[0] which is sorted by index)
            if ch_ids:
                self._home_channels[cid] = int(best_ch) if best_ch is not None else int(ch_ids[0])
                self._template_channels[cid] = [int(c) for c in ch_ids]
        self._update_probe_view()
        if set(templates.keys()) == set(self._current_cluster_ids):
            self._set_status(
                f"Cluster(s) {self._fmt_ids(list(templates.keys()))}  — loading waveforms…"
            )
            self._dispatch_waveforms(templates)
        else:
            logger.info("Templates discarded (selection changed to %s)", self._current_cluster_ids)

    @pyqtSlot(dict)
    def _on_features_ready(self, features: dict) -> None:
        for cid, arr in features.items():
            self._features[cid] = arr
        if set(features.keys()) == set(self._current_cluster_ids):
            self._feature_cloud_view.set_feature_data(
                features, spike_times=self._spike_times_for_current()
            )

    @pyqtSlot(dict)
    def _on_template_features_ready(self, template_features: dict) -> None:
        for cid, arr in template_features.items():
            self._template_features[cid] = arr
        if set(template_features.keys()) == set(self._current_cluster_ids):
            self._template_feat_view.set_feature_data(template_features)

    # Tick colours (index 0 = primary/selected cluster)
    _TICK_COLORS = [
        (0.20, 0.55, 0.95, 1.0),   # 0 primary — blue
        (0.92, 0.20, 0.20, 1.0),   # 1 compare — red
        (0.15, 0.72, 0.35, 1.0),   # 2          — green
        (0.92, 0.58, 0.08, 1.0),   # 3          — orange
        (0.65, 0.15, 0.90, 1.0),   # 4          — purple
    ]

    @pyqtSlot(dict)
    def _on_spike_data_ready(self, spike_data: dict) -> None:
        logger.info("Spike data ready for clusters %s", list(spike_data.keys()))
        # Cache spike times and full spike data
        for cid, data in spike_data.items():
            self._spike_times[cid] = data[:, 0].astype(np.float64)
            self._spike_data[cid] = data

        if set(spike_data.keys()) == set(self._current_cluster_ids):
            self._update_trace_view()
            self._isi_view.set_spike_times(self._spike_times_for_current())
            self._amp_view.set_spike_data(spike_data)
            self._raster_view.set_selected_clusters(list(spike_data.keys()))
            cur_features = self._features_for_current()
            if cur_features:
                self._feature_cloud_view.set_feature_data(
                    cur_features, spike_times=self._spike_times_for_current()
                )
            self._fetch_amp_background(list(spike_data.keys()))

    @pyqtSlot(dict)
    def _on_spike_times_ready(self, spike_times: dict) -> None:
        for cid, times in spike_times.items():
            self._spike_times[cid] = times
        if set(spike_times.keys()) == set(self._current_cluster_ids):
            times = self._spike_times_for_current()
            self._isi_view.set_spike_times(times)
            self._correlogram_view.set_spike_times(times)
            self._update_trace_view()

    @pyqtSlot(dict)
    def _on_feat_spike_data_ready(self, feat_spike_data: dict) -> None:
        for cid, data in feat_spike_data.items():
            self._feat_spike_data[cid] = data
        if set(feat_spike_data.keys()) == set(self._current_cluster_ids):
            self._feat_amp_view.set_spike_data(feat_spike_data)

    @pyqtSlot(object)
    def _on_background_features_ready(self, arr) -> None:
        self._feature_cloud_view.set_background_features(arr)

    @pyqtSlot(dict)
    def _on_waveforms_ready(self, waveforms: dict) -> None:
        # waveforms: {cid: (array, ch_ids)}
        logger.info("Waveforms ready for clusters %s", list(waveforms.keys()))
        debug_parts = []
        for cid, (arr, ch_ids) in waveforms.items():
            shape = tuple(np.asarray(arr).shape)
            debug_parts.append(f"{cid}:shape={shape},ch={len(ch_ids)}")
        if debug_parts:
            msg = "Waveforms debug: " + " | ".join(debug_parts)
            logger.info(msg)
            self._console_view.log(msg)
        for cid, val in waveforms.items():
            self._waveform_cache.put(cid, val)
        if set(waveforms.keys()) == set(self._current_cluster_ids):
            self._set_status(f"Cluster(s) {self._fmt_ids(list(waveforms.keys()))}")
            self._dispatch_waveforms(waveforms)
            self._start_prefetch(self._current_cluster_ids)
        else:
            logger.info("Waveforms discarded (selection changed to %s)", self._current_cluster_ids)

    @pyqtSlot(str)
    def _on_fetch_error(self, msg: str) -> None:
        logger.error("Fetch error: %s", msg)
        self._set_status(f"Error: {msg}")

    @pyqtSlot(int, object)
    def _on_prefetch_template(self, cluster_id: int, template) -> None:
        self._template_cache.put(cluster_id, template)
        logger.debug("Prefetched template for cluster %d", cluster_id)

    # ------------------------------------------------------------------
    # Labelling
    # ------------------------------------------------------------------

    def _apply_label(self, label: str) -> None:
        cluster_ids = self._cluster_dock.selected_cluster_ids()
        if not cluster_ids:
            self._set_status("No clusters selected")
            return
        try:
            self.transport.label_cluster(cluster_ids, label)
        except TransportError as exc:
            self._set_status(f"Label error: {exc}")
            return
        for cid in cluster_ids:
            self._cluster_dock.update_label(cid, label)
        clu_str = ", ".join(str(c) for c in cluster_ids)
        self._set_status(f"Labelled {clu_str} → {label}  [unsaved — Ctrl+S to save]")
        self._mark_unsaved()
        logger.info("Labelled cluster(s) %s as %r", cluster_ids, label)

    # ------------------------------------------------------------------
    # Merge / undo / redo / save
    # ------------------------------------------------------------------

    def _smart_g(self) -> None:
        """g: label good when 1 cluster selected; merge when 2+ selected."""
        ids = self._cluster_dock.selected_cluster_ids()
        if len(ids) >= 2:
            self._merge_clusters()
        else:
            self._apply_label("good")

    def _merge_clusters(self) -> None:
        cluster_ids = self._cluster_dock.selected_cluster_ids()
        if len(cluster_ids) < 2:
            self._set_status("Select at least 2 clusters to merge")
            return
        # Evict merged clusters from caches so stale data isn't shown
        for cid in cluster_ids:
            self._template_cache._data.pop(cid, None)
            self._waveform_cache._data.pop(cid, None)
            self._similarity_cache.pop(cid, None)
        self._similarity_cache.clear()  # all similarity rows change after a merge
        try:
            result = self.transport.merge_clusters(cluster_ids)
        except TransportError as exc:
            self._set_status(f"Merge error: {exc}")
            return
        new_id = result.get("new_cluster_id")
        self._console_view.log(f"Merged {self._fmt_ids(cluster_ids)} → {new_id}")
        self._set_status(f"Merged {self._fmt_ids(cluster_ids)} → cluster {new_id}  [unsaved — Ctrl+S to save]")
        self._mark_unsaved()
        clusters = result.get("clusters")
        if clusters:
            self._reload_from_cluster_list(clusters)
        else:
            self._load_cluster_info()
        # Select the newly created cluster so its waveforms load automatically
        if new_id is not None:
            self._cluster_dock.select_cluster_by_id(new_id)

    def _undo(self) -> None:
        try:
            result = self.transport.undo()
        except TransportError as exc:
            self._set_status(f"Undo: {exc}")
            return
        self._set_status("Undo")
        clusters = result.get("clusters")
        if clusters:
            self._reload_from_cluster_list(clusters)
        else:
            self._load_cluster_info()

    def _redo(self) -> None:
        try:
            result = self.transport.redo()
        except TransportError as exc:
            self._set_status(f"Redo: {exc}")
            return
        self._set_status("Redo")
        clusters = result.get("clusters")
        if clusters:
            self._reload_from_cluster_list(clusters)
        else:
            self._load_cluster_info()

    def _mark_unsaved(self) -> None:
        title = self.windowTitle()
        if not title.endswith(" *"):
            self.setWindowTitle(title + " *")

    def _mark_saved(self) -> None:
        title = self.windowTitle()
        if title.endswith(" *"):
            self.setWindowTitle(title[:-2])

    def _save(self) -> None:
        try:
            self.transport.save()
            self._mark_saved()
            self._set_status("Saved")
            self._console_view.log("Saved spike_clusters.npy + cluster_group.tsv")
        except TransportError as exc:
            self._set_status(f"Save error: {exc}")

    def _reload_from_cluster_list(self, clusters: list[dict]) -> None:
        """Update cluster table from a server-supplied cluster list."""
        self._cluster_dock.populate(clusters)
        self._similarity_view.set_cluster_info(clusters)
        n_good = sum(1 for c in clusters if c["label"] == "good")
        self._set_status(f"{len(clusters)} clusters  —  {n_good} good")
        # Raster data may have changed (merge/undo/redo) — re-fetch
        threading.Thread(target=self._fetch_raster_data_bg, daemon=True).start()

    # ------------------------------------------------------------------
    # Label similar / label all
    # ------------------------------------------------------------------

    def _label_similar(self, label: str) -> None:
        """Label all clusters shown in the similarity view as *label*."""
        rows = self._similarity_view.current_rows()
        if not rows:
            self._set_status("No similar clusters to label")
            return
        cluster_ids = [r["id"] for r in rows]
        try:
            self.transport.label_cluster(cluster_ids, label)
        except TransportError as exc:
            self._set_status(f"Label error: {exc}")
            return
        for cid in cluster_ids:
            self._cluster_dock.update_label(cid, label)
        self._set_status(f"Labelled {len(cluster_ids)} similar clusters → {label}")

    def _label_all(self, label: str) -> None:
        """Label every cluster in the table as *label*."""
        cluster_ids = self._cluster_dock.all_cluster_ids()
        if not cluster_ids:
            return
        try:
            self.transport.label_cluster(cluster_ids, label)
        except TransportError as exc:
            self._set_status(f"Label error: {exc}")
            return
        for cid in cluster_ids:
            self._cluster_dock.update_label(cid, label)
        self._set_status(f"Labelled all {len(cluster_ids)} clusters → {label}")

    # ------------------------------------------------------------------
    # Waveform zoom helpers
    # ------------------------------------------------------------------

    def _dispatch_waveforms(self, data: dict) -> None:
        """Send cluster 0 to left panel, cluster 1 to right panel."""
        items = list(data.items())
        if len(items) >= 1:
            self._waveform_view_0.set_waveforms({items[0][0]: items[0][1]})
        else:
            self._waveform_view_0.clear()
        if len(items) >= 2:
            self._waveform_view_1.set_waveforms({items[1][0]: items[1][1]})
        else:
            self._waveform_view_1.clear()

    def _waveform_amp_zoom(self, factor: float) -> None:
        """Scale waveform amplitude up or down on both panels."""
        for wv in (self._waveform_view_0, self._waveform_view_1):
            wv._set_y_zoom(wv._y_zoom * factor)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_status(self, text: str) -> None:
        self._status_label.setText(text)

    @pyqtSlot(str)
    def _console_view_log(self, text: str) -> None:
        if hasattr(self, "_console_view"):
            self._console_view.log(text)

    def _spike_times_for_current(self) -> dict[int, np.ndarray]:
        out = {}
        for cid in self._current_cluster_ids:
            if cid in self._spike_times:
                out[cid] = self._spike_times[cid]
        return out

    def _update_secondary_views_from_cache(self) -> None:
        """Refresh ISI, amplitude, raster, correlogram, and feature views from cache."""
        sd = self._spike_data_for_current()
        if sd:
            times = self._spike_times_for_current()
            self._isi_view.set_spike_times(times)
            self._correlogram_view.set_spike_times(times)
            self._amp_view.set_spike_data(sd)
            fsd = self._feat_spike_data_for_current()
            if fsd:
                self._feat_amp_view.set_spike_data(fsd)
            self._raster_view.set_selected_clusters(self._current_cluster_ids)
        cur_features = self._features_for_current()
        if cur_features:
            self._feature_cloud_view.set_feature_data(
                cur_features, spike_times=self._spike_times_for_current()
            )
        cur_tf = {cid: self._template_features[cid]
                  for cid in self._current_cluster_ids if cid in self._template_features}
        if cur_tf:
            self._template_feat_view.set_feature_data(cur_tf)

    def _spike_data_for_current(self) -> dict[int, np.ndarray]:
        return {cid: self._spike_data[cid]
                for cid in self._current_cluster_ids if cid in self._spike_data}

    def _feat_spike_data_for_current(self) -> dict[int, np.ndarray]:
        return {cid: self._feat_spike_data[cid]
                for cid in self._current_cluster_ids if cid in self._feat_spike_data}

    def _features_for_current(self) -> dict[int, np.ndarray]:
        out = {}
        for cid in self._current_cluster_ids:
            if cid in self._features:
                out[cid] = self._features[cid]
        return out

    def _set_comparison_cluster(self, cluster_id: int) -> None:
        """Keep the primary cluster selected; replace any comparison with cluster_id."""
        primary = self._cluster_dock._primary_cluster_id
        if primary is None:
            return

        # Stop any queued debounce before touching the selection model
        self._selection_debounce.stop()
        self._pending_cluster_ids = [primary, cluster_id]

        # Block selection signals so table updates don't restart the debounce
        sm = self._cluster_dock._view.selectionModel()
        sm.blockSignals(True)
        try:
            self._cluster_dock.select_cluster_by_id(primary)
            self._cluster_dock.add_cluster_to_selection(cluster_id)
        finally:
            sm.blockSignals(False)

        # Schedule apply for next event-loop iteration (safe from within a slot)
        QTimer.singleShot(0, self._apply_cluster_selection)

    def _fetch_raster_data_bg(self) -> None:
        """Fetch global raster data for all clusters in a daemon thread."""
        try:
            tr = PhyTransport(host=self.transport.host, port=self.transport.port)
            try:
                header, data = tr.get_raster_data(n_spikes=200)
            finally:
                tr.close()
            self._raster_ready.emit({
                "data":        data,
                "cluster_ids": header.get("cluster_ids", []),
                "duration":    float(header.get("duration", 0.0)),
            })
        except Exception as exc:
            logger.warning("Raster data fetch failed: %s", exc)

    @pyqtSlot(object)
    def _on_raster_ready(self, payload: dict) -> None:
        self._raster_view.set_raster_data(
            payload["data"],
            payload["cluster_ids"],
            payload["duration"],
        )
        # Highlight the current selection if any is already active
        if self._current_cluster_ids:
            self._raster_view.set_selected_clusters(self._current_cluster_ids)

    def _fetch_similarity_bg(self, cluster_id: int) -> None:
        """Fetch similar-cluster rows in a daemon thread; emit _similarity_ready when done."""
        def _worker():
            try:
                tr = PhyTransport(host=self.transport.host, port=self.transport.port)
                try:
                    rows = tr.get_similar_clusters(cluster_id, limit=100)
                finally:
                    tr.close()
                self._similarity_ready.emit(cluster_id, rows)
            except Exception as exc:
                logger.warning("Similarity fetch failed for %d: %s", cluster_id, exc)
        threading.Thread(target=_worker, daemon=True).start()

    @pyqtSlot(int, object)
    def _on_similarity_ready(self, cluster_id: int, rows: list) -> None:
        self._similarity_cache[cluster_id] = rows
        # Only update the view if this cluster is still the active primary.
        if self._current_cluster_ids and self._current_cluster_ids[0] == cluster_id:
            self._similarity_view.set_similarity_rows(cluster_id, rows)

    def _fetch_amp_background(self, selected_ids: list[int]) -> None:
        """Fetch grey background spike data for the amplitude view in a daemon thread."""
        if not selected_ids:
            return
        self._amp_bg_seq += 1
        seq     = self._amp_bg_seq
        primary = selected_ids[0]

        def _worker() -> None:
            from phy_remote.client.transport import PhyTransport
            try:
                tr = PhyTransport(host=self.transport.host, port=self.transport.port)
                try:
                    data = tr.get_background_spike_data(primary, excluded_ids=selected_ids)
                finally:
                    tr.close()
                if seq == self._amp_bg_seq:
                    self._amp_bg_ready.emit(data)
            except Exception as exc:
                logger.debug("Amplitude background fetch failed: %s", exc)

        threading.Thread(target=_worker, daemon=True).start()

    def _update_probe_view(self) -> None:
        """Push current selected-channel info to the probe view."""
        sel = []
        for idx, cid in enumerate(self._current_cluster_ids):
            if cid in self._template_channels:
                color = self._TICK_COLORS[idx % len(self._TICK_COLORS)]
                sel.append((self._template_channels[cid], color))
        if sel:
            self._probe_view.set_selected_channels(sel)
        self._update_trace_view()

    def _update_trace_view(self) -> None:
        """Push spike overlay data for the current clusters to the trace view."""
        if self._trace_view is None:
            return
        cluster_data = {}
        for idx, cid in enumerate(self._current_cluster_ids):
            if cid not in self._spike_times:
                continue
            # Pass all template channel ids so the trace view can overlay
            # waveform snippets across the full channel neighbourhood,
            # matching phy's _iter_spike_waveforms(channel_ids=get_best_channels(c)).
            ch_ids = self._template_channels.get(cid)
            if not ch_ids:
                home_ch = self._home_channels.get(cid)
                if home_ch is None:
                    continue
                ch_ids = [home_ch]
            color = self._TICK_COLORS[idx % len(self._TICK_COLORS)]
            cluster_data[cid] = (self._spike_times[cid], ch_ids, color)
        self._trace_view.set_cluster_data(cluster_data)
        if cluster_data:
            self._trace_view.go_to_first_spike()

    def closeEvent(self, event) -> None:
        if self._fetch_worker is not None:
            self._fetch_worker.cancel()
        if self._prefetch_worker is not None:
            self._prefetch_worker.cancel()
        # Close all fastplotlib figures before Qt destroys the widgets.
        # rendercanvas connects an aboutToQuit handler that accesses the
        # canvas C++ object — if we let Qt destroy widgets first that handler
        # crashes with "wrapped C/C++ object has been deleted".
        _fpl_views = (
            self._waveform_view_0, self._waveform_view_1,
            self._amp_view, self._isi_view, self._raster_view,
            self._correlogram_view, self._feature_cloud_view,
            self._template_feat_view, self._trace_view,
        )
        for view in _fpl_views:
            fig = getattr(view, '_fig', None)
            if fig is not None:
                try:
                    fig.close()
                except Exception:
                    pass
        super().closeEvent(event)
