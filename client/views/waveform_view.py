"""WaveformWidget — fastplotlib waveform view with probe spatial layout.

All selected clusters are rendered together in one figure:
  • semi-transparent raw spike traces behind the mean (shown when "Show raw" toggled)
  • mean ± std band (std = two thin transparent lines flanking the mean)
  • solid mean waveform on top

Cluster colors (matching original phy convention):
  index 0 — primary / selected  → red
  index 1 — comparison          → blue
  index 2+                      → green, orange, …

Channels are positioned by real probe geometry (depth on Y, shank offset on X).
The 2D camera has maintain_aspect=False so X (time) and Y (depth) zoom independently.
"""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from phy_remote.client.views._graphics import safe_delete, slim_subplot

logger = logging.getLogger(__name__)

# Cluster colors: 0 = selected (red), 1 = comparison (blue), 2+ = green, orange…
_CLUSTER_COLORS = [
    (0.90, 0.15, 0.15),   # red
    (0.15, 0.47, 0.90),   # blue
    (0.15, 0.70, 0.35),   # green
    (0.92, 0.58, 0.08),   # orange
]
_RAW_ALPHA  = 0.10
_STD_ALPHA  = 0.30
_MEAN_ALPHA = 1.00


class WaveformWidget(QWidget):
    """
    Waveform widget that displays mean ± std and optional raw spike traces
    for one or more clusters, laid out spatially by probe geometry.

    Parameters
    ----------
    max_spikes : cap on raw spike traces drawn per cluster (subsampled if exceeded)
    """

    render_debug = pyqtSignal(str)

    def __init__(self, max_spikes: int = 500, cluster_slot: int = 0, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.max_spikes   = max_spikes
        self.cluster_slot = cluster_slot
        self._y_zoom     = 1.0

        self._channel_positions: np.ndarray | None = None
        self._last_normalised: dict | None = None

        # fastplotlib handles
        self._fig            = None
        self._subplot        = None
        self._canvas_widget: QWidget | None = None
        self._fpl_ready      = False

        # Per-cluster graphic stores
        self._raw_lines: dict[int, object] = {}

        self.setMinimumHeight(120)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- toolbar ----
        bar = QWidget()
        bar.setFixedHeight(26)
        bar.setStyleSheet("background:#252525;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(6, 2, 6, 2)
        bl.setSpacing(6)

        self._y_plus_btn  = self._make_btn("Y+")
        self._y_minus_btn = self._make_btn("Y-")

        self._y_plus_btn.clicked.connect(lambda: self._set_y_zoom(self._y_zoom * 1.2))
        self._y_minus_btn.clicked.connect(lambda: self._set_y_zoom(self._y_zoom / 1.2))

        self._cluster_label = QLabel("")
        self._cluster_label.setStyleSheet("color:#aaa;font-size:11px;padding-right:4px;")

        for w in (self._y_minus_btn, self._y_plus_btn):
            bl.addWidget(w)
        bl.addStretch()
        bl.addWidget(self._cluster_label)
        outer.addWidget(bar)

        # ---- canvas area ----
        self._canvas_area = QWidget()
        self._canvas_area.setStyleSheet("background:#1e1e1e;")
        self._canvas_layout = QVBoxLayout(self._canvas_area)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel("Select a cluster to display waveforms")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._placeholder.setStyleSheet("color:#666;font-size:14px;")
        self._canvas_layout.addWidget(self._placeholder)
        outer.addWidget(self._canvas_area)

        try:
            import fastplotlib as fpl
            self._init_fpl(fpl)
        except Exception as exc:
            logger.warning("WaveformWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channel_positions(self, positions: np.ndarray) -> None:
        """Supply probe channel positions (n_channels, 2) [x, y] µm."""
        self._channel_positions = np.asarray(positions, dtype=np.float32)
        logger.info("WaveformWidget: %d channel positions loaded", len(positions))

    def set_waveforms(
        self,
        data_per_cluster: "dict[int, tuple[np.ndarray, list[int]] | np.ndarray]",
    ) -> None:
        """
        Render waveforms for one or more clusters.

        Parameters
        ----------
        data_per_cluster
            {cluster_id: (array, channel_ids)} where array is shaped
            (n_spikes, n_samples, n_channels) or (n_samples, n_channels).
        """
        if not data_per_cluster:
            self.clear()
            return

        normalised: dict[int, tuple[np.ndarray, list[int]]] = {}
        for cid, val in data_per_cluster.items():
            if isinstance(val, tuple):
                arr, ch_ids = val[0], list(val[1])
            else:
                arr = val
                ch_ids = list(range(arr.shape[-1] if arr.ndim >= 2 else 1))
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 2:
                arr = arr[np.newaxis]   # (1, n_samples, n_ch)
            if arr.ndim != 3:
                logger.error("Cluster %d: unexpected shape %s — skipped", cid, arr.shape)
                continue
            normalised[cid] = (arr, ch_ids)

        if not normalised:
            return

        self._last_normalised = normalised
        cids = list(normalised.keys())
        self._cluster_label.setText(
            f"Cluster {cids[0]}" if len(cids) == 1
            else "Clusters " + ", ".join(str(c) for c in cids)
        )

        if self._fpl_ready:
            try:
                self._render(normalised, rescale_camera=True)
            except Exception as exc:
                logger.exception("WaveformWidget render failed: %s", exc)
                self.render_debug.emit(f"fpl error: {exc}")

    def clear(self) -> None:
        self._cluster_label.setText("")
        self._last_normalised = None
        sp = self._subplot
        if sp is not None:
            for cid in list(self._raw_lines.keys()):
                safe_delete(sp, self._raw_lines.pop(cid))
        self._show_placeholder("Select a cluster to display waveforms")

    # ------------------------------------------------------------------
    # fastplotlib
    # ------------------------------------------------------------------

    def _init_fpl(self, fpl) -> None:
        self._fig     = fpl.Figure(canvas="qt")
        self._subplot = self._fig[0, 0]
        try:
            self._subplot.camera = "2d"
            self._subplot.camera.maintain_aspect = False
        except Exception:
            pass
        try:
            self._subplot.axes.visible = False
            self._subplot.title.visible = False
            slim_subplot(self._subplot)
        except Exception:
            pass
        self._fig.show()
        self._canvas_widget = self._fig.canvas
        self._canvas_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas_widget.setMinimumSize(240, 180)
        self._canvas_layout.addWidget(self._canvas_widget)
        self._wheel_filter = _WaveformWheelFilter(self, self._canvas_widget, self)
        QApplication.instance().installEventFilter(self._wheel_filter)
        self._fpl_ready = True

    def _render(self, normalised: dict, rescale_camera: bool = False) -> None:
        sp = self._subplot
        self._placeholder.hide()
        self._canvas_widget.show()

        # Union of all channels across all clusters, preserving order
        seen: set[int] = set()
        all_ch_ids: list[int] = []
        for _, (_, ch_ids) in normalised.items():
            for c in ch_ids:
                if c not in seen:
                    seen.add(c)
                    all_ch_ids.append(c)

        positions, x_scale, amp_scale = self._probe_layout(all_ch_ids, normalised)

        # Remove graphics for clusters no longer selected
        active = set(normalised.keys())
        for cid in list(self._raw_lines.keys()):
            if cid not in active:
                safe_delete(sp, self._raw_lines.pop(cid))

        for clu_idx, (cid, (arr3, ch_ids)) in enumerate(normalised.items()):
            rgb = _CLUSTER_COLORS[(self.cluster_slot + clu_idx) % len(_CLUSTER_COLORS)]

            ch_pos = self._positions_for_channels(ch_ids, all_ch_ids, positions)
            n_ch   = min(arr3.shape[2], len(ch_pos))
            arr3   = arr3[:, :, :n_ch]
            ch_pos = ch_pos[:n_ch]

            n_spikes = arr3.shape[0]
            if n_spikes > self.max_spikes:
                rng  = np.random.default_rng(seed=cid)
                idxs = np.sort(rng.choice(n_spikes, self.max_spikes, replace=False))
                arr3 = arr3[idxs]

            raw_data = self._build_raw_lines(arr3, ch_pos, x_scale, amp_scale)
            self._upsert(self._raw_lines, sp, cid, raw_data, (*rgb, 0.25), 1.0)

        if rescale_camera:
            QTimer.singleShot(50, self._auto_scale)
        else:
            try:
                self._fig.canvas.request_draw()
            except Exception:
                pass

    def _upsert(
        self,
        store: dict,
        sp,
        cid: int,
        data: np.ndarray,
        color: tuple,
        thickness: float,
    ) -> None:
        """Update an existing line in-place or create a new one."""
        if len(data) <= 1:
            if cid in store:
                safe_delete(sp, store.pop(cid))
            return
        if cid in store:
            old = store[cid]
            if old.data.value.shape[0] == len(data):
                xyz = np.zeros((len(data), 3), dtype=np.float32)
                fin = np.isfinite(data[:, 0]) & np.isfinite(data[:, 1])
                xyz[fin, 0]  = data[fin, 0]
                xyz[fin, 1]  = data[fin, 1]
                xyz[~fin, :] = np.nan
                old.data[:] = xyz
                try:
                    old.colors = color
                except Exception:
                    pass
                return
            safe_delete(sp, old)
        store[cid] = sp.add_line(data, colors=color, thickness=thickness)

    # ------------------------------------------------------------------
    # Line-data builders (fully vectorised)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_channel_lines(
        waveform_2d: np.ndarray,   # (n_samples, n_ch)
        positions:   np.ndarray,   # (n_ch, 2) µm
        x_scale:     float,
        amp_scale:   float,
    ) -> np.ndarray:
        """NaN-separated (N, 2) line for a single waveform (mean or std)."""
        n_samples, n_ch = waveform_2d.shape
        n_ch = min(n_ch, len(positions))
        x_base = np.linspace(-x_scale / 2, x_scale / 2, n_samples, dtype=np.float32)
        xs = positions[:n_ch, 0:1] + x_base[np.newaxis, :]          # (n_ch, n_samples)
        ys = positions[:n_ch, 1:2] + waveform_2d[:, :n_ch].T.astype(np.float32) * amp_scale
        n_pts = n_samples + 1
        out = np.full((n_ch, n_pts, 2), np.nan, dtype=np.float32)
        out[:, :n_samples, 0] = xs
        out[:, :n_samples, 1] = ys
        return out.reshape(-1, 2)

    @staticmethod
    def _build_raw_lines(
        arr3:      np.ndarray,   # (n_spikes, n_samples, n_ch)
        positions: np.ndarray,   # (n_ch, 2) µm
        x_scale:   float,
        amp_scale: float,
    ) -> np.ndarray:
        """NaN-separated (N, 2) line for all raw spike traces."""
        n_spikes, n_samples, n_ch = arr3.shape
        n_ch = min(n_ch, len(positions))
        x_base = np.linspace(-x_scale / 2, x_scale / 2, n_samples, dtype=np.float32)
        xs = positions[:n_ch, 0:1] + x_base[np.newaxis, :]          # (n_ch, n_samples)
        arr_t = arr3[:, :, :n_ch].transpose(0, 2, 1).astype(np.float32)  # (n_sp, n_ch, n_s)
        ys = positions[:n_ch, 1][np.newaxis, :, np.newaxis] + arr_t * amp_scale
        n_pts = n_samples + 1
        out = np.full((n_spikes, n_ch, n_pts, 2), np.nan, dtype=np.float32)
        out[:, :, :n_samples, 0] = xs[np.newaxis, :, :]   # broadcast
        out[:, :, :n_samples, 1] = ys
        return out.reshape(-1, 2)

    # ------------------------------------------------------------------
    # Probe layout
    # ------------------------------------------------------------------

    def _probe_layout(
        self,
        ch_ids:    list[int],
        normalised: dict,
    ) -> tuple[np.ndarray, float, float]:
        """Return (positions, x_scale, amp_scale) for the given channel list."""
        n_ch = len(ch_ids)
        if (self._channel_positions is not None and n_ch > 0
                and max(ch_ids) < len(self._channel_positions)):
            positions = self._channel_positions[np.array(ch_ids)].copy()
        else:
            positions = np.zeros((n_ch, 2), dtype=np.float32)
            positions[:, 1] = np.linspace(n_ch * 40.0, 0.0, n_ch)

        if n_ch > 1:
            y = np.sort(positions[:, 1])
            diffs = np.diff(y)
            diffs = diffs[diffs > 0.5]
            y_pitch = float(np.min(diffs)) if len(diffs) else 40.0
        else:
            y_pitch = 40.0

        # Shrink waveform width so adjacent x-columns don't overlap.
        # For probes with multiple x-positions (e.g. Neuropixels stagger),
        # x_gap is the minimum horizontal separation between columns.
        x_vals = np.unique(np.round(positions[:, 0]).astype(int))
        if len(x_vals) > 1:
            x_gap = float(np.min(np.diff(np.sort(x_vals))))
            x_scale = min(y_pitch * 0.85, x_gap * 0.85)
        else:
            x_scale = y_pitch * 0.85

        ptp_max = max(
            (float(np.ptp(arr3.mean(axis=0)[:, ch]))
             for arr3, _ in normalised.values()
             for ch in range(arr3.shape[2])),
            default=1.0,
        )
        amp_scale = (y_pitch * 0.45 / ptp_max) if ptp_max > 1e-9 else 1.0
        amp_scale *= self._y_zoom

        # Stretch channel Y spacing for visual clarity
        positions = positions.copy()
        positions[:, 1] *= 3.0

        return positions, x_scale, amp_scale

    @staticmethod
    def _positions_for_channels(
        ch_ids:     list[int],
        all_ch_ids: list[int],
        positions:  np.ndarray,
    ) -> np.ndarray:
        """Slice the positions array to match ch_ids ordering."""
        idx_map = {c: i for i, c in enumerate(all_ch_ids)}
        rows = [idx_map[c] for c in ch_ids if c in idx_map]
        if not rows:
            return np.zeros((0, 2), dtype=np.float32)
        return positions[np.array(rows)].astype(np.float32)

    # ------------------------------------------------------------------
    # Camera
    # ------------------------------------------------------------------

    def _auto_scale(self) -> None:
        if self._fig is None:
            return
        try:
            sp = self._subplot
            sp.auto_scale()
            state = sp.camera.get_state()
            pos = state['position'].copy()
            pos[2] = state['depth'] / 2
            sp.camera.set_state({
                'position':        pos,
                'fov':             0.0,
                'width':           abs(state['width'])  * 1.1,
                'height':          abs(state['height']) * 1.1,
                'depth':           state['depth'],
                'zoom':            1.0,
                'maintain_aspect': False,
            })
            self._fig.canvas.request_draw()
        except Exception as exc:
            logger.debug("WaveformWidget auto_scale: %s", exc)

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def _set_y_zoom(self, value: float) -> None:
        self._y_zoom = float(np.clip(value, 0.05, 50.0))
        if self._last_normalised and self._fpl_ready:
            try:
                self._render(self._last_normalised, rescale_camera=False)
            except Exception as exc:
                logger.debug("WaveformWidget y-zoom render: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_btn(label: str, checkable: bool = False) -> QPushButton:
        b = QPushButton(label)
        b.setCheckable(checkable)
        b.setFixedHeight(20)
        b.setStyleSheet(
            "QPushButton{font-size:11px;padding:0 6px;background:#333;"
            "color:#bbb;border:1px solid #555;border-radius:2px;}"
            "QPushButton:checked{background:#555;color:#fff;}"
            "QPushButton:hover{background:#444;}"
        )
        return b

    def _show_placeholder(self, text: str) -> None:
        self._placeholder.setText(text)
        if self._canvas_widget is not None:
            self._canvas_widget.hide()
        self._placeholder.show()


# ---------------------------------------------------------------------------
# Wheel event filter: Ctrl+scroll or Cmd+scroll → Y amplitude zoom
# ---------------------------------------------------------------------------

class _WaveformWheelFilter(QObject):
    _STEP = 1.15

    def __init__(self, widget: WaveformWidget, canvas, parent=None):
        super().__init__(parent)
        self._widget = widget
        self._canvas = canvas

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel:
            return False
        # Walk up the parent chain to check the event is over our canvas
        w = obj
        while w is not None:
            if w is self._canvas:
                break
            w = w.parent()
        else:
            return False

        mods  = event.modifiers()
        delta = event.angleDelta().y()
        if delta == 0:
            return False

        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        meta = bool(mods & Qt.KeyboardModifier.MetaModifier)   # Command on Mac

        factor = self._STEP ** (delta / 120)
        v = self._widget

        if ctrl or meta:
            v._set_y_zoom(v._y_zoom * factor)

        # Always consume — prevent fastplotlib camera zoom from spreading channels
        return True
