"""TraceView — raw voltage trace display using fastplotlib.

Matches phy2 TraceView behaviour exactly (minus GPU backend):
- Default window 0.25 s  (phy: interval_duration = 0.25)
- Spike overlay = waveform snippets cut from the already-loaded trace buffer,
  one NaN-separated line per cluster — no extra network round-trip
  (phy: _iter_spike_waveforms cuts traces_interval[s-k:s+k, channel_ids])
- Only selected clusters shown  (phy: show_all_spikes = False by default)
- HP filter: 3rd-order Butterworth 300 Hz zero-phase
- Navigation: Alt+←/→ (scroll 50%), Ctrl+wheel (time zoom),
  Alt+wheel (amplitude), right-click drag (pan time)
"""
from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QPushButton,
    QSizePolicy, QVBoxLayout, QWidget,
)

from phy_remote.client.views._graphics import slim_subplot

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — match phy defaults where named
# ---------------------------------------------------------------------------

_WINDOW_S        = 0.25     # phy: interval_duration = 0.25
_SHIFT_FRAC      = 0.5
_TRACE_COLOR     = (0.45, 0.45, 0.45, 0.90)
_SNIPPET_SAMPLES = 82       # phy: n_samples_waveforms; ~2.7 ms @ 30 kHz
_MAX_SAMPLES     = 3_000    # server-side downsampling cap

_CLUSTER_COLORS = [
    (0.92, 0.15, 0.15, 1.0),
    (0.15, 0.47, 0.90, 1.0),
    (0.15, 0.72, 0.35, 1.0),
    (0.92, 0.58, 0.08, 1.0),
    (0.65, 0.15, 0.90, 1.0),
]


class TraceWidget(QWidget):
    spike_clicked      = pyqtSignal(float)
    time_range_changed = pyqtSignal(float, float)   # (t_start, t_end) → amp view

    def __init__(self, host: str, port: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._host     = host
        self._port     = port
        self._t_start  = 0.0
        self._window_s = _WINDOW_S
        self._y_scale  = 1.0
        self._filtered = False

        # Probe geometry
        self._ch_positions: np.ndarray | None = None
        self._ch_y:         np.ndarray | None = None   # (n_ch,) probe y per channel id

        # Trace buffer — reloaded only on time navigation
        self._buf_traces:   np.ndarray | None = None   # (n_ch, n_s) CMR + scaled
        self._buf_t_arr:    np.ndarray | None = None   # (n_s,) seconds
        self._buf_ch_ids:   list[int] = []
        self._ch_id_to_row: dict[int, int] = {}        # channel_id → row in buf

        # Selected cluster data: {cid: (spike_times, channel_ids, rgba)}
        #   spike_times  — (n,) float64 seconds (already in client memory)
        #   channel_ids  — list of best channel ids from the cluster's template
        #   rgba         — 4-tuple float color
        self._cluster_data: dict[int, tuple] = {}

        # Right-click pan state
        self._pan_last_x: float | None = None

        # fastplotlib handles
        self._fig:            Any = None
        self._subplot:        Any = None
        self._canvas_widget:  QWidget | None = None
        self._line_collection: Any = None
        self._snippet_lines:   dict[int, Any] = {}   # one line per selected cluster
        self._fpl_ready        = False

        # Async fetch
        self._fetch_seq = 0
        self._pending:  dict | None = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._do_fetch)

        self._poll = QTimer(self)
        self._poll.setInterval(40)
        self._poll.timeout.connect(self._apply_pending)
        self._poll.start()

        self._build_ui()

        try:
            import fastplotlib as fpl
            self._init_fpl(fpl)
        except Exception as exc:
            logger.warning("TraceWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_channel_positions(self, positions: np.ndarray) -> None:
        self._ch_positions = np.asarray(positions, dtype=np.float32)
        self._ch_y = self._ch_positions[:, 1].copy()
        logger.info("TraceWidget: %d channels", len(positions))
        self._schedule_fetch()

    def set_cluster_data(
        self,
        cluster_data: "dict[int, tuple[np.ndarray, list[int], tuple]]",
    ) -> None:
        """Update spike overlay without reloading raw data.

        Parameters
        ----------
        cluster_data : {cid: (spike_times, channel_ids, rgba)}
            spike_times  — (n,) float64 seconds, already in client memory
            channel_ids  — best channel ids for this cluster (from template)
            rgba         — 4-tuple float color
        """
        self._cluster_data = dict(cluster_data)
        if self._fpl_ready and self._buf_traces is not None:
            self._rebuild_snippets()
            try:
                self._fig.canvas.request_draw()
            except Exception:
                pass

    def set_all_best_channels(self, best: "dict[int, int]") -> None:
        """No-op — retained for API compatibility.

        We no longer fetch or display all-cluster spike overlays.
        Spike data comes from the client-side spike_times cache via set_cluster_data.
        """

    def go_to_time(self, t: float) -> None:
        self._t_start = max(0.0, t - self._window_s / 2)
        self._schedule_fetch()

    def go_to_first_spike(self) -> None:
        """Jump to the first spike of the primary selected cluster."""
        for _cid, (times, _ch_ids, _col) in self._cluster_data.items():
            if len(times):
                self.go_to_time(float(times[0]))
                return

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet("background:#252525;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(6, 2, 6, 2)
        bl.setSpacing(6)

        self._filter_btn = self._make_btn("HP filter", checkable=True)
        self._filter_btn.toggled.connect(self._on_filter_toggled)
        self._spike_btn  = self._make_btn("⟩| spike")
        self._spike_btn.clicked.connect(self.go_to_first_spike)

        self._status = QLabel("Waiting for data…")
        self._status.setStyleSheet("color:#666;font-size:11px;")

        bl.addWidget(self._filter_btn)
        bl.addWidget(self._spike_btn)
        bl.addStretch()
        bl.addWidget(self._status)
        outer.addWidget(bar)

        self._canvas_area = QWidget()
        self._canvas_area.setStyleSheet("background:#111;")
        self._canvas_layout = QVBoxLayout(self._canvas_area)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel("Select a cluster — traces will load automatically")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._placeholder.setStyleSheet("color:#555;font-size:13px;")
        self._canvas_layout.addWidget(self._placeholder)
        outer.addWidget(self._canvas_area)

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

    # ------------------------------------------------------------------
    # fastplotlib setup
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
        self._canvas_widget.setMinimumHeight(60)
        self._canvas_layout.addWidget(self._canvas_widget)
        self._canvas_widget.hide()

        self._event_filter = _TraceEventFilter(self, self._canvas_widget, self)
        QApplication.instance().installEventFilter(self._event_filter)

        try:
            self._fig.canvas.add_event_handler(self._on_pointer_down, "pointer_down")
            self._fig.canvas.add_event_handler(self._on_pointer_move, "pointer_move")
            self._fig.canvas.add_event_handler(self._on_pointer_up,   "pointer_up")
        except Exception as exc:
            logger.debug("TraceWidget: canvas events not available: %s", exc)

        self._fpl_ready = True

    # ------------------------------------------------------------------
    # Fetch pipeline — traces only, no spike RPC
    # (phy finds spikes in-memory via searchsorted on pre-loaded spike_times)
    # ------------------------------------------------------------------

    def _schedule_fetch(self) -> None:
        self._fetch_seq += 1
        self._debounce.start()

    def _do_fetch(self) -> None:
        if self._ch_positions is None:
            return
        t0  = self._t_start
        t1  = t0 + self._window_s
        seq = self._fetch_seq

        def _worker() -> None:
            from phy_remote.client.transport import PhyTransport
            try:
                tr = PhyTransport(host=self._host, port=self._port)
                try:
                    traces, hdr = tr.get_traces(
                        t0, t1,
                        channel_ids=None,
                        filtered=self._filtered,
                        max_samples=_MAX_SAMPLES,
                    )
                finally:
                    tr.close()
            except Exception as exc:
                if seq == self._fetch_seq:
                    self._pending = {"error": str(exc)}
                return
            if seq == self._fetch_seq:
                self._pending = {
                    "traces": traces,
                    "t_start": float(hdr.get("t_start", t0)),
                    "t_end":   float(hdr.get("t_end",   t1)),
                    "sr":      float(hdr.get("sample_rate", 30_000)),
                    "ch_ids":  hdr.get("channel_ids", list(range(traces.shape[0]))),
                }

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_pending(self) -> None:
        if self._pending is None:
            return
        data = self._pending
        self._pending = None

        if "error" in data:
            self._status.setText(f"Error: {data['error']}")
            return

        traces  = np.array(data["traces"], dtype=np.float32)
        t_start = data["t_start"]
        t_end   = data["t_end"]
        ch_ids  = list(data["ch_ids"])

        self._buf_ch_ids   = ch_ids
        self._ch_id_to_row = {ch: i for i, ch in enumerate(ch_ids)}

        n_ch, n_s = traces.shape
        self._buf_t_arr = np.linspace(t_start, t_end, n_s, dtype=np.float32)

        # CMR: subtract median across channels  (phy: select_traces)
        traces -= np.median(traces, axis=0)

        # Auto-scale: normalise so 1%–99% range spans one channel pitch
        # (phy: trace_quantile = 0.01, auto_scale = True)
        q01  = float(np.quantile(traces, 0.01))
        q99  = float(np.quantile(traces, 0.99))
        span = max(abs(q01), abs(q99), 1e-9)
        if self._ch_y is not None and len(ch_ids) > 1:
            ys    = self._ch_y[np.array(ch_ids)]
            diffs = np.abs(np.diff(np.sort(ys)))
            diffs = diffs[diffs > 0.5]
            pitch = float(np.min(diffs)) if len(diffs) else 40.0
        else:
            pitch = 40.0
        self._buf_traces = traces * (pitch * 0.45 / span) * self._y_scale

        self._render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if not self._fpl_ready or self._buf_traces is None:
            return
        try:
            self._update_line_collection()
            self._rebuild_snippets()
            self._placeholder.hide()
            self._canvas_widget.show()
            t0 = float(self._buf_t_arr[0])
            t1 = float(self._buf_t_arr[-1])
            self._status.setText(
                f"{t0:.3f}–{t1:.3f} s  ·  {len(self._buf_ch_ids)} ch  ·  "
                f"{'HP' if self._filtered else 'raw'}"
            )
            QTimer.singleShot(60, self._fit_camera)
            self.time_range_changed.emit(t0, t1)
        except Exception as exc:
            logger.exception("TraceWidget render error: %s", exc)

    def _channel_offsets(self, ch_ids: list[int]) -> np.ndarray:
        """Y offset (µm) for each channel id."""
        if self._ch_y is not None:
            return np.array(
                [self._ch_y[c] if c < len(self._ch_y) else float(i)
                 for i, c in enumerate(ch_ids)],
                dtype=np.float32,
            )
        return np.arange(len(ch_ids), dtype=np.float32) * 40.0

    def _update_line_collection(self) -> None:
        """Rebuild the trace line collection — one (n_s, 2) array per channel."""
        traces  = self._buf_traces
        t_arr   = self._buf_t_arr
        ch_ids  = self._buf_ch_ids
        offsets = self._channel_offsets(ch_ids)
        n_ch, n_s = traces.shape

        lines = []
        for i in range(n_ch):
            xy = np.empty((n_s, 2), dtype=np.float32)
            xy[:, 0] = t_arr
            xy[:, 1] = offsets[i] + traces[i]
            lines.append(xy)

        sp = self._subplot
        if self._line_collection is not None:
            try:
                sp.delete_graphic(self._line_collection)
            except Exception:
                pass
            self._line_collection = None

        try:
            self._line_collection = sp.add_line_collection(
                data=lines, colors=_TRACE_COLOR, thickness=1.0,
            )
        except AttributeError:
            # Fallback: NaN-separated single line
            out = np.full((n_ch * (n_s + 1), 2), np.nan, dtype=np.float32)
            for i in range(n_ch):
                s = i * (n_s + 1)
                out[s:s + n_s, 0] = t_arr
                out[s:s + n_s, 1] = offsets[i] + traces[i]
            self._line_collection = sp.add_line(out, colors=_TRACE_COLOR, thickness=1.0)

    def _rebuild_snippets(self) -> None:
        """Overlay waveform snippets for selected clusters.

        Matches phy's _plot_spike / _iter_spike_waveforms:
          - find spikes in window via searchsorted on in-memory spike_times
          - cut snippet from already-loaded trace buffer: traces[row, s-k:s+k]
          - render as one NaN-separated line per cluster (single GPU object)
        No extra network round-trip needed.
        """
        if not self._fpl_ready or self._buf_traces is None:
            return

        sp      = self._subplot
        t_arr   = self._buf_t_arr
        traces  = self._buf_traces
        ch_ids  = self._buf_ch_ids
        offsets = self._channel_offsets(ch_ids)
        n_s     = len(t_arr)
        k       = _SNIPPET_SAMPLES // 2
        seg_len = _SNIPPET_SAMPLES + 1   # +1 for NaN separator

        t0 = float(t_arr[0])
        t1 = float(t_arr[-1])

        # Remove snippets for clusters no longer selected
        for cid in list(self._snippet_lines.keys()):
            if cid not in self._cluster_data:
                try:
                    sp.delete_graphic(self._snippet_lines.pop(cid))
                except Exception:
                    self._snippet_lines.pop(cid, None)

        for cid, (spike_times, cluster_ch_ids, color) in self._cluster_data.items():
            # Find spikes in current window — phy: spike_times.searchsorted(interval)
            i0 = int(np.searchsorted(spike_times, t0))
            i1 = int(np.searchsorted(spike_times, t1))
            vis_times = spike_times[i0:i1]

            if len(vis_times) == 0:
                if cid in self._snippet_lines:
                    try:
                        sp.delete_graphic(self._snippet_lines.pop(cid))
                    except Exception:
                        self._snippet_lines.pop(cid, None)
                continue

            # Map cluster channel ids → buffer rows (only channels we fetched)
            ch_rows = [
                (self._ch_id_to_row[ch], offsets[self._ch_id_to_row[ch]])
                for ch in cluster_ch_ids
                if ch in self._ch_id_to_row
            ]
            if not ch_rows:
                continue

            n_spikes   = len(vis_times)
            n_ch_local = len(ch_rows)
            # One segment per (spike × channel): _SNIPPET_SAMPLES points + NaN
            line = np.full((n_spikes * n_ch_local * seg_len, 2), np.nan, dtype=np.float32)

            ptr = 0
            for t_spike in vis_times:
                s       = int(np.searchsorted(t_arr, t_spike))
                s_start = s - k
                s_end   = s_start + _SNIPPET_SAMPLES
                if s_start < 0 or s_end > n_s:
                    # Skip partial spikes at edges — same as phy
                    ptr += n_ch_local * seg_len
                    continue
                x_seg = t_arr[s_start:s_end]
                for row, y_off in ch_rows:
                    line[ptr : ptr + _SNIPPET_SAMPLES, 0] = x_seg
                    line[ptr : ptr + _SNIPPET_SAMPLES, 1] = traces[row, s_start:s_end] + y_off
                    # line[ptr + _SNIPPET_SAMPLES] stays NaN (pen-lift)
                    ptr += seg_len

            # Remove stale graphic, upload new one — one line per cluster
            if cid in self._snippet_lines:
                try:
                    sp.delete_graphic(self._snippet_lines.pop(cid))
                except Exception:
                    self._snippet_lines.pop(cid, None)
            try:
                self._snippet_lines[cid] = sp.add_line(
                    line, colors=color, thickness=2.0,
                )
            except Exception as exc:
                logger.warning("TraceWidget: snippet line failed for cluster %d: %s", cid, exc)

    def _fit_camera(self) -> None:
        if self._fig is None or self._buf_t_arr is None:
            return
        try:
            sp      = self._subplot
            offsets = self._channel_offsets(self._buf_ch_ids)
            t0      = float(self._buf_t_arr[0])
            t1      = float(self._buf_t_arr[-1])
            y0      = float(offsets.min())
            y1      = float(offsets.max())
            mid_x   = (t0 + t1) / 2
            mid_y   = (y0 + y1) / 2
            w = (t1 - t0) * 1.04
            h = (y1 - y0 + 40) * 1.08

            state = sp.camera.get_state()
            sp.camera.set_state({
                'position':        np.array([mid_x, mid_y, state['position'][2]]),
                'fov':             0.0,
                'width':           w,
                'height':          h,
                'depth':           state['depth'],
                'zoom':            1.0,
                'maintain_aspect': False,
            })
            self._fig.canvas.request_draw()
        except Exception as exc:
            logger.debug("TraceWidget fit_camera: %s", exc)

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _scroll(self, frac: float) -> None:
        self._t_start = max(0.0, self._t_start + frac * self._window_s)
        self._schedule_fetch()

    def _zoom_time(self, factor: float) -> None:
        mid = self._t_start + self._window_s / 2
        self._window_s = max(0.01, self._window_s * factor)
        self._t_start  = max(0.0, mid - self._window_s / 2)
        self._schedule_fetch()

    def _zoom_amp(self, factor: float) -> None:
        self._y_scale = float(np.clip(self._y_scale * factor, 0.01, 200.0))
        if self._buf_traces is not None:
            self._buf_traces *= factor
            if self._fpl_ready:
                try:
                    self._update_line_collection()
                    self._rebuild_snippets()
                    self._fig.canvas.request_draw()
                except Exception as exc:
                    logger.debug("amp zoom redraw: %s", exc)

    def _on_filter_toggled(self, checked: bool) -> None:
        self._filtered = checked
        self._schedule_fetch()

    # ------------------------------------------------------------------
    # Right-click pan (fpl pointer events)
    # ------------------------------------------------------------------

    def _on_pointer_down(self, event) -> None:
        if getattr(event, "button", None) == 2:
            self._pan_last_x = getattr(event, "x", None)

    def _on_pointer_move(self, event) -> None:
        if self._pan_last_x is None:
            return
        if getattr(event, "button", None) != 2:
            self._pan_last_x = None
            return
        x = getattr(event, "x", None)
        if x is None:
            return
        dx_px = x - self._pan_last_x
        self._pan_last_x = x
        try:
            w_px = self._canvas_widget.width()
            if w_px > 0:
                dt = -(dx_px / w_px) * self._window_s
                self._t_start = max(0.0, self._t_start + dt)
                self._schedule_fetch()
        except Exception:
            pass

    def _on_pointer_up(self, event) -> None:
        self._pan_last_x = None

    # ------------------------------------------------------------------
    # Qt lifecycle
    # ------------------------------------------------------------------

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if self._ch_positions is not None and self._buf_traces is None:
            self._schedule_fetch()


# ---------------------------------------------------------------------------
# Qt event filter: wheel + keyboard navigation
# ---------------------------------------------------------------------------

class _TraceEventFilter(QObject):
    _WHEEL_STEP = 1.20

    def __init__(self, widget: TraceWidget, canvas, parent=None):
        super().__init__(parent)
        self._w      = widget
        self._canvas = canvas

    def _over_canvas(self, obj) -> bool:
        w = obj
        while w is not None:
            if w is self._canvas:
                return True
            w = w.parent()
        return False

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            if not self._over_canvas(obj):
                return False
            mods  = event.modifiers()
            delta = event.angleDelta().y()
            if delta == 0:
                return False

            ctrl   = bool(mods & Qt.KeyboardModifier.ControlModifier)
            meta   = bool(mods & Qt.KeyboardModifier.MetaModifier)
            alt    = bool(mods & Qt.KeyboardModifier.AltModifier)
            factor = self._WHEEL_STEP ** (delta / 120)

            if ctrl or meta:
                self._w._zoom_time(1.0 / factor)
            elif alt:
                self._w._zoom_amp(factor)
            else:
                self._w._scroll(0.1 * (-1 if delta > 0 else 1))
            return True

        if event.type() == QEvent.Type.KeyPress:
            if not self._over_canvas(obj):
                return False
            key = event.key()
            alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)

            if key == Qt.Key.Key_Left and alt:
                self._w._scroll(-_SHIFT_FRAC)
                return True
            if key == Qt.Key.Key_Right and alt:
                self._w._scroll(+_SHIFT_FRAC)
                return True
            if key == Qt.Key.Key_F:
                self._w._scroll(+1.0)
                return True
            if key == Qt.Key.Key_B:
                self._w._scroll(-1.0)
                return True

        return False
