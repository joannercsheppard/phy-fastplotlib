"""ISI histogram widget — inter-spike interval distribution (fastplotlib).

Matches phy ISIView (phy/cluster/views/histogram.py ISIView):
  x_min  = 0
  x_max  = 0.05 s  (50 ms default)
  n_bins = 50      (1 ms per bin)
  data   = np.diff(spike_times)
  norm   = density  (histogram / (sum × bin_width))

Both selected clusters are overlaid in one subplot (different colors) so the
full dock height goes to a single plot — easier to read in a short panel.

Ctrl+scroll  → widen / narrow x_max
Alt+scroll   → more / fewer bins
"""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer
from PyQt6.QtGui import QColor, QPainter
from PyQt6.QtWidgets import QApplication, QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete, slim_subplot
from phy_remote.client.views._zoom import install_zoom_filter

logger = logging.getLogger(__name__)

# Defaults matching phy ISIView
_X_MAX_S  = 0.050   # phy: x_max = .05  (50 ms)
_N_BINS   = 50      # phy: n_bins = int(0.05 / .001) → 1 ms per bin
_REFRAC_S = 0.002   # 2 ms refractory period
_MAX_CLUS = 2


class ISIWidget(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel("ISI")
        hdr.setFixedHeight(20)
        hdr.setStyleSheet(
            "background:#252525;color:#aaa;font-size:11px;"
            "padding-left:6px;border-bottom:1px solid #333;"
        )
        layout.addWidget(hdr)

        self._fig     = None
        self._subplot = None
        self._zoom_filter = None

        # One histogram line + one refractory line per cluster slot
        self._hist_lines:   list = [None] * _MAX_CLUS
        self._refrac_line:  object = None   # single shared refractory marker

        self._x_max_s = _X_MAX_S
        self._n_bins  = _N_BINS
        self._last_spike_times: dict[int, np.ndarray] = {}

        try:
            import fastplotlib as fpl
            self._fig = fpl.Figure(canvas="qt")
            sp = self._fig[0, 0]
            try:
                sp.camera = "2d"
                sp.camera.maintain_aspect = False
            except Exception:
                pass
            try:
                sp.axes.visible = True
                sp.title.visible = False
                slim_subplot(sp)
            except Exception:
                pass
            self._subplot = sp
            self._fig.show()
            canvas = self._fig.canvas
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumSize(160, 80)

            # Row: [rotated "density" label] [canvas]
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(0)
            y_lbl = _AxisLabel("density", vertical=True)
            row.addWidget(y_lbl)
            row.addWidget(canvas)
            layout.addLayout(row)

            # Bottom x label
            x_lbl = QLabel("ISI (ms)")
            x_lbl.setFixedHeight(14)
            x_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            x_lbl.setStyleSheet("color:#888;font-size:10px;background:transparent;")
            layout.addWidget(x_lbl)
            # PanZoomFilter: right-drag zoom + double-click reset
            self._zoom_filter = install_zoom_filter(self._fig, [sp], self)
            # WheelFilter on top: Ctrl = change x_max, Alt = change n_bins
            # (installed after PanZoomFilter so it gets first crack at wheel events)
            self._wheel_filter = _WheelFilter(self, canvas, self)
            QApplication.instance().installEventFilter(self._wheel_filter)
        except Exception as exc:
            logger.warning("ISIWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_spike_times(self, spike_times: "dict[int, np.ndarray]") -> None:
        """spike_times: {cluster_id → (n_spikes,) float64 seconds}

        Matches phy: data = np.diff(spike_times)
        """
        if self._fig is None:
            return
        self._last_spike_times = dict(spike_times)
        self._rerender()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _rerender(self) -> None:
        try:
            self._render(self._last_spike_times)
        except Exception as exc:
            logger.exception("ISIWidget render failed: %s", exc)

    def _render(self, spike_times: "dict[int, np.ndarray]") -> None:
        sp    = self._subplot
        items = list(spike_times.items())[:_MAX_CLUS]

        # Work in milliseconds throughout so axis ticks read 0–50 ms
        x_max_s  = max(self._x_max_s, 1e-6)
        x_max_ms = x_max_s * 1000.0
        bins_s   = np.linspace(0.0, x_max_s,  self._n_bins + 1)
        bins_ms  = bins_s * 1000.0
        bin_w_s  = bins_s[1] - bins_s[0]   # bin width in seconds (for density norm)

        y_global_max = 0.0

        for slot in range(_MAX_CLUS):
            if slot >= len(items):
                safe_delete(sp, self._hist_lines[slot])
                self._hist_lines[slot] = None
                continue

            cid, times = items[slot]
            # phy: intervals = np.diff(st)
            isis_s = np.diff(np.sort(times.astype(np.float64)))
            isis_s = isis_s[(isis_s > 0) & (isis_s <= x_max_s)]

            counts, _ = np.histogram(isis_s, bins=bins_s)

            # phy: normalize → density (integral = 1, in seconds)
            hist_sum = float(counts.sum()) * bin_w_s
            density  = (counts / hist_sum).astype(np.float32) if hist_sum > 0 \
                       else counts.astype(np.float32)

            y_global_max = max(y_global_max, float(density.max()))

            # Plot in ms on X axis
            xy   = _step_outline(density, bins_ms.astype(np.float32))
            rgba = np.array(cluster_color(slot, alpha=0.8), dtype=np.float32)

            safe_delete(sp, self._hist_lines[slot])
            try:
                self._hist_lines[slot] = sp.add_line(xy, colors=rgba, thickness=1.5)
            except Exception as exc:
                logger.debug("ISIWidget hist line slot %d: %s", slot, exc)

        # Single yellow refractory line shared across clusters
        safe_delete(sp, self._refrac_line)
        self._refrac_line = None
        if y_global_max > 0:
            y_top      = y_global_max * 1.1
            refrac_ms  = _REFRAC_S * 1000.0
            refrac_xy  = np.array(
                [[refrac_ms, 0.0], [refrac_ms, y_top]], dtype=np.float32
            )
            try:
                self._refrac_line = sp.add_line(
                    refrac_xy, colors=(1.0, 0.9, 0.0, 1.0), thickness=2.0
                )
            except Exception as exc:
                logger.debug("ISIWidget refrac line: %s", exc)

        QTimer.singleShot(50, self._fit_camera)

    def _fit_camera(self) -> None:
        if self._subplot is None:
            return
        x_max_ms = max(self._x_max_s, 1e-6) * 1000.0
        try:
            # auto_scale fits the camera to the actual data bounds (correct Y centering)
            self._subplot.auto_scale(maintain_aspect=False)
            # Then lock the X range to [0, x_max_ms] with a small margin
            state = self._subplot.camera.get_state()
            new = dict(state)
            new['position'] = np.array([x_max_ms / 2,
                                        float(state['position'][1]),
                                        float(state['position'][2])])
            new['width'] = x_max_ms * 1.08
            new['zoom']  = 1.0
            self._subplot.camera.set_state(new)
        except Exception:
            try:
                self._subplot.auto_scale()
            except Exception:
                pass
        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass
        if self._zoom_filter is not None:
            self._zoom_filter.save_initial_states()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _step_outline(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Step-function outline (N, 2) — same shape as phy's HistogramVisual."""
    n   = len(counts)
    pts = np.empty((2 * n + 2, 2), dtype=np.float32)
    pts[0] = [edges[0], 0.0]
    for i in range(n):
        pts[2 * i + 1] = [edges[i],     counts[i]]
        pts[2 * i + 2] = [edges[i + 1], counts[i]]
    pts[-1] = [edges[-1], 0.0]
    return pts


# ---------------------------------------------------------------------------
# Wheel filter
# ---------------------------------------------------------------------------

class _AxisLabel(QWidget):
    """Small axis-title label. vertical=True paints text rotated 90° CCW."""

    _STYLE = "color:#888;font-size:10px;"

    def __init__(self, text: str, vertical: bool = False, parent=None):
        super().__init__(parent)
        self._text     = text
        self._vertical = vertical
        self.setStyleSheet(self._STYLE)
        fm = self.fontMetrics()
        text_px = fm.horizontalAdvance(text)
        if vertical:
            self.setFixedWidth(14)
            self.setMinimumHeight(text_px + 8)
        else:
            self.setFixedHeight(14)
            self.setMinimumWidth(text_px + 8)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setFont(self.font())
        p.setPen(QColor("#888888"))
        if self._vertical:
            p.translate(self.width(), self.height())
            p.rotate(-90)
            p.drawText(0, 0, self.height(), self.width(),
                       Qt.AlignmentFlag.AlignCenter, self._text)
        else:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._text)
        p.end()


class _WheelFilter(QObject):
    _STEP = 1.2

    def __init__(self, view: ISIWidget, canvas, parent=None):
        super().__init__(parent)
        self._view   = view
        self._canvas = canvas

    def _over_canvas(self, obj) -> bool:
        w = obj
        while w is not None:
            if w is self._canvas:
                return True
            w = w.parent()
        return False

    def eventFilter(self, obj, event) -> bool:
        if event.type() != QEvent.Type.Wheel:
            return False
        if not self._over_canvas(obj):
            return False
        mods  = event.modifiers()
        delta = event.angleDelta().y()
        if delta == 0:
            return False
        ctrl = bool(mods & Qt.KeyboardModifier.ControlModifier)
        alt  = bool(mods & Qt.KeyboardModifier.AltModifier)
        if not ctrl and not alt:
            return False
        factor = self._STEP ** (delta / 120)
        v = self._view
        if ctrl:
            v._x_max_s = max(0.002, v._x_max_s / factor)
            v._rerender()
        elif alt:
            v._n_bins = max(5, int(v._n_bins * factor))
            v._rerender()
        return True
