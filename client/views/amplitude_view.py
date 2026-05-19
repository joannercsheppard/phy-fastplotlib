"""Amplitude-over-time widget — GPU rendering via fastplotlib.

Matches phy's AmplitudeView (phy/cluster/views/amplitude.py):

  ScatterVisual            → sp.add_scatter  (one graphic per cluster)
  HistogramVisual (right)  → sp.add_line     (step-function, one per cluster)
  PatchVisual (time bar)   → sp.add_line     (two yellow verticals for window edges)
  on_mouse_click Alt       → time_clicked signal → TraceWidget.go_to_time
  quantile = 0.99          → same clipping for y-axis bounds
  y_min = min(0, m)        → zero always visible

Not ported (requires fetching all-cluster spike data over the network):
  Background grey spikes from other clusters on the same channel.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QApplication, QLabel, QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._zoom import install_zoom_filter
from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete, slim_subplot

logger = logging.getLogger(__name__)

# phy constants
_QUANTILE        = 0.99    # phy: quantile = 0.99
_N_BINS          = 100     # phy: n_bins   = 100
_HIST_ALPHA      = 0.4     # phy: histogram_alpha = 0.5
_SCATTER_ALPHA   = 0.8
_SCATTER_SIZE    = 6
_HIST_WIDTH_FRAC = 0.10    # histogram bar width as fraction of time range
_HIST_GAP_FRAC   = 0.015   # gap between scatter right edge and histogram
_TIME_BAR_COLOR  = (1.0, 1.0, 0.0, 0.30)   # phy: time_range_color = (1,1,0,.25)


class AmplitudeWidget(QWidget):

    # Emitted on Alt+click with the clicked time in seconds.
    # Connect to TraceWidget.go_to_time  (phy: emit('select_time', self, time))
    time_clicked = pyqtSignal(float)

    def __init__(self, title: str = "Template amplitude", parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        hdr = QLabel(title)
        hdr.setFixedHeight(20)
        hdr.setStyleSheet(
            "background:#252525;color:#aaa;font-size:11px;"
            "padding-left:6px;border-bottom:1px solid #333;"
        )
        layout.addWidget(hdr)

        self._fig:     Any = None
        self._subplot: Any = None
        self._fpl_ready = False

        self._scatter_graphics: dict[int, Any] = {}
        self._hist_graphics:    dict[int, Any] = {}
        self._cid_to_slot:      dict[int, int] = {}
        self._time_bar_graphics: list[Any] = []
        self._bg_scatter: Any = None   # grey background spikes (other clusters, same channel)

        # Cached bounds for Alt+click coordinate conversion
        self._t_min = 0.0
        self._t_max = 1.0
        self._y_min = 0.0
        self._y_max = 1.0

        try:
            import fastplotlib as fpl
            self._fig     = fpl.Figure(canvas="qt")
            self._subplot = self._fig[0, 0]
            try:
                self._subplot.camera = "2d"
            except Exception:
                pass
            try:
                self._subplot.axes.visible = False
                self._subplot.title.visible = False
                slim_subplot(self._subplot)
            except Exception:
                pass
            self._fig.show()
            canvas = self._fig.canvas
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumSize(160, 120)
            layout.addWidget(canvas)
            self._zoom_filter = install_zoom_filter(self._fig, [self._subplot], self)
            try:
                self._fig.canvas.add_event_handler(self._on_pointer_down, "pointer_down")
            except Exception:
                pass
            self._fpl_ready = True
        except Exception as exc:
            logger.warning("AmplitudeWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_background_spike_data(self, data: np.ndarray) -> None:
        """Render grey background scatter for other clusters on the same channel."""
        if not self._fpl_ready:
            return
        sp = self._subplot
        safe_delete(sp, self._bg_scatter)
        self._bg_scatter = None

        if data is None or len(data) == 0:
            return

        xy = data[:, :2].astype(np.float32)
        try:
            self._bg_scatter = sp.add_scatter(
                xy, sizes=_SCATTER_SIZE - 1,
                colors=np.array([0.5, 0.5, 0.5, 0.3], dtype=np.float32),
            )
            # Push behind cluster scatter in depth so it never occludes it
            try:
                self._bg_scatter.world_object.local.position = (0, 0, -1)
            except Exception:
                pass
        except Exception as exc:
            logger.debug("AmplitudeWidget background scatter: %s", exc)
        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass

    def set_spike_data(self, spike_data: "dict[int, np.ndarray]") -> None:
        """spike_data: {cluster_id → (n_spikes, 2) array [time_s, amplitude]}"""
        if self._fig is None or not spike_data:
            return
        # Clear stale background scatter — new background will arrive shortly
        safe_delete(self._subplot, self._bg_scatter)
        self._bg_scatter = None
        try:
            self._render(spike_data)
        except Exception as exc:
            logger.exception("AmplitudeWidget render failed: %s", exc)

    def show_time_range(self, t_start: float, t_end: float) -> None:
        """Highlight the trace view's current window with yellow vertical bars.

        Matches phy's AmplitudeView.show_time_range / PatchVisual.
        """
        if not self._fpl_ready:
            return
        sp = self._subplot
        for g in self._time_bar_graphics:
            safe_delete(sp, g)
        self._time_bar_graphics.clear()

        y0, y1 = self._y_min, self._y_max
        if y0 == y1:
            return
        for t in (t_start, t_end):
            try:
                g = sp.add_line(
                    np.array([[t, y0], [t, y1]], dtype=np.float32),
                    colors=_TIME_BAR_COLOR,
                    thickness=2.0,
                )
                self._time_bar_graphics.append(g)
            except Exception as exc:
                logger.debug("AmplitudeWidget time bar: %s", exc)
        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self, spike_data: "dict[int, np.ndarray]") -> None:
        sp = self._subplot

        # Remove graphics for clusters no longer selected
        for cid in list(self._scatter_graphics.keys()):
            if cid not in spike_data:
                safe_delete(sp, self._scatter_graphics.pop(cid))
                safe_delete(sp, self._hist_graphics.pop(cid, None))
                self._cid_to_slot.pop(cid, None)

        # --- Data bounds  (phy: _get_data_bounds, quantile = 0.99) ---
        all_amps  = np.concatenate([d[:, 1] for d in spike_data.values()])
        all_times = np.concatenate([d[:, 0] for d in spike_data.values()])

        y_lo = float(np.quantile(all_amps, 1.0 - _QUANTILE))
        y_hi = float(np.quantile(all_amps, _QUANTILE))
        y_lo = min(0.0, y_lo)          # phy: m = min(0, m)
        y_hi = max(y_hi, y_lo + 1e-9)

        t_lo = float(all_times.min())
        t_hi = float(all_times.max())
        if t_hi <= t_lo:
            t_hi = t_lo + 1.0

        self._t_min, self._t_max = t_lo, t_hi
        self._y_min, self._y_max = y_lo, y_hi

        dt       = t_hi - t_lo
        hist_x0  = t_hi + dt * _HIST_GAP_FRAC
        hist_w   = dt * _HIST_WIDTH_FRAC

        # --- Per-cluster scatter + histogram ---
        for idx, (cid, data) in enumerate(spike_data.items()):
            times = data[:, 0].astype(np.float32)
            amps  = data[:, 1].astype(np.float32)
            xy    = np.column_stack([times, amps]).astype(np.float32)
            rgba  = np.array(cluster_color(idx, alpha=_SCATTER_ALPHA), dtype=np.float32)
            colors_arr = np.broadcast_to(rgba, (len(xy), 4)).copy()

            slot_changed = self._cid_to_slot.get(cid) != idx

            # Scatter  (phy: ScatterVisual)
            if cid in self._scatter_graphics and not slot_changed:
                sc = self._scatter_graphics[cid]
                try:
                    if sc.data.value.shape[0] == len(xy):
                        xyz = np.zeros((len(xy), 3), dtype=np.float32)
                        xyz[:, :2] = xy
                        sc.data[:] = xyz
                    else:
                        safe_delete(sp, self._scatter_graphics.pop(cid))
                        self._scatter_graphics[cid] = _add_scatter(
                            sp, xy, _SCATTER_SIZE, colors_arr)
                except Exception:
                    safe_delete(sp, self._scatter_graphics.pop(cid, None))
                    self._scatter_graphics[cid] = _add_scatter(
                        sp, xy, _SCATTER_SIZE, colors_arr)
            else:
                if cid in self._scatter_graphics:
                    safe_delete(sp, self._scatter_graphics.pop(cid))
                self._scatter_graphics[cid] = _add_scatter(
                    sp, xy, _SCATTER_SIZE, colors_arr)
            self._cid_to_slot[cid] = idx

            # Histogram step-function  (phy: HistogramVisual, rotated 90°, right edge)
            safe_delete(sp, self._hist_graphics.pop(cid, None))
            pts = _histogram_line(amps, y_lo, y_hi, hist_x0, hist_w)
            if pts is not None:
                hist_rgba = cluster_color(idx, alpha=_HIST_ALPHA)
                try:
                    self._hist_graphics[cid] = sp.add_line(
                        pts, colors=hist_rgba, thickness=1.5)
                except Exception as exc:
                    logger.debug("AmplitudeWidget histogram: %s", exc)

        # Fit camera to show scatter + histogram  (phy: panzoom zoom=.75, pan=-.25)
        self._fit_camera(t_lo, t_hi + hist_w + dt * _HIST_GAP_FRAC, y_lo, y_hi)
        self._zoom_filter.save_initial_states()

    def _fit_camera(self, x0: float, x1: float, y0: float, y1: float) -> None:
        try:
            sp    = self._subplot
            mid_x = (x0 + x1) / 2
            mid_y = (y0 + y1) / 2
            state = sp.camera.get_state()
            sp.camera.set_state({
                'position':        np.array([mid_x, mid_y, state['position'][2]]),
                'fov':             0.0,
                'width':           (x1 - x0) * 1.04,
                'height':          (y1 - y0) * 1.15,
                'depth':           state['depth'],
                'zoom':            1.0,
                'maintain_aspect': False,
            })
            self._fig.canvas.request_draw()
        except Exception as exc:
            logger.debug("AmplitudeWidget fit_camera: %s", exc)

    # ------------------------------------------------------------------
    # Alt+click → select_time  (phy: on_mouse_click with 'Alt' in modifiers)
    # ------------------------------------------------------------------

    def _on_pointer_down(self, event) -> None:
        if getattr(event, "button", None) != 1:
            return
        mods = QApplication.keyboardModifiers()
        if not (mods & Qt.KeyboardModifier.AltModifier):
            return
        x_px = getattr(event, "x", None)
        if x_px is None:
            return
        self.time_clicked.emit(self._pixel_to_time(float(x_px)))

    def _pixel_to_time(self, x_px: float) -> float:
        """Convert canvas x pixel → data time (seconds)."""
        try:
            w_px  = self._fig.canvas.width()
            if w_px <= 0:
                return self._t_min
            state = self._subplot.camera.get_state()
            cx    = float(state['position'][0])
            cw    = float(state['width'])
            ndc_x = 2.0 * x_px / w_px - 1.0   # [-1, +1]
            return cx + ndc_x * cw / 2.0
        except Exception:
            return self._t_min


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _add_scatter(subplot, xy, size, colors):
    """Add scatter without point outline (edge_width=0)."""
    try:
        return subplot.add_scatter(xy, sizes=size, colors=colors, edge_width=0)
    except TypeError:
        return subplot.add_scatter(xy, sizes=size, colors=colors)


def _histogram_line(
    amps: np.ndarray,
    y_min: float,
    y_max: float,
    x_start: float,
    x_width: float,
) -> "np.ndarray | None":
    """Step-function polygon tracing the amplitude histogram.

    Matches phy's _compute_histogram (normalize=True, ignore_zeros=True) +
    HistogramVisual rendered rotated 90° at the right edge of the scatter.

    Returns (N, 2) float32 [x, y] or None if nothing to draw.
    """
    data = amps[amps != 0]
    if len(data) == 0:
        return None

    bins = np.linspace(y_min, y_max, _N_BINS + 1)
    counts, _ = np.histogram(data, bins=bins)

    # phy: normalize by integral (density), then we re-normalise to [0,1] for width
    hist_sum = counts.sum() * (bins[1] - bins[0])
    if hist_sum > 0:
        counts = counts / hist_sum
    max_c = counts.max()
    if max_c == 0:
        return None
    counts = counts / max_c   # scale to [0, 1] for display width

    # Build step outline (counterclockwise):
    #   (x_start, bins[0])
    #   for each bin i:  → (x_start + c[i]*W, bins[i])
    #                    ↑ (x_start + c[i]*W, bins[i+1])
    #   (x_start, bins[-1])
    n   = len(counts)
    pts = np.empty((2 * n + 2, 2), dtype=np.float32)
    pts[0] = [x_start, bins[0]]
    for i in range(n):
        x = x_start + float(counts[i]) * x_width
        pts[2 * i + 1] = [x, bins[i]]
        pts[2 * i + 2] = [x, bins[i + 1]]
    pts[-1] = [x_start, bins[-1]]
    return pts
