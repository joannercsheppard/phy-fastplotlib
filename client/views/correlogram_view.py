"""Correlogram view — auto/cross-correlograms (fastplotlib).

Computed locally on the client from spike times already fetched for ISI.
No extra server round-trip needed.

Rendering details (matching phy):
- Auto-correlogram (diagonal): full cluster colour
- Cross-correlogram (off-diagonal): desaturated blend of both colours
- Refractory period markers: two vertical grey lines at ±2 ms
- Per-cell max-normalisation (phy default)

Interactive:
- Ctrl+scroll  : widen / narrow the lag window
- Alt+scroll   : increase / decrease bin size (finer / coarser)
"""
from __future__ import annotations

import logging
import threading

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication, QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete, slim_subplot

logger = logging.getLogger(__name__)

# phy defaults
_DEFAULT_BIN_MS    = 1.0    # 1 ms
_DEFAULT_WINDOW_MS = 50.0   # 50 ms total window
_REFRACTORY_MS     = 2.0    # 2 ms refractory period
_MAX_CLUSTERS      = 2      # fixed 2×2 figure


class CorrelogramWidget(QWidget):
    _ccg_ready = pyqtSignal(object, object, float, float)  # ccg, cluster_ids, bin_s, window_s

    def __init__(self, parent=None):
        super().__init__(parent)
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(0, 0, 0, 0)

        self._fig = None
        self._line_grid:    dict[tuple[int, int], object] = {}
        self._ref_line_keys: dict[tuple[int, int], list]  = {}

        self._bin_ms    = _DEFAULT_BIN_MS
        self._window_ms = _DEFAULT_WINDOW_MS

        # Cached spike times: {cluster_id → (n_spikes,) float64 seconds}
        self._spike_times: dict[int, np.ndarray] = {}
        self._cluster_ids: list[int] = []

        # Signal delivers CCG result from worker thread → main thread safely
        self._ccg_ready.connect(self._render)

        # Debounce for scroll-triggered recomputes
        self._recompute_timer = QTimer(self)
        self._recompute_timer.setSingleShot(True)
        self._recompute_timer.setInterval(150)
        self._recompute_timer.timeout.connect(self._recompute)

        try:
            import fastplotlib as fpl
            self._fig = fpl.Figure(shape=(_MAX_CLUSTERS, _MAX_CLUSTERS), canvas="qt")
            for r in range(_MAX_CLUSTERS):
                for c in range(_MAX_CLUSTERS):
                    sp = self._fig[r, c]
                    try:
                        sp.camera = "2d"
                        sp.axes.visible = False
                        sp.title.visible = False
                        slim_subplot(sp)
                    except Exception:
                        pass
            self._fig.show()
            canvas = self._fig.canvas
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumSize(160, 120)
            self._layout.addWidget(canvas)
            self._wheel_filter = _CCGWheelFilter(self, canvas, self)
            QApplication.instance().installEventFilter(self._wheel_filter)
        except Exception as exc:
            logger.warning("CorrelogramWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_spike_times(self, spike_times: dict[int, np.ndarray]) -> None:
        """Called by main window with full spike times for selected clusters."""
        self._spike_times = dict(spike_times)
        self._cluster_ids = list(spike_times.keys())
        self._recompute()

    # ------------------------------------------------------------------
    # Compute + render
    # ------------------------------------------------------------------

    def _recompute(self) -> None:
        if not self._spike_times or self._fig is None:
            return
        cluster_ids = list(self._spike_times.keys())[:_MAX_CLUSTERS]
        bin_s    = self._bin_ms    / 1000.0
        window_s = self._window_ms / 1000.0

        spike_times = {cid: self._spike_times[cid] for cid in cluster_ids}

        def _worker():
            try:
                ccg = _compute_ccg(spike_times, bin_s, window_s)
                self._ccg_ready.emit(ccg, cluster_ids, bin_s, window_s)
            except Exception as exc:
                logger.warning("CorrelogramWidget compute failed: %s", exc, exc_info=True)

        threading.Thread(target=_worker, daemon=True).start()

    def _render(
        self,
        ccg: np.ndarray,
        cluster_ids: list[int],
        bin_size: float,
        window_size: float,
    ) -> None:
        try:
            self._render_inner(ccg, cluster_ids, bin_size, window_size)
        except Exception as exc:
            logger.warning("CorrelogramWidget render failed: %s", exc, exc_info=True)

    def _render_inner(
        self,
        ccg: np.ndarray,
        cluster_ids: list[int],
        bin_size: float,
        window_size: float,
    ) -> None:
        if self._fig is None:
            return

        n_cl   = len(cluster_ids)
        n_bins = ccg.shape[2] if (ccg.ndim == 3 and ccg.shape[0] > 0) else 0
        if n_bins == 0:
            return

        window_ms = window_size * 1000.0
        edges = np.linspace(-window_ms / 2, window_ms / 2, n_bins + 1, dtype=np.float32)

        active_keys: set[tuple[int, int]] = set()

        for i in range(min(n_cl, _MAX_CLUSTERS)):
            for j in range(min(n_cl, _MAX_CLUSTERS)):
                sp  = self._fig[i, j]
                key = (i, j)
                active_keys.add(key)

                counts = ccg[i, j, :].astype(np.float32)

                # Per-cell max-normalisation
                m = counts.max()
                if m > 0:
                    counts = counts / m

                # Zero the centre bin for auto-correlogram
                if i == j:
                    counts[n_bins // 2] = 0.0

                xy = _step_hist(counts, edges)

                if i == j:
                    color = cluster_color(i, alpha=0.85)
                else:
                    r0, g0, b0, _ = cluster_color(i, alpha=1.0)
                    r1, g1, b1, _ = cluster_color(j, alpha=1.0)
                    grey = 0.5
                    blend = 0.1
                    color = (
                        r0 * blend + grey * (1 - blend),
                        (g0 + g1) / 2 * blend + grey * (1 - blend),
                        b1 * blend + grey * (1 - blend),
                        0.75,
                    )

                if key in self._line_grid:
                    old = self._line_grid[key]
                    if len(xy) > 1 and old.data.value.shape[0] == len(xy):
                        xyz = np.zeros((len(xy), 3), dtype=np.float32)
                        xyz[:, :2] = xy
                        old.data[:] = xyz
                    else:
                        safe_delete(sp, old)
                        self._line_grid.pop(key)
                        if len(xy) > 1:
                            self._line_grid[key] = sp.add_line(xy, colors=color, thickness=1.5)
                elif len(xy) > 1:
                    self._line_grid[key] = sp.add_line(xy, colors=color, thickness=1.5)

                self._update_refractory_lines(sp, key, _REFRACTORY_MS, window_ms)

        for key in list(self._line_grid.keys()):
            if key not in active_keys:
                i, j = key
                safe_delete(self._fig[i, j], self._line_grid.pop(key))

        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass
        QTimer.singleShot(50, lambda: self._fit_cameras(window_ms))

    def _update_refractory_lines(self, sp, key, ref_ms, window_ms):
        for g in self._ref_line_keys.get(key, []):
            safe_delete(sp, g)
        self._ref_line_keys[key] = []
        grey = (0.6, 0.6, 0.6, 0.6)
        for x in (-ref_ms, ref_ms):
            if abs(x) < window_ms / 2:
                try:
                    g = sp.add_line(
                        np.array([[x, 0.0], [x, 1.05]], dtype=np.float32),
                        colors=grey, thickness=1.0,
                    )
                    self._ref_line_keys[key].append(g)
                except Exception:
                    pass

    def _fit_cameras(self, window_ms: float) -> None:
        if self._fig is None:
            return
        half = window_ms / 2.0
        for r in range(_MAX_CLUSTERS):
            for c in range(_MAX_CLUSTERS):
                try:
                    sp    = self._fig[r, c]
                    state = sp.camera.get_state()
                    sp.camera.set_state({
                        'position':        np.array([0.0, 0.55, float(state['position'][2])]),
                        'fov':             0.0,
                        'width':           half * 2.0 * 1.12,
                        'height':          1.20,
                        'depth':           state['depth'],
                        'zoom':            1.0,
                        'maintain_aspect': False,
                    })
                except Exception as exc:
                    logger.debug("CorrelogramWidget fit_cameras [%d,%d]: %s", r, c, exc)
        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Wheel filter
# ---------------------------------------------------------------------------

class _CCGWheelFilter(QObject):
    _STEP = 1.3

    def __init__(self, view, canvas, parent=None):
        super().__init__(parent)
        self._view   = view
        self._canvas = canvas

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel:
            return False
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
        alt  = bool(mods & Qt.KeyboardModifier.AltModifier)
        if not ctrl and not alt:
            return False

        factor = self._STEP ** (delta / 120)
        v = self._view

        if ctrl:
            v._window_ms = max(10.0, min(500.0, v._window_ms / factor))
        elif alt:
            v._bin_ms = max(0.5, min(10.0, v._bin_ms * (1.0 / factor)))
            v._bin_ms = min(v._bin_ms, v._window_ms / 4)

        v._recompute_timer.start()
        return True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_ccg(
    spike_times: dict[int, np.ndarray],
    bin_size: float,
    window_size: float,
) -> np.ndarray:
    """Compute (n_cl, n_cl, n_bins) CCG array from spike time dicts.

    Tries phylib's fast C implementation first; falls back to pure numpy.
    """
    cluster_ids = list(spike_times.keys())
    n_cl  = len(cluster_ids)
    n_bins = max(1, round(window_size / bin_size))

    # Merge all spikes into two arrays required by phylib
    all_times:    list[np.ndarray] = []
    all_clusters: list[np.ndarray] = []
    for cid, times in spike_times.items():
        t = np.asarray(times, dtype=np.float64)
        all_times.append(t)
        all_clusters.append(np.full(len(t), cid, dtype=np.int32))

    st = np.concatenate(all_times)
    sc = np.concatenate(all_clusters)
    order = np.argsort(st)
    st = st[order]
    sc = sc[order]

    try:
        from phylib.stats import correlograms as _ccg_fn
        ccg = _ccg_fn(
            st, sc,
            sample_rate=20_000.0,   # value only affects unit conversion inside phylib;
                                    # times are already in seconds so any rate works
            cluster_ids=cluster_ids,
            bin_size=bin_size,
            window_size=window_size,
        )
        return np.asarray(ccg, dtype=np.float32)
    except Exception as exc:
        logger.debug("phylib CCG failed (%s), using numpy fallback", exc)

    # Pure numpy fallback
    ccg = np.zeros((n_cl, n_cl, n_bins), dtype=np.float32)
    cid_to_idx = {cid: i for i, cid in enumerate(cluster_ids)}
    half = window_size / 2.0

    for i, cid_i in enumerate(cluster_ids):
        ti = st[sc == cid_i]
        for j, cid_j in enumerate(cluster_ids):
            tj = st[sc == cid_j]
            if len(ti) == 0 or len(tj) == 0:
                continue
            # For each spike in ti, collect lags to all spikes in tj within ±half
            lags = []
            for t in ti:
                lo = np.searchsorted(tj, t - half)
                hi = np.searchsorted(tj, t + half)
                lags.append(tj[lo:hi] - t)
            if lags:
                all_lags = np.concatenate(lags)
                counts, _ = np.histogram(
                    all_lags,
                    bins=np.linspace(-half, half, n_bins + 1),
                )
                ccg[i, j] = counts.astype(np.float32)

    return ccg


def _step_hist(counts: np.ndarray, edges: np.ndarray) -> np.ndarray:
    x_stair = np.repeat(edges, 2)[1:-1]
    y_stair = np.repeat(counts, 2).astype(np.float32)
    x_full  = np.concatenate([[edges[0]], x_stair, [edges[-1]]]).astype(np.float32)
    y_full  = np.concatenate([[0.0], y_stair, [0.0]]).astype(np.float32)
    return np.column_stack([x_full, y_full])
