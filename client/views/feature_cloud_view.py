"""Feature view — 4×4 grid of PC projections (fastplotlib)."""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._zoom import install_zoom_filter
from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete, slim_subplot

logger = logging.getLogger(__name__)
_N_DIMS     = 4
_MAX_POINTS = 10_000


class FeatureViewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._fig  = None
        self._scatter_grid: dict[tuple[int, int], dict[int, object]] = {}
        self._cid_to_slot: dict[int, int] = {}
        self._bg_scatters:  dict[tuple[int, int], object] = {}  # grey background scatter per cell

        try:
            import fastplotlib as fpl
            self._fig = fpl.Figure(shape=(_N_DIMS, _N_DIMS), canvas="qt")
            for r in range(_N_DIMS):
                for c in range(_N_DIMS):
                    sp = self._fig[r, c]
                    try:
                        sp.camera = "2d"
                    except Exception:
                        pass
                    try:
                        sp.axes.visible = False
                        sp.title.visible = False
                        slim_subplot(sp)
                    except Exception:
                        pass
                    self._scatter_grid[(r, c)] = {}
            self._fig.show()   # wires up wgpu render loop
            canvas = self._fig.canvas
            canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            canvas.setMinimumSize(200, 200)
            layout.addWidget(canvas)  # reparents canvas into our widget
            all_subplots = [self._fig[r, c] for r in range(_N_DIMS) for c in range(_N_DIMS)]
            self._zoom_filter = install_zoom_filter(
                self._fig, all_subplots, self, centered_zoom=True
            )
        except Exception as exc:
            logger.warning("FeatureViewWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_background_features(self, arr: "np.ndarray | None") -> None:
        """
        Render grey background scatter on off-diagonal cells.

        arr : (n_spikes, n_channels, n_pcs) float32, or None to clear.
        """
        if self._fig is None:
            return
        # Clear existing background scatters
        for (r, c), sc in list(self._bg_scatters.items()):
            if r != c:
                safe_delete(self._fig[r, c], sc)
        self._bg_scatters.clear()

        if arr is None or arr.ndim != 3 or arr.shape[0] == 0:
            return

        try:
            dims = self._to_dims(arr)  # (n_spikes, 4)
        except Exception as exc:
            logger.debug("set_background_features _to_dims failed: %s", exc)
            return

        grey = np.array([0.5, 0.5, 0.5, 0.3], dtype=np.float32)

        for r in range(_N_DIMS):
            for c in range(_N_DIMS):
                if r == c:
                    continue  # skip diagonal — no spike times for background
                sp = self._fig[r, c]
                x, y = dims[:, c], dims[:, r]
                xy = np.column_stack([x, y]).astype(np.float32)
                if len(xy) == 0:
                    continue
                colors_arr = np.broadcast_to(grey, (len(xy), 4)).copy()
                try:
                    sc = _add_scatter(sp, xy, 2, colors_arr)
                    self._bg_scatters[(r, c)] = sc
                except Exception as exc:
                    logger.debug("set_background_features add_scatter failed: %s", exc)

    def set_feature_data(
        self,
        feature_data: dict[int, np.ndarray],
        spike_times: dict[int, np.ndarray] | None = None,
    ) -> None:
        if self._fig is None:
            return
        if not feature_data:
            # Clear all graphics when called with empty dict
            for (r, c), cell in self._scatter_grid.items():
                for sc in list(cell.values()):
                    safe_delete(self._fig[r, c], sc)
                cell.clear()
            self._cid_to_slot.clear()
            return
        try:
            self._render(feature_data, spike_times or {})
        except Exception as exc:
            logger.exception("FeatureViewWidget render failed: %s", exc)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(
        self,
        feature_data: dict[int, np.ndarray],
        spike_times: dict[int, np.ndarray],
    ) -> None:
        projected = {cid: self._to_dims(arr) for cid, arr in feature_data.items()}

        for r in range(_N_DIMS):
            for c in range(_N_DIMS):
                sp   = self._fig[r, c]
                cell = self._scatter_grid[(r, c)]

                for old_cid in list(cell.keys()):
                    if old_cid not in projected:
                        safe_delete(sp, cell.pop(old_cid))

                for idx, (cid, dims) in enumerate(projected.items()):
                    n = dims.shape[0]
                    if r == c and cid in spike_times:
                        t = spike_times[cid].astype(np.float32)
                        n_pair = min(len(t), n)
                        x, y = t[:n_pair], dims[:n_pair, r]
                    else:
                        x, y = dims[:, c], dims[:, r]

                    if len(x) > _MAX_POINTS:
                        rng = np.random.default_rng(seed=cid * 31 + r * 7 + c)
                        sel = rng.choice(len(x), _MAX_POINTS, replace=False)
                        x, y = x[sel], y[sel]

                    xy = np.column_stack([x, y]).astype(np.float32)
                    rgba = np.array(cluster_color(idx, alpha=0.35), dtype=np.float32)
                    colors_arr = np.broadcast_to(rgba, (len(xy), 4)).copy()

                    slot_changed = self._cid_to_slot.get(cid) != idx
                    if cid in cell and not slot_changed:
                        sc = cell[cid]
                        if sc.data.value.shape[0] == len(xy):
                            xyz = np.zeros((len(xy), 3), dtype=np.float32)
                            xyz[:, :2] = xy
                            sc.data[:] = xyz
                        else:
                            safe_delete(sp, sc)
                            cell[cid] = _add_scatter(sp, xy, 3, colors_arr)
                    else:
                        if cid in cell:
                            safe_delete(sp, cell.pop(cid))
                        cell[cid] = _add_scatter(sp, xy, 3, colors_arr)
                    self._cid_to_slot[cid] = idx

        QTimer.singleShot(50, self._auto_scale)

    def _auto_scale(self) -> None:
        if self._fig is None:
            return
        try:
            for r in range(_N_DIMS):
                for c in range(_N_DIMS):
                    sp = self._fig[r, c]
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
            self._zoom_filter.save_initial_states()
        except Exception as exc:
            logger.debug("FeatureViewWidget auto_scale: %s", exc)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_dims(arr: np.ndarray) -> np.ndarray:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim == 3:
            n_spikes, n_ch, n_pc = a.shape
            cols = [
                a[:, min(ch, n_ch - 1), min(pc, n_pc - 1)]
                for ch, pc in [(0, 0), (1, 0), (0, 1), (1, 1)]
            ]
            return np.column_stack(cols)
        if a.ndim == 2:
            if a.shape[1] < _N_DIMS:
                pad = np.zeros((a.shape[0], _N_DIMS - a.shape[1]), dtype=np.float32)
                a = np.concatenate([a, pad], axis=1)
            return a[:, :_N_DIMS]
        return np.zeros((len(a), _N_DIMS), dtype=np.float32)


def _add_scatter(subplot, xy, size, colors):
    """Add scatter without point outline (edge_width=0)."""
    try:
        return subplot.add_scatter(xy, sizes=size, colors=colors, edge_width=0)
    except TypeError:
        return subplot.add_scatter(xy, sizes=size, colors=colors)


FeatureCloudWidget = FeatureViewWidget
