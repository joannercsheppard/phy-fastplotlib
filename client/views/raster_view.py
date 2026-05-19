"""Raster view — global spike raster matching phy's architecture (fastplotlib).

Architecture (matching phy):
- At startup, fetch subsampled spike times for ALL clusters in one shot.
- Render a single grey scatter (all clusters, static background).
- On selection change, overlay bright-colored scatter(s) for the selected
  cluster(s) — no refetch, just index into the cached per-cluster arrays.

This means the view is populated once and stays live across selection changes.
"""
from __future__ import annotations

import logging

import numpy as np
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QSizePolicy, QVBoxLayout, QWidget

from phy_remote.client.views._zoom import install_zoom_filter
from phy_remote.client.views._colors import cluster_color
from phy_remote.client.views._graphics import safe_delete, slim_subplot

logger = logging.getLogger(__name__)


class RasterWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._fig = None
        self._subplot = None

        # Static grey background scatter (all clusters)
        self._bg_scatter = None

        # Bright overlay scatters for currently selected clusters
        self._sel_scatters: dict[int, object] = {}

        # Per-cluster cached (n, 2) float32 arrays: [time, y_row]
        self._all_xy: dict[int, np.ndarray] = {}

        # cluster_id → y-row (float) and stable color index
        self._cluster_y:   dict[int, float] = {}
        self._cluster_idx: dict[int, int]   = {}

        try:
            import fastplotlib as fpl
            self._fig = fpl.Figure(canvas="qt")
            self._subplot = self._fig[0, 0]
            try:
                self._subplot.camera = "2d"
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
        except Exception as exc:
            logger.warning("RasterWidget: fastplotlib init failed: %s", exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_raster_data(
        self,
        data: np.ndarray,
        cluster_ids: list[int],
        duration: float,
    ) -> None:
        """
        Called once at startup (and after merge/undo) with global raster data.

        Parameters
        ----------
        data        : (n_spikes, 2) float32  [time_s, cluster_id]
        cluster_ids : ordered list of all cluster ids (determines y ordering)
        duration    : recording length in seconds (for x-axis scale)
        """
        if self._fig is None:
            return
        try:
            self._build_raster(data, cluster_ids, duration)
        except Exception as exc:
            logger.exception("RasterWidget.set_raster_data failed: %s", exc)

    def set_selected_clusters(self, cluster_ids: list[int]) -> None:
        """
        Highlight one or more clusters in bright colors.  No data fetch needed.

        Parameters
        ----------
        cluster_ids : list of currently selected cluster ids
        """
        if self._fig is None:
            return
        try:
            self._update_selection(cluster_ids)
        except Exception as exc:
            logger.exception("RasterWidget.set_selected_clusters failed: %s", exc)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _build_raster(
        self,
        data: np.ndarray,
        cluster_ids: list[int],
        duration: float,
    ) -> None:
        sp = self._subplot

        # Remove any existing scatters
        safe_delete(sp, self._bg_scatter)
        self._bg_scatter = None
        for sc in self._sel_scatters.values():
            safe_delete(sp, sc)
        self._sel_scatters.clear()
        self._all_xy.clear()
        self._cluster_y.clear()

        if len(data) == 0 or not cluster_ids:
            return

        # Assign each cluster a y-row and a stable color index
        self._cluster_y   = {int(cid): float(idx) for idx, cid in enumerate(cluster_ids)}
        self._cluster_idx = {int(cid): idx         for idx, cid in enumerate(cluster_ids)}

        # Split data into per-cluster (time, y) arrays and cache them
        times_col   = data[:, 0]
        cid_col_raw = data[:, 1]

        bg_xys:    list[np.ndarray] = []
        bg_colors: list[np.ndarray] = []
        for idx, cid in enumerate(cluster_ids):
            mask = cid_col_raw == float(cid)
            if not mask.any():
                continue
            t  = times_col[mask]
            y  = np.full(len(t), self._cluster_y[cid], dtype=np.float32)
            xy = np.column_stack([t, y]).astype(np.float32)
            self._all_xy[cid] = xy
            rgba = np.array(cluster_color(idx, alpha=0.4), dtype=np.float32)
            bg_xys.append(xy)
            bg_colors.append(np.broadcast_to(rgba, (len(xy), 4)).copy())

        if not bg_xys:
            return

        all_xy     = np.concatenate(bg_xys,    axis=0)
        all_colors = np.concatenate(bg_colors, axis=0)
        self._bg_scatter = sp.add_scatter(all_xy, sizes=2, colors=all_colors)

        QTimer.singleShot(50, self._auto_scale)

    def _update_selection(self, cluster_ids: list[int]) -> None:
        sp = self._subplot

        # Remove old selection overlays
        for sc in list(self._sel_scatters.values()):
            safe_delete(sp, sc)
        self._sel_scatters.clear()

        for cid in cluster_ids:
            xy = self._all_xy.get(cid)
            if xy is None or len(xy) == 0:
                continue
            # Use the same global color index as the background so colours match
            global_idx = self._cluster_idx.get(cid, 0)
            rgba = np.array(cluster_color(global_idx, alpha=1.0), dtype=np.float32)
            colors_arr = np.broadcast_to(rgba, (len(xy), 4)).copy()
            self._sel_scatters[cid] = sp.add_scatter(xy, sizes=4, colors=colors_arr)

        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass

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
                'width':           abs(state['width'])  * 1.05,
                'height':          abs(state['height']) * 1.1,
                'depth':           state['depth'],
                'zoom':            1.0,
                'maintain_aspect': False,
            })
            self._fig.canvas.request_draw()
            self._zoom_filter.save_initial_states()
        except Exception as exc:
            logger.debug("RasterWidget auto_scale: %s", exc)
