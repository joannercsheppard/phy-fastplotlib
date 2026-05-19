"""Template features view — 2D projection scatter (fastplotlib).

Works for 1 or more selected clusters:
- 1 cluster : scatter template-feature col 0 vs col 1 (the two strongest
              template projections for each spike).
- 2+ clusters: project every cluster's spikes onto two reference directions
              (the mean feature vector of cluster 0 and cluster 1), then
              scatter.  Extra clusters beyond the first two are still plotted
              using the same reference axes.
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
_MAX_POINTS = 5_000


class TemplateFeaturesWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._fig = None
        self._subplot = None
        self._scatter_graphics: dict[int, object] = {}
        self._cid_to_slot: dict[int, int] = {}

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
            logger.warning("TemplateFeaturesWidget: fastplotlib init failed: %s", exc)

    def set_feature_data(self, feature_data: dict[int, np.ndarray]) -> None:
        if self._fig is None:
            return
        if not feature_data:
            for sc in list(self._scatter_graphics.values()):
                safe_delete(self._subplot, sc)
            self._scatter_graphics.clear()
            self._cid_to_slot.clear()
            return
        try:
            self._render(feature_data)
        except Exception as exc:
            logger.exception("TemplateFeaturesWidget render failed: %s", exc)

    def _render(self, feature_data: dict[int, np.ndarray]) -> None:
        sp = self._subplot
        items = list(feature_data.items())

        # Flatten each cluster's feature array to (n_spikes, n_features)
        vecs_list = [self._flatten(arr) for _, arr in items]

        # Build two reference directions for the 2-D projection
        if len(items) == 1:
            # Single cluster: project onto the first two feature columns
            n_feat = vecs_list[0].shape[1]
            r0 = np.zeros(n_feat, dtype=np.float32); r0[0] = 1.0
            r1 = np.zeros(n_feat, dtype=np.float32)
            if n_feat >= 2:
                r1[1] = 1.0
            else:
                # Only one feature dimension — use spike index as y
                r1 = None
        else:
            # 2+ clusters: project onto mean feature vectors of first two clusters
            r0 = vecs_list[0].mean(axis=0).astype(np.float32)
            r1 = vecs_list[1].mean(axis=0).astype(np.float32)
            norm0 = float(np.linalg.norm(r0))
            norm1 = float(np.linalg.norm(r1))
            r0 /= norm0 if norm0 > 1e-12 else 1.0
            r1 /= norm1 if norm1 > 1e-12 else 1.0

        # Project and subsample
        projected: dict[int, np.ndarray] = {}
        for (cid, _), vecs in zip(items, vecs_list):
            x = (vecs @ r0).astype(np.float32)
            if r1 is None:
                y = np.arange(len(vecs), dtype=np.float32)
            else:
                y = (vecs @ r1).astype(np.float32)
            xy = np.column_stack([x, y]).astype(np.float32)
            if len(xy) > _MAX_POINTS:
                rng = np.random.default_rng(seed=cid)
                xy = xy[rng.choice(len(xy), _MAX_POINTS, replace=False)]
            projected[cid] = xy

        # Remove graphics for clusters no longer selected
        for cid in list(self._scatter_graphics.keys()):
            if cid not in projected:
                safe_delete(sp, self._scatter_graphics.pop(cid))
                self._cid_to_slot.pop(cid, None)

        # Add / update scatter per cluster
        for idx, (cid, xy) in enumerate(projected.items()):
            rgba = np.array(cluster_color(idx, alpha=0.5), dtype=np.float32)
            colors_arr = np.broadcast_to(rgba, (len(xy), 4)).copy()

            slot_changed = self._cid_to_slot.get(cid) != idx
            if cid in self._scatter_graphics and not slot_changed:
                sc = self._scatter_graphics[cid]
                if sc.data.value.shape[0] == len(xy):
                    xyz = np.zeros((len(xy), 3), dtype=np.float32)
                    xyz[:, :2] = xy
                    sc.data[:] = xyz
                else:
                    safe_delete(sp, sc)
                    self._scatter_graphics[cid] = sp.add_scatter(xy, sizes=2, colors=colors_arr)
            else:
                if cid in self._scatter_graphics:
                    safe_delete(sp, self._scatter_graphics.pop(cid))
                self._scatter_graphics[cid] = sp.add_scatter(xy, sizes=2, colors=colors_arr)
            self._cid_to_slot[cid] = idx

        QTimer.singleShot(50, self._auto_scale)

    @staticmethod
    def _flatten(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            return arr[:, np.newaxis]
        if arr.ndim > 2:
            return arr.reshape(arr.shape[0], -1)
        return arr

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
            self._zoom_filter.save_initial_states()
        except Exception as exc:
            logger.debug("TemplateFeaturesWidget auto_scale: %s", exc)
