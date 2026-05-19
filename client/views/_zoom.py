"""
Phy-compatible pan/zoom event filter for fastplotlib Qt canvases.

Gestures (matching phy's PanZoom):
  Scroll (no mod)   → zoom both axes toward cursor
  Ctrl/Cmd + scroll → zoom Y axis only
  Shift + scroll    → zoom X axis only
  Right drag        → zoom X (horizontal) and Y (vertical) independently,
                      anchored at the drag-start point
  Double-click      → reset to initial view (saved when fit_camera is called)
  Left drag         → pan  (handled by fastplotlib's built-in controller)

On macOS, Qt maps Command (⌘) → ControlModifier.
"""
from __future__ import annotations

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt
from PyQt6.QtWidgets import QApplication


class PanZoomFilter(QObject):
    """Phy-compatible pan/zoom filter for a fastplotlib figure."""

    _SCROLL_STEP = 1.12   # zoom factor per scroll notch (120 delta units)
    _DRAG_SCALE  = 8.0    # right-drag sensitivity (higher = faster zoom)

    def __init__(self, fig, subplots: list, canvas, parent=None,
                 centered_zoom: bool = False):
        super().__init__(parent)
        self._fig          = fig
        self._subplots     = subplots
        self._canvas       = canvas
        self._centered_zoom = centered_zoom   # zoom around each subplot's own centre
        self._initial_states: dict[int, dict] = {}
        # Right-drag state
        self._rdrag_pos:    tuple[float, float] | None = None
        self._rdrag_states: list[dict | None]  | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_initial_states(self) -> None:
        """Save current camera states for double-click reset.
        Call this at the end of fit_camera after data has been plotted.
        """
        self._initial_states.clear()
        for i, sp in enumerate(self._subplots):
            try:
                state = sp.camera.get_state()
                self._initial_states[i] = _copy_state(state)
            except Exception:
                pass

    def reset(self) -> None:
        """Restore all subplots to the saved initial camera state."""
        for i, sp in enumerate(self._subplots):
            if i not in self._initial_states:
                continue
            try:
                sp.camera.set_state(self._initial_states[i])
            except Exception:
                pass
        self._request_draw()

    # ------------------------------------------------------------------
    # Qt event filter
    # ------------------------------------------------------------------

    def eventFilter(self, obj, event) -> bool:
        if not self._over_canvas(obj):
            return False

        t = event.type()

        if t == QEvent.Type.Wheel:
            return self._on_wheel(event)

        if t == QEvent.Type.MouseButtonDblClick:
            if event.button() == Qt.MouseButton.LeftButton:
                self.reset()
                return True

        if t == QEvent.Type.MouseButtonPress:
            if event.button() == Qt.MouseButton.RightButton:
                pos = event.position()
                self._rdrag_pos = (pos.x(), pos.y())
                self._rdrag_states = [
                    _copy_state_safe(sp) for sp in self._subplots
                ]
                return True  # consume — prevents Qt context menu

        if t == QEvent.Type.MouseMove and self._rdrag_pos is not None:
            return self._on_right_drag(event)

        if t == QEvent.Type.MouseButtonRelease:
            if event.button() == Qt.MouseButton.RightButton:
                self._rdrag_pos   = None
                self._rdrag_states = None
                return True

        return False

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_wheel(self, event) -> bool:
        delta = event.angleDelta().y()
        if delta == 0:
            return False

        mods  = event.modifiers()
        ctrl  = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        factor = self._SCROLL_STEP ** (delta / 120.0)

        if self._centered_zoom:
            ndc_x, ndc_y = 0.0, 0.0
        else:
            try:
                pos = event.position()
                ndc_x, ndc_y = self._px_to_ndc(pos.x(), pos.y())
            except Exception:
                ndc_x, ndc_y = 0.0, 0.0

        for sp in self._subplots:
            if ctrl and not shift:
                _zoom_toward(sp, 0.0, ndc_y, 1.0, factor)
            elif shift and not ctrl:
                _zoom_toward(sp, ndc_x, 0.0, factor, 1.0)
            else:
                _zoom_toward(sp, ndc_x, ndc_y, factor, factor)

        self._request_draw()
        return True  # always consume — prevents fastplotlib double-handling

    def _on_right_drag(self, event) -> bool:
        if self._rdrag_pos is None or not self._rdrag_states:
            return False

        pos = event.position()
        dx  =  (pos.x() - self._rdrag_pos[0])  # right → zoom in X
        dy  = -(pos.y() - self._rdrag_pos[1])  # up    → zoom in Y (screen Y-down flip)

        factor_x = self._SCROLL_STEP ** (dx * self._DRAG_SCALE / 120.0)
        factor_y = self._SCROLL_STEP ** (dy * self._DRAG_SCALE / 120.0)

        for i, sp in enumerate(self._subplots):
            if i >= len(self._rdrag_states) or self._rdrag_states[i] is None:
                continue
            try:
                init = self._rdrag_states[i]
                new  = _copy_state(init)
                new['width']  = abs(float(init['width']))  / factor_x
                new['height'] = abs(float(init['height'])) / factor_y
                new['zoom']   = 1.0
                sp.camera.set_state(new)
            except Exception:
                pass

        self._request_draw()
        return True

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _over_canvas(self, obj) -> bool:
        w = obj
        while w is not None:
            if w is self._canvas:
                return True
            w = w.parent()
        return False

    def _px_to_ndc(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert canvas pixel position → NDC [-1, +1] (Y-up)."""
        cw = max(self._fig.canvas.width(),  1)
        ch = max(self._fig.canvas.height(), 1)
        ndc_x =  2.0 * x_px / cw - 1.0
        ndc_y = -(2.0 * y_px / ch - 1.0)  # flip: screen Y-down → NDC Y-up
        return ndc_x, ndc_y

    def _request_draw(self) -> None:
        try:
            self._fig.canvas.request_draw()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Module-level helpers (shared by views that do their own camera work)
# ---------------------------------------------------------------------------

def _copy_state(state: dict) -> dict:
    return {k: (np.array(v) if isinstance(v, np.ndarray) else v)
            for k, v in state.items()}


def _copy_state_safe(sp) -> dict | None:
    try:
        return _copy_state(sp.camera.get_state())
    except Exception:
        return None


def _zoom_toward(sp, ndc_x: float, ndc_y: float,
                 factor_x: float, factor_y: float) -> None:
    """Zoom a subplot, keeping the NDC point (ndc_x, ndc_y) fixed in view."""
    try:
        state = sp.camera.get_state()
        w  = abs(float(state['width']))
        h  = abs(float(state['height']))
        cx = float(state['position'][0])
        cy = float(state['position'][1])

        new_w = w / factor_x
        new_h = h / factor_y
        # Shift center so the data point under the cursor stays fixed
        new_cx = cx + ndc_x * (w - new_w) / 2.0
        new_cy = cy + ndc_y * (h - new_h) / 2.0

        new = dict(state)
        new['width']    = new_w
        new['height']   = new_h
        new['position'] = np.array([new_cx, new_cy, float(state['position'][2])])
        new['zoom']     = 1.0
        sp.camera.set_state(new)
    except Exception:
        pass


def install_zoom_filter(fig, subplots: list, parent,
                        centered_zoom: bool = False) -> PanZoomFilter:
    """Create and register the filter at app level. Call once per view.

    centered_zoom=True: zoom around each subplot's own centre (good for grids).
    centered_zoom=False (default): zoom toward the cursor position.
    """
    filt = PanZoomFilter(fig, subplots, fig.canvas, parent,
                         centered_zoom=centered_zoom)
    QApplication.instance().installEventFilter(filt)
    return filt
