"""Shared fastplotlib graphic helpers."""
from __future__ import annotations
import logging

logger = logging.getLogger(__name__)


def slim_subplot(subplot) -> None:
    """Eliminate fastplotlib's built-in dead space around a subplot.

    fastplotlib always reserves space for a title TextGraphic (font_size=16 →
    24 px at the top) and a resize-handle grip (13 px at the bottom), even
    when title.visible = False.  This reclaims that space by zeroing the title
    font size and forcing a viewport recalculation.

    Call once after subplot creation, before the first draw.
    """
    try:
        subplot.frame._title_graphic.font_size = 0
        subplot.frame.reset_viewport()
    except Exception as exc:
        logger.debug("slim_subplot: %s", exc)


def safe_delete(subplot, graphic) -> None:
    """Delete a fastplotlib graphic from a subplot.

    Hides the graphic first (instant visual effect), then calls delete_graphic
    to free GPU memory.  If delete_graphic raises the graphic is already hidden
    so it won't accumulate visually.
    """
    try:
        graphic.visible = False
    except Exception:
        pass
    try:
        subplot.delete_graphic(graphic)
    except Exception as exc:
        logger.debug("delete_graphic failed after hide: %s", exc)
