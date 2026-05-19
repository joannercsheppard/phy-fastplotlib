"""Shared cluster colour palette.

Index 0 = primary (blue), index 1 = compare (red), then green / orange.
Each view picks its own alpha but uses the same hue per cluster index.
"""
from __future__ import annotations

# (R, G, B) — no alpha
_RGB = [
    (0.20, 0.55, 0.95),   # 0 primary  — blue
    (0.92, 0.20, 0.20),   # 1 compare  — red
    (0.15, 0.72, 0.35),   # 2          — green
    (0.92, 0.58, 0.08),   # 3          — orange
]


def cluster_color(idx: int, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """Return (R, G, B, A) for cluster slot *idx*."""
    r, g, b = _RGB[idx % len(_RGB)]
    return (r, g, b, alpha)
