"""Namespace bridge to top-level `raw_viewer` package."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "raw_viewer")]
