"""Namespace bridge to top-level `shared` package."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "shared")]
