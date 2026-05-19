"""Namespace bridge to top-level `server` package."""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[2] / "server")]
