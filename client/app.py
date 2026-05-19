"""
phy-remote client entry point.

Run with:
    python -m phy_remote.client.app --port 5555

Or via the installed console script:
    phy-remote --port 5555

Prerequisites
-------------
1.  Start the server on the cluster:
        python -m phy_remote.server.server --params /path/to/params.py

2.  Open the SSH tunnel on your MacBook:
        ssh -N -L 5555:localhost:5555 user@cluster

3.  Launch the client (this file):
        python -m phy_remote.client.app

Dependencies (client-side only):
    pip install fastplotlib[notebook] wgpu PyQt6 pyzmq numpy
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="phy-remote",
        description="phy-remote client — renders spike-sorting data from a remote cluster",
    )
    p.add_argument("--host", default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    p.add_argument("--port", type=int, default=5555, help="ZMQ port (default: 5555)")
    p.add_argument(
        "--n-spikes",
        type=int,
        default=100,
        metavar="N",
        help="Max spikes to fetch per cluster for display (default: 100)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    # ------------------------------------------------------------------ #
    # 1.  Verify dependencies before starting Qt
    # ------------------------------------------------------------------ #
    try:
        from PyQt6.QtWidgets import QApplication
    except ImportError:
        logger.error(
            "PyQt6 is not installed.  Run: pip install PyQt6"
        )
        sys.exit(1)

    try:
        import fastplotlib  # noqa — imported here only to fail early
    except ImportError:
        logger.warning(
            "fastplotlib not installed — waveform rendering will show a placeholder.  "
            "Run: pip install fastplotlib[notebook] wgpu"
        )

    # ------------------------------------------------------------------ #
    # 2.  Connect to server
    # ------------------------------------------------------------------ #
    from phy_remote.client.transport import PhyTransport, TransportError

    logger.info("Connecting to %s:%d …", args.host, args.port)
    try:
        transport = PhyTransport(host=args.host, port=args.port, timeout_ms=60_000)
        if not transport.ping():
            logger.error("Server did not respond to ping")
            transport.close()
            sys.exit(1)
        logger.info("Server reachable — ping OK")
    except Exception as exc:
        logger.error("Cannot reach server: %s  (is the tunnel running?)", exc)
        try:
            transport.close()
        except Exception:
            pass
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # 3.  Launch Qt GUI
    # ------------------------------------------------------------------ #
    app = QApplication.instance() or QApplication(sys.argv)
    # macOS: use native style
    app.setStyle("macos") if sys.platform == "darwin" else None

    from phy_remote.client.main_window import MainWindow

    try:
        window = MainWindow(transport=transport, n_spikes=args.n_spikes)
        window.show()
    except Exception as exc:
        logger.exception("Failed to initialize main window: %s", exc)
        transport.close()
        sys.exit(1)

    exit_code = app.exec()

    # Close ZMQ transport before the process exits.
    transport.close()

    # Prefer normal interpreter shutdown so Qt/plugin errors remain visible.
    # Keep an opt-in hard-exit mode for environments that still hit the
    # known rendercanvas shutdown crash.
    if os.environ.get("PHY_REMOTE_FORCE_OS_EXIT", "0") == "1":
        os._exit(exit_code or 0)
    sys.exit(exit_code or 0)


if __name__ == "__main__":
    main()
