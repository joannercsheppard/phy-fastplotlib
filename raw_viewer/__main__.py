"""Entry point: python -m phy_remote.raw_viewer.server"""
from __future__ import annotations

import argparse
import logging
import socket
import sys


def _check_not_login_node() -> None:
    hostname = socket.gethostname()
    if hostname.startswith("login") or "login" in hostname.split(".")[0]:
        print(f"\nERROR: Refusing to run on login node ({hostname}).\n", file=sys.stderr)
        sys.exit(1)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m phy_remote.raw_viewer")
    p.add_argument("raw_file", help="Path to the .raw / .bin file")
    p.add_argument("--n-channels",  type=int,   required=True, help="Total channels in file")
    p.add_argument("--sample-rate", type=float, default=30000.0)
    p.add_argument("--dtype",       default="int16")
    p.add_argument("--port",        type=int,   default=5560)
    p.add_argument("--host",        default="0.0.0.0")
    p.add_argument("--video-file",  default=None, help="Path to video file (optional)")
    p.add_argument("--log-level",   default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    _check_not_login_node()

    from phy_remote.raw_viewer.server import RawServer
    srv = RawServer(
        raw_path=args.raw_file,
        n_channels=args.n_channels,
        sample_rate=args.sample_rate,
        dtype=args.dtype,
        port=args.port,
        host=args.host,
        video_path=args.video_file,
    )
    srv.serve_forever()


if __name__ == "__main__":
    main()
