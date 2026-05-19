"""
Server entry point — run as:

    python -m phy_remote.server /path/to/params.py

Tunnel strategy (same pattern as Janelia Jupyter jobs)
-------------------------------------------------------
Bind to 0.0.0.0 on the compute node, then forward from the login node to
the compute node over plain TCP.  No SSH keys between nodes required.

  Compute node:   python -m phy_remote.server params.py
                  (binds to 0.0.0.0:5555 by default)

    MacBook:        ssh -N -L 5555:<compute_node>:5555 <user>@login1.int.janelia.org
                  python -m phy_remote.client.app --port 5555
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import sys

logger = logging.getLogger(__name__)


def _extract_spike_waveforms_if_needed(model, params_path) -> None:
    """
    Pre-extract spike waveforms to _phy_spikes_subset.waveforms.npy if not present.

    This makes subsequent waveform requests near-instant (sequential read from a
    small pre-built file rather than random seeks into the raw .bin).
    Runs in the background after model load — server stays responsive throughout.
    """
    from pathlib import Path
    data_dir = Path(params_path).parent
    waveforms_path = data_dir / "_phy_spikes_subset.waveforms.npy"

    if waveforms_path.exists():
        logger.info("Spike waveforms already cached: %s", waveforms_path)
        return

    if getattr(model, "traces", None) is None:
        logger.warning("Cannot pre-extract waveforms: no raw traces file available")
        return

    import time
    import threading

    def _run():
        # Wait before starting — avoid competing with early cluster clicks for NFS bandwidth
        time.sleep(30)
        logger.info("Pre-extracting spike waveforms → %s …", waveforms_path)
        t0 = time.time()
        try:
            model.save_spikes_subset_waveforms(max_n_spikes_per_template=100)
            mb = waveforms_path.stat().st_size / 1e6
            logger.info(
                "Spike waveforms cached in %.1f s (%.0f MB) → %s",
                time.time() - t0, mb, waveforms_path,
            )
            # Hot-load so subsequent waveform requests use the fast pre-extracted path
            try:
                model.spike_waveforms = model._load_spike_waveforms()
                logger.info(
                    "spike_waveforms hot-loaded (%d spikes) — waveform view now instant",
                    len(model.spike_waveforms.spike_ids),
                )
            except Exception as exc2:
                logger.debug("Could not hot-load spike_waveforms after extraction: %s", exc2)
        except Exception as exc:
            logger.warning("Spike waveform pre-extraction failed: %s", exc)

    threading.Thread(target=_run, daemon=True).start()


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m phy_remote.server",
        description="phy-remote ZMQ server — runs headless on the HPC cluster",
    )
    p.add_argument(
        "params_path",
        help="Path to params.py (Kilosort / phy-format dataset)",
    )
    p.add_argument("--port", type=int, default=5555, help="ZMQ port (default: 5555)")
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="Interface to bind (default: 0.0.0.0)",
    )
    p.add_argument(
        "--login-node",
        default="login1.int.janelia.org",
        help="Login node hostname, used to print the tunnel command",
    )
    p.add_argument(
        "--max-datasets",
        type=int,
        default=0,
        help="Limit number of datasets shown in the switcher (0 = all)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def _check_not_login_node() -> None:
    """Refuse to run if the hostname looks like a Janelia login node."""
    hostname = socket.gethostname()
    is_login = hostname.startswith("login") or "login" in hostname.split(".")[0]
    if is_login:
        print(
            f"\nERROR: Refusing to run on login node ({hostname}).\n\n"
            f"  Start an interactive job first:\n\n"
            f"    bsub -Is -q interactive -n 1 -R \"rusage[mem=8000]\" bash\n\n"
            f"  Then re-run this command.\n",
            file=sys.stderr,
        )
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )

    _check_not_login_node()

    # ------------------------------------------------------------------ #
    # Load the TemplateModel
    # ------------------------------------------------------------------ #
    from pathlib import Path
    params_path = Path(args.params_path).resolve()
    if not params_path.exists():
        logger.error("params.py not found: %s", params_path)
        sys.exit(1)

    try:
        from phylib.io.model import load_model
    except ImportError:
        logger.error(
            "phylib is not installed in this environment.  "
            "Activate the phy conda environment first."
        )
        sys.exit(1)

    # If a directory was given, find the first params.py using direct iterdir (no rglob)
    if params_path.is_dir():
        found = None
        try:
            for letter_path in sorted(params_path.iterdir()):
                if not letter_path.is_dir() or len(letter_path.name) != 1:
                    continue
                for shank_path in sorted(letter_path.iterdir()):
                    if not shank_path.is_dir() or not shank_path.name.startswith("shank_"):
                        continue
                    for sorter_dir in sorted(shank_path.iterdir()):
                        pp = sorter_dir / "params.py"
                        if pp.exists():
                            found = pp
                            break
                    if found:
                        break
                if found:
                    break
        except Exception as exc:
            logger.error("Could not scan directory %s: %s", params_path, exc)
            sys.exit(1)
        if not found:
            logger.error("No params.py found under %s", params_path)
            sys.exit(1)
        logger.info("Directory given — using first dataset: %s", found)
        params_path = found

    # Log key file sizes so user knows what to expect
    data_dir = params_path.parent
    big_files = ["templates.npy", "spike_times.npy", "spike_clusters.npy",
                 "pc_features.npy", "template_features.npy"]
    for fname in big_files:
        p = data_dir / fname
        if p.exists():
            mb = p.stat().st_size / 1e6
            logger.info("  %-30s  %.0f MB", fname, mb)

    # ------------------------------------------------------------------ #
    # Start server immediately (model loads in background)
    # ------------------------------------------------------------------ #
    from phy_remote.server.server import PhyServer

    server = PhyServer(model=None, port=args.port, host=args.host,
                       params_path=str(params_path),
                       max_datasets=args.max_datasets)

    def _load_model_bg():
        import time
        logger.info("Loading model from %s …", params_path)
        t0 = time.time()
        try:
            model = load_model(params_path)
        except Exception as exc:
            logger.error("Failed to load model: %s", exc)
            return
        elapsed = time.time() - t0
        server.model = model
        server._model_ready = True
        # Defer _init_mutable_spike_clusters to first merge — avoids copying
        # the full spike_clusters memmap (can be 100s of MB) at startup.
        logger.info("Model ready in %.1f s", elapsed)

        # If waveforms were pre-extracted in a previous run, reload them now.
        # phylib._load_spike_waveforms() requires all three sidecar files:
        #   _phy_spikes_subset.waveforms.npy
        #   _phy_spikes_subset.channels.npy
        #   _phy_spikes_subset.spikes.npy
        from pathlib import Path as _Path
        _data_dir = _Path(params_path).parent
        _sidecar_files = [
            _data_dir / "_phy_spikes_subset.waveforms.npy",
            _data_dir / "_phy_spikes_subset.channels.npy",
            _data_dir / "_phy_spikes_subset.spikes.npy",
        ]
        if all(f.exists() for f in _sidecar_files) and getattr(model, "spike_waveforms", None) is None:
            try:
                model.spike_waveforms = model._load_spike_waveforms()
                logger.info(
                    "Reloaded cached spike waveforms (%d spikes) from %s",
                    len(model.spike_waveforms.spike_ids), _data_dir,
                )
            except Exception as exc:
                logger.debug("Could not reload spike waveforms: %s", exc)
        elif not all(f.exists() for f in _sidecar_files):
            missing = [f.name for f in _sidecar_files if not f.exists()]
            logger.debug("Spike waveform sidecar files missing: %s — will extract from raw traces", missing)
        dat_path    = getattr(model, "dat_path", None) or getattr(model, "path", "?")
        traces_shape = getattr(model, "traces", None)
        traces_shape = traces_shape.shape if traces_shape is not None else "?"
        logger.info(
            "Model loaded: %d clusters, %d spikes, %d probe channels",
            len(model.cluster_ids), len(model.spike_times), model.n_channels,
        )
        logger.info("Binary file: %s  traces shape: %s", dat_path, traces_shape)

        # Pre-extract spike waveforms if not already done — makes waveform
        # view near-instant on subsequent runs (sequential read vs random .bin seeks)
        _extract_spike_waveforms_if_needed(model, params_path)

        # Wait a few seconds so the client's first burst of requests (cluster_info,
        # channel_positions, best_channels) are handled before we consume CPU/RAM
        # for pre-warming.  The template-map bincount is O(n_spikes) and allocates
        # a large temporary array — doing it while the client is idle avoids competing.
        time.sleep(5)

        # Warm the cluster→template map — O(n_spikes) bincount, cached for the session.
        # Powers _best_template_for_cluster (O(1) lookup) and _get_channel_to_clusters.
        try:
            t1 = time.time()
            server._get_cluster_template_map()
            logger.info("Cluster→template map cached in %.1f s", time.time() - t1)
        except Exception as exc:
            logger.debug("Could not pre-warm template map: %s", exc)

        # Warm the channel→clusters map (O(n_unique_templates) reads — fast).
        try:
            t1 = time.time()
            server._get_channel_to_clusters()
            logger.info("Channel→cluster map cached in %.1f s", time.time() - t1)
        except Exception as exc:
            logger.debug("Could not pre-warm channel map: %s", exc)

        # Pre-warm the spike_ids cache using a 1%-random subsample of all spikes.
        # Most views (waveforms, features, correlogram) only need ~100 spike_ids per
        # cluster; a 1% sample gives ~1900 ids for a typical 190K-spike cluster —
        # more than enough.  get_spike_times (full raster) needs all spikes: it will
        # miss the cache on first call, do one O(n_spikes) full scan, then cache the
        # full list so subsequent calls are instant.
        # Peak RAM: ~8 MB (vs ~1 GB for the full argsort).  Takes < 1 s.
        try:
            import numpy as np
            t1 = time.time()
            sc = np.asarray(model.spike_clusters)          # memmap → read sequentially
            n_spikes = len(sc)
            logger.info(
                "Pre-warming spike_ids cache via 1%% sample (%d clusters, %d spikes) …",
                len(model.cluster_ids), n_spikes,
            )
            rng  = np.random.default_rng(seed=0)
            idx  = np.sort(rng.choice(n_spikes, size=max(1, n_spikes // 100), replace=False))
            sc_s = sc[idx]                                 # sampled cluster labels
            del sc
            order  = np.argsort(sc_s, kind="stable")      # sort the tiny sample
            unique, counts = np.unique(sc_s, return_counts=True)
            splits = np.split(idx[order], np.cumsum(counts)[:-1])
            del sc_s, order
            n_warmed = 0
            for cid, ids in zip(unique.tolist(), splits):
                cid = int(cid)
                if cid not in server._spike_ids_cache:
                    server._spike_ids_cache[cid] = ids.astype(np.int64)
                    server._spike_ids_sampled.add(cid)   # mark as subsample
                    n_warmed += 1
            del splits
            logger.info(
                "spike_ids cache warm: %d clusters, %.0f ids/cluster avg, in %.1f s",
                n_warmed,
                (n_spikes // 100) / max(1, n_warmed),
                time.time() - t1,
            )
        except Exception as exc:
            logger.warning("spike_ids pre-warm failed: %s", exc)

    import threading as _threading
    _threading.Thread(target=_load_model_bg, daemon=True).start()

    hostname = socket.gethostname()
    user = os.environ.get("USER", "<user>")
    login_node = args.login_node

    print(f"PHY_REMOTE_READY host={hostname} port={args.port}", flush=True)
    print(
        f"\n  On your MacBook:\n\n"
        f"    ssh -N -L {args.port}:{hostname}:{args.port} "
        f"{user}@{login_node}\n\n"
        f"    python -m phy_remote.client.app --port {args.port}\n",
        file=sys.stderr,
        flush=True,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
