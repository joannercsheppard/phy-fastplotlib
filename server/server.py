"""
ZMQ REP server — runs headless on the HPC cluster.

Usage
-----
    from phy_remote.server.server import PhyServer
    server = PhyServer(model=my_template_model, port=5555)
    server.serve_forever()

The server is intentionally synchronous (one request at a time) to keep the
code simple and because phy's model layer is not thread-safe.  If latency
becomes an issue, move to a ROUTER/DEALER pattern later.
"""

import json
import logging
import signal
import socket
import threading

import numpy as np
import zmq

from phy_remote.shared.protocol import (
    CMD_PING,
    CMD_GET_WAVEFORMS,
    CMD_GET_SPIKE_TIMES,
    CMD_GET_FEATURES,
    CMD_GET_TEMPLATES,
    CMD_GET_CLUSTER_IDS,
    CMD_GET_CLUSTER_INFO,
    CMD_LABEL_CLUSTER,
    CMD_GET_SPIKE_DATA,
    CMD_GET_CHANNEL_POSITIONS,
    CMD_GET_TRACES,
    CMD_GET_SPIKES_IN_WINDOW,
    CMD_GET_SIMILAR_CLUSTERS,
    CMD_GET_TEMPLATE_FEATURES,
    CMD_GET_CLUSTER_BEST_CHANNELS,
    CMD_GET_BACKGROUND_SPIKE_DATA,
    CMD_GET_FEATURE_SPIKE_DATA,
    CMD_GET_CORRELOGRAMS,
    CMD_GET_RASTER_DATA,
    CMD_GET_BACKGROUND_FEATURES,
    CMD_GET_DATASET_LIST,
    CMD_SWITCH_DATASET,
    CMD_MERGE,
    CMD_UNDO,
    CMD_REDO,
    CMD_SAVE,
    decode_request,
    encode_response,
)

logger = logging.getLogger(__name__)
_TOP_WAVEFORM_CHANNELS = 10


def _safe_float(s: str) -> float:
    try:
        return round(float(s), 4)
    except (ValueError, TypeError):
        return 0.0


class PhyServer:
    """
    Minimal ZMQ REP server wrapping a phy TemplateModel.

    Parameters
    ----------
    model : TemplateModel or any object with the expected attributes
        The data source.  Pass *None* to start without a model (useful for
        testing the transport layer only — only CMD_PING will work).
    port : int
        TCP port to bind.  SSH-tunnel this to the client.
    host : str
        Interface to bind.  Defaults to loopback; change to "0.0.0.0" only
        on a trusted network.
    """

    def __init__(self, model=None, port: int = 5555, host: str = "127.0.0.1",
                 params_path: "str | None" = None, max_datasets: int = 0):
        self.model = model
        self.host = host
        self._running = False
        self._model_ready = model is not None  # False while background-loading

        # Dataset switching: {label: Path}
        self._datasets: "dict[str, object]" = {}  # label → Path
        self._current_dataset: str = ""
        if params_path is not None:
            self._discover_datasets(params_path, max_datasets=max_datasets)  # fast — filesystem scan only

        # Per-cluster result cache — avoids re-reading NFS files for recently seen clusters.
        # Invalidated on merge/undo/redo/dataset switch.
        # Keys: cluster_id (int). Values: dict of cached response bytes per command.
        self._cluster_cache: "dict[int, dict[str, list[bytes]]]" = {}

        # Per-cluster spike_ids cache — each model.get_cluster_spikes() call is O(n_spikes).
        # Caching means the scan only happens once per cluster per session, not once per handler.
        # Invalidated on merge/undo/redo/dataset switch.
        self._spike_ids_cache: "dict[int, np.ndarray]" = {}
        # Cluster IDs whose cache entry is a pre-warm subsample (not the full list).
        # _get_all_spike_ids() checks this set and recomputes when needed (e.g. spike_times).
        self._spike_ids_sampled: "set[int]" = set()

        # Global caches invalidated by merge/undo/redo/switch.
        self._ch_to_cl_cache: "dict[int, list[int]] | None" = None
        self._spike_counts_cache: "tuple[np.ndarray, np.ndarray] | None" = None
        self._cluster_tmpl_map_cache: "dict[int, int] | None" = None

        # Undo/redo stacks for merge operations.
        # Each entry: (spike_clusters_copy, cluster_groups_copy)
        self._undo_stack: list[tuple[np.ndarray, dict]] = []
        self._redo_stack: list[tuple[np.ndarray, dict]] = []

        # Keep a writable copy of spike_clusters so merges are in-memory.
        # We try to shadow the model attribute so all existing handlers pick
        # up the merged version automatically.
        if model is not None:
            self._init_mutable_spike_clusters()

        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)

        # Try the requested port; if it's taken walk up until one is free.
        for candidate in range(port, port + 20):
            try:
                self._sock.bind(f"tcp://{host}:{candidate}")
                self.port = candidate
                break
            except zmq.ZMQError:
                continue
        else:
            raise OSError(f"No free port found in range {port}–{port + 19}")

        logger.info("PhyServer bound to tcp://%s:%d", host, self.port)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def serve_forever(self) -> None:
        """
        Block and serve requests until stop() is called or a signal arrives.

        The socket is created, polled, and destroyed entirely within this
        method so it is always owned by a single thread (ZMQ requirement).
        """
        self._running = True
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)

        import os
        hostname = socket.getfqdn()
        logger.info("PhyServer ready (pid %d, host %s)", os.getpid(), hostname)
        # Machine-parseable ready token (also emitted by __main__.py before
        # calling serve_forever, but repeated here for library use)
        print(f"PHY_REMOTE_READY host={hostname} port={self.port}", flush=True)

        poller = zmq.Poller()
        poller.register(self._sock, zmq.POLLIN)
        try:
            while self._running:
                events = dict(poller.poll(timeout=100))  # 100 ms poll interval
                if self._sock not in events:
                    continue
                frames = self._sock.recv_multipart()
                self._sock.send_multipart(self._dispatch(frames))
        finally:
            self._sock.close(linger=0)
            logger.info("PhyServer loop exited")

    def stop(self) -> None:
        """Thread-safe: signal the serve_forever loop to exit."""
        self._running = False

    def close(self) -> None:
        """Stop the server and release the ZMQ context.

        Call *after* the thread running serve_forever has been joined so the
        socket is guaranteed closed before ctx.term().
        """
        self._running = False
        self._ctx.term()
        logger.info("PhyServer context terminated")

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _dispatch(self, frames: list[bytes]) -> list[bytes]:
        try:
            req = decode_request(frames)
        except Exception as exc:
            return encode_response(status="error", error=f"bad request: {exc}")

        cmd = req.get("cmd", "")
        logger.debug("cmd=%s args=%s", cmd, {k: v for k, v in req.items() if k != "cmd"})

        # Respond immediately while model is still loading so client socket never sticks
        if not self._model_ready and cmd not in (CMD_PING, CMD_GET_DATASET_LIST):
            return encode_response(status="loading", error="model still loading")

        handler = {
            CMD_PING: self._handle_ping,
            CMD_GET_WAVEFORMS: self._handle_get_waveforms,
            CMD_GET_SPIKE_TIMES: self._handle_get_spike_times,
            CMD_GET_FEATURES: self._handle_get_features,
            CMD_GET_TEMPLATES: self._handle_get_templates,
            CMD_GET_CLUSTER_IDS: self._handle_get_cluster_ids,
            CMD_GET_CLUSTER_INFO: self._handle_get_cluster_info,
            CMD_LABEL_CLUSTER: self._handle_label_cluster,
            CMD_GET_SPIKE_DATA: self._handle_get_spike_data,
            CMD_GET_CHANNEL_POSITIONS: self._handle_get_channel_positions,
            CMD_GET_TRACES: self._handle_get_traces,
            CMD_GET_SPIKES_IN_WINDOW: self._handle_get_spikes_in_window,
            CMD_GET_SIMILAR_CLUSTERS: self._handle_get_similar_clusters,
            CMD_GET_TEMPLATE_FEATURES: self._handle_get_template_features,
            CMD_GET_CLUSTER_BEST_CHANNELS: self._handle_get_cluster_best_channels,
            CMD_GET_BACKGROUND_SPIKE_DATA: self._handle_get_background_spike_data,
            CMD_GET_FEATURE_SPIKE_DATA:    self._handle_get_feature_spike_data,
            CMD_GET_CORRELOGRAMS:          self._handle_get_correlograms,
            CMD_GET_RASTER_DATA: self._handle_get_raster_data,
            CMD_GET_BACKGROUND_FEATURES: self._handle_get_background_features,
            CMD_GET_DATASET_LIST: self._handle_get_dataset_list,
            CMD_SWITCH_DATASET:   self._handle_switch_dataset,
            CMD_MERGE: self._handle_merge,
            CMD_UNDO:  self._handle_undo,
            CMD_REDO:  self._handle_redo,
            CMD_SAVE:  self._handle_save,
        }.get(cmd)

        if handler is None:
            return encode_response(status="error", error=f"unknown command: {cmd!r}")

        try:
            return handler(req)
        except Exception as exc:
            logger.exception("error handling %s", cmd)
            return encode_response(status="error", error=str(exc))

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _handle_ping(self, req: dict) -> list[bytes]:
        return encode_response(pong=True)

    def _handle_get_waveforms(self, req: dict) -> list[bytes]:
        """
        Return waveforms for a cluster, restricted to template channels.

        Reads only the ~10-20 channels near the probe site rather than all
        channels — same as phy does locally, and essential for long recordings
        where seeking across all channels is prohibitively slow.

        Request fields
        --------------
        cluster_id : int
        n_spikes   : int, optional  (default 50)
        """
        self._require_model()
        cluster_id = int(req["cluster_id"])
        n_spikes = int(req.get("n_spikes", 50))
        return self._cluster_cached(
            cluster_id, f"waveforms_{n_spikes}",
            lambda: self._do_get_waveforms(cluster_id, n_spikes),
        )

    def _do_get_waveforms(self, cluster_id: int, n_spikes: int) -> list[bytes]:
        # Get top channels for this cluster's dominant template.
        _, ch_top, _ = self._template_waveform_and_channels_for_cluster(
            cluster_id, top_n=_TOP_WAVEFORM_CHANNELS
        )
        channel_ids = np.asarray(ch_top, dtype=np.int64) if ch_top else None

        all_spike_ids = self._get_spike_ids_cached(cluster_id)

        # If pre-extracted spike waveforms exist, restrict to spike_ids that are
        # present in the pre-extracted subset — otherwise get_waveforms() raises
        # AssertionError (requested ids not in subset) and falls back to raw
        # traces, which fails if traces is None.
        sw = getattr(self.model, "spike_waveforms", None)
        if sw is not None and hasattr(sw, "spike_ids"):
            avail = np.intersect1d(all_spike_ids, sw.spike_ids)
            if len(avail) > 0:
                spike_ids = avail[:n_spikes]
            else:
                # No pre-extracted spikes for this cluster — fall back to raw
                spike_ids = all_spike_ids[:n_spikes]
        else:
            spike_ids = all_spike_ids
            if len(spike_ids) > n_spikes:
                rng = np.random.default_rng(seed=cluster_id)
                spike_ids = rng.choice(spike_ids, size=n_spikes, replace=False)
                spike_ids.sort()

        # shape: (n_spikes, n_samples, n_template_channels)
        waveforms = self.model.get_waveforms(spike_ids, channel_ids)
        if waveforms is None:
            return encode_response(status="error", error="no waveforms available")

        arr = np.asarray(waveforms, dtype=np.float32)

        # Match phy's _get_waveforms_with_n_spikes processing:
        # 1. Median subtraction across time (axis=1) — removes DC offset per spike per channel
        arr = arr - np.median(arr, axis=1, keepdims=True)

        # 2. Zero-phase 3rd-order Butterworth 150 Hz high-pass filter (phy's add_default_filter)
        try:
            from scipy.signal import butter, lfilter
            sr = float(self.model.sample_rate)
            b, a = butter(3, 150.0 / (sr / 2.0), btype="high")
            arr = lfilter(b, a, arr, axis=1).astype(np.float32)
            arr = np.flip(arr, axis=1)
            arr = lfilter(b, a, arr, axis=1).astype(np.float32)
            arr = np.flip(arr, axis=1).copy()
        except Exception as exc:
            logger.warning("Waveform HP filter failed: %s", exc)
        ch_list = channel_ids.tolist() if channel_ids is not None else []
        return encode_response(
            array=arr,
            cluster_id=cluster_id,
            n_spikes=len(spike_ids),
            channel_ids=ch_list,
        )

    def _handle_get_spike_times(self, req: dict) -> list[bytes]:
        """Return spike times (seconds) for a cluster — needs ALL spikes, not a sample."""
        self._require_model()
        cluster_id = int(req["cluster_id"])
        # The pre-warm stores only a 1% sample in _spike_ids_cache so that views like
        # waveforms and features start instantly.  Spike-times needs the full list
        # for correct ISI / raster.  Check whether the cached entry looks complete
        # (≥ 95% of expected count via spike_counts); if not, do a full scan and
        # replace the cache entry so subsequent calls are instant.
        spike_ids = self._get_all_spike_ids(cluster_id)
        times = np.asarray(self.model.spike_times[spike_ids], dtype=np.float64)
        return encode_response(array=times, cluster_id=cluster_id)

    def _handle_get_features(self, req: dict) -> list[bytes]:
        """Return PC features for a cluster."""
        self._require_model()
        cluster_id = int(req["cluster_id"])
        return self._cluster_cached(
            cluster_id, "features",
            lambda: self._do_get_features(cluster_id),
        )

    def _do_get_features(self, cluster_id: int) -> list[bytes]:
        spike_ids = self._get_spike_ids_cached(cluster_id)
        # Match Phy's behavior: use top channels for this cluster.
        _, ch_top, _ = self._template_waveform_and_channels_for_cluster(
            cluster_id, top_n=_TOP_WAVEFORM_CHANNELS
        )
        channel_ids = np.asarray(ch_top, dtype=np.int64) if ch_top else None
        features = self.model.get_features(spike_ids, channel_ids)
        if features is None:
            return encode_response(status="error", error="no features available")
        arr = np.asarray(getattr(features, "data", features), dtype=np.float32)
        ch_list = channel_ids.tolist() if channel_ids is not None else []
        return encode_response(array=arr, cluster_id=cluster_id, channel_ids=ch_list)

    def _handle_get_template_features(self, req: dict) -> list[bytes]:
        """Return template_features for a cluster (KiloSort template_features.npy path).

        Returns available=False (no array) when template_features.npy is absent,
        so the client can silently skip rather than treating it as an error.
        """
        self._require_model()
        cluster_id = int(req["cluster_id"])
        return self._cluster_cached(
            cluster_id, "template_features",
            lambda: self._do_get_template_features(cluster_id),
        )

    def _do_get_template_features(self, cluster_id: int) -> list[bytes]:
        if not hasattr(self.model, "get_template_features"):
            return encode_response(available=False, cluster_id=cluster_id)
        spike_ids = self._get_spike_ids_cached(cluster_id)
        tf = self.model.get_template_features(spike_ids)
        if tf is None:
            return encode_response(available=False, cluster_id=cluster_id)
        arr = np.asarray(tf, dtype=np.float32)
        return encode_response(array=arr, available=True, cluster_id=cluster_id)

    def _handle_get_templates(self, req: dict) -> list[bytes]:
        """
        Return the mean template waveform for a cluster.

        Request fields
        --------------
        cluster_id : int

        Response array shape: (n_samples, n_channels), float32
        """
        self._require_model()
        cluster_id = int(req["cluster_id"])
        return self._cluster_cached(
            cluster_id, "templates",
            lambda: self._do_get_templates(cluster_id),
        )

    def _do_get_templates(self, cluster_id: int) -> list[bytes]:
        # Use dominant template + top channels (Phy-like best channels).
        arr, channel_ids, best_ch = self._template_waveform_and_channels_for_cluster(
            cluster_id, top_n=_TOP_WAVEFORM_CHANNELS
        )
        if arr is None or channel_ids is None:
            return encode_response(status="error", error="no template available")
        return encode_response(
            array=arr, cluster_id=cluster_id,
            channel_ids=channel_ids, best_ch=best_ch,
        )

    def _handle_get_cluster_ids(self, req: dict) -> list[bytes]:
        """Return the list of all cluster ids as a 1-D int32 array."""
        self._require_model()
        cluster_ids = np.asarray(self.model.cluster_ids, dtype=np.int32)
        return encode_response(array=cluster_ids)

    def _handle_get_cluster_info(self, req: dict) -> list[bytes]:
        """
        Return a summary table for all clusters.

        Response header contains a 'clusters' list, each entry:
          { id, label, n_spikes, amplitude, fr }

        No array frame — all data fits in the JSON header.
        """
        self._require_model()
        m = self.model

        # spike counts per cluster — cached bincount, O(n) only on first call
        spike_clusters, bc = self._get_spike_counts()
        cluster_ids = m.cluster_ids.tolist()
        counts = {int(cid): int(bc[cid]) if cid < len(bc) else 0 for cid in cluster_ids}

        # labels / groups (good / mua / noise / unsorted)
        groups = {int(k): str(v) for k, v in self._get_groups().items()}

        # Resolve the dataset directory (params.py lives inside it)
        import csv
        from pathlib import Path as _Path
        _dir_path = None
        for attr in ("dir_path", "dat_path"):
            p = getattr(m, attr, None)
            if p is not None:
                p = _Path(p)
                _dir_path = p.parent if p.is_file() else p
                break
        if _dir_path is None:
            _dir_path = _Path(".")

        # KiloSort labels from cluster_KSLabel.tsv (read-only, best-effort)
        ks_labels: dict[int, str] = {}
        try:
            ks_tsv = _Path(_dir_path) / "cluster_KSLabel.tsv"
            logger.debug("cluster_KSLabel.tsv path: %s exists=%s", ks_tsv, ks_tsv.exists())
            if ks_tsv.exists():
                with open(ks_tsv, newline="") as f:
                    reader = csv.DictReader(f, delimiter="\t")
                    for row in reader:
                        ks_labels[int(row["cluster_id"])] = str(row["KSLabel"])
                logger.debug("Loaded %d KS labels", len(ks_labels))
        except Exception as exc:
            logger.warning("cluster_KSLabel.tsv read failed: %s", exc)

        # unit_labels.tsv — sua/noise predictions + confidences (comma-separated, unnamed id col)
        unit_labels: dict[int, dict] = {}
        try:
            ul_csv = _Path(_dir_path) / "unit_labels.tsv"
            logger.debug("unit_labels.tsv path: %s exists=%s", ul_csv, ul_csv.exists())
            if ul_csv.exists():
                with open(ul_csv, newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # unnamed first column holds cluster_id
                        cid_str = row.get("") or row.get("cluster_id", "")
                        if not cid_str:
                            continue
                        cid = int(cid_str)
                        unit_labels[cid] = {
                            "sua_pred":   str(row.get("sua_prediction",  "")),
                            "sua_conf":   _safe_float(row.get("sua_confidence",  "")),
                            "noise_pred": str(row.get("noise_prediction", "")),
                            "noise_conf": _safe_float(row.get("noise_confidence", "")),
                        }
                logger.debug("Loaded %d unit labels", len(unit_labels))
        except Exception as exc:
            logger.warning("unit_labels.tsv read failed: %s", exc)

        # mean amplitude per cluster — single O(n) pass, cached weights
        amplitudes = {}
        if hasattr(m, 'amplitudes') and m.amplitudes is not None:
            amps = np.asarray(m.amplitudes)
            amp_sum = np.bincount(spike_clusters, weights=amps,
                                  minlength=int(spike_clusters.max()) + 1)
            for cid in cluster_ids:
                cid = int(cid)
                if bc[cid] > 0:
                    amplitudes[cid] = float(amp_sum[cid] / bc[cid])

        # firing rate: n_spikes / recording duration
        duration = float(m.spike_times[-1]) if len(m.spike_times) else 1.0

        clusters = []
        for cid in cluster_ids:
            cid = int(cid)
            n = counts.get(cid, 0)
            ul = unit_labels.get(cid, {})
            clusters.append({
                "id": cid,
                "label": groups.get(cid, "unsorted"),
                "ks_label": ks_labels.get(cid, ""),
                "sua_pred":   ul.get("sua_pred",   ""),
                "sua_conf":   ul.get("sua_conf",   0.0),
                "noise_pred": ul.get("noise_pred", ""),
                "noise_conf": ul.get("noise_conf", 0.0),
                "n_spikes": n,
                "amplitude": round(amplitudes.get(cid, 0.0), 1),
                "fr": round(n / duration, 2),
            })

        return encode_response(clusters=clusters)

    def _handle_get_spike_data(self, req: dict) -> list[bytes]:
        """
        Return per-spike times (s) and amplitudes for a cluster.

        Response array shape: (n_spikes, 2), float64
          column 0 = spike time in seconds
          column 1 = spike amplitude (arbitrary units from model.amplitudes,
                     or 0 if amplitudes are unavailable)

        Matches phy: n_spikes_amplitudes = 10_000 (random subsample when larger).
        """
        self._require_model()
        cluster_id = int(req["cluster_id"])
        n_max = int(req.get("n_spikes", 2_500))   # subsample for fast response
        return self._cluster_cached(
            cluster_id, f"spike_data_{n_max}",
            lambda: self._do_get_spike_data(cluster_id, n_max),
        )

    def _do_get_spike_data(self, cluster_id: int, n_max: int) -> list[bytes]:
        all_spike_ids = self._get_spike_ids_cached(cluster_id)
        n_total = len(all_spike_ids)

        if n_max > 0 and n_total > n_max:
            rng = np.random.default_rng(seed=cluster_id)
            spike_ids = np.sort(rng.choice(all_spike_ids, size=n_max, replace=False))
        else:
            spike_ids = all_spike_ids

        times = np.asarray(self.model.spike_times[spike_ids], dtype=np.float64)

        if hasattr(self.model, 'amplitudes') and self.model.amplitudes is not None:
            amps = np.asarray(self.model.amplitudes[spike_ids], dtype=np.float64)
        else:
            amps = np.zeros(len(spike_ids), dtype=np.float64)

        data = np.column_stack([times, amps])   # (n_spikes, 2)
        return encode_response(array=data, cluster_id=cluster_id, n_total=n_total)

    def _handle_get_feature_spike_data(self, req: dict) -> list[bytes]:
        """
        Return per-spike times and feature amplitude for a cluster.

        Matches phy's 'feature' amplitude type: PC-0 coefficient on the best
        channel from pc_features.npy.

        Response array shape: (n_spikes, 2), float64  [time_s, PC0_best_channel]
        Returns available=False (no array) when pc_features.npy is absent.
        """
        self._require_model()
        cluster_id = int(req["cluster_id"])
        n_max = int(req.get("n_spikes", 10_000))
        return self._cluster_cached(
            cluster_id, f"feature_spike_data_{n_max}",
            lambda: self._do_get_feature_spike_data(cluster_id, n_max),
        )

    def _do_get_feature_spike_data(self, cluster_id: int, n_max: int) -> list[bytes]:
        all_spike_ids = self._get_spike_ids_cached(cluster_id)
        n_total = len(all_spike_ids)
        if n_max > 0 and n_total > n_max:
            rng = np.random.default_rng(seed=cluster_id)
            spike_ids = np.sort(rng.choice(all_spike_ids, size=n_max, replace=False))
        else:
            spike_ids = all_spike_ids

        times = np.asarray(self.model.spike_times[spike_ids], dtype=np.float64)

        _, ch_top, best_ch = self._template_waveform_and_channels_for_cluster(
            cluster_id, top_n=_TOP_WAVEFORM_CHANNELS
        )
        channel_ids = np.asarray(ch_top, dtype=np.int64) if ch_top else None
        features = self.model.get_features(spike_ids, channel_ids)
        if features is None:
            return encode_response(available=False, cluster_id=cluster_id)

        arr = np.asarray(getattr(features, "data", features), dtype=np.float32)

        if arr.ndim == 3:
            # (n_spikes, n_channels, n_pcs) — take PC-0 on best channel
            best_local = 0
            if best_ch is not None and ch_top:
                try:
                    best_local = list(ch_top).index(int(best_ch))
                except ValueError:
                    best_local = 0
            feat_amps = arr[:, best_local, 0].astype(np.float64)
        elif arr.ndim == 2:
            feat_amps = arr[:, 0].astype(np.float64)
        else:
            feat_amps = arr.astype(np.float64)

        data = np.column_stack([times, feat_amps])
        return encode_response(
            array=data, cluster_id=cluster_id, n_total=n_total, available=True
        )

    def _handle_get_traces(self, req: dict) -> list[bytes]:
        """
        Return raw (or high-pass filtered) voltage traces for a time window.

        Request fields
        --------------
        t_start     : float  seconds
        t_end       : float  seconds
        channel_ids : list[int]  optional — all channels if omitted
        filter      : bool  optional  high-pass filter at 300 Hz (default False)
        max_samples : int   optional  downsample to at most this many columns
                            (default 3000 — enough for a ~1000 px wide display)

        Response array shape: (n_channels, n_samples) float32
        Response header extras: sample_rate, t_start, t_end, channel_ids
        """
        self._require_model()
        m = self.model
        sr = float(m.sample_rate)

        t_start = float(req["t_start"])
        t_end   = float(req["t_end"])
        channel_ids = req.get("channel_ids", None)
        do_filter   = bool(req.get("filter", False))
        max_samples = int(req.get("max_samples", 3000))

        n_total = m.traces.shape[0]
        s_start = max(0, int(t_start * sr))
        s_end   = min(n_total, int(t_end * sr))
        if s_start >= s_end:
            return encode_response(status="error", error="empty time range")

        # Read the trace chunk — force into a plain numpy array
        chunk = np.array(m.traces[s_start:s_end], dtype=np.float32)  # (n_s, n_ch)
        if channel_ids is not None:
            chunk = chunk[:, list(channel_ids)]
        else:
            channel_ids = list(range(chunk.shape[1]))

        chunk = chunk.T  # (n_ch, n_s) — channel-first for efficient wire transfer

        # Optional high-pass filter (300 Hz, zero-phase)
        if do_filter:
            try:
                from scipy.signal import butter, sosfiltfilt
                sos = butter(3, 150.0 / (sr / 2.0), btype="high", output="sos")
                chunk = sosfiltfilt(sos, chunk, axis=1).astype(np.float32)
            except Exception as exc:
                logger.warning("HP filter failed: %s", exc)

        # Downsample for display if needed
        n_s = chunk.shape[1]
        if n_s > max_samples:
            step = n_s // max_samples
            chunk = chunk[:, ::step]
            actual_t_end = t_start + (chunk.shape[1] * step) / sr
        else:
            actual_t_end = s_end / sr

        actual_t_start = s_start / sr
        return encode_response(
            array=chunk,
            sample_rate=sr,
            t_start=actual_t_start,
            t_end=actual_t_end,
            channel_ids=list(channel_ids),
        )

    def _handle_get_spikes_in_window(self, req: dict) -> list[bytes]:
        """
        Return spike times and cluster ids within a time window.

        Request fields
        --------------
        t_start     : float  seconds
        t_end       : float  seconds
        cluster_ids : list[int]  optional — if given, filter to these clusters

        Response array shape: (n_spikes, 2) float64  [time_s, cluster_id]
        """
        self._require_model()
        m = self.model
        times    = m.spike_times
        clusters = m.spike_clusters

        t_start = float(req["t_start"])
        t_end   = float(req["t_end"])

        mask = (times >= t_start) & (times <= t_end)
        if "cluster_ids" in req:
            mask &= np.isin(clusters, req["cluster_ids"])

        result = np.column_stack([
            times[mask].astype(np.float64),
            clusters[mask].astype(np.float64),
        ]) if mask.any() else np.empty((0, 2), dtype=np.float64)

        return encode_response(
            array=result,
            t_start=t_start,
            t_end=t_end,
        )

    def _handle_get_cluster_best_channels(self, req: dict) -> list[bytes]:
        """
        Return the single best (highest-amplitude) channel for every cluster.

        Uses the cached cluster→dominant-template map so each unique template is
        read from the model only once rather than once per cluster.

        Response header: best_channels = {str(cluster_id): best_ch_int, ...}
        No array frame.
        """
        self._require_model()
        cluster_tmpl_map = self._get_cluster_template_map()

        # Deduplicate: call get_template once per unique template id.
        tmpl_to_best_ch: dict[int, int] = {}
        for cid in self.model.cluster_ids:
            dom_t = cluster_tmpl_map.get(int(cid))
            if dom_t is None or dom_t in tmpl_to_best_ch:
                continue
            template = self.model.get_template(dom_t)
            if template is None:
                continue
            wave = np.asarray(template.template, dtype=np.float32)
            ch_ids = np.asarray(template.channel_ids, dtype=np.int64)
            if wave.ndim == 2 and len(ch_ids) == wave.shape[1]:
                ptp = np.ptp(wave, axis=0)
                tmpl_to_best_ch[dom_t] = int(ch_ids[np.argmax(ptp)])

        best_channels: dict[str, int] = {}
        for cid in self.model.cluster_ids:
            cid = int(cid)
            dom_t = cluster_tmpl_map.get(cid)
            if dom_t is not None and dom_t in tmpl_to_best_ch:
                best_channels[str(cid)] = tmpl_to_best_ch[dom_t]
        return encode_response(best_channels=best_channels)

    def _handle_get_correlograms(self, req: dict) -> list[bytes]:
        """
        Compute cross- and auto-correlograms using phylib's fast C implementation.

        Matches phy: n_spikes_correlograms=100_000, bin_size=1ms, window_size=50ms.

        Request fields
        --------------
        cluster_ids  : list[int]
        bin_size     : float  seconds  (default 1e-3 = 1 ms)
        window_size  : float  seconds  (default 50e-3 = 50 ms)
        n_spikes     : int    max spikes to use (default 100_000)

        Response array shape: (n_clusters, n_clusters, n_bins) float32
        Response header extras: cluster_ids, bin_size, window_size, n_bins, firing_rates
        """
        self._require_model()
        m = self.model
        cluster_ids = [int(c) for c in req["cluster_ids"]]
        bin_size    = float(req.get("bin_size",    1e-3))   # 1 ms
        window_size = float(req.get("window_size", 50e-3))  # 50 ms
        n_max       = int(req.get("n_spikes", 100_000))

        # Select spikes from the relevant clusters (uniform subsample matching phy)
        mask = np.isin(m.spike_clusters, cluster_ids)
        spike_ids = np.where(mask)[0]
        if len(spike_ids) > n_max:
            rng = np.random.default_rng(seed=0)
            spike_ids = np.sort(rng.choice(spike_ids, n_max, replace=False))

        st = np.asarray(m.spike_times[spike_ids], dtype=np.float64)
        sc = np.asarray(m.spike_clusters[spike_ids], dtype=np.int32)

        try:
            from phylib.stats import correlograms as _ccg
            ccg = _ccg(
                st, sc,
                sample_rate=float(m.sample_rate),
                cluster_ids=cluster_ids,
                bin_size=bin_size,
                window_size=window_size,
            )
            ccg = np.asarray(ccg, dtype=np.float32)
        except Exception as exc:
            logger.warning("phylib correlograms failed (%s) — falling back to zeros", exc)
            n_bins = max(1, round(window_size / bin_size))
            n_cl   = len(cluster_ids)
            ccg    = np.zeros((n_cl, n_cl, n_bins), dtype=np.float32)

        n_bins = ccg.shape[2] if ccg.ndim == 3 else 1

        # Per-cluster firing rates (spike count / recording duration)
        duration = float(m.spike_times[-1]) if len(m.spike_times) > 0 else 1.0
        firing_rates = []
        for cid in cluster_ids:
            n = int((sc == cid).sum())
            firing_rates.append(round(n / max(duration, 1e-9), 4))

        return encode_response(
            array=ccg,
            cluster_ids=cluster_ids,
            bin_size=bin_size,
            window_size=window_size,
            n_bins=n_bins,
            firing_rates=firing_rates,
        )

    def _handle_get_background_spike_data(self, req: dict) -> list[bytes]:
        """
        Return merged spike data for all OTHER clusters on the same best channel
        as *cluster_id*.  Matches phy's grey backdrop in AmplitudeView.

        Request fields
        --------------
        cluster_id   : int   reference cluster (determines the channel)
        excluded_ids : list[int]  clusters to exclude (typically the selected ones)
        n_spikes     : int   total cap across all background clusters (default 10_000)

        Response array shape: (n_spikes, 2), float64  [time_s, amplitude]
        """
        self._require_model()
        cluster_id   = int(req["cluster_id"])
        excluded_ids = set(int(x) for x in req.get("excluded_ids", []))
        excluded_ids.add(cluster_id)
        n_max = int(req.get("n_spikes", 10_000))

        _, _, best_ch = self._template_waveform_and_channels_for_cluster(cluster_id, top_n=1)
        if best_ch is None:
            return encode_response(array=np.empty((0, 2), dtype=np.float64), n_clusters=0)

        ch_to_cl = self._get_channel_to_clusters()
        bg_ids = [cid for cid in ch_to_cl.get(int(best_ch), []) if cid not in excluded_ids]

        if not bg_ids:
            return encode_response(array=np.empty((0, 2), dtype=np.float64), n_clusters=0)

        n_per = max(1, n_max // len(bg_ids))
        parts = []
        for cid in bg_ids:
            spike_ids = self.model.get_cluster_spikes(cid)
            if len(spike_ids) > n_per:
                rng = np.random.default_rng(seed=cid)
                spike_ids = rng.choice(spike_ids, size=n_per, replace=False)
            times = np.asarray(self.model.spike_times[spike_ids], dtype=np.float64)
            if hasattr(self.model, 'amplitudes') and self.model.amplitudes is not None:
                amps = np.asarray(self.model.amplitudes[spike_ids], dtype=np.float64)
            else:
                amps = np.zeros(len(spike_ids), dtype=np.float64)
            parts.append(np.column_stack([times, amps]))

        data = np.concatenate(parts, axis=0) if parts else np.empty((0, 2), dtype=np.float64)
        return encode_response(array=data, n_clusters=len(bg_ids))

    def _cluster_cached(self, cluster_id: int, key: str, fn) -> "list[bytes]":
        """Return cached response for (cluster_id, key), computing via fn() on miss.

        Error responses are never cached — a transient failure (NFS hiccup, waveforms
        not yet extracted) should not permanently block the view for that cluster.
        """
        entry = self._cluster_cache.get(cluster_id)
        if entry is None:
            self._cluster_cache[cluster_id] = {}
            entry = self._cluster_cache[cluster_id]
        if key not in entry:
            result = fn()
            try:
                status = json.loads(result[0]).get("status", "ok")
            except Exception:
                status = "ok"
            if status != "error":
                entry[key] = result
            return result
        return entry[key]

    def _invalidate_cluster_cache(self) -> None:
        self._cluster_cache.clear()
        self._spike_ids_cache.clear()
        self._spike_ids_sampled.clear()
        self._ch_to_cl_cache = None
        self._spike_counts_cache = None
        self._cluster_tmpl_map_cache = None

    def _get_spike_ids_cached(self, cluster_id: int) -> np.ndarray:
        """
        Return spike indices for cluster_id (may be a pre-warm subsample).

        The pre-warm stores a 1%-random subsample so that first clicks on any cluster
        are instant for views that only need ~100 spikes (waveforms, features…).
        Handlers that need ALL spikes should call _get_all_spike_ids() instead.
        """
        ids = self._spike_ids_cache.get(cluster_id)
        if ids is None:
            ids = self.model.get_cluster_spikes(cluster_id)
            self._spike_ids_cache[cluster_id] = ids
        return ids

    def _get_all_spike_ids(self, cluster_id: int) -> np.ndarray:
        """
        Return ALL spike indices for cluster_id (never a subsample).

        The pre-warm populates _spike_ids_cache with a 1%-subsample and records
        those cluster_ids in _spike_ids_sampled.  If the requested cluster is marked
        as sampled, recompute the full list, replace the cache entry, and clear the
        flag so subsequent calls are O(1).
        """
        if cluster_id in self._spike_ids_sampled:
            ids = self.model.get_cluster_spikes(cluster_id)
            self._spike_ids_cache[cluster_id] = ids
            self._spike_ids_sampled.discard(cluster_id)
            return ids
        ids = self._spike_ids_cache.get(cluster_id)
        if ids is None:
            ids = self.model.get_cluster_spikes(cluster_id)
            self._spike_ids_cache[cluster_id] = ids
        return ids

    def _get_cluster_template_map(self) -> "dict[int, int]":
        """
        Return {cluster_id: dominant_template_id} for all clusters.

        Uses a single O(n_spikes) bincount pass instead of per-cluster NFS scans.
        Result cached; invalidated by _invalidate_cluster_cache() after merge/undo/redo.
        """
        if getattr(self, '_cluster_tmpl_map_cache', None) is not None:
            return self._cluster_tmpl_map_cache

        m = self.model
        if not hasattr(m, 'spike_templates') or m.spike_templates is None:
            result = {int(cid): int(cid) for cid in m.cluster_ids}
            self._cluster_tmpl_map_cache = result
            return result

        sc = np.asarray(m.spike_clusters, dtype=np.int32)
        st = np.asarray(m.spike_templates, dtype=np.int32)
        n_cl = int(sc.max()) + 1
        n_t  = int(st.max()) + 1

        if n_cl * n_t <= 10_000_000:
            # Fast path: single bincount over combined (cluster, template) index.
            # Peak extra memory = n_spikes × 8 bytes (temporary int64 array).
            combined = sc.astype(np.int64) * n_t + st.astype(np.int64)
            co = np.bincount(combined, minlength=n_cl * n_t).reshape(n_cl, n_t)
            del combined  # free the large temporary
            result = {int(cid): int(np.argmax(co[int(cid)])) for cid in m.cluster_ids}
        else:
            # Fallback for unusually large ID spaces (many post-merge cluster IDs).
            result = {}
            for cid in m.cluster_ids:
                cid = int(cid)
                spike_ids = m.get_cluster_spikes(cid)
                if len(spike_ids) == 0:
                    result[cid] = cid
                    continue
                st_cl = np.asarray(m.spike_templates[spike_ids], dtype=np.int32)
                result[cid] = int(np.bincount(st_cl).argmax())

        self._cluster_tmpl_map_cache = result
        logger.debug("Built cluster→template map for %d clusters", len(result))
        return result

    def _get_spike_counts(self) -> "tuple[np.ndarray, np.ndarray]":
        """Return (spike_clusters, bincount) cached — recomputed only after merge/undo/redo."""
        if getattr(self, '_spike_counts_cache', None) is None:
            sc = np.asarray(self.model.spike_clusters)
            bc = np.bincount(sc, minlength=int(sc.max()) + 1)
            self._spike_counts_cache = (sc, bc)
        return self._spike_counts_cache

    def _get_channel_to_clusters(self) -> "dict[int, list[int]]":
        """
        Return {best_channel: [cluster_ids]} mapping, built lazily and cached.

        Uses the cached cluster→template map + deduplication of get_template() calls
        so this is O(n_unique_templates) NFS reads rather than O(n_clusters × n_spikes).
        """
        if getattr(self, '_ch_to_cl_cache', None) is not None:
            return self._ch_to_cl_cache

        cluster_tmpl_map = self._get_cluster_template_map()

        # Compute best channel for each unique dominant template only once.
        tmpl_to_best_ch: dict[int, int] = {}
        for cid in self.model.cluster_ids:
            dom_t = cluster_tmpl_map.get(int(cid))
            if dom_t is None or dom_t in tmpl_to_best_ch:
                continue
            try:
                template = self.model.get_template(dom_t)
                if template is None:
                    continue
                wave = np.asarray(template.template, dtype=np.float32)
                ch_ids = np.asarray(template.channel_ids, dtype=np.int64)
                if wave.ndim == 2 and len(ch_ids) == wave.shape[1]:
                    ptp = np.ptp(wave, axis=0)
                    tmpl_to_best_ch[dom_t] = int(ch_ids[np.argmax(ptp)])
            except Exception:
                pass

        ch_to_cl: dict[int, list[int]] = {}
        for cid in self.model.cluster_ids:
            cid = int(cid)
            dom_t = cluster_tmpl_map.get(cid)
            best_ch = tmpl_to_best_ch.get(dom_t) if dom_t is not None else None
            if best_ch is not None:
                ch_to_cl.setdefault(best_ch, []).append(cid)

        self._ch_to_cl_cache = ch_to_cl
        return ch_to_cl

    # ------------------------------------------------------------------
    # Dataset discovery and switching
    # ------------------------------------------------------------------

    def _discover_datasets(self, params_path: str, max_datasets: int = 0) -> None:
        """
        Given a params.py path like .../output/a/shank_0/kilosort4/params.py,
        scan the output/ directory for all sibling shank datasets and build
        self._datasets = {label: Path}.

        If the path doesn't fit the expected pattern the single dataset is
        still registered so the client can display the current shank name.
        """
        from pathlib import Path
        p = Path(params_path).resolve()
        self._current_dataset = ""

        # Walk up looking for a directory whose children look like letter/shank_N
        # Expected structure: <output_root>/<letter>/<shank_N>/kilosort4/params.py
        # p.parents: kilosort4, shank_N, letter, output_root, ...
        try:
            shank_dir  = p.parent.parent   # .../output/a/shank_0
            letter_dir = shank_dir.parent  # .../output/a
            output_dir = letter_dir.parent # .../output

            datasets: dict[str, Path] = {}
            for letter_path in sorted(output_dir.iterdir()):
                if not letter_path.is_dir() or len(letter_path.name) != 1:
                    continue
                for shank_path in sorted(letter_path.iterdir()):
                    if not shank_path.is_dir() or not shank_path.name.startswith("shank_"):
                        continue
                    # Check one level deeper directly — avoid rglob over network fs
                    candidate = None
                    try:
                        for sorter_dir in sorted(shank_path.iterdir()):
                            pp = sorter_dir / "params.py"
                            if pp.exists():
                                candidate = pp
                                break
                        # Also check params.py directly in shank dir
                        if candidate is None and (shank_path / "params.py").exists():
                            candidate = shank_path / "params.py"
                    except Exception:
                        pass
                    if candidate is not None:
                        label = f"{letter_path.name}/{shank_path.name}"
                        datasets[label] = candidate

            if datasets:
                if max_datasets > 0:
                    datasets = dict(list(datasets.items())[:max_datasets])
                self._datasets = datasets
                # Find our current label
                for lbl, path in datasets.items():
                    if path == p:
                        self._current_dataset = lbl
                        break
                if not self._current_dataset and datasets:
                    self._current_dataset = next(iter(datasets))
                logger.info("Discovered %d datasets: %s", len(datasets), list(datasets))
                return
        except Exception as exc:
            logger.debug("Dataset discovery failed: %s", exc)

        # Fallback: register just the one dataset with its shank dir name as label
        label = p.parent.parent.name  # e.g. "shank_0" or just the dir name
        self._datasets = {label: p}
        self._current_dataset = label

    def _handle_get_dataset_list(self, req: dict) -> list[bytes]:
        """Return available datasets and current selection."""
        datasets = [
            {"label": lbl, "params_path": str(path)}
            for lbl, path in self._datasets.items()
        ]
        return encode_response(datasets=datasets, current=self._current_dataset)

    def _handle_switch_dataset(self, req: dict) -> list[bytes]:
        """
        Reload the model from a new params.py.

        Request fields
        --------------
        label : str  dataset label (must be in self._datasets)
        """
        label = req.get("label", "")
        if label not in self._datasets:
            return encode_response(
                status="error",
                error=f"Unknown dataset label '{label}'. Available: {list(self._datasets)}"
            )
        if label == self._current_dataset:
            return encode_response(current=self._current_dataset)

        params_path = self._datasets[label]
        logger.info("Switching dataset → %s (%s)", label, params_path)

        # Mark as loading immediately and kick off background load
        self._model_ready = False
        self._current_dataset = label

        def _load():
            try:
                from phylib.io.model import load_model
                new_model = load_model(params_path)
            except Exception as exc:
                logger.error("Failed to load model for %s: %s", label, exc)
                self._model_ready = True  # unblock so client gets real errors
                return
            self.model = new_model
            self._undo_stack.clear()
            self._redo_stack.clear()
            self._invalidate_cluster_cache()
            self._model_ready = True  # defer _init_mutable_spike_clusters to first merge
            logger.info("Dataset switched to %s: %d clusters, %d spikes",
                        label, len(new_model.cluster_ids), len(new_model.spike_times))
            # Pre-extract waveforms for this shank if not already done
            try:
                from phy_remote.server.__main__ import _extract_spike_waveforms_if_needed
                _extract_spike_waveforms_if_needed(new_model, params_path)
            except Exception as exc:
                logger.debug("Waveform pre-extraction skipped: %s", exc)

        threading.Thread(target=_load, daemon=True).start()
        return encode_response(current=self._current_dataset)

    def _handle_get_background_features(self, req: dict) -> list[bytes]:
        """
        Return PC features for a random sample of spikes from ALL clusters on
        specific probe channels.  Used to draw grey background scatter in the
        feature view.

        Request fields
        --------------
        channel_ids : list[int]  global channel indices (up to 4)
        n_spikes    : int        total random spikes to return (default 2500)

        Response array shape: (n_spikes, n_channels, n_pcs), float32
        Returns available=False when pc_features.npy is absent.
        """
        self._require_model()
        m = self.model
        channel_ids = [int(c) for c in req.get("channel_ids", [])]
        n_max = int(req.get("n_spikes", 500))

        # Random sample from all spikes
        all_spike_ids = np.arange(len(m.spike_times), dtype=np.int64)
        if len(all_spike_ids) > n_max:
            rng = np.random.default_rng(seed=42)
            all_spike_ids = rng.choice(all_spike_ids, size=n_max, replace=False)

        ch_arr = np.asarray(channel_ids, dtype=np.int64) if channel_ids else None
        features = m.get_features(all_spike_ids, ch_arr)
        if features is None:
            return encode_response(available=False)

        arr = np.asarray(getattr(features, "data", features), dtype=np.float32)
        return encode_response(array=arr, available=True, channel_ids=channel_ids)

    def _handle_get_raster_data(self, req: dict) -> list[bytes]:
        """
        Return evenly-spaced spike times for every cluster (for the raster view).

        Request fields
        --------------
        n_spikes : int  spikes per cluster (default 200)

        Response array shape: (n_total_spikes, 2) float32  [time_s, cluster_id]
        Response header extras: cluster_ids (ordered list), duration (float seconds)
        """
        self._require_model()
        m = self.model
        n_per = int(req.get("n_spikes", 200))

        cluster_ids = [int(c) for c in m.cluster_ids]
        duration = float(m.spike_times[-1]) if len(m.spike_times) > 0 else 0.0

        parts: list[np.ndarray] = []
        for cid in cluster_ids:
            spike_ids = m.get_cluster_spikes(cid)
            n = len(spike_ids)
            if n == 0:
                continue
            if n > n_per:
                # Evenly-spaced subsample (same as phy's raster strategy)
                idx = np.round(np.linspace(0, n - 1, n_per)).astype(np.int64)
                spike_ids = spike_ids[idx]
            times = np.asarray(m.spike_times[spike_ids], dtype=np.float32)
            cid_col = np.full(len(times), float(cid), dtype=np.float32)
            parts.append(np.column_stack([times, cid_col]).astype(np.float32))

        data = np.concatenate(parts, axis=0) if parts else np.empty((0, 2), dtype=np.float32)
        return encode_response(
            array=data,
            cluster_ids=cluster_ids,
            duration=duration,
        )

    def _handle_get_channel_positions(self, req: dict) -> list[bytes]:
        """Return probe channel positions as (n_channels, 2) float32 array [x, y] in µm."""
        self._require_model()
        positions = np.asarray(self.model.channel_positions, dtype=np.float32)
        return encode_response(array=positions)

    def _handle_get_similar_clusters(self, req: dict) -> list[bytes]:
        """
        Return a ranked similarity list for one selected cluster (Phy-style).

        Uses the cached cluster→dominant-template map so the whole computation is
        O(n_clusters) rather than O(n_clusters × n_spikes).

        Request fields
        --------------
        cluster_id : int
        limit      : int, optional (default 100)
        """
        self._require_model()
        m = self.model
        cluster_id = int(req["cluster_id"])
        limit = int(req.get("limit", 100))

        if not hasattr(m, "similar_templates") or m.similar_templates is None:
            return encode_response(similar_clusters=[])
        if not hasattr(m, "spike_templates") or m.spike_templates is None:
            return encode_response(similar_clusters=[])

        cluster_ids = [int(c) for c in np.asarray(m.cluster_ids, dtype=np.int64)]
        if cluster_id not in cluster_ids:
            return encode_response(similar_clusters=[])

        # Dominant template per cluster — O(n_spikes) once, then cached.
        cluster_tmpl_map = self._get_cluster_template_map()

        dom_t_i = cluster_tmpl_map.get(cluster_id)
        if dom_t_i is None:
            return encode_response(similar_clusters=[])

        similar_templates = np.asarray(m.similar_templates)  # (n_templates, n_templates)
        if dom_t_i >= similar_templates.shape[0]:
            return encode_response(similar_clusters=[])

        # Similarity of selected cluster's dominant template to every template.
        sims = similar_templates[dom_t_i, :]  # (n_templates,)

        # Precompute cluster info in one O(n_spikes) pass.
        sc_arr, bc = self._get_spike_counts()
        counts = {int(cid): int(bc[cid]) if cid < len(bc) else 0 for cid in cluster_ids}
        groups = {int(k): str(v) for k, v in self._get_groups().items()}
        amplitudes: dict[int, float] = {}
        if hasattr(m, "amplitudes") and m.amplitudes is not None:
            amps = np.asarray(m.amplitudes)
            amp_sum = np.bincount(sc_arr, weights=amps, minlength=int(sc_arr.max()) + 1)
            for cid in cluster_ids:
                if bc[cid] > 0:
                    amplitudes[cid] = float(amp_sum[cid] / bc[cid])
        duration = float(m.spike_times[-1]) if len(m.spike_times) else 1.0

        rows = []
        for cid in cluster_ids:
            if cid == cluster_id:
                continue
            dom_t_j = cluster_tmpl_map.get(cid)
            s = float(sims[dom_t_j]) if (dom_t_j is not None and dom_t_j < len(sims)) else 0.0
            n = counts.get(cid, 0)
            rows.append({
                "id": cid,
                "similarity": round(s, 3),
                "n_spikes": n,
                "label": groups.get(cid, "unsorted"),
                "amplitude": round(amplitudes.get(cid, 0.0), 1),
                "fr": round(n / duration, 2),
            })

        rows.sort(key=lambda r: r["similarity"], reverse=True)
        if limit > 0:
            rows = rows[:limit]
        return encode_response(primary_cluster_id=cluster_id, similar_clusters=rows)

    def _handle_label_cluster(self, req: dict) -> list[bytes]:
        """
        Set the label for one or more clusters (in memory only — not saved to disk).
        Call CMD_SAVE to persist changes.

        Request fields
        --------------
        cluster_ids : list[int]  (or a single int as 'cluster_id')
        label       : str  one of good / mua / noise / unsorted
        """
        self._require_model()
        VALID = {"good", "mua", "noise", "unsorted"}
        label = str(req.get("label", ""))
        if label not in VALID:
            return encode_response(
                status="error",
                error=f"invalid label {label!r}, must be one of {sorted(VALID)}",
            )

        # Accept either cluster_ids (list) or cluster_id (scalar)
        if "cluster_ids" in req:
            cluster_ids = [int(c) for c in req["cluster_ids"]]
        else:
            cluster_ids = [int(req["cluster_id"])]

        m = self.model
        groups = self._get_groups()
        for cid in cluster_ids:
            groups[cid] = label

        logger.info("Labelled cluster(s) %s as %r (unsaved)", cluster_ids, label)
        return encode_response(cluster_ids=cluster_ids, label=label)

    def _handle_merge(self, req: dict) -> list[bytes]:
        """
        Merge two or more clusters into a new cluster.

        Request fields
        --------------
        cluster_ids : list[int]  (must have at least 2 entries)

        Response
        --------
        new_cluster_id : int
        merged_ids     : list[int]
        clusters       : list[dict]  updated cluster info (same format as GET_CLUSTER_INFO)
        """
        self._require_model()
        cluster_ids = [int(c) for c in req.get("cluster_ids", [])]
        if len(cluster_ids) < 2:
            return encode_response(status="error", error="need at least 2 clusters to merge")

        # Always work with a guaranteed-writeable array stored in __dict__
        sc = self._get_mutable_spike_clusters()
        existing = set(np.unique(sc).tolist())
        missing = [c for c in cluster_ids if c not in existing]
        if missing:
            return encode_response(status="error", error=f"clusters not found: {missing}")

        # Save state for undo
        groups = self._get_groups()
        self._undo_stack.append((sc.copy(), dict(groups)))
        self._redo_stack.clear()

        # New cluster id is max existing + 1
        new_id = int(np.max(sc)) + 1

        # Inherit label from the largest source cluster
        best_label = "unsorted"
        best_count = 0
        for cid in cluster_ids:
            n = int(np.sum(sc == cid))
            if n > best_count:
                best_count = n
                best_label = groups.get(cid, "unsorted")

        # Reassign spikes in our mutable copy
        mask = np.isin(sc, cluster_ids)
        sc[mask] = new_id

        # Update the live groups dict
        for cid in cluster_ids:
            groups.pop(cid, None)
        groups[new_id] = best_label

        logger.info("Merged clusters %s → %d", cluster_ids, new_id)
        self._invalidate_cluster_cache()
        clusters = self._build_cluster_info()
        return encode_response(new_cluster_id=new_id, merged_ids=cluster_ids, clusters=clusters)

    def _handle_undo(self, req: dict) -> list[bytes]:
        """Undo the last merge. Returns updated cluster info."""
        self._require_model()
        if not self._undo_stack:
            return encode_response(status="error", error="nothing to undo")

        sc_now = self._get_mutable_spike_clusters()
        self._redo_stack.append((sc_now.copy(), dict(self._get_groups())))

        sc_prev, groups_prev = self._undo_stack.pop()
        sc_now[:] = sc_prev
        self._get_groups().clear()
        self._get_groups().update(groups_prev)

        logger.info("Undo: restored spike_clusters")
        self._invalidate_cluster_cache()
        clusters = self._build_cluster_info()
        return encode_response(clusters=clusters)

    def _handle_redo(self, req: dict) -> list[bytes]:
        """Redo the last undone merge. Returns updated cluster info."""
        self._require_model()
        if not self._redo_stack:
            return encode_response(status="error", error="nothing to redo")

        sc_now = self._get_mutable_spike_clusters()
        self._undo_stack.append((sc_now.copy(), dict(self._get_groups())))

        sc_next, groups_next = self._redo_stack.pop()
        sc_now[:] = sc_next
        self._get_groups().clear()
        self._get_groups().update(groups_next)

        logger.info("Redo: restored spike_clusters")
        self._invalidate_cluster_cache()
        clusters = self._build_cluster_info()
        return encode_response(clusters=clusters)

    def _handle_save(self, req: dict) -> list[bytes]:
        """
        Persist spike_clusters.npy and cluster_group.tsv to the dataset directory.
        """
        self._require_model()
        m = self.model
        dir_path = None
        for attr in ("dir_path", "dat_path"):
            if hasattr(m, attr) and getattr(m, attr) is not None:
                import pathlib
                p = pathlib.Path(getattr(m, attr))
                dir_path = p.parent if p.is_file() else p
                break

        if dir_path is None:
            return encode_response(status="error", error="cannot determine dataset directory")

        import pathlib
        dir_path = pathlib.Path(dir_path)

        # spike_clusters.npy
        try:
            np.save(str(dir_path / "spike_clusters.npy"), m.spike_clusters)
        except Exception as exc:
            logger.warning("Could not save spike_clusters.npy: %s", exc)

        # cluster_group.tsv  (via model if possible, fallback to manual write)
        saved_via_model = False
        if hasattr(m, 'save_metadata'):
            try:
                groups = self._get_groups()
                m.save_metadata("group", groups)
                saved_via_model = True
            except Exception as exc:
                logger.warning("save_metadata failed, falling back: %s", exc)

        if not saved_via_model:
            try:
                groups = self._get_groups()
                tsv_path = dir_path / "cluster_group.tsv"
                with open(tsv_path, "w") as f:
                    f.write("cluster_id\tgroup\n")
                    for cid, grp in sorted(groups.items()):
                        f.write(f"{cid}\t{grp}\n")
            except Exception as exc:
                return encode_response(status="error", error=f"could not write cluster_group.tsv: {exc}")

        logger.info("Saved clustering to %s", dir_path)
        return encode_response(saved=True)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_groups(self) -> dict:
        """Return the live {cluster_id (int): label (str)} dict.

        Uses model.metadata['group'] as the canonical in-memory store so that
        it stays in sync with phylib's own machinery.  Falls back to a plain
        dict attached to the model when metadata is unavailable.
        """
        m = self.model
        if hasattr(m, 'metadata') and isinstance(m.metadata, dict):
            if 'group' not in m.metadata:
                m.metadata['group'] = {}
            return m.metadata['group']
        # Fallback: use a plain dict attached to the model instance
        if not hasattr(m, '_phy_remote_groups'):
            m._phy_remote_groups = {}
        return m._phy_remote_groups

    def _save_cluster_groups(self) -> None:
        """Write cluster labels to cluster_group.tsv.

        Strategy (in order):
          1. Call model.save_metadata("group", ...) — phylib's native method.
          2. Write cluster_group.tsv directly next to the data file.

        Both paths are attempted so the file is always written even if phylib's
        method exists but fails internally.
        """
        groups = {int(k): str(v) for k, v in self._get_groups().items()}

        # --- attempt 1: phylib's native save ---
        native_ok = False
        if hasattr(m, 'save_metadata'):
            try:
                m.save_metadata("group", groups)
                native_ok = True
                logger.debug("Labels saved via save_metadata (%d clusters)", len(groups))
            except Exception as exc:
                logger.warning("save_metadata failed (%s), falling back to direct write", exc)

        # --- attempt 2: write TSV ourselves ---
        # Resolve the dataset directory from the model's known path attributes.
        dir_path = None
        for attr in ("dir_path", "dat_path", "path"):
            val = getattr(m, attr, None)
            if val is not None:
                import pathlib
                p = pathlib.Path(val)
                dir_path = p.parent if p.suffix else p
                break

        if dir_path is None:
            # Last resort: look for params.py location via __file__ attribute
            if hasattr(m, 'params_path'):
                import pathlib
                dir_path = pathlib.Path(m.params_path).parent

        if dir_path is not None:
            import pathlib
            tsv_path = pathlib.Path(dir_path) / "cluster_group.tsv"
            try:
                with open(tsv_path, "w") as f:
                    f.write("cluster_id\tgroup\n")
                    for cid, grp in sorted(groups.items()):
                        f.write(f"{cid}\t{grp}\n")
                logger.debug("Labels written directly to %s (%d clusters)", tsv_path, len(groups))
            except Exception as exc:
                logger.error("Could not write cluster_group.tsv: %s", exc)
        elif not native_ok:
            logger.error(
                "Labels NOT saved: save_metadata unavailable and dataset directory unknown"
            )

    def _get_mutable_spike_clusters(self) -> np.ndarray:
        """Return the writeable spike_clusters array stored in the model's __dict__.

        If it was never installed (e.g. init failed) we install it now so that
        the first merge still works.
        """
        sc = self.model.__dict__.get('spike_clusters')
        if sc is None or not isinstance(sc, np.ndarray) or not sc.flags.writeable:
            self._init_mutable_spike_clusters()
            sc = self.model.__dict__.get('spike_clusters')
        if sc is None:
            # Absolute fallback: copy from the model property each time
            sc = np.array(self.model.spike_clusters)
            self.model.__dict__['spike_clusters'] = sc
        return sc

    def _init_mutable_spike_clusters(self) -> None:
        """Replace model.spike_clusters with a writable numpy array copy.

        Handles three cases:
          1. Regular instance attribute (memmap or ndarray) — set via instance dict.
          2. Read-only property — set via instance __dict__ to shadow it.
          3. Both fail — warn and carry on (merge will still attempt the copy trick).
        """
        try:
            sc = np.array(self.model.spike_clusters)   # always a writeable plain array
            # Set via __dict__ to bypass any property descriptor
            self.model.__dict__['spike_clusters'] = sc
            logger.debug("Installed mutable spike_clusters copy (%d spikes)", len(sc))
        except Exception as exc:
            logger.warning("Could not install mutable spike_clusters: %s", exc)

    def _build_cluster_info(self) -> list[dict]:
        """Return the same cluster summary as _handle_get_cluster_info."""
        m = self.model
        sc, bc = self._get_spike_counts()
        cluster_ids = m.cluster_ids.tolist()
        counts = {int(cid): int(bc[cid]) if cid < len(bc) else 0 for cid in cluster_ids}
        groups = {int(k): str(v) for k, v in self._get_groups().items()}
        amplitudes = {}
        if hasattr(m, 'amplitudes') and m.amplitudes is not None:
            amps = np.asarray(m.amplitudes)
            amp_sum = np.bincount(sc, weights=amps, minlength=int(sc.max()) + 1)
            for cid in cluster_ids:
                cid = int(cid)
                if bc[cid] > 0:
                    amplitudes[cid] = float(amp_sum[cid] / bc[cid])
        duration = float(m.spike_times[-1]) if len(m.spike_times) else 1.0
        clusters = []
        for cid in cluster_ids:
            cid = int(cid)
            n = counts.get(cid, 0)
            clusters.append({
                "id": cid,
                "label": groups.get(cid, "unsorted"),
                "ks_label": "",   # not re-read on refresh; loaded at startup
                "n_spikes": n,
                "amplitude": round(amplitudes.get(cid, 0.0), 1),
                "fr": round(n / duration, 2),
            })
        return clusters

    def _require_model(self) -> None:
        if self.model is None:
            raise RuntimeError("server started without a model")

    def _best_template_for_cluster(self, cluster_id: int) -> int:
        """
        Return the dominant template id for a cluster.
        Mirrors Phy's get_template_for_cluster behavior.

        Uses the cached cluster→template map when available (O(1)); only falls
        back to an O(n_spikes) scan if the cache hasn't been built yet.
        """
        # Fast path: use precomputed map (built once for all clusters at model load).
        tmpl_map = getattr(self, '_cluster_tmpl_map_cache', None)
        if tmpl_map is not None:
            t = tmpl_map.get(int(cluster_id))
            if t is not None:
                return t

        m = self.model
        if not hasattr(m, "spike_templates") or m.spike_templates is None:
            return int(cluster_id)
        spike_ids = self._get_spike_ids_cached(cluster_id)
        if spike_ids is None or len(spike_ids) == 0:
            raise RuntimeError(f"cluster {cluster_id} has no spikes")
        st = np.asarray(m.spike_templates[spike_ids], dtype=np.int64)
        template_ids, counts = np.unique(st, return_counts=True)
        return int(template_ids[int(np.argmax(counts))])

    def _best_channel_ids_for_cluster(self, cluster_id: int) -> "np.ndarray | None":
        """
        Return best channel ids for the dominant template of a cluster.
        """
        template_id = self._best_template_for_cluster(cluster_id)
        template = self.model.get_template(int(template_id))
        return template.channel_ids if template is not None else None

    def _template_waveform_and_channels_for_cluster(
        self, cluster_id: int, top_n: int = _TOP_WAVEFORM_CHANNELS
    ) -> "tuple[np.ndarray | None, list[int] | None]":
        """
        Return template waveform restricted to top-N amplitude channels.
        """
        template_id = self._best_template_for_cluster(cluster_id)
        template = self.model.get_template(int(template_id))
        if template is None:
            return None, None

        wave = np.asarray(template.template, dtype=np.float32)  # (n_samples, n_ch_local)
        ch_ids = np.asarray(template.channel_ids, dtype=np.int64)
        if wave.ndim != 2 or len(ch_ids) != wave.shape[1]:
            return None, None

        # Rank channels by template peak-to-peak amplitude.
        ptp = np.ptp(wave, axis=0)
        order = np.argsort(ptp)[::-1]
        k = min(max(1, int(top_n)), len(order))
        keep = np.sort(order[:k])  # keep stable channel order for probe mapping

        wave_top = wave[:, keep].astype(np.float32)
        ch_top = ch_ids[keep].astype(np.int64).tolist()
        # Also expose the single best (highest-amplitude) channel id
        best_ch = int(ch_ids[order[0]])
        return wave_top, ch_top, best_ch

    def _handle_signal(self, signum, frame) -> None:
        logger.info("received signal %d, stopping", signum)
        self._running = False  # poller loop checks this flag and exits cleanly
