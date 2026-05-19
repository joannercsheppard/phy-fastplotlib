"""
ZMQ REQ transport — runs on the MacBook client.

Typical setup
-------------
    # On cluster:  already running PhyServer on port 5555 (loopback)
    # On MacBook:  ssh -L 5555:localhost:5555 user@cluster

    from phy_remote.client.transport import PhyTransport
    t = PhyTransport(port=5555)
    header, waveforms = t.get_waveforms(cluster_id=42)
    t.close()

    # Or use as a context manager:
    with PhyTransport() as t:
        header, waveforms = t.get_waveforms(cluster_id=42)
"""

import logging

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
    encode_request,
    decode_response,
)

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_MS = 60_000  # 60 s — on-the-fly waveform extraction can be slow


class TransportError(RuntimeError):
    """Raised when the server returns an error response."""

class ServerLoadingError(RuntimeError):
    """Raised when the server is still loading its model."""


class PhyTransport:
    """
    Thin ZMQ REQ wrapper around the phy-remote protocol.

    Parameters
    ----------
    host : str
        Hostname / IP of the server.  When using an SSH tunnel, keep this
        as "127.0.0.1" (the local tunnel endpoint).
    port : int
        Port number (must match ``PhyServer`` / SSH tunnel).
    timeout_ms : int
        Socket receive timeout in milliseconds.  Raises ``zmq.Again`` on
        expiry so the UI can display a meaningful error instead of hanging.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 5555,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
    ):
        self.host = host
        self.port = port
        self._ctx = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        addr = f"tcp://{host}:{port}"
        self._sock.connect(addr)
        logger.info("PhyTransport connected to %s", addr)

    # ------------------------------------------------------------------
    # Context-manager support
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctx.term()

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def ping(self) -> bool:
        """Return True if the server responds to a ping."""
        header, _ = self._call(CMD_PING)
        return header.get("pong", False)

    def get_waveforms(
        self, cluster_id: int, n_spikes: int = 100
    ) -> tuple[dict, np.ndarray]:
        """
        Fetch waveforms for *cluster_id*.

        Returns
        -------
        header : dict
            Response metadata (cluster_id, n_spikes, dtype, shape).
        waveforms : np.ndarray, shape (n_spikes, n_samples, n_channels), float32
        """
        header, array = self._call(
            CMD_GET_WAVEFORMS, cluster_id=cluster_id, n_spikes=n_spikes
        )
        if array is None:
            raise TransportError("server sent no array for get_waveforms")
        return header, array

    def get_spike_times(self, cluster_id: int) -> tuple[dict, np.ndarray]:
        """
        Fetch spike times (seconds) for *cluster_id*.

        Returns
        -------
        header : dict
        times : np.ndarray, shape (n_spikes,), float64
        """
        header, array = self._call(CMD_GET_SPIKE_TIMES, cluster_id=cluster_id)
        if array is None:
            raise TransportError("server sent no array for get_spike_times")
        return header, array

    def get_features(self, cluster_id: int) -> tuple[dict, np.ndarray]:
        """
        Fetch PC features for *cluster_id*.

        Returns
        -------
        header : dict
        features : np.ndarray, float32
        """
        header, array = self._call(CMD_GET_FEATURES, cluster_id=cluster_id)
        if array is None:
            raise TransportError("server sent no array for get_features")
        return header, array

    def get_template_features(self, cluster_id: int) -> tuple[dict, np.ndarray] | None:
        """
        Fetch per-spike template features for *cluster_id*.

        Returns (header, array) on success, or None if the dataset has no
        template_features.npy (server sets available=False).
        """
        header, array = self._call(CMD_GET_TEMPLATE_FEATURES, cluster_id=cluster_id)
        if not header.get("available", True):
            return None
        if array is None:
            raise TransportError("server sent no array for get_template_features")
        return header, array

    def get_templates(self, cluster_id: int) -> tuple[dict, np.ndarray]:
        """
        Fetch the mean template waveform for *cluster_id*.

        Returns
        -------
        header : dict
            Includes ``channel_ids`` list.
        template : np.ndarray, shape (n_samples, n_channels), float32
        """
        header, array = self._call(CMD_GET_TEMPLATES, cluster_id=cluster_id)
        if array is None:
            raise TransportError("server sent no array for get_templates")
        return header, array

    def get_cluster_ids(self) -> np.ndarray:
        """Return all cluster ids as a 1-D int32 array."""
        _, array = self._call(CMD_GET_CLUSTER_IDS)
        if array is None:
            raise TransportError("server sent no array for get_cluster_ids")
        return array

    def get_cluster_info(self) -> list[dict]:
        """
        Return summary info for all clusters.

        Returns
        -------
        list of dicts, each with keys:
            id, label, n_spikes, amplitude, fr
        """
        header, _ = self._call(CMD_GET_CLUSTER_INFO)
        return header["clusters"]

    def get_traces(
        self,
        t_start: float,
        t_end: float,
        channel_ids: "list[int] | None" = None,
        filtered: bool = False,
        max_samples: int = 3000,
    ) -> "tuple[np.ndarray, dict]":
        """
        Fetch raw (or HP-filtered) voltage traces for a time window.

        Returns
        -------
        traces  : np.ndarray, shape (n_channels, n_samples), float32
        header  : dict  with keys sample_rate, t_start, t_end, channel_ids
        """
        kwargs: dict = dict(t_start=t_start, t_end=t_end,
                            filter=filtered, max_samples=max_samples)
        if channel_ids is not None:
            kwargs["channel_ids"] = channel_ids
        header, array = self._call(CMD_GET_TRACES, **kwargs)
        if array is None:
            raise TransportError("server sent no array for get_traces")
        return array, header

    def get_spikes_in_window(
        self,
        t_start: float,
        t_end: float,
        cluster_ids: "list[int] | None" = None,
    ) -> np.ndarray:
        """
        Fetch spike times and cluster ids in a time window.

        Returns
        -------
        data : np.ndarray, shape (n_spikes, 2), float64
            column 0 = spike time (s), column 1 = cluster_id
        """
        kwargs: dict = dict(t_start=t_start, t_end=t_end)
        if cluster_ids is not None:
            kwargs["cluster_ids"] = cluster_ids
        _, array = self._call(CMD_GET_SPIKES_IN_WINDOW, **kwargs)
        return array if array is not None else np.empty((0, 2), dtype=np.float64)

    def get_similar_clusters(self, cluster_id: int, limit: int = 100) -> list[dict]:
        """
        Return ranked similar clusters for a primary cluster.

        Returns
        -------
        list[dict]
            Each row contains at least:
            id, similarity, n_spikes, label, amplitude, fr
        """
        header, _ = self._call(
            CMD_GET_SIMILAR_CLUSTERS, cluster_id=int(cluster_id), limit=int(limit)
        )
        return header.get("similar_clusters", [])

    def get_channel_positions(self) -> np.ndarray:
        """Return probe channel positions as (n_channels, 2) float32 array [x, y] µm."""
        _, array = self._call(CMD_GET_CHANNEL_POSITIONS)
        if array is None:
            raise TransportError("server sent no array for get_channel_positions")
        return array

    def get_cluster_best_channels(self) -> "dict[int, int]":
        """Return {cluster_id: best_channel_index} for every cluster."""
        header, _ = self._call(CMD_GET_CLUSTER_BEST_CHANNELS)
        raw = header.get("best_channels", {})
        return {int(k): int(v) for k, v in raw.items()}

    def get_spike_data(self, cluster_id: int) -> np.ndarray:
        """
        Fetch per-spike times and amplitudes for *cluster_id*.

        Returns
        -------
        data : np.ndarray, shape (n_spikes, 2), float64
            column 0 = spike time (seconds)
            column 1 = spike amplitude
        """
        _, array = self._call(CMD_GET_SPIKE_DATA, cluster_id=cluster_id)
        if array is None:
            raise TransportError("server sent no array for get_spike_data")
        return array

    def get_correlograms(
        self,
        cluster_ids: "list[int]",
        bin_size: float = 1e-3,
        window_size: float = 50e-3,
    ) -> "tuple[dict, np.ndarray]":
        """
        Fetch cross- and auto-correlograms for *cluster_ids*.

        Returns
        -------
        header : dict  with keys cluster_ids, bin_size, window_size, n_bins, firing_rates
        ccg    : np.ndarray, shape (n_clusters, n_clusters, n_bins), float32
        """
        header, array = self._call(
            CMD_GET_CORRELOGRAMS,
            cluster_ids=[int(c) for c in cluster_ids],
            bin_size=bin_size,
            window_size=window_size,
        )
        if array is None:
            raise TransportError("server sent no array for get_correlograms")
        return header, array

    def get_raster_data(self, n_spikes: int = 200) -> "tuple[dict, np.ndarray]":
        """
        Fetch evenly-subsampled spike data for every cluster (for the raster view).

        Returns
        -------
        header : dict  with keys cluster_ids (list[int]) and duration (float seconds)
        data   : np.ndarray, shape (n_total, 2), float32  [time_s, cluster_id]
        """
        header, array = self._call(CMD_GET_RASTER_DATA, n_spikes=n_spikes)
        if array is None:
            array = np.empty((0, 2), dtype=np.float32)
        return header, array

    def get_feature_spike_data(self, cluster_id: int) -> "np.ndarray | None":
        """
        Fetch per-spike times and PC-0 feature amplitude for *cluster_id*.

        Matches phy's 'feature' amplitude type (PC-0 on best channel).

        Returns None if pc_features.npy is not available for this dataset.
        """
        header, array = self._call(CMD_GET_FEATURE_SPIKE_DATA, cluster_id=cluster_id)
        if not header.get("available", True):
            return None
        if array is None:
            raise TransportError("server sent no array for get_feature_spike_data")
        return array

    def get_background_features(
        self,
        channel_ids: "list[int]",
        n_spikes: int = 2500,
    ) -> "np.ndarray | None":
        """
        Fetch PC features for a random background sample of all spikes on *channel_ids*.

        Returns None if pc_features.npy is not available.
        Returns (n_spikes, n_channels, n_pcs) float32 array on success.
        """
        header, array = self._call(
            CMD_GET_BACKGROUND_FEATURES,
            channel_ids=[int(c) for c in channel_ids],
            n_spikes=n_spikes,
        )
        if not header.get("available", True):
            return None
        if array is None:
            raise TransportError("server sent no array for get_background_features")
        return array

    def get_background_spike_data(
        self,
        cluster_id: int,
        excluded_ids: "list[int]",
        n_spikes: int = 10_000,
    ) -> np.ndarray:
        """
        Fetch merged spike data for all other clusters on the same best channel.

        Returns
        -------
        data : np.ndarray, shape (n_spikes, 2), float64
            column 0 = spike time (seconds), column 1 = amplitude
            Empty (0, 2) array when no background clusters exist.
        """
        _, array = self._call(
            CMD_GET_BACKGROUND_SPIKE_DATA,
            cluster_id=cluster_id,
            excluded_ids=[int(x) for x in excluded_ids],
            n_spikes=n_spikes,
        )
        return array if array is not None else np.empty((0, 2), dtype=np.float64)

    def merge_clusters(self, cluster_ids: "list[int]") -> dict:
        """
        Merge two or more clusters into a new one.

        Returns
        -------
        header : dict  with keys new_cluster_id, merged_ids, clusters
        """
        header, _ = self._call(CMD_MERGE, cluster_ids=[int(c) for c in cluster_ids])
        return header

    def undo(self) -> dict:
        """Undo the last merge. Returns header with 'clusters' list."""
        header, _ = self._call(CMD_UNDO)
        return header

    def redo(self) -> dict:
        """Redo the last undone merge. Returns header with 'clusters' list."""
        header, _ = self._call(CMD_REDO)
        return header

    def save(self) -> None:
        """Save spike_clusters.npy and cluster_group.tsv to disk on the server."""
        self._call(CMD_SAVE)

    def label_cluster(
        self, cluster_ids: "int | list[int]", label: str
    ) -> None:
        """
        Set the label for one or more clusters and save to disk on the server.

        Parameters
        ----------
        cluster_ids : int or list[int]
        label : "good" | "mua" | "noise" | "unsorted"
        """
        if isinstance(cluster_ids, int):
            cluster_ids = [cluster_ids]
        self._call(CMD_LABEL_CLUSTER, cluster_ids=cluster_ids, label=label)

    def get_dataset_list(self) -> "tuple[list[dict], str]":
        """
        Return (datasets, current_label).

        datasets : list of dicts with keys 'label' and 'params_path'
        current_label : str  label of the currently loaded dataset
        """
        header, _ = self._call(CMD_GET_DATASET_LIST)
        return header.get("datasets", []), header.get("current", "")

    def switch_dataset(self, label: str) -> str:
        """
        Tell the server to reload from the dataset identified by *label*.

        Returns the new current label on success.
        Raises TransportError on failure.
        """
        header, _ = self._call(CMD_SWITCH_DATASET, label=label)
        return header.get("current", label)

    # ------------------------------------------------------------------
    # Low-level send/recv
    # ------------------------------------------------------------------

    def _call(self, cmd: str, **kwargs) -> tuple[dict, "np.ndarray | None"]:
        frames = encode_request(cmd, **kwargs)
        self._sock.send_multipart(frames)
        reply = self._sock.recv_multipart()
        header, array = decode_response(reply)
        if header.get("status") == "error":
            raise TransportError(header.get("error", "unknown server error"))
        if header.get("status") == "loading":
            raise ServerLoadingError(header.get("error", "model still loading"))
        return header, array
