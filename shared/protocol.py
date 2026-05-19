"""
ZMQ message protocol for phy-remote.

Wire format (multipart message):
  Frame 0 — JSON-encoded header  (always present)
  Frame 1 — raw numpy bytes      (only when header["has_array"] is True)

Request header fields:
  cmd      str    command name (see CMD_* constants)
  **kwargs        command-specific keyword arguments

Response header fields:
  status   str    "ok" | "error"
  error    str    error message (only when status == "error")
  has_array bool  True when frame 1 is present
  dtype    str    numpy dtype string  (only when has_array)
  shape    list   array shape as list of ints (only when has_array)
"""

import json
import numpy as np

# ---------------------------------------------------------------------------
# Command constants
# ---------------------------------------------------------------------------

CMD_PING = "ping"
CMD_GET_WAVEFORMS = "get_waveforms"
CMD_GET_SPIKE_TIMES = "get_spike_times"
CMD_GET_FEATURES = "get_features"
CMD_GET_TEMPLATES = "get_templates"
CMD_GET_CLUSTER_IDS = "get_cluster_ids"
CMD_GET_CLUSTER_INFO = "get_cluster_info"   # all clusters: id, label, n_spikes, amplitude
CMD_LABEL_CLUSTER = "label_cluster"          # set good/mua/noise/unsorted, save to disk
CMD_GET_SPIKE_DATA = "get_spike_data"        # (n_spikes, 2) float64: [time_s, amplitude]
CMD_GET_CHANNEL_POSITIONS = "get_channel_positions"  # (n_channels, 2) float32: [x, y] µm
CMD_GET_TRACES = "get_traces"                        # (n_ch, n_samples) float32 raw/HP traces
CMD_GET_SPIKES_IN_WINDOW = "get_spikes_in_window"    # (n_spikes, 2) float64: [time, cluster_id]
CMD_GET_SIMILAR_CLUSTERS = "get_similar_clusters"    # header-only ranked list for one cluster
CMD_GET_TEMPLATE_FEATURES = "get_template_features"  # per-spike template feature vectors
CMD_GET_CLUSTER_BEST_CHANNELS = "get_cluster_best_channels"  # header: best_channels dict
CMD_GET_BACKGROUND_SPIKE_DATA = "get_background_spike_data"  # grey backdrop for amp view
CMD_GET_FEATURE_SPIKE_DATA = "get_feature_spike_data"        # (n,2) [time, PC0_best_ch]
CMD_GET_CORRELOGRAMS = "get_correlograms"  # (n_cl, n_cl, n_bins) float32 CCG array
CMD_GET_RASTER_DATA = "get_raster_data"   # all-cluster subsampled spike times for raster view
CMD_GET_BACKGROUND_FEATURES = "get_background_features"  # (n_spikes, n_ch, n_pc) float32 on given channels
CMD_GET_DATASET_LIST = "get_dataset_list"  # header: datasets list + current label
CMD_SWITCH_DATASET   = "switch_dataset"    # reload model from params_path; header: ok/error
CMD_MERGE = "merge"         # merge cluster_ids → new cluster, returns new_cluster_id
CMD_UNDO  = "undo"          # undo last merge/split
CMD_REDO  = "redo"          # redo
CMD_SAVE  = "save"          # write spike_clusters.npy + cluster_group.tsv to disk

# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_request(cmd: str, **kwargs) -> list[bytes]:
    """Return a single-frame multipart message for a command request."""
    header = {"cmd": cmd, **kwargs}
    return [json.dumps(header).encode()]


def decode_request(frames: list[bytes]) -> dict:
    """Parse an incoming request from the server side."""
    return json.loads(frames[0])


def encode_response(
    *,
    status: str = "ok",
    error: str = "",
    array: "np.ndarray | None" = None,
    **extra,
) -> list[bytes]:
    """
    Build a response multipart message.

    Parameters
    ----------
    status : "ok" | "error"
    error  : error string (when status == "error")
    array  : optional numpy array to attach as frame 1
    **extra: additional metadata fields written into the header
    """
    header: dict = {"status": status, "has_array": array is not None, **extra}
    if status == "error":
        header["error"] = error
    if array is not None:
        header["dtype"] = array.dtype.str   # e.g. "<f4"
        header["shape"] = list(array.shape)
        return [json.dumps(header).encode(), array.tobytes()]
    return [json.dumps(header).encode()]


def decode_response(frames: list[bytes]) -> tuple[dict, "np.ndarray | None"]:
    """
    Parse a server response.

    Returns
    -------
    header : dict
    array  : np.ndarray or None
    """
    header = json.loads(frames[0])
    array = None
    if header.get("has_array"):
        dtype = np.dtype(header["dtype"])
        shape = tuple(header["shape"])
        array = np.frombuffer(frames[1], dtype=dtype).reshape(shape)
    return header, array
