"""
Round-trip integration tests for the ZMQ transport layer.

A PhyServer is started in a background thread with a FakeModel; a
PhyTransport connects to it over tcp://127.0.0.1:<ephemeral port>.

No phy dataset or GPU is required — the server is headless by design.
"""

import threading
import time

import numpy as np
import pytest

from phy_remote.server.server import PhyServer
from phy_remote.client.transport import PhyTransport, TransportError


# ---------------------------------------------------------------------------
# Fake model — mimics the TemplateModel surface used by the server
# ---------------------------------------------------------------------------

class FakeModel:
    """Minimal stand-in for phylib TemplateModel."""

    N_SPIKES = 200
    N_SAMPLES = 82
    N_CHANNELS = 4
    SAMPLE_RATE = 30_000.0

    def __init__(self):
        rng = np.random.default_rng(0)
        # spike_times in seconds, sorted
        self.spike_times = np.sort(rng.uniform(0, 100, self.N_SPIKES))
        # cluster 0 owns all spikes; cluster 1 owns the first 50
        self._cluster_spikes = {
            0: np.arange(self.N_SPIKES),
            1: np.arange(50),
        }
        # waveforms: (N_SPIKES, N_SAMPLES, N_CHANNELS)
        self._waveforms = rng.standard_normal(
            (self.N_SPIKES, self.N_SAMPLES, self.N_CHANNELS)
        ).astype(np.float32)

    def get_cluster_spikes(self, cluster_id: int) -> np.ndarray:
        return self._cluster_spikes[cluster_id]

    def get_waveforms(self, spike_ids: np.ndarray) -> np.ndarray:
        return self._waveforms[spike_ids]

    def get_features(self, spike_ids, channel_ids):
        # Return a Bunch-like object with a .data attribute
        from types import SimpleNamespace
        rng = np.random.default_rng(1)
        data = rng.standard_normal((len(spike_ids), 3)).astype(np.float32)
        return SimpleNamespace(data=data)


# ---------------------------------------------------------------------------
# Pytest fixture: server + transport pair on an ephemeral port
# ---------------------------------------------------------------------------

def _find_free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def server_transport():
    port = _find_free_port()
    model = FakeModel()
    server = PhyServer(model=model, port=port, host="127.0.0.1")

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)  # give the server time to enter its recv loop

    transport = PhyTransport(host="127.0.0.1", port=port, timeout_ms=5_000)
    yield transport, model

    transport.close()
    server.stop()           # sets _running=False; poller loop exits within 100 ms
    thread.join(timeout=2)  # wait for serve_forever to close the socket
    server.close()          # term the context now that the socket is gone


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPing:
    def test_ping_returns_true(self, server_transport):
        transport, _ = server_transport
        assert transport.ping() is True


class TestGetWaveforms:
    def test_shape_all_spikes(self, server_transport):
        transport, model = server_transport
        header, waveforms = transport.get_waveforms(cluster_id=0, n_spikes=200)

        assert waveforms.ndim == 3
        assert waveforms.shape[1] == model.N_SAMPLES
        assert waveforms.shape[2] == model.N_CHANNELS
        assert waveforms.dtype == np.float32

    def test_n_spikes_capped(self, server_transport):
        """Requesting more spikes than exist should return all available."""
        transport, model = server_transport
        _, waveforms = transport.get_waveforms(cluster_id=1, n_spikes=200)
        # cluster 1 has only 50 spikes
        assert waveforms.shape[0] == 50

    def test_n_spikes_subsampled(self, server_transport):
        transport, model = server_transport
        n = 30
        _, waveforms = transport.get_waveforms(cluster_id=0, n_spikes=n)
        assert waveforms.shape[0] == n

    def test_values_match_model(self, server_transport):
        """Values returned must exactly match what the fake model holds."""
        transport, model = server_transport
        # Ask for all spikes so order is deterministic (no subsampling)
        _, waveforms = transport.get_waveforms(cluster_id=1, n_spikes=200)
        expected = model.get_waveforms(model.get_cluster_spikes(1))
        np.testing.assert_array_equal(waveforms, expected)

    def test_dtype_is_float32(self, server_transport):
        transport, _ = server_transport
        _, waveforms = transport.get_waveforms(cluster_id=0, n_spikes=10)
        assert waveforms.dtype == np.float32

    def test_header_contains_cluster_id(self, server_transport):
        transport, _ = server_transport
        header, _ = transport.get_waveforms(cluster_id=0, n_spikes=10)
        assert header["cluster_id"] == 0


class TestGetSpikeTimes:
    def test_shape_and_dtype(self, server_transport):
        transport, model = server_transport
        header, times = transport.get_spike_times(cluster_id=0)
        assert times.shape == (model.N_SPIKES,)
        assert times.dtype == np.float64

    def test_values_match_model(self, server_transport):
        transport, model = server_transport
        _, times = transport.get_spike_times(cluster_id=0)
        np.testing.assert_array_equal(times, model.spike_times)


class TestGetFeatures:
    def test_shape(self, server_transport):
        transport, _ = server_transport
        _, features = transport.get_features(cluster_id=1)
        # cluster 1 has 50 spikes, FakeModel returns 3 PCs
        assert features.shape == (50, 3)
        assert features.dtype == np.float32


class TestErrorHandling:
    def test_unknown_command(self, server_transport):
        """Protocol-level unknown command should raise TransportError."""
        from phy_remote.shared.protocol import encode_request, decode_response
        transport, _ = server_transport
        # Bypass the high-level API and send a raw unknown command
        frames = encode_request("not_a_real_command")
        transport._sock.send_multipart(frames)
        reply = transport._sock.recv_multipart()
        header, _ = decode_response(reply)
        assert header["status"] == "error"
        assert "unknown command" in header["error"]

    def test_missing_cluster_id_raises(self, server_transport):
        """Server should return an error (not crash) on missing required arg."""
        from phy_remote.shared.protocol import encode_request, decode_response
        transport, _ = server_transport
        frames = encode_request("get_waveforms")  # no cluster_id
        transport._sock.send_multipart(frames)
        reply = transport._sock.recv_multipart()
        header, _ = decode_response(reply)
        assert header["status"] == "error"
