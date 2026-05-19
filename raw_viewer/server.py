"""Raw-file viewer server — runs headless on the HPC cluster.

Serves chunks from a flat binary file (int16, channels-last) over ZMQ REQ/REP.

Run as:
    python -m phy_remote.raw_viewer.server /path/to/file.raw \\
        --n-channels 384 --sample-rate 30000 --port 5560
"""
from __future__ import annotations

import json
import logging
import socket
import sys

import numpy as np
import zmq

logger = logging.getLogger(__name__)

CMD_PING       = "ping"
CMD_GET_INFO   = "get_info"
CMD_GET_TRACES = "get_traces"
CMD_GET_FRAME  = "get_frame"


def _encode(*, array=None, **fields) -> list[bytes]:
    hdr = {"status": "ok", "has_array": array is not None, **fields}
    if array is not None:
        hdr["dtype"] = array.dtype.str
        hdr["shape"] = list(array.shape)
        return [json.dumps(hdr).encode(), array.tobytes()]
    return [json.dumps(hdr).encode()]


def _error(msg: str) -> list[bytes]:
    return [json.dumps({"status": "error", "has_array": False, "error": msg}).encode()]


class RawServer:
    def __init__(
        self,
        raw_path: str,
        n_channels: int,
        sample_rate: float,
        dtype: str = "int16",
        port: int = 5560,
        host: str = "0.0.0.0",
        video_path: str | None = None,
    ):
        self.n_channels  = n_channels
        self.sample_rate = sample_rate
        self.port        = port

        # Video (optional)
        self._video_cap  = None
        self._video_fps  = 0.0
        self._video_n    = 0
        if video_path:
            self._open_video(video_path)

        logger.info("Mapping %s  n_ch=%d  sr=%.0f  dtype=%s", raw_path, n_channels, sample_rate, dtype)
        raw = np.memmap(raw_path, dtype=dtype, mode="r")
        n_samples = len(raw) // n_channels
        self._traces = raw[:n_samples * n_channels].reshape(n_samples, n_channels)
        self._duration = n_samples / sample_rate
        logger.info("  %d samples  %.1f s", n_samples, self._duration)

        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.bind(f"tcp://{host}:{port}")

    def _open_video(self, path: str) -> None:
        try:
            import cv2
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise RuntimeError(f"cv2 could not open {path}")
            self._video_cap = cap
            self._video_fps = cap.get(5) or 30.0
            self._video_n   = int(cap.get(7))
            logger.info("Video: %s  %.2f fps  %d frames", path, self._video_fps, self._video_n)
        except ImportError:
            logger.warning("cv2 not available — video serving disabled")

    def serve_forever(self) -> None:
        hostname = socket.gethostname().split(".")[0]
        print(f"\nPHY_REMOTE_READY host={hostname} port={self.port}\n", flush=True)
        print(f"On your MacBook:\n"
              f"  ssh -N -L {self.port}:{hostname}:{self.port} "
              f"<user>@login1.int.janelia.org\n"
              f"  python -m phy_remote.raw_viewer.client --port {self.port} "
              f"--n-channels {self.n_channels} --sample-rate {self.sample_rate:.0f} "
              f"--duration {self._duration:.2f}\n", flush=True)

        while True:
            try:
                frames = self._sock.recv_multipart()
                req    = json.loads(frames[0])
                cmd    = req.get("cmd", "")
                resp   = self._dispatch(cmd, req)
                self._sock.send_multipart(resp)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.exception("Handler error: %s", exc)
                try:
                    self._sock.send_multipart(_error(str(exc)))
                except Exception:
                    pass

    def _dispatch(self, cmd: str, req: dict) -> list[bytes]:
        if cmd == CMD_PING:
            return _encode(pong=True)
        if cmd == CMD_GET_INFO:
            return _encode(
                n_channels=self.n_channels,
                sample_rate=self.sample_rate,
                duration=self._duration,
                n_samples=int(self._traces.shape[0]),
                video_fps=self._video_fps,
                video_n_frames=self._video_n,
                has_video=self._video_cap is not None,
            )
        if cmd == CMD_GET_TRACES:
            return self._handle_get_traces(req)
        if cmd == CMD_GET_FRAME:
            return self._handle_get_frame(req)
        return _error(f"unknown command: {cmd!r}")

    def _handle_get_traces(self, req: dict) -> list[bytes]:
        sr          = self.sample_rate
        t_start     = float(req.get("t_start", 0.0))
        t_end       = float(req.get("t_end",   t_start + 0.25))
        channel_ids = req.get("channel_ids", None)
        do_filter   = bool(req.get("filter",   False))
        max_samples = int(req.get("max_samples", 3000))

        s_start = max(0, int(t_start * sr))
        s_end   = min(self._traces.shape[0], int(t_end * sr))
        if s_start >= s_end:
            return _error("empty time range")

        chunk = np.array(self._traces[s_start:s_end], dtype=np.float32)  # (n_s, n_ch)

        if channel_ids is not None:
            chunk = chunk[:, list(channel_ids)]
            out_ch_ids = list(channel_ids)
        else:
            out_ch_ids = list(range(chunk.shape[1]))

        chunk = chunk.T  # (n_ch, n_s)

        if do_filter:
            try:
                from scipy.signal import butter, sosfiltfilt
                sos   = butter(3, 300.0 / (sr / 2.0), btype="high", output="sos")
                chunk = sosfiltfilt(sos, chunk, axis=1).astype(np.float32)
            except Exception as exc:
                logger.warning("HP filter failed: %s", exc)

        if chunk.shape[1] > max_samples:
            step  = chunk.shape[1] // max_samples
            chunk = chunk[:, ::step]
            actual_t_end = t_start + (chunk.shape[1] * step) / sr
        else:
            actual_t_end = s_end / sr

        return _encode(
            array=chunk,
            sample_rate=sr,
            t_start=s_start / sr,
            t_end=actual_t_end,
            channel_ids=out_ch_ids,
        )

    def _handle_get_frame(self, req: dict) -> list[bytes]:
        if self._video_cap is None:
            return _error("no video loaded on server")
        import cv2
        t       = float(req.get("t", 0.0))
        quality = int(req.get("quality", 85))
        idx     = int(round(t * self._video_fps))
        idx     = max(0, min(idx, self._video_n - 1))
        self._video_cap.set(1, idx)
        ok, bgr = self._video_cap.read()
        if not ok:
            return _error(f"could not read frame {idx}")
        ok2, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok2:
            return _error("jpeg encode failed")
        jpeg_bytes = buf.tobytes()
        hdr = json.dumps({
            "status": "ok", "has_array": False,
            "frame_idx": idx, "t": t,
            "fps": self._video_fps, "n_frames": self._video_n,
            "jpeg": True,
        }).encode()
        return [hdr, jpeg_bytes]
