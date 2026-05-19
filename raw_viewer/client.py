"""Raw-file viewer client — fully automated launch from MacBook.

Submits a bsub job on the Janelia cluster, waits for the server to start,
opens an SSH tunnel automatically, then connects.

Run as:
    python -m phy_remote.raw_viewer.client
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
from typing import Any



import numpy as np
import zmq
from PyQt6.QtCore import (
    QEvent, QObject, QSettings, Qt, QThread, QTimer, pyqtSignal,
)
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPlainTextEdit,
    QPushButton, QScrollBar, QSizePolicy, QSpinBox, QSplitter, QTabWidget,
    QVBoxLayout, QWidget,
)

logger = logging.getLogger(__name__)

_WINDOW_S     = 1.0
_MAX_SAMPLES  = 4_000
_TRACE_COLOR  = (0.45, 0.45, 0.45, 0.90)
_N_CH_SHOW    = 48
_POLL_INTERVAL = 3          # seconds between bpeek polls
_LAUNCH_TIMEOUT = 360       # seconds before giving up
_SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=15",
]

# ---------------------------------------------------------------------------
# ZMQ transport
# ---------------------------------------------------------------------------

class RawTransport:
    def __init__(self, host: str = "127.0.0.1", port: int = 5560,
                 timeout_ms: int = 30_000):
        self._ctx  = zmq.Context()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
        self._sock.connect(f"tcp://{host}:{port}")

    def close(self) -> None:
        self._sock.close(linger=0)
        self._ctx.term()

    def _call(self, **req) -> tuple[dict, "np.ndarray | None"]:
        self._sock.send_multipart([json.dumps(req).encode()])
        frames = self._sock.recv_multipart()
        hdr    = json.loads(frames[0])
        if hdr.get("status") == "error":
            raise RuntimeError(hdr.get("error", "unknown error"))
        arr = None
        if hdr.get("has_array") and len(frames) > 1:
            arr = np.frombuffer(frames[1], dtype=hdr["dtype"]).reshape(hdr["shape"])
        return hdr, arr

    def ping(self) -> bool:
        hdr, _ = self._call(cmd="ping")
        return hdr.get("pong", False)

    def get_info(self) -> dict:
        hdr, _ = self._call(cmd="get_info")
        return hdr

    def get_traces(self, t_start: float, t_end: float,
                   channel_ids: "list[int] | None" = None,
                   filtered: bool = False,
                   max_samples: int = _MAX_SAMPLES) -> "tuple[np.ndarray, dict]":
        req: dict = dict(cmd="get_traces", t_start=t_start, t_end=t_end,
                         filter=filtered, max_samples=max_samples)
        if channel_ids is not None:
            req["channel_ids"] = channel_ids
        hdr, arr = self._call(**req)
        if arr is None:
            raise RuntimeError("server returned no array")
        return arr, hdr

    def get_frame(self, t: float, quality: int = 85) -> "tuple[bytes, dict] | None":
        """Fetch a JPEG-encoded video frame at time t (seconds).
        Returns (jpeg_bytes, header) or None if no video on server."""
        self._sock.send_multipart([json.dumps(
            dict(cmd="get_frame", t=t, quality=quality)
        ).encode()])
        frames = self._sock.recv_multipart()
        hdr = json.loads(frames[0])
        if hdr.get("status") == "error":
            return None
        jpeg = frames[1] if len(frames) > 1 else None
        return jpeg, hdr


# ---------------------------------------------------------------------------
# Cluster launcher (runs in a QThread)
# ---------------------------------------------------------------------------

class ClusterLauncher(QThread):
    """
    Submits a bsub job on the Janelia cluster, polls bpeek until
    PHY_REMOTE_READY appears, then emits ready(compute_node, port).
    """
    log       = pyqtSignal(str)
    ready     = pyqtSignal(str, int, str)   # (compute_node, port, job_id)
    failed    = pyqtSignal(str)

    def __init__(self, cfg: dict, parent=None):
        super().__init__(parent)
        self._cfg = cfg
        self._stop = False
        self.job_id: str = ""

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        cfg = self._cfg
        user       = cfg["user"]
        login_node = cfg["login_node"]
        raw_file   = cfg["raw_file"]
        n_channels = cfg["n_channels"]
        sample_rate = cfg["sample_rate"]
        port       = cfg["port"]
        python_path = cfg["python_path"]
        queue      = cfg["queue"]
        walltime   = cfg["walltime"]

        ssh_target = f"{user}@{login_node}"

        # ------------------------------------------------------------------ #
        # 1. Submit bsub job
        # ------------------------------------------------------------------ #
        server_cmd = (
            f"{python_path} -m phy_remote.raw_viewer "
            f"{raw_file} "
            f"--n-channels {n_channels} "
            f"--sample-rate {sample_rate:.0f} "
            f"--port {port}"
        )
        bsub_cmd = (
            f"bsub -J phy-raw -n 1 -q {queue} -gpu 'num=1' -W {walltime} '{server_cmd}'"
        )
        self.log.emit(f"→ Submitting job on {login_node} …")
        self.log.emit(f"  {bsub_cmd}")

        try:
            result = subprocess.run(
                ["ssh"] + _SSH_OPTS + [ssh_target, bsub_cmd],
                capture_output=True, text=True, timeout=30,
            )
        except subprocess.TimeoutExpired:
            self.failed.emit("SSH timed out when submitting bsub job.")
            return
        except Exception as exc:
            self.failed.emit(f"SSH error: {exc}")
            return

        combined = result.stdout + result.stderr
        self.log.emit(f"  bsub reply: {combined.strip()}")

        m = re.search(r"Job <(\d+)>", combined)
        if not m:
            self.failed.emit(f"Could not parse job ID from bsub output:\n{combined}")
            return

        job_id = m.group(1)
        self.job_id = job_id
        self.log.emit(f"  Job ID: {job_id}")

        # ------------------------------------------------------------------ #
        # 2. Poll bpeek until PHY_REMOTE_READY
        # ------------------------------------------------------------------ #
        self.log.emit("Waiting for job to start (polling bpeek) …")
        deadline = time.monotonic() + _LAUNCH_TIMEOUT
        compute_node = None

        while time.monotonic() < deadline:
            if self._stop:
                return

            time.sleep(_POLL_INTERVAL)

            try:
                peek = subprocess.run(
                    ["ssh"] + _SSH_OPTS + [ssh_target, f"bpeek {job_id} 2>&1"],
                    capture_output=True, text=True, timeout=20,
                )
                output = peek.stdout + peek.stderr
            except Exception as exc:
                self.log.emit(f"  bpeek error (retrying): {exc}")
                continue

            # Check for EXITED / error states
            if "is not found" in output or "EXITED" in output:
                self.failed.emit(f"Job {job_id} exited before becoming ready:\n{output}")
                return

            # Detect port conflict
            if "Address already in use" in output:
                self.failed.emit(
                    f"Port {port} is already in use on the cluster.\n\n"
                    f"A previous phy-raw job is probably still running. Kill it with:\n"
                    f"  ssh {ssh_target} \"bkill -J phy-raw\"\n\n"
                    f"Or change the port number in the dialog and try again."
                )
                # Also kill this new (failed) job
                subprocess.Popen(
                    ["ssh"] + _SSH_OPTS + [ssh_target, f"bkill {job_id}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                return

            # Show any new lines (avoid spamming "No output yet")
            if output.strip() and "No output yet" not in output:
                last_lines = output.strip().splitlines()[-4:]
                for line in last_lines:
                    self.log.emit(f"  [server] {line}")

            m2 = re.search(r"PHY_REMOTE_READY\s+host=(\S+)\s+port=(\d+)", output)
            if m2:
                compute_node = m2.group(1)
                actual_port  = int(m2.group(2))
                self.log.emit(f"  Server ready on {compute_node}:{actual_port}")
                self.ready.emit(compute_node, actual_port, job_id)
                return

        self.failed.emit(
            f"Timed out after {_LAUNCH_TIMEOUT}s waiting for job {job_id} to start.\n"
            f"Check with:  bjobs {job_id}\n"
            f"             bpeek {job_id}"
        )


# ---------------------------------------------------------------------------
# Launch dialog
# ---------------------------------------------------------------------------

_SETTINGS_ORG  = "phy-remote"
_SETTINGS_APP  = "raw-viewer"


class LaunchDialog(QDialog):
    """Two-tab dialog: 'Launch on Cluster' and 'Connect (manual tunnel)'."""

    # emitted when connection is fully established
    connected = pyqtSignal(str, int, dict)     # host, port, info
    tunnel_proc = None                          # kept alive until window closes

    _job_id:    str = ""
    _ssh_target: str = ""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Raw Viewer — Connect")
        self.setMinimumWidth(520)

        self._s = QSettings(_SETTINGS_ORG, _SETTINGS_APP)
        self._launcher: ClusterLauncher | None = None
        self._tunnel:   subprocess.Popen | None = None

        tabs = QTabWidget()
        tabs.addTab(self._build_launch_tab(), "Launch on Cluster")
        tabs.addTab(self._build_manual_tab(), "Connect (manual tunnel)")

        vl = QVBoxLayout(self)
        vl.addWidget(tabs)

    # ------------------------------------------------------------------
    # Tab: Launch on Cluster
    # ------------------------------------------------------------------

    def _build_launch_tab(self) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)

        form = QFormLayout()
        s = self._s

        self._user       = _le(s.value("user",       os.environ.get("USER", "")))
        self._login_node = _le(s.value("login_node", "login1.int.janelia.org"))
        self._raw_file   = _le(s.value("raw_file",   ""))
        self._n_channels = _spin(1, 10000, int(s.value("n_channels", 384)))
        self._sample_rate = _dspin(1000, 200000, float(s.value("sample_rate", 30000)), step=100)
        self._port_launch = _spin(1024, 65535, int(s.value("port_launch", 5560)))
        default_cluster_user = os.environ.get("USER", "<user>")
        self._python_path = _le(s.value("python_path",
            f"/groups/scicompsoft/home/{default_cluster_user}/miniconda3/envs/phy/bin/python"))
        self._queue      = _le(s.value("queue",    "gpu_l4"))
        self._walltime   = _le(s.value("walltime",  "120"))

        self._raw_file.setPlaceholderText("/groups/voigts/…/recording.raw")

        # Probe file — local MacBook path OR cluster path (will be scp'd automatically)
        self._probe_local  = _le(s.value("probe_local",   ""))
        self._probe_local.setPlaceholderText("local channel_positions.npy  (browse →)")
        self._probe_remote = _le(s.value("probe_remote",  ""))
        self._probe_remote.setPlaceholderText("cluster path, e.g. /groups/…/channel_positions.npy")

        probe_local_row = QWidget()
        pl_hl = QHBoxLayout(probe_local_row)
        pl_hl.setContentsMargins(0, 0, 0, 0); pl_hl.setSpacing(4)
        pl_hl.addWidget(self._probe_local)
        probe_browse = QPushButton("Browse…")
        probe_browse.setFixedWidth(70)
        probe_browse.clicked.connect(self._browse_probe)
        pl_hl.addWidget(probe_browse)

        form.addRow("SSH user:",               self._user)
        form.addRow("Login node:",             self._login_node)
        form.addRow("Raw file (cluster):",     self._raw_file)
        form.addRow("n_channels:",             self._n_channels)
        form.addRow("sample_rate (Hz):",       self._sample_rate)
        form.addRow("Port:",                   self._port_launch)
        form.addRow("Probe — local file:",     probe_local_row)
        form.addRow("Probe — cluster path:",   self._probe_remote)
        form.addRow("Python path:",            self._python_path)
        form.addRow("LSF queue:",              self._queue)
        form.addRow("Walltime (min):",         self._walltime)
        vl.addLayout(form)

        # Log area
        self._log = QPlainTextEdit()
        self._log.setReadOnly(True)
        self._log.setFixedHeight(150)
        self._log.setStyleSheet(
            "background:#111;color:#aaa;font-family:monospace;font-size:11px;"
        )
        vl.addWidget(self._log)

        # Buttons
        hl = QHBoxLayout()
        self._launch_btn = QPushButton("Launch && Connect")
        self._launch_btn.setDefault(True)
        self._launch_btn.clicked.connect(self._on_launch)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.clicked.connect(self._on_cancel)
        hl.addWidget(self._launch_btn)
        hl.addWidget(self._cancel_btn)
        vl.addLayout(hl)

        return w

    # ------------------------------------------------------------------
    # Tab: manual tunnel
    # ------------------------------------------------------------------

    def _build_manual_tab(self) -> QWidget:
        w  = QWidget()
        vl = QVBoxLayout(w)

        form = QFormLayout()
        s = self._s
        self._man_host = _le(s.value("man_host", "127.0.0.1"))
        self._man_port = _spin(1024, 65535, int(s.value("man_port", 5560)))
        form.addRow("Host:", self._man_host)
        form.addRow("Port:", self._man_port)
        vl.addLayout(form)

        hint = QLabel(
            "Start the server on the cluster:\n"
            "  python -m phy_remote.raw_viewer /path/to/file.raw \\\n"
            "      --n-channels 384 --port 5560\n\n"
            "Open the tunnel:\n"
            "  ssh -N -L 5560:<compute_node>:5560 <user>@login1.int.janelia.org"
        )
        hint.setStyleSheet(
            "font-family:monospace;font-size:11px;color:#888;"
            "background:#1a1a1a;padding:8px;border-radius:3px;"
        )
        hint.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        vl.addWidget(hint)
        vl.addStretch()

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        btns.accepted.connect(self._on_manual_connect)
        btns.rejected.connect(self.reject)
        vl.addWidget(btns)
        return w

    # ------------------------------------------------------------------
    # Launch flow
    # ------------------------------------------------------------------

    def _on_launch(self) -> None:
        self._save_settings()
        self._log.clear()
        self._launch_btn.setEnabled(False)

        cfg = dict(
            user        = self._user.text().strip(),
            login_node  = self._login_node.text().strip(),
            raw_file    = self._raw_file.text().strip(),
            n_channels  = self._n_channels.value(),
            sample_rate = self._sample_rate.value(),
            port        = self._port_launch.value(),
            python_path = self._python_path.text().strip(),
            queue       = self._queue.text().strip(),
            walltime    = self._walltime.text().strip(),
        )

        if not cfg["raw_file"]:
            self._append_log("ERROR: Raw file path is required.")
            self._launch_btn.setEnabled(True)
            return

        self._launcher = ClusterLauncher(cfg, parent=self)
        self._launcher.log.connect(self._append_log)
        self._launcher.ready.connect(self._on_server_ready)
        self._launcher.failed.connect(self._on_launch_failed)
        self._launcher.start()

        self._pending_cfg = cfg

    def _on_server_ready(self, compute_node: str, port: int, job_id: str) -> None:
        """Server is up — scp probe if needed, open SSH tunnel, then connect ZMQ."""
        cfg = self._pending_cfg
        ssh_target = f"{cfg['user']}@{cfg['login_node']}"
        self._job_id     = job_id
        self._ssh_target = ssh_target

        self._resolve_probe()   # scp cluster probe file if specified

        self._append_log(f"Opening SSH tunnel: {port} → {compute_node}:{port} …")
        self._tunnel = subprocess.Popen(
            ["ssh"] + _SSH_OPTS + [
                "-N",
                "-L", f"{port}:{compute_node}:{port}",
                ssh_target,
            ],
        )

        # Give the tunnel a moment to establish before connecting ZMQ
        QTimer.singleShot(1500, lambda: self._try_connect("127.0.0.1", port))

    def _try_connect(self, host: str, port: int, attempts: int = 0) -> None:
        try:
            tr = RawTransport(host=host, port=port, timeout_ms=5_000)
            ok = tr.ping()
            info = tr.get_info() if ok else {}
            tr.close()
        except Exception as exc:
            if attempts < 10:
                self._append_log(f"  Tunnel not ready yet, retrying … ({exc})")
                QTimer.singleShot(1500, lambda: self._try_connect(host, port, attempts + 1))
                return
            self._on_launch_failed(f"Could not connect after {attempts} retries: {exc}")
            return

        self._append_log("Connected!")
        self.connected.emit(host, port, info)
        self.accept()

    def _on_launch_failed(self, msg: str) -> None:
        self._append_log(f"FAILED: {msg}")
        self._launch_btn.setEnabled(True)
        if self._tunnel:
            self._tunnel.terminate()
            self._tunnel = None

    def _on_cancel(self) -> None:
        if self._launcher:
            self._launcher.stop()
            self._launcher.quit()
        if self._tunnel:
            self._tunnel.terminate()
            self._tunnel = None
        self.reject()

    # ------------------------------------------------------------------
    # Manual connect
    # ------------------------------------------------------------------

    def _browse_probe(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Select channel_positions.npy",
            os.path.expanduser("~"),
            "NumPy (*.npy);;All files (*)",
        )
        if path:
            self._probe_local.setText(path)

    def get_probe_path(self) -> str:
        """Return resolved local probe path (scp already done if needed)."""
        return getattr(self, "_resolved_probe_path", "")

    def _resolve_probe(self) -> None:
        """scp cluster probe file if provided, otherwise use local path."""
        local  = self._probe_local.text().strip()
        remote = self._probe_remote.text().strip()

        if local and os.path.isfile(local):
            self._resolved_probe_path = local
            return

        if remote:
            cfg        = self._pending_cfg
            ssh_target = f"{cfg['user']}@{cfg['login_node']}"
            import tempfile
            ext = os.path.splitext(remote)[1] or ".json"
            tmp = tempfile.mktemp(suffix=f"_channel_positions{ext}")
            self._append_log(f"Downloading probe file from cluster …")
            try:
                result = subprocess.run(
                    ["scp"] + _SSH_OPTS + [f"{ssh_target}:{remote}", tmp],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and os.path.isfile(tmp):
                    self._resolved_probe_path = tmp
                    self._append_log(f"  Probe downloaded → {tmp}")
                else:
                    self._append_log(f"  scp failed: {result.stderr.strip()}")
                    self._resolved_probe_path = ""
            except Exception as exc:
                self._append_log(f"  scp error: {exc}")
                self._resolved_probe_path = ""
        else:
            self._resolved_probe_path = ""

    def _on_manual_connect(self) -> None:
        host = self._man_host.text().strip() or "127.0.0.1"
        port = self._man_port.value()
        self._s.setValue("man_host", host)
        self._s.setValue("man_port", port)
        try:
            tr   = RawTransport(host=host, port=port, timeout_ms=8_000)
            ok   = tr.ping()
            info = tr.get_info() if ok else {}
            tr.close()
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Connection failed", str(exc))
            return
        self.connected.emit(host, port, info)
        self.accept()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _append_log(self, text: str) -> None:
        self._log.appendPlainText(text)
        self._log.verticalScrollBar().setValue(
            self._log.verticalScrollBar().maximum()
        )

    def _save_settings(self) -> None:
        s = self._s
        s.setValue("user",        self._user.text())
        s.setValue("login_node",  self._login_node.text())
        s.setValue("raw_file",    self._raw_file.text())
        s.setValue("n_channels",  self._n_channels.value())
        s.setValue("sample_rate", self._sample_rate.value())
        s.setValue("port_launch", self._port_launch.value())
        s.setValue("python_path", self._python_path.text())
        s.setValue("queue",       self._queue.text())
        s.setValue("walltime",     self._walltime.text())
        s.setValue("probe_local",  self._probe_local.text())
        s.setValue("probe_remote", self._probe_remote.text())

    def get_tunnel(self) -> "subprocess.Popen | None":
        return self._tunnel

    def get_cluster_info(self) -> "tuple[str, str]":
        """Return (ssh_target, job_id) for bkill on close. Empty strings if manual connect."""
        return self._ssh_target, self._job_id


# ---------------------------------------------------------------------------
# Probe file loading
# ---------------------------------------------------------------------------

def _load_probe_file(path: str, n_ch_file: int) -> "tuple[np.ndarray, list]":
    """
    Load probe geometry from either:
      - channel_positions.npy  : (n_ch, 2) float array [x, y] µm, channel index = row
      - probeinterface JSON    : reads contact_positions + device_channel_indices + shank_ids

    Returns
    -------
    pos       : np.ndarray (n_ch_file, 2) float32 — [x, y] µm indexed by recording channel
    shank_ids : list[str|int] length n_ch_file — shank label per recording channel
    """
    if path.endswith(".npy"):
        pos = np.load(path)
        if pos.ndim != 2 or pos.shape[1] != 2:
            raise ValueError(f"Expected shape (n_ch, 2), got {pos.shape}")
        if pos.shape[0] != n_ch_file:
            raise ValueError(
                f"Probe has {pos.shape[0]} channels but recording has {n_ch_file}"
            )
        # Detect shanks by x-position gaps > 100 µm
        x_vals = np.round(pos[:, 0]).astype(int)
        shank_ids = [0] * n_ch_file
        shank_idx = 0
        prev_x = None
        x_to_shank: dict[int, int] = {}
        for x in sorted(np.unique(x_vals)):
            if prev_x is not None and x - prev_x > 100:
                shank_idx += 1
            x_to_shank[x] = shank_idx
            prev_x = x
        shank_ids = [x_to_shank[x] for x in x_vals]
        return pos.astype(np.float32), shank_ids

    elif path.endswith(".json"):
        import json as _json
        with open(path) as f:
            probe_dict = _json.load(f)

        # Support both bare probe dict and probeinterface container
        if "probes" in probe_dict:
            probe = probe_dict["probes"][0]
        else:
            probe = probe_dict

        dev_ch  = np.array(probe["device_channel_indices"])
        contact_pos = np.array(probe["contact_positions"])   # (n_contacts, 2)
        raw_shank_ids = probe.get("shank_ids", ["0"] * len(dev_ch))

        # Filter to active channels (device_channel_indices != -1)
        active = dev_ch != -1
        dev_ch        = dev_ch[active]
        contact_pos   = contact_pos[active]
        raw_shank_ids = np.array(raw_shank_ids)[active]

        if contact_pos.shape[1] != 2:
            raise ValueError("contact_positions must have shape (n, 2)")

        # Build full (n_ch_file, 2) array; unconnected channels get (0, 0)
        pos       = np.zeros((n_ch_file, 2), dtype=np.float32)
        shank_ids = ["?"] * n_ch_file
        for i, ch in enumerate(dev_ch):
            if 0 <= ch < n_ch_file:
                pos[ch]       = contact_pos[i]
                shank_ids[ch] = str(raw_shank_ids[i])

        # Only keep channels that are actually mapped
        mapped = set(int(c) for c in dev_ch if 0 <= c < n_ch_file)
        if len(mapped) != n_ch_file:
            # Some channels unmapped — caller will only display mapped ones
            # Mark unmapped channels with shank_id "?"
            pass

        return pos, shank_ids

    else:
        raise ValueError(f"Unsupported probe file format: {path!r}  (use .npy or .json)")


# ---------------------------------------------------------------------------
# Probe map widget
# ---------------------------------------------------------------------------

class ProbeMapWidget(QWidget):
    """Draws a contact map for one shank.

    All contacts shown as small circles.
    Contacts inside the current depth+column selection highlighted bright.
    Click to re-centre the depth window.
    """

    depth_clicked = pyqtSignal(float)   # emitted with µm depth when user clicks

    _CONTACT_R  = 4    # px radius
    _MARGIN_L   = 38   # left margin for depth labels
    _MARGIN_R   = 8
    _MARGIN_T   = 8
    _MARGIN_B   = 8

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(110)
        self.setMaximumWidth(160)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background:#111;")
        self.setToolTip("Click to jump to depth")

        # Set by update_probe()
        self._all_pos:  "np.ndarray | None" = None  # (n, 2) all contacts in shank
        self._sel_mask: "np.ndarray | None" = None  # (n,) bool — currently shown
        self._y_min = 0.0
        self._y_max = 1.0

    def update_probe(
        self,
        all_pos:  np.ndarray,   # (n, 2) x/y µm — all contacts in this shank
        sel_mask: np.ndarray,   # (n,) bool — which are currently displayed
    ) -> None:
        self._all_pos  = all_pos
        self._sel_mask = sel_mask
        if len(all_pos):
            self._y_min = float(all_pos[:, 1].min())
            self._y_max = float(all_pos[:, 1].max())
        self.update()

    # ------------------------------------------------------------------

    def _to_px(self, x_um: float, y_um: float) -> tuple[float, float]:
        """Map µm → widget pixels."""
        w = self.width()  - self._MARGIN_L - self._MARGIN_R
        h = self.height() - self._MARGIN_T  - self._MARGIN_B
        y_range = max(self._y_max - self._y_min, 1.0)

        # x: fit all column x values into available width
        x_all = self._all_pos[:, 0]
        x_range = max(float(x_all.max() - x_all.min()), 1.0)
        px = self._MARGIN_L + (x_um - float(x_all.min())) / x_range * w

        # y: flip so tip (low y) is at bottom
        py = self._MARGIN_T + (1.0 - (y_um - self._y_min) / y_range) * h
        return px, py

    def paintEvent(self, event) -> None:
        if self._all_pos is None:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor("#111111"))

        w = self.width()
        h = self.height() - self._MARGIN_T - self._MARGIN_B
        y_range = max(self._y_max - self._y_min, 1.0)

        # Depth labels every ~500 µm
        p.setPen(QPen(QColor("#555555"), 1))
        step = 500 if y_range > 1000 else 200
        y_label = int(self._y_min // step) * step
        while y_label <= self._y_max + step:
            _, py = self._to_px(0, y_label)
            if self._MARGIN_T <= py <= self._MARGIN_T + h:
                p.drawLine(self._MARGIN_L - 4, int(py), self._MARGIN_L, int(py))
                p.setPen(QPen(QColor("#666666"), 1))
                p.drawText(0, int(py) + 4, self._MARGIN_L - 6, 12,
                           Qt.AlignmentFlag.AlignRight, str(y_label))
                p.setPen(QPen(QColor("#555555"), 1))
            y_label += step

        r = self._CONTACT_R
        for i, (xc, yc) in enumerate(self._all_pos):
            px, py = self._to_px(float(xc), float(yc))
            selected = bool(self._sel_mask[i]) if self._sel_mask is not None else False
            if selected:
                p.setBrush(QColor("#f0c040"))
                p.setPen(QPen(QColor("#ffffff"), 0.5))
            else:
                p.setBrush(QColor("#333333"))
                p.setPen(QPen(QColor("#555555"), 0.5))
            p.drawEllipse(int(px - r), int(py - r), r * 2, r * 2)

        p.end()

    def mousePressEvent(self, event) -> None:
        if self._all_pos is None:
            return
        py = event.position().y()
        h  = self.height() - self._MARGIN_T - self._MARGIN_B
        frac = 1.0 - (py - self._MARGIN_T) / max(h, 1)
        depth = self._y_min + frac * (self._y_max - self._y_min)
        self.depth_clicked.emit(float(depth))


# ---------------------------------------------------------------------------
# Video panel
# ---------------------------------------------------------------------------

def _open_video(path: str):
    """Return a video capture object. Tries cv2 then imageio."""
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise RuntimeError(f"cv2 could not open {path}")
        return ("cv2", cap)
    except ImportError:
        pass
    try:
        import imageio.v3 as iio
        return ("imageio", path)
    except ImportError:
        pass
    raise RuntimeError("Install opencv-python or imageio to show video frames.")


def _read_frame(backend_cap, frame_idx: int) -> "np.ndarray | None":
    """Return (H, W, 3) uint8 RGB array for frame_idx, or None on error."""
    backend, cap = backend_cap
    try:
        if backend == "cv2":
            cap.set(1, frame_idx)          # CAP_PROP_POS_FRAMES
            ok, bgr = cap.read()
            if not ok:
                return None
            import cv2
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        else:
            import imageio.v3 as iio
            return iio.imread(cap, index=frame_idx)
    except Exception as exc:
        logger.debug("video read error at frame %d: %s", frame_idx, exc)
        return None


class VideoPanel(QWidget):
    """Fetches video frames from the RawServer over ZMQ and displays them."""

    def __init__(self, host: str, port: int, info: dict, parent=None):
        super().__init__(parent)
        self._host     = host
        self._port     = port
        self._fps      = float(info.get("video_fps", 0))
        self._n_frames = int(info.get("video_n_frames", 0))
        self._has_video = bool(info.get("has_video", False))
        self._t_offset = 0.0
        self._last_t   = None
        self._fetch_seq = 0
        self._pending_pixmap = None

        self.setMinimumHeight(80)
        self.setStyleSheet("background:#0a0a0a;")

        vl = QVBoxLayout(self)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(0)

        bar = QWidget()
        bar.setFixedHeight(28)
        bar.setStyleSheet("background:#1a1a1a;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(6, 2, 6, 2)
        bl.setSpacing(8)

        bl.addWidget(QLabel("t offset (s):"))
        self._offset_spin = _dspin(-9999, 9999, 0.0, step=0.1)
        self._offset_spin.setDecimals(3)
        self._offset_spin.setFixedWidth(85)
        self._offset_spin.setToolTip(
            "t_video = t_ephys − offset\n"
            "Positive: video starts after ephys\n"
            "Negative: video starts before ephys"
        )
        self._offset_spin.valueChanged.connect(self._on_offset_changed)
        bl.addWidget(self._offset_spin)

        bl.addStretch()
        self._info_label = QLabel()
        self._info_label.setStyleSheet("color:#555;font-size:11px;")
        bl.addWidget(self._info_label)
        vl.addWidget(bar)

        self._frame_label = QLabel()
        self._frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._frame_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._frame_label.setStyleSheet("background:#0a0a0a;color:#444;font-size:12px;")
        vl.addWidget(self._frame_label, stretch=1)

        self._poll = QTimer(self)
        self._poll.setInterval(40)
        self._poll.timeout.connect(self._apply_pending)
        self._poll.start()

        if self._has_video:
            dur = self._n_frames / max(self._fps, 1)
            self._info_label.setText(
                f"{self._fps:.2f} fps  ·  {self._n_frames} frames  ·  {dur:.1f} s"
            )
            self._frame_label.setText("")
        else:
            self._frame_label.setText(
                "No video on server\n"
                "Add --video-file /path/to/video.mp4 when launching the server"
            )

    def _on_offset_changed(self, val: float) -> None:
        self._t_offset = val
        self._last_t   = None   # force redraw

    def show_time(self, t_center: float) -> None:
        if not self._has_video:
            return
        t_video = t_center - self._t_offset
        if t_video < 0 or (self._n_frames and t_video * self._fps > self._n_frames):
            return
        # Skip if same frame
        if self._last_t is not None:
            last_idx = int(round(self._last_t * self._fps))
            new_idx  = int(round(t_video * self._fps))
            if last_idx == new_idx:
                return
        self._last_t = t_video
        seq = self._fetch_seq = self._fetch_seq + 1
        host, port = self._host, self._port

        def _fetch():
            try:
                tr = RawTransport(host=host, port=port, timeout_ms=5_000)
                try:
                    result = tr.get_frame(t_video)
                finally:
                    tr.close()
            except Exception as exc:
                logger.debug("video fetch error: %s", exc)
                return
            if result is None or seq != self._fetch_seq:
                return
            jpeg, hdr = result
            px = QPixmap()
            px.loadFromData(jpeg, "JPEG")
            self._pending_pixmap = (px, hdr, t_video)

        threading.Thread(target=_fetch, daemon=True).start()

    def _apply_pending(self) -> None:
        if self._pending_pixmap is None:
            return
        px, hdr, t_video = self._pending_pixmap
        self._pending_pixmap = None
        scaled = px.scaled(
            self._frame_label.width(), self._frame_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._frame_label.setPixmap(scaled)
        self._info_label.setText(
            f"{self._fps:.2f} fps  ·  "
            f"frame {hdr.get('frame_idx', '?')}  ·  "
            f"t_video={t_video:.3f} s"
        )


# ---------------------------------------------------------------------------
# Raw trace widget
# ---------------------------------------------------------------------------

class RawTraceWidget(QWidget):
    def __init__(self, host: str, port: int, info: dict,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._host      = host
        self._port      = port
        self._info      = info
        self._n_ch      = int(info["n_channels"])
        self._sr        = float(info["sample_rate"])
        self._duration  = float(info["duration"])

        self._t_start   = 0.0
        self._window_s  = min(_WINDOW_S, self._duration)
        self._y_scale   = 1.0
        self._filtered  = False

        self._ch_start  = 0
        self._ch_count  = min(_N_CH_SHOW, self._n_ch)

        # Probe geometry (optional) — loaded from channel_positions.npy
        # _probe_pos: (n_ch, 2) float32 [x, y] in µm, indexed by channel id
        # _shanks: list of sorted channel-id arrays, one per shank
        # _shank_idx: currently displayed shank index
        self._probe_pos:  "np.ndarray | None" = None
        self._shanks:     "list[np.ndarray]"  = []
        self._shank_idx:  int = 0
        self._col_x:      "float | None" = None   # None = all columns

        self._buf_traces:  np.ndarray | None = None
        self._buf_raw:     np.ndarray | None = None   # unscaled traces for hover readout
        self._buf_t_arr:   np.ndarray | None = None
        self._buf_ch_ids:  list[int] = []

        # Camera world rect (updated by _fit_camera)
        self._cam_x0 = 0.0
        self._cam_x1 = 1.0
        self._cam_y0 = 0.0
        self._cam_y1 = 1.0

        self._fig:             Any = None
        self._subplot:         Any = None
        self._canvas_widget:   QWidget | None = None
        self._line_collection: Any = None
        self._fpl_ready        = False

        self._fetch_seq  = 0
        self._pending:   dict | None = None
        self._pan_last_x: float | None = None

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._do_fetch)

        self._poll = QTimer(self)
        self._poll.setInterval(40)
        self._poll.timeout.connect(self._apply_pending)
        self._poll.start()

        self._build_ui()

        try:
            import fastplotlib as fpl
            self._init_fpl(fpl)
        except Exception as exc:
            logger.warning("fastplotlib init failed: %s", exc)

        self._schedule_fetch()

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- Top toolbar ------------------------------------------------
        bar = QWidget()
        bar.setFixedHeight(34)
        bar.setStyleSheet("background:#252525;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(6, 3, 6, 3)
        bl.setSpacing(8)

        self._filter_btn = _btn("HP filter", checkable=True)
        self._filter_btn.toggled.connect(self._on_filter_toggled)
        bl.addWidget(self._filter_btn)

        self._export_btn = _btn("Export…")
        self._export_btn.clicked.connect(self._on_export)
        bl.addWidget(self._export_btn)

        self._probe_btn = _btn("Load probe…")
        self._probe_btn.clicked.connect(self._on_load_probe)
        bl.addWidget(self._probe_btn)

        bl.addWidget(_sep())

        # Channel controls (no probe)
        self._no_probe_widget = QWidget()
        np_hl = QHBoxLayout(self._no_probe_widget)
        np_hl.setContentsMargins(0, 0, 0, 0)
        np_hl.setSpacing(6)
        np_hl.addWidget(QLabel("Ch start:"))
        self._ch_start_spin = QSpinBox()
        self._ch_start_spin.setRange(0, max(0, self._n_ch - 1))
        self._ch_start_spin.setValue(0)
        self._ch_start_spin.setFixedWidth(60)
        self._ch_start_spin.valueChanged.connect(self._on_ch_start_changed)
        np_hl.addWidget(self._ch_start_spin)
        np_hl.addWidget(QLabel("count:"))
        self._ch_count_spin = QSpinBox()
        self._ch_count_spin.setRange(1, self._n_ch)
        self._ch_count_spin.setValue(self._ch_count)
        self._ch_count_spin.setFixedWidth(60)
        self._ch_count_spin.valueChanged.connect(self._on_ch_count_changed)
        np_hl.addWidget(self._ch_count_spin)
        bl.addWidget(self._no_probe_widget)

        bl.addWidget(_sep())
        bl.addWidget(QLabel("Window (s):"))
        self._window_spin = _dspin(0.05, min(60.0, self._duration),
                                   self._window_s, step=0.1)
        self._window_spin.valueChanged.connect(self._on_window_changed)
        bl.addWidget(self._window_spin)
        bl.addWidget(_sep())
        bl.addWidget(QLabel("Go to (s):"))
        self._goto_spin = _dspin(0.0, max(0.0, self._duration), 0.0, step=1.0)
        self._goto_spin.setDecimals(3)
        self._goto_spin.setFixedWidth(90)
        self._goto_spin.editingFinished.connect(self._on_goto)
        bl.addWidget(self._goto_spin)
        bl.addStretch()
        self._status = QLabel("Loading…")
        self._status.setStyleSheet("color:#666;font-size:11px;")
        bl.addWidget(self._status)
        outer.addWidget(bar)

        # ---- Probe toolbar (hidden until probe loaded) -------------------
        from PyQt6.QtWidgets import QComboBox
        self._probe_bar = QWidget()
        self._probe_bar.setFixedHeight(34)
        self._probe_bar.setStyleSheet("background:#1e1e1e;border-bottom:1px solid #333;")
        self._probe_bar.hide()
        pb = QHBoxLayout(self._probe_bar)
        pb.setContentsMargins(6, 3, 6, 3)
        pb.setSpacing(8)

        pb.addWidget(QLabel("Shank:"))
        self._shank_combo = QComboBox()
        self._shank_combo.setFixedWidth(100)
        self._shank_combo.currentIndexChanged.connect(self._on_shank_changed)
        pb.addWidget(self._shank_combo)

        pb.addWidget(_sep())
        pb.addWidget(QLabel("Column:"))
        self._col_combo = QComboBox()
        self._col_combo.setFixedWidth(110)
        self._col_combo.currentIndexChanged.connect(self._on_col_changed)
        pb.addWidget(self._col_combo)

        pb.addWidget(_sep())
        pb.addWidget(QLabel("Depth (µm):"))
        self._depth_top_spin = QSpinBox()
        self._depth_top_spin.setRange(-10000, 10000)
        self._depth_top_spin.setValue(0)
        self._depth_top_spin.setFixedWidth(75)
        self._depth_top_spin.valueChanged.connect(self._on_depth_changed)
        pb.addWidget(self._depth_top_spin)
        pb.addWidget(QLabel("–"))
        self._depth_bot_spin = QSpinBox()
        self._depth_bot_spin.setRange(-10000, 10000)
        self._depth_bot_spin.setValue(4000)
        self._depth_bot_spin.setFixedWidth(75)
        self._depth_bot_spin.valueChanged.connect(self._on_depth_changed)
        pb.addWidget(self._depth_bot_spin)
        pb.addStretch()
        outer.addWidget(self._probe_bar)

        # ---- Main splitter: traces (top) / video (bottom) ---------------
        self._splitter = QSplitter(Qt.Orientation.Vertical)
        self._splitter.setHandleWidth(4)
        self._splitter.setStyleSheet(
            "QSplitter::handle { background: #333; }"
        )

        # Top half: trace canvas (left) + probe map (right)
        top_widget = QWidget()
        top_hl     = QHBoxLayout(top_widget)
        top_hl.setContentsMargins(0, 0, 0, 0)
        top_hl.setSpacing(0)

        self._canvas_area = QWidget()
        self._canvas_area.setStyleSheet("background:#111;")
        self._canvas_layout = QVBoxLayout(self._canvas_area)
        self._canvas_layout.setContentsMargins(0, 0, 0, 0)

        self._placeholder = QLabel("Fetching traces…")
        self._placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._placeholder.setStyleSheet("color:#555;font-size:13px;")
        self._canvas_layout.addWidget(self._placeholder)

        self._probe_map = ProbeMapWidget()
        self._probe_map.depth_clicked.connect(self._on_probe_map_clicked)
        self._probe_map.hide()

        top_hl.addWidget(self._canvas_area, stretch=1)
        top_hl.addWidget(self._probe_map)

        # Bottom half: video panel
        self._video_panel = VideoPanel(host=self._host, port=self._port, info=self._info)

        self._splitter.addWidget(top_widget)
        self._splitter.addWidget(self._video_panel)
        self._splitter.setSizes([600, 200])
        self._splitter.setCollapsible(1, True)   # video pane collapsible
        outer.addWidget(self._splitter, stretch=1)

        self._scrollbar = QScrollBar(Qt.Orientation.Horizontal)
        self._scrollbar.setRange(0, 10000)
        self._scrollbar.setValue(0)
        self._scrollbar.setPageStep(
            max(1, int(10000 * self._window_s / max(self._duration, 1e-6)))
        )
        self._scrollbar.valueChanged.connect(self._on_scrollbar)
        self._scrollbar.setStyleSheet("background:#1a1a1a;")
        outer.addWidget(self._scrollbar)
        self._scrollbar_updating = False

    # ------------------------------------------------------------------
    # fastplotlib
    # ------------------------------------------------------------------

    def _init_fpl(self, fpl) -> None:
        from phy_remote.client.views._graphics import slim_subplot
        self._fig     = fpl.Figure(canvas="qt")
        self._subplot = self._fig[0, 0]
        try:
            self._subplot.camera = "2d"
            self._subplot.camera.maintain_aspect = False
        except Exception:
            pass
        try:
            self._subplot.axes.visible = False
            self._subplot.title.visible = False
            slim_subplot(self._subplot)
        except Exception:
            pass
        self._fig.show()
        self._canvas_widget = self._fig.canvas
        self._canvas_widget.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self._canvas_widget.setMinimumHeight(60)
        self._canvas_widget.setMouseTracking(True)
        self._canvas_layout.addWidget(self._canvas_widget)
        self._canvas_widget.hide()

        self._ef = _RawEventFilter(self, self._canvas_widget, self)
        QApplication.instance().installEventFilter(self._ef)

        try:
            self._fig.canvas.add_event_handler(self._on_pointer_down, "pointer_down")
            self._fig.canvas.add_event_handler(self._on_pointer_move, "pointer_move")
            self._fig.canvas.add_event_handler(self._on_pointer_up,   "pointer_up")
        except Exception as exc:
            logger.debug("canvas pointer events unavailable: %s", exc)

        self._fpl_ready = True

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def _schedule_fetch(self) -> None:
        self._fetch_seq += 1
        self._debounce.start()

    def _channel_ids(self) -> list[int]:
        if self._probe_pos is not None and self._shanks:
            return self._probe_channel_ids()
        end = min(self._ch_start + self._ch_count, self._n_ch)
        return list(range(self._ch_start, end))

    def _probe_channel_ids(self) -> list[int]:
        """Channels for the current shank, depth range, and column, sorted tip→surface."""
        shank_chs = self._shanks[self._shank_idx]
        y_vals    = self._probe_pos[shank_chs, 1]
        x_vals    = self._probe_pos[shank_chs, 0]
        y_top     = float(self._depth_top_spin.value())
        y_bot     = float(self._depth_bot_spin.value())
        y_lo, y_hi = min(y_top, y_bot), max(y_top, y_bot)
        mask = (y_vals >= y_lo) & (y_vals <= y_hi)
        if self._col_x is not None:
            mask &= np.abs(x_vals - self._col_x) < 5.0   # 5 µm tolerance
        return shank_chs[mask].tolist()

    def _channel_offsets_for(self, ch_ids: list[int]) -> np.ndarray:
        """Y display offsets (µm) for a list of channel ids."""
        if self._probe_pos is not None:
            return self._probe_pos[np.array(ch_ids), 1].astype(np.float32)
        return np.arange(len(ch_ids), dtype=np.float32) * 40.0

    def _do_fetch(self) -> None:
        t0       = self._t_start
        t1       = min(t0 + self._window_s, self._duration)
        ch_ids   = self._channel_ids()
        seq      = self._fetch_seq
        host     = self._host
        port     = self._port
        filtered = self._filtered
        sr       = self._sr

        def _worker() -> None:
            try:
                tr = RawTransport(host=host, port=port)
                try:
                    traces, hdr = tr.get_traces(
                        t0, t1, channel_ids=ch_ids,
                        filtered=filtered, max_samples=_MAX_SAMPLES,
                    )
                finally:
                    tr.close()
            except Exception as exc:
                if seq == self._fetch_seq:
                    self._pending = {"error": str(exc)}
                return
            if seq == self._fetch_seq:
                self._pending = {
                    "traces":  traces,
                    "t_start": float(hdr.get("t_start", t0)),
                    "t_end":   float(hdr.get("t_end",   t1)),
                    "sr":      float(hdr.get("sample_rate", sr)),
                    "ch_ids":  hdr.get("channel_ids", ch_ids),
                }

        threading.Thread(target=_worker, daemon=True).start()

    def _apply_pending(self) -> None:
        if self._pending is None:
            return
        data = self._pending
        self._pending = None

        if "error" in data:
            self._status.setText(f"Error: {data['error']}")
            return

        traces  = np.array(data["traces"], dtype=np.float32)
        t_start = data["t_start"]
        t_end   = data["t_end"]
        ch_ids  = list(data["ch_ids"])
        n_ch, n_s = traces.shape

        self._buf_t_arr  = np.linspace(t_start, t_end, n_s, dtype=np.float32)
        self._buf_ch_ids = ch_ids

        self._buf_raw = traces.copy()              # keep raw for hover readout
        traces -= np.median(traces, axis=0)
        q01  = float(np.quantile(traces, 0.01))
        q99  = float(np.quantile(traces, 0.99))
        span = max(abs(q01), abs(q99), 1e-9)
        self._buf_traces = traces * (40.0 * 0.45 / span) * self._y_scale

        self._render()
        self._update_scrollbar(t_start)
        # Show video frame at centre of current window
        t_center = (t_start + data["t_end"]) / 2
        self._video_panel.show_time(t_center)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def _render(self) -> None:
        if not self._fpl_ready or self._buf_traces is None:
            return
        try:
            self._update_line_collection()
            self._placeholder.hide()
            self._canvas_widget.show()
            t0 = float(self._buf_t_arr[0])
            t1 = float(self._buf_t_arr[-1])
            ch_ids = self._buf_ch_ids
            if self._probe_pos is not None and ch_ids:
                y_vals = self._probe_pos[np.array(ch_ids), 1]
                shank_label = f"Shank {self._shank_combo.currentText()}"
                col_label   = self._col_combo.currentText()
                depth_str   = (f"{shank_label}  ·  {col_label}  ·  "
                               f"{y_vals.min():.0f}–{y_vals.max():.0f} µm")
            else:
                ch_end    = self._ch_start + len(ch_ids) - 1
                depth_str = f"ch {self._ch_start}–{ch_end}"
            self._status.setText(
                f"{t0:.3f}–{t1:.3f} s  ·  {depth_str}  ·  "
                f"{'HP' if self._filtered else 'raw'}"
            )
            QTimer.singleShot(60, self._fit_camera)
        except Exception as exc:
            logger.exception("render error: %s", exc)

    def _col_colors(self, ch_ids: list[int]) -> "list[tuple]":
        """Return a colour per channel — different colour per column when probe loaded."""
        _COL_PALETTE = [
            (0.55, 0.55, 0.55, 0.90),   # grey   — col 0
            (0.25, 0.55, 0.90, 0.90),   # blue   — col 1
            (0.30, 0.80, 0.45, 0.90),   # green  — col 2
            (0.90, 0.55, 0.20, 0.90),   # orange — col 3
        ]
        if self._probe_pos is None or self._col_x is not None:
            # single column or no probe — uniform grey
            return [_TRACE_COLOR] * len(ch_ids)
        # map each channel to a column index by x position
        x_vals   = np.round(self._probe_pos[np.array(ch_ids), 0]).astype(int)
        unique_x = sorted(np.unique(x_vals))
        x_to_idx = {x: i for i, x in enumerate(unique_x)}
        return [_COL_PALETTE[x_to_idx[x] % len(_COL_PALETTE)] for x in x_vals]

    def _update_line_collection(self) -> None:
        traces  = self._buf_traces
        t_arr   = self._buf_t_arr
        ch_ids  = self._buf_ch_ids
        n_ch, n_s = traces.shape
        offsets = self._channel_offsets_for(ch_ids)
        colors  = self._col_colors(ch_ids)

        lines = []
        for i in range(n_ch):
            xy = np.empty((n_s, 2), dtype=np.float32)
            xy[:, 0] = t_arr
            xy[:, 1] = offsets[i] + traces[i]
            lines.append(xy)

        sp = self._subplot
        if self._line_collection is not None:
            try:
                sp.delete_graphic(self._line_collection)
            except Exception:
                pass
            self._line_collection = None

        try:
            self._line_collection = sp.add_line_collection(
                data=lines, colors=colors, thickness=1.0,
            )
        except AttributeError:
            out = np.full((n_ch * (n_s + 1), 2), np.nan, dtype=np.float32)
            for i in range(n_ch):
                s = i * (n_s + 1)
                out[s:s + n_s, 0] = t_arr
                out[s:s + n_s, 1] = offsets[i] + traces[i]
            self._line_collection = sp.add_line(out, colors=colors[0], thickness=1.0)

    def _fit_camera(self) -> None:
        if self._fig is None or self._buf_t_arr is None:
            return
        try:
            sp      = self._subplot
            ch_ids  = self._buf_ch_ids
            offsets = self._channel_offsets_for(ch_ids)
            t0      = float(self._buf_t_arr[0])
            t1      = float(self._buf_t_arr[-1])
            y0      = float(offsets.min()) - 20
            y1      = float(offsets.max()) + 20
            mid_x   = (t0 + t1) / 2
            mid_y   = (y0 + y1) / 2
            self._cam_x0 = t0 - 0.02 * (t1 - t0)
            self._cam_x1 = t1 + 0.02 * (t1 - t0)
            self._cam_y0 = y0
            self._cam_y1 = y1
            state   = sp.camera.get_state()
            sp.camera.set_state({
                "position":        np.array([mid_x, mid_y, state["position"][2]]),
                "fov":             0.0,
                "width":           (t1 - t0) * 1.04,
                "height":          (y1 - y0),
                "depth":           state["depth"],
                "zoom":            1.0,
                "maintain_aspect": False,
            })
            self._fig.canvas.request_draw()
        except Exception as exc:
            logger.debug("fit_camera: %s", exc)

    # ------------------------------------------------------------------
    # Scrollbar
    # ------------------------------------------------------------------

    def _update_scrollbar(self, t_start: float) -> None:
        self._scrollbar_updating = True
        try:
            dur  = max(self._duration, 1e-6)
            pos  = int(10000 * t_start / dur)
            page = max(1, int(10000 * self._window_s / dur))
            self._scrollbar.setPageStep(page)
            self._scrollbar.setValue(pos)
        finally:
            self._scrollbar_updating = False

    def _on_scrollbar(self, val: int) -> None:
        if self._scrollbar_updating:
            return
        self._t_start = (val / 10000.0) * self._duration
        self._schedule_fetch()

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _scroll(self, frac: float) -> None:
        self._t_start = max(0.0, min(
            self._t_start + frac * self._window_s,
            max(0.0, self._duration - self._window_s),
        ))
        self._schedule_fetch()

    def _zoom_time(self, factor: float) -> None:
        mid = self._t_start + self._window_s / 2
        self._window_s = float(np.clip(self._window_s * factor, 0.01, self._duration))
        self._t_start  = max(0.0, min(mid - self._window_s / 2,
                                      self._duration - self._window_s))
        self._window_spin.blockSignals(True)
        self._window_spin.setValue(self._window_s)
        self._window_spin.blockSignals(False)
        self._schedule_fetch()

    def _zoom_amp(self, factor: float) -> None:
        self._y_scale = float(np.clip(self._y_scale * factor, 0.01, 200.0))
        if self._buf_traces is not None:
            self._buf_traces *= factor
            if self._fpl_ready:
                try:
                    self._update_line_collection()
                    self._fig.canvas.request_draw()
                except Exception:
                    pass

    def _on_filter_toggled(self, checked: bool) -> None:
        self._filtered = checked
        self._schedule_fetch()

    def _on_ch_start_changed(self, val: int) -> None:
        self._ch_start = val
        self._schedule_fetch()

    def _on_ch_count_changed(self, val: int) -> None:
        self._ch_count = val
        self._schedule_fetch()

    def _on_window_changed(self, val: float) -> None:
        self._window_s = float(val)
        self._schedule_fetch()

    def _on_load_probe(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Load probe file",
            os.path.expanduser("~"),
            "Probe files (*.json *.npy);;ProbeInterface JSON (*.json);;NumPy (*.npy);;All files (*)",
        )
        if not path:
            return
        self._on_load_probe_from_path(path)

    def _on_load_probe_from_path(self, path: str) -> None:
        try:
            pos, shank_ids = _load_probe_file(path, self._n_ch)
        except Exception as exc:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Probe load error", str(exc))
            return

        self._probe_pos = pos.astype(np.float32)

        # Group channels by shank_id, sort each group tip→surface (ascending y)
        unique_shanks = sorted(set(shank_ids))
        self._shanks = []
        unique_shanks = sorted(s for s in set(shank_ids) if s != "?")
        for sid in unique_shanks:
            chs = np.array([i for i, s in enumerate(shank_ids) if s == sid])
            chs = chs[np.argsort(pos[chs, 1])]
            self._shanks.append(chs)

        # Populate shank combo
        unique_shanks = sorted(set(shank_ids))
        self._shank_combo.blockSignals(True)
        self._shank_combo.clear()
        for sid, chs in zip(unique_shanks, self._shanks):
            x_center = int(np.mean(pos[chs, 0]))
            self._shank_combo.addItem(f"Shank {sid} (x≈{x_center}µm)")
        self._shank_combo.blockSignals(False)
        self._shank_idx = 0

        # Set depth range to full shank
        y_all = pos[:, 1]
        self._depth_top_spin.blockSignals(True)
        self._depth_bot_spin.blockSignals(True)
        self._depth_top_spin.setRange(int(y_all.min()) - 100, int(y_all.max()) + 100)
        self._depth_bot_spin.setRange(int(y_all.min()) - 100, int(y_all.max()) + 100)
        self._depth_top_spin.setValue(int(y_all.min()))
        self._depth_bot_spin.setValue(int(y_all.max()))
        self._depth_top_spin.blockSignals(False)
        self._depth_bot_spin.blockSignals(False)

        # Swap UI
        self._no_probe_widget.hide()
        self._probe_bar.show()
        self._probe_btn.setText("Probe ✓")
        self._col_x = None
        self._populate_col_combo()
        self._update_probe_map()

        n_shanks = len(self._shanks)
        n_ch_per = len(self._shanks[0]) if self._shanks else 0
        logger.info("Probe loaded: %d shanks × %d ch, y=%.0f–%.0f µm",
                    n_shanks, n_ch_per, y_all.min(), y_all.max())
        self._schedule_fetch()

    def _on_shank_changed(self, idx: int) -> None:
        self._shank_idx = idx
        self._populate_col_combo()
        self._col_x = None
        self._schedule_fetch()

    def _on_col_changed(self, idx: int) -> None:
        if idx <= 0:
            self._col_x = None
        else:
            # label is "x=NN µm"
            text = self._col_combo.itemText(idx)
            try:
                self._col_x = float(text.replace("x=", "").replace(" µm", "").strip())
            except Exception:
                self._col_x = None
        self._update_probe_map()
        self._schedule_fetch()

    def _on_depth_changed(self) -> None:
        self._update_probe_map()
        self._schedule_fetch()

    def _on_probe_map_clicked(self, depth_um: float) -> None:
        """Centre the depth window on the clicked depth."""
        span = abs(self._depth_bot_spin.value() - self._depth_top_spin.value())
        half = span / 2
        y_all = self._probe_pos[:, 1]
        new_top = max(int(y_all.min()), int(depth_um - half))
        new_bot = min(int(y_all.max()), int(depth_um + half))
        self._depth_top_spin.blockSignals(True)
        self._depth_bot_spin.blockSignals(True)
        self._depth_top_spin.setValue(new_top)
        self._depth_bot_spin.setValue(new_bot)
        self._depth_top_spin.blockSignals(False)
        self._depth_bot_spin.blockSignals(False)
        self._update_probe_map()
        self._schedule_fetch()

    def _populate_col_combo(self) -> None:
        """Fill column combo from x-positions of current shank. Default to first column."""
        if self._probe_pos is None or not self._shanks:
            return
        chs    = self._shanks[self._shank_idx]
        x_vals = np.round(self._probe_pos[chs, 0]).astype(int)
        unique_x = sorted(np.unique(x_vals))
        self._col_combo.blockSignals(True)
        self._col_combo.clear()
        self._col_combo.addItem("All cols")
        for x in unique_x:
            self._col_combo.addItem(f"x={x} µm")
        # Default to first column
        if len(unique_x) > 0:
            self._col_combo.setCurrentIndex(1)
            self._col_x = float(unique_x[0])
        self._col_combo.blockSignals(False)

    def _update_probe_map(self) -> None:
        if self._probe_pos is None or not self._shanks:
            return
        chs     = self._shanks[self._shank_idx]
        pos     = self._probe_pos[chs]          # (n, 2) for this shank
        ch_ids  = self._probe_channel_ids()     # currently shown channel ids

        sel_set  = set(ch_ids)
        sel_mask = np.array([c in sel_set for c in chs])

        self._probe_map.update_probe(pos, sel_mask)
        self._probe_map.show()

    def _on_goto(self) -> None:
        t = float(self._goto_spin.value())
        self._t_start = max(0.0, min(t - self._window_s / 2,
                                     self._duration - self._window_s))
        self._schedule_fetch()

    def _on_export(self) -> None:
        if self._buf_traces is None:
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "traces.pdf",
            "PDF (*.pdf);;PNG 300 dpi (*.png);;SVG (*.svg)",
        )
        if not path:
            return
        self._export_matplotlib(path)

    def _export_matplotlib(self, path: str) -> None:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        traces  = self._buf_traces.copy()   # (n_ch, n_s) — already CMR + scaled
        t_arr   = self._buf_t_arr.copy()    # seconds
        ch_ids  = self._buf_ch_ids
        n_ch, n_s = traces.shape

        # Compute µV scale from the scaling we applied:
        # buf_traces = raw * (40 * 0.45 / span) * y_scale
        # So 40 * 0.45 * y_scale display units ≈ span µV (int16 → ~uV for Neuropixels: ×0.195)
        # Just show a 100 µV scale bar — good enough for debugging
        scale_uv   = 100
        # figure out how many display units = scale_uv µV
        # raw int16 * 0.195 = µV (Neuropixels); our scaling: display = raw * k
        # k = 40*0.45*y_scale / span; span = max(|q01|,|q99|) in raw int16 units
        # Instead, derive from the actual data range shown
        disp_range = float(np.percentile(np.abs(traces), 95))  # typical excursion
        # We'll just draw a fixed-height scale bar in data units
        # 1 display unit ≈ pitch/2 µV is a rough guess; label it accordingly
        # Better: ask server for raw scale, but for now annotate "100 µV" next to
        # a bar that is 40/2 = 20 display units tall (half a channel pitch)
        bar_height = 20.0   # display units = ~half pitch, label as approx µV

        fig_h = max(4, n_ch * 0.18)
        fig_w = max(8, (t_arr[-1] - t_arr[0]) * 12)
        fig_w = min(fig_w, 20)   # cap at 20 inches wide

        fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=300)
        ax.set_facecolor("#111111")
        fig.patch.set_facecolor("#111111")

        offsets = self._channel_offsets_for(ch_ids)

        for i in range(n_ch):
            ax.plot(t_arr, offsets[i] + traces[i],
                    color="#888888", linewidth=0.4, rasterized=True)

        # Y-axis labels
        tick_rows = list(range(0, n_ch, max(1, n_ch // 10)))
        tick_y    = [float(offsets[r]) for r in tick_rows]
        if self._probe_pos is not None:
            tick_label = [f"{self._probe_pos[ch_ids[r], 1]:.0f}" for r in tick_rows]
            y_label    = "Depth (µm)"
        else:
            tick_label = [str(ch_ids[r]) for r in tick_rows]
            y_label    = "Channel"
        ax.set_yticks(tick_y)
        ax.set_yticklabels(tick_label, fontsize=7, color="#aaaaaa")
        ax.set_ylabel(y_label, color="#aaaaaa", fontsize=9)

        ax.set_xlabel("Time (s)", color="#aaaaaa", fontsize=9)
        ax.tick_params(axis="x", colors="#aaaaaa", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#444444")

        # Scale bar (bottom-right corner)
        x_bar = t_arr[-1] - (t_arr[-1] - t_arr[0]) * 0.02
        y_bar = offsets[0] + bar_height / 2
        ax.annotate("", xy=(x_bar, y_bar + bar_height / 2),
                    xytext=(x_bar, y_bar - bar_height / 2),
                    arrowprops=dict(arrowstyle="-", color="white", lw=1.5))
        ax.text(x_bar + (t_arr[-1] - t_arr[0]) * 0.005,
                y_bar, "~100 µV", color="white", fontsize=7, va="center")

        filt_label = "HP filtered" if self._filtered else "raw"
        if self._probe_pos is not None:
            y_vals = self._probe_pos[np.array(ch_ids), 1]
            loc_label = (f"shank {self._shank_idx}  ·  "
                         f"{y_vals.min():.0f}–{y_vals.max():.0f} µm depth")
        else:
            loc_label = f"ch {ch_ids[0]}–{ch_ids[-1]}"
        ax.set_title(
            f"{loc_label}  ·  {t_arr[0]:.3f}–{t_arr[-1]:.3f} s  ·  {filt_label}",
            color="#cccccc", fontsize=9,
        )

        fig.tight_layout()
        fig.savefig(path, dpi=300, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info("Exported to %s", path)

        # Open the file with the system viewer
        import subprocess as _sp
        try:
            _sp.Popen(["open", path])
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Right-click pan
    # ------------------------------------------------------------------

    def _on_pointer_down(self, event) -> None:
        if getattr(event, "button", None) == 2:
            self._pan_last_x = getattr(event, "x", None)

    def _on_pointer_move(self, event) -> None:
        # Right-click drag → pan
        if self._pan_last_x is None:
            return
        if getattr(event, "button", None) != 2:
            self._pan_last_x = None
            return
        x = getattr(event, "x", None)
        if x is None:
            return
        dx_px = x - self._pan_last_x
        self._pan_last_x = x
        try:
            w_px = self._canvas_widget.width()
            if w_px > 0:
                dt = -(dx_px / w_px) * self._window_s
                self._t_start = max(0.0, min(
                    self._t_start + dt,
                    max(0.0, self._duration - self._window_s),
                ))
                self._schedule_fetch()
        except Exception:
            pass

    def _on_hover(self, ex: float, ey: float) -> None:
        """Called by Qt event filter on mouse move over canvas."""
        if self._buf_raw is None or self._buf_t_arr is None:
            return
        try:
            w_px = self._canvas_widget.width()
            h_px = self._canvas_widget.height()
            if w_px <= 0 or h_px <= 0:
                return

            frac_x = ex / w_px
            frac_y = 1.0 - (ey / h_px)
            t_world = self._cam_x0 + frac_x * (self._cam_x1 - self._cam_x0)
            y_world = self._cam_y0 + frac_y * (self._cam_y1 - self._cam_y0)

            t_arr = self._buf_t_arr
            si = int(np.clip(np.searchsorted(t_arr, t_world), 0, len(t_arr) - 1))

            offsets = self._channel_offsets_for(self._buf_ch_ids)
            ci = int(np.argmin(np.abs(offsets - y_world)))

            ch_id = self._buf_ch_ids[ci]
            raw_val = float(self._buf_raw[ci, si])
            t_val = float(t_arr[si])
            self._status.setText(
                f"ch {ch_id}  ·  t={t_val:.4f} s  ·  raw={raw_val:.0f}"
            )
        except Exception as exc:
            logger.debug("hover error: %s", exc)

    def _on_pointer_up(self, event) -> None:
        self._pan_last_x = None


# ---------------------------------------------------------------------------
# Qt event filter
# ---------------------------------------------------------------------------

class _RawEventFilter(QObject):
    _WHEEL_STEP = 1.20

    def __init__(self, widget: RawTraceWidget, canvas, parent=None):
        super().__init__(parent)
        self._w      = widget
        self._canvas = canvas

    def _over_canvas(self, obj) -> bool:
        w = obj
        while w is not None:
            if w is self._canvas:
                return True
            w = w.parent()
        return False

    def eventFilter(self, obj, event) -> bool:
        if event.type() == QEvent.Type.Wheel:
            if not self._over_canvas(obj):
                return False
            mods  = event.modifiers()
            delta = event.angleDelta().y()
            if delta == 0:
                return False
            ctrl   = bool(mods & Qt.KeyboardModifier.ControlModifier)
            meta   = bool(mods & Qt.KeyboardModifier.MetaModifier)
            alt    = bool(mods & Qt.KeyboardModifier.AltModifier)
            factor = self._WHEEL_STEP ** (delta / 120)
            if ctrl or meta:
                self._w._zoom_time(1.0 / factor)
            elif alt:
                self._w._zoom_amp(factor)
            else:
                self._w._scroll(0.1 * (-1 if delta > 0 else 1))
            return True

        if event.type() == QEvent.Type.KeyPress:
            if not self._over_canvas(obj):
                return False
            key = event.key()
            alt = bool(event.modifiers() & Qt.KeyboardModifier.AltModifier)
            if key == Qt.Key.Key_Left and alt:
                self._w._scroll(-0.5); return True
            if key == Qt.Key.Key_Right and alt:
                self._w._scroll(+0.5); return True
            if key == Qt.Key.Key_F:
                self._w._scroll(+1.0); return True
            if key == Qt.Key.Key_B:
                self._w._scroll(-1.0); return True
            if key == Qt.Key.Key_Plus or key == Qt.Key.Key_Equal:
                self._w._zoom_amp(1.5); return True
            if key == Qt.Key.Key_Minus:
                self._w._zoom_amp(1.0 / 1.5); return True

        if event.type() == QEvent.Type.MouseMove:
            if self._over_canvas(obj):
                pos = event.position()
                self._w._on_hover(pos.x(), pos.y())
        return False


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class RawViewerWindow(QMainWindow):
    def __init__(self, host: str, port: int, info: dict,
                 tunnel: "subprocess.Popen | None" = None,
                 ssh_target: str = "",
                 job_id: str = "") -> None:
        super().__init__()
        self._tunnel     = tunnel
        self._ssh_target = ssh_target
        self._job_id     = job_id
        n_ch = info["n_channels"]
        dur  = info["duration"]
        sr   = info["sample_rate"]
        self.setWindowTitle(
            f"Raw Viewer — {n_ch} ch  {sr/1000:.1f} kHz  {dur:.1f} s"
        )
        self.resize(1280, 720)
        self._trace = RawTraceWidget(host=host, port=port, info=info, parent=self)
        self.setCentralWidget(self._trace)

    def load_probe(self, path: str) -> None:
        """Load a channel_positions.npy after the window is shown."""
        self._trace._probe_file_path = path
        self._trace._on_load_probe_from_path(path)

    def closeEvent(self, event) -> None:
        # Kill the cluster job
        if self._job_id and self._ssh_target:
            try:
                subprocess.Popen(
                    ["ssh"] + _SSH_OPTS + [self._ssh_target, f"bkill {self._job_id}"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                logger.info("bkill %s sent to %s", self._job_id, self._ssh_target)
            except Exception as exc:
                logger.warning("bkill failed: %s", exc)
        # Close the SSH tunnel
        if self._tunnel and self._tunnel.poll() is None:
            self._tunnel.terminate()
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _le(text: str = "") -> QLineEdit:
    w = QLineEdit(text)
    return w


def _spin(min_val: int, max_val: int, val: int) -> QSpinBox:
    s = QSpinBox()
    s.setRange(min_val, max_val)
    s.setValue(val)
    s.setFixedWidth(80)
    return s


def _dspin(min_val: float, max_val: float, val: float, step: float = 0.1) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(min_val, max_val)
    s.setSingleStep(step)
    s.setValue(val)
    s.setDecimals(1)
    s.setFixedWidth(90)
    return s


def _btn(label: str, checkable: bool = False) -> QPushButton:
    b = QPushButton(label)
    b.setCheckable(checkable)
    b.setFixedHeight(22)
    b.setStyleSheet(
        "QPushButton{font-size:11px;padding:0 6px;background:#333;"
        "color:#bbb;border:1px solid #555;border-radius:2px;}"
        "QPushButton:checked{background:#555;color:#fff;}"
        "QPushButton:hover{background:#444;}"
    )
    return b


def _sep() -> QWidget:
    w = QWidget()
    w.setFixedWidth(1)
    w.setFixedHeight(20)
    w.setStyleSheet("background:#444;")
    return w


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    import argparse
    import os
    import sys

    p = argparse.ArgumentParser(prog="python -m phy_remote.raw_viewer.client")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--n-channels",  type=int,   default=384)
    p.add_argument("--sample-rate", type=float, default=30000.0)
    p.add_argument("--dtype",       default="int16")
    p.add_argument("--port",        type=int,   default=5560)
    p.add_argument("--video-file",  default="",  help="Cluster path to video file")
    p.add_argument("--probe-file",  default="",  help="Local path to channel_positions.npy")
    p.add_argument("--sleap-file",  default="",  help="Local path to SLEAP .h5 file")
    # Cluster args — if provided, skip dialog and submit bsub automatically
    p.add_argument("--raw-file",    default="",  help="Cluster path to raw binary file")
    p.add_argument("--user",        default=os.environ.get("USER", ""))
    p.add_argument("--login-node",  default="login1.int.janelia.org")
    p.add_argument("--python-path", default=
                   f"/groups/scicompsoft/home/{os.environ.get('USER', '<user>')}/miniconda3/envs/phy/bin/python")
    p.add_argument("--queue",       default="gpu_l4")
    p.add_argument("--walltime",    default="120")
    args = p.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    app = QApplication.instance() or QApplication(sys.argv)
    win_ref: list[RawViewerWindow] = []

    if args.raw_file:
        # ---- Headless cluster mode: submit bsub, tunnel, connect ----
        cfg = dict(
            user        = args.user,
            login_node  = args.login_node,
            raw_file    = args.raw_file,
            video_file  = args.video_file,
            n_channels  = args.n_channels,
            sample_rate = args.sample_rate,
            port        = args.port,
            python_path = args.python_path,
            queue       = args.queue,
            walltime    = args.walltime,
        )

        # Show a minimal progress window while launching
        progress = QDialog()
        progress.setWindowTitle("Launching…")
        progress.resize(560, 200)
        log_box = QPlainTextEdit()
        log_box.setReadOnly(True)
        log_box.setStyleSheet("background:#111;color:#aaa;font-family:monospace;font-size:11px;")
        QVBoxLayout(progress).addWidget(log_box)
        progress.show()

        def _append(msg: str):
            log_box.appendPlainText(msg)
            log_box.verticalScrollBar().setValue(log_box.verticalScrollBar().maximum())

        launcher = ClusterLauncher(cfg)

        def _on_ready(compute_node: str, port: int, job_id: str):
            ssh_target = f"{args.user}@{args.login_node}"
            tunnel = subprocess.Popen(
                ["ssh"] + _SSH_OPTS + ["-N", "-L",
                 f"{port}:{compute_node}:{port}", ssh_target],
            )
            time.sleep(1.5)
            try:
                tr   = RawTransport(host="127.0.0.1", port=port, timeout_ms=10_000)
                info = tr.get_info()
                tr.close()
            except Exception as exc:
                _append(f"ERROR: {exc}")
                return
            progress.accept()
            win = RawViewerWindow(host="127.0.0.1", port=port, info=info,
                                  tunnel=tunnel, ssh_target=ssh_target, job_id=job_id)
            win_ref.append(win)
            win.show()
            if args.probe_file and os.path.isfile(args.probe_file):
                win.load_probe(args.probe_file)
            if args.sleap_file and os.path.isfile(args.sleap_file):
                win.load_sleap(args.sleap_file)

        def _on_failed(msg: str):
            _append(f"FAILED: {msg}")

        launcher.log.connect(_append)
        launcher.ready.connect(_on_ready)
        launcher.failed.connect(_on_failed)
        launcher.start()
        progress.exec()

    else:
        # ---- Dialog mode ----
        def _on_connected(host: str, port: int, info: dict) -> None:
            tunnel = dlg.get_tunnel()
            ssh_target, job_id = dlg.get_cluster_info()
            probe_path = dlg.get_probe_path()
            win = RawViewerWindow(host=host, port=port, info=info,
                                  tunnel=tunnel, ssh_target=ssh_target, job_id=job_id)
            win_ref.append(win)
            win.show()
            if probe_path and os.path.isfile(probe_path):
                win.load_probe(probe_path)

        dlg = LaunchDialog()
        dlg.connected.connect(_on_connected)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            sys.exit(0)

    exit_code = app.exec()
    os._exit(exit_code or 0)


if __name__ == "__main__":
    main()
