# phy-fastplotlib

Remote Phy-like GUI using fastplotlib.

This repo is designed to:
- run the data/model server on the Janelia cluster,
- open an SSH tunnel automatically,
- launch the local Mac GUI.

For most users, the only command you need is `./phy-launch.sh ...`.

## First-time setup (one time only)

### 1) Clone and enter the repo

```bash
git clone <this-repo-url> phy-fastplotlib
cd phy-fastplotlib
```

### 2) Make launcher executable

```bash
chmod +x phy-launch.sh
chmod +x scripts/setup.sh
```

### 3) Confirm local Python env exists

The launcher defaults to your shell python:
- first tries `python`, then `python3` from `PATH`

If your path is different, set `CLIENT_PYTHON` when launching:

```bash
CLIENT_PYTHON=/path/to/your/env/bin/python ./phy-launch.sh ...
```

### 3a) Install the project (recommended)

```bash
python -m pip install -e .
```

This installs CLI entrypoints:
- `phy-remote-server`
- `phy-remote-client`
- `phy-raw-viewer-server`
- `phy-raw-viewer-client`

Optional one-shot setup (installs + checks SSH/cluster tools):

```bash
./scripts/setup.sh
```

### 3b) Set cluster identity + remote python (recommended)

Set these once in your shell profile (`~/.zshrc` or `~/.bashrc`):

```bash
export CLUSTER_USER=<your_cluster_username>
export LOGIN_NODE=login1.int.janelia.org
export REMOTE_PYTHON=/groups/scicompsoft/home/<your_cluster_username>/miniconda3/envs/phy/bin/python
```

Then open a new terminal (or `source ~/.zshrc`).

### 4) Confirm cluster access

You need passwordless or normal SSH access to:
- `<your_cluster_username>@login1.int.janelia.org`

Quick test:

```bash
ssh <your_cluster_username>@login1.int.janelia.org "hostname"
```

## Fastest way to run (recommended)

Use **path mode** and point to a directory that contains (or is under) your shank `params.py`.

Example:

```bash
./phy-launch.sh /groups/voigts/voigtslab/neuropixels_2025/npx12/2026_04_29_npx12_box/output/a/shank_0/
```

Preview what would run without submitting jobs:

```bash
./phy-launch.sh --dry-run /groups/voigts/voigtslab/neuropixels_2025/npx12/2026_04_29_npx12_box/output/a/shank_0/
```

What this does:
1. finds `params.py`,
2. submits cluster job (`bsub`) for `phy_remote.server`,
3. waits for `PHY_REMOTE_READY`,
4. opens SSH tunnel,
5. launches GUI (`phy_remote.client.app`).

When you close the GUI, the tunnel and cluster job are cleaned up automatically.

## Other launch modes

### Probe/shank shortcut mode

```bash
./phy-launch.sh <probe_letter> <shank_number> [port]
# example
./phy-launch.sh b 0
```

Dry-run this mode:

```bash
./phy-launch.sh --dry-run b 0
```

This mode searches under `OUTPUT_DIR` (defaults to a hardcoded lab path in `phy-launch.sh`).

### Interactive menu mode

```bash
./phy-launch.sh
```

Scans `OUTPUT_DIR`, then prompts you to pick a dataset.

## Common first-run issues

### “no params.py found ...”

Cause: wrong dataset root or wrong probe/shank path.

Fix:
- use path mode with the exact directory you know contains your dataset,
- or set `OUTPUT_DIR` first.

```bash
OUTPUT_DIR=/groups/.../output ./phy-launch.sh a 0
```

### “no gui opening”

Most common causes:
- server job never reached ready state,
- tunnel failed,
- wrong local Python environment.

Use path mode first (most reliable), and if needed run with explicit python:

```bash
CLIENT_PYTHON=/path/to/local/env/bin/python ./phy-launch.sh /groups/.../shank_0/
```

### Client exits with code 146 / no visible error

The client now defaults to normal shutdown and logs startup failures. If this still happens, rerun with debug output:

```bash
CLIENT_PYTHON=/path/to/local/env/bin/python \
PYTHONPATH=$PWD \
/path/to/local/env/bin/python -m phy_remote.client.app --port 5557 --log-level DEBUG
```

## Manual run (without launcher)

Use this only for debugging.

1) On cluster (inside job):

```bash
/path/to/cluster/env/bin/python -m phy_remote.server /abs/path/to/params.py --port 5557
```

Or if installed with `pip install -e .`:

```bash
phy-remote-server /abs/path/to/params.py --port 5557
```

2) On Mac (new terminal):

```bash
ssh -N -L 5557:<compute-host>:5557 <your_cluster_username>@login1.int.janelia.org
```

3) On Mac (another terminal):

```bash
PYTHONPATH=$PWD /path/to/local/env/bin/python -m phy_remote.client.app --port 5557
```

Or if installed with `pip install -e .`:

```bash
phy-remote-client --port 5557
```

