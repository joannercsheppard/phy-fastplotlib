# phy-fastplotlib

Remote Phy-like GUI using fastplotlib.

This repo is designed to:
- run the data/model server on the Janelia cluster,
- open an SSH tunnel automatically,
- launch the local Mac GUI.

For most users, the only command you need is `./phy-launch.sh ...`.

## 30-second quickstart

If this is your first run, do this:

```bash
git clone https://github.com/joannercsheppard/phy-fastplotlib.git phy-fastplotlib
cd phy-fastplotlib
chmod +x phy-launch.sh scripts/bootstrap_cluster.sh
cp phy-launch.env.example phy-launch.env
conda create -n phy-fastplotlib python=3.11 -y
conda activate phy-fastplotlib
python -m pip install -e .
```

Edit `phy-launch.env` **before running anything else**:
- required: `CLUSTER_USER`, `REMOTE_PYTHON`
- optional: `CLIENT_PYTHON`, `REMOTE_REPO_DIR`, `OUTPUT_DIR`

Open it in an editor:

```bash
nano phy-launch.env
```

Do not leave placeholder values like `<your_cluster_username>`.

Quick sanity check (should print your values, not placeholders):

```bash
grep -E '^(CLUSTER_USER|REMOTE_PYTHON|CLIENT_PYTHON|REMOTE_REPO_DIR|OUTPUT_DIR)=' phy-launch.env
```

Preflight check (safe, no job submission):

```bash
./phy-launch.sh --dry-run /path/to/output/<probe>/shank_<n>/
```

```bash
./scripts/bootstrap_cluster.sh
./phy-launch.sh /path/to/output/<probe>/shank_<n>/
```

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
chmod +x scripts/bootstrap_cluster.sh
```

### 2a) Create your launcher config file (recommended)

```bash
cp phy-launch.env.example phy-launch.env
```

Edit `phy-launch.env` once with your cluster username and Python paths.
`phy-launch.sh` loads this file automatically.

### 3) Create and activate a local environment

Choose one option:

**Option A: conda (recommended)**

```bash
conda create -n phy-fastplotlib python=3.11 -y
conda activate phy-fastplotlib
```

**Option B: venv**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4) Confirm local Python env is active

`phy-launch.sh` needs a local Python to run the GUI client.

Default behavior:
- uses `CLIENT_PYTHON` if set,
- otherwise tries `python`, then `python3` from your `PATH`.

Recommended (set once in `phy-launch.env`):

```bash
CLIENT_PYTHON=/path/to/your/local/env/bin/python
```

Quick check:

```bash
"${CLIENT_PYTHON:-python}" --version
```

### 4a) Install the project (recommended)

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

### 4b) Set cluster identity + remote python (recommended)

You can set these either in `phy-launch.env` (preferred) or your shell profile (`~/.zshrc` / `~/.bashrc`):

```bash
export CLUSTER_USER=<your_cluster_username>
export LOGIN_NODE=login1.int.janelia.org
export REMOTE_PYTHON=/groups/scicompsoft/home/<your_cluster_username>/miniconda3/envs/phy/bin/python
```

Then open a new terminal (or `source ~/.zshrc`).

### 5) Confirm cluster access

You need passwordless or normal SSH access to:
- `<your_cluster_username>@login1.int.janelia.org`

Quick test:

```bash
ssh <your_cluster_username>@login1.int.janelia.org "hostname"
```

### 6) One-time cluster setup (required)

`phy-launch.sh` starts the **server on the cluster**, so cluster-side dependencies must exist.

Recommended (automated) option:

```bash
./scripts/bootstrap_cluster.sh
```

This will:
- SSH to the cluster,
- sync this repo to `REMOTE_REPO_DIR`,
- create the remote conda env if needed,
- install with `pip install -e .`,
- verify required imports.

Run once on the cluster:

```bash
conda activate phy
cd /path/to/phy-fastplotlib
python -m pip install -e .
python -c "import phy_remote, phylib, zmq, numpy; print('cluster ok')"
```

Then set `REMOTE_PYTHON` in your local shell to that cluster python path.

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

