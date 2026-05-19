# Troubleshooting & Advanced Usage

## First-time setup (detailed)

### 1) Clone and enter the repo

```bash
git clone https://github.com/joannercsheppard/phy-fastplotlib.git phy-fastplotlib
cd phy-fastplotlib
```

### 2) Make scripts executable

```bash
chmod +x phy-launch.sh
chmod +x scripts/setup.sh
chmod +x scripts/bootstrap_cluster.sh
```

### 3) Create launcher config

```bash
cp phy-launch.env.example phy-launch.env
nano phy-launch.env
```

Required values:
- `CLUSTER_USER`
- `REMOTE_PYTHON`

Useful optional values:
- `CLIENT_PYTHON`
- `REMOTE_REPO_DIR`
- `OUTPUT_DIR`

### 4) Create local environment

```bash
conda create -n phy-fastplotlib python=3.11 -y
conda activate phy-fastplotlib
python -m pip install -e .
```

### 5) Bootstrap cluster

```bash
./scripts/bootstrap_cluster.sh
```

This script:
- verifies SSH,
- syncs code to cluster,
- creates remote conda env if needed,
- installs package remotely,
- checks imports (`phy_remote`, `phylib`, `zmq`, `numpy`).

## Launcher modes

### Path mode (recommended)

```bash
./phy-launch.sh /path/to/output/<probe>/shank_<n>/
```

### Probe/shank mode

```bash
./phy-launch.sh <probe_letter> <shank_number> [port]
```

### Interactive mode

```bash
./phy-launch.sh
```

### Dry-run (no submission)

```bash
./phy-launch.sh --dry-run /path/to/output/<probe>/shank_<n>/
```

## Common issues

### `no params.py found`

- Wrong dataset root or wrong probe/shank.
- Use path mode with exact directory, or set `OUTPUT_DIR`.

```bash
OUTPUT_DIR=/groups/.../output ./phy-launch.sh a 0
```

### GUI does not open

Most common causes:
- server not ready,
- tunnel failed,
- wrong local Python.

Use explicit client Python:

```bash
CLIENT_PYTHON=/path/to/local/env/bin/python ./phy-launch.sh /groups/.../shank_0/
```

### Exit code 146 or silent client exit

Run client directly with debug logs:

```bash
CLIENT_PYTHON=/path/to/local/env/bin/python \
PYTHONPATH=$PWD \
/path/to/local/env/bin/python -m phy_remote.client.app --port 5557 --log-level DEBUG
```

## Manual run (debug only)

1) Start server on cluster:

```bash
/path/to/cluster/env/bin/python -m phy_remote.server /abs/path/to/params.py --port 5557
```

or

```bash
phy-remote-server /abs/path/to/params.py --port 5557
```

2) Open tunnel on Mac:

```bash
ssh -N -L 5557:<compute-host>:5557 <your_cluster_username>@login1.int.janelia.org
```

3) Start client on Mac:

```bash
PYTHONPATH=$PWD /path/to/local/env/bin/python -m phy_remote.client.app --port 5557
```

or

```bash
phy-remote-client --port 5557
```
