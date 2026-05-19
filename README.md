# phy-fastplotlib

Remote Phy-like GUI using fastplotlib.

This repo is designed to:
- run the data/model server on the Janelia cluster,
- open an SSH tunnel automatically,
- launch the local Mac GUI.

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

```bash
nano phy-launch.env
```

Quick sanity check (should print real values, not placeholders):

```bash
grep -E '^(CLUSTER_USER|REMOTE_PYTHON|CLIENT_PYTHON|REMOTE_REPO_DIR|OUTPUT_DIR)=' phy-launch.env
```

Preflight (safe, no job submission):

```bash
./phy-launch.sh --dry-run /path/to/output/<probe>/shank_<n>/
```

First real run:

```bash
./scripts/bootstrap_cluster.sh
./phy-launch.sh /path/to/output/<probe>/shank_<n>/
```

## Day-to-day run

After setup, this is usually enough:

```bash
conda activate phy-fastplotlib
cd phy-fastplotlib
./phy-launch.sh /path/to/output/<probe>/shank_<n>/
```

## Other launch modes

```bash
./phy-launch.sh <probe_letter> <shank_number> [port]
./phy-launch.sh
```

## Help

- Detailed setup, troubleshooting, and manual debug commands:
  [docs/troubleshooting.md](docs/troubleshooting.md)
- See launcher options:

```bash
./phy-launch.sh --help
```
