#!/bin/bash
# bsub_template.sh — LSF job script for phy-remote server (Janelia cluster)
#
# Same tunnel pattern as Janelia Jupyter jobs:
#   - Server binds to 0.0.0.0 on the compute node
#   - MacBook forwards through the login node to the compute node via TCP
#   - No SSH keys between nodes required
#
# Usage (interactive — recommended):
#   bsub -Is -q interactive -n 1 -R "rusage[mem=8000]" bash
#   bash bsub_template.sh /path/to/params.py
#
#BSUB -J phy-remote
#BSUB -q local
#BSUB -n 1
#BSUB -R "rusage[mem=8000]"
#BSUB -W 480
#BSUB -o /tmp/phy-remote.%J.out
#BSUB -e /tmp/phy-remote.%J.err

set -euo pipefail

PARAMS_PATH="${1:-${PHY_PARAMS:-/path/to/params.py}}"
ZMQ_PORT="${PHY_PORT:-5555}"
CONDA_ENV="${PHY_CONDA_ENV:-phy}"
LOGIN_NODE="login1.int.janelia.org"

CONDA_BASE="${HOME}/miniconda3"
[[ ! -d "$CONDA_BASE" ]] && CONDA_BASE="${HOME}/anaconda3"
PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"

exec "${PYTHON}" -m phy_remote.server "${PARAMS_PATH}" \
    --port "${ZMQ_PORT}" \
    --login-node "${LOGIN_NODE}"
