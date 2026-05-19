#!/usr/bin/env bash
# Bootstrap cluster-side environment + code for phy-fastplotlib.
#
# What it does:
# 1) Loads local config from phy-launch.env (or PHY_LAUNCH_CONFIG)
# 2) Syncs this repo to cluster (rsync if available, else tar over ssh)
# 3) Creates remote conda env if missing
# 4) Installs package on cluster with pip -e
# 5) Verifies imports needed by server

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${PHY_LAUNCH_CONFIG:-${REPO_ROOT}/phy-launch.env}"

if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
else
  echo "WARNING: config file not found: ${CONFIG_FILE}"
  echo "Using environment variables + defaults."
fi

CLUSTER_USER="${CLUSTER_USER:-${USER}}"
LOGIN_NODE="${LOGIN_NODE:-login1.int.janelia.org}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-phy}"
REMOTE_REPO_DIR="${REMOTE_REPO_DIR:-$HOME/phy-fastplotlib}"
REMOTE_PYTHON="${REMOTE_PYTHON:-/groups/scicompsoft/home/${CLUSTER_USER}/miniconda3/envs/${REMOTE_CONDA_ENV}/bin/python}"

_placeholder() {
  [[ "$1" == *"<"* || "$1" == *">"* ]]
}

if _placeholder "${CLUSTER_USER}"; then
  echo "ERROR: CLUSTER_USER looks unset (${CLUSTER_USER})."
  echo "Set CLUSTER_USER in phy-launch.env before running bootstrap."
  exit 1
fi

if _placeholder "${LOGIN_NODE}"; then
  echo "ERROR: LOGIN_NODE looks unset (${LOGIN_NODE})."
  echo "Set LOGIN_NODE in phy-launch.env before running bootstrap."
  exit 1
fi

if _placeholder "${REMOTE_PYTHON}"; then
  echo "ERROR: REMOTE_PYTHON looks unset (${REMOTE_PYTHON})."
  echo "Set REMOTE_PYTHON in phy-launch.env before running bootstrap."
  exit 1
fi

if _placeholder "${REMOTE_REPO_DIR}"; then
  echo "ERROR: REMOTE_REPO_DIR looks unset (${REMOTE_REPO_DIR})."
  echo "Set REMOTE_REPO_DIR in phy-launch.env before running bootstrap."
  exit 1
fi

SSH_TARGET="${CLUSTER_USER}@${LOGIN_NODE}"

echo "[1/5] Checking SSH access to ${SSH_TARGET}..."
ssh "${SSH_TARGET}" "hostname" >/dev/null

echo "[2/5] Syncing repo to cluster: ${REMOTE_REPO_DIR}"
if command -v rsync >/dev/null 2>&1; then
  rsync -az --delete \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='phy-launch.env' \
    "${REPO_ROOT}/" "${SSH_TARGET}:${REMOTE_REPO_DIR}/"
else
  echo "rsync not found locally; falling back to tar stream copy"
  ssh "${SSH_TARGET}" "mkdir -p '${REMOTE_REPO_DIR}'"
  tar -C "${REPO_ROOT}" \
      --exclude='.git' \
      --exclude='.venv' \
      --exclude='__pycache__' \
      --exclude='*.pyc' \
      --exclude='phy-launch.env' \
      -cf - . | ssh "${SSH_TARGET}" "tar -C '${REMOTE_REPO_DIR}' -xf -"
fi

echo "[3/5] Ensuring remote conda env '${REMOTE_CONDA_ENV}' exists"
ssh "${SSH_TARGET}" "bash -lc '
set -euo pipefail
CONDA_BASE=\"$HOME/miniconda3\"
[[ -d \"$CONDA_BASE\" ]] || CONDA_BASE=\"$HOME/anaconda3\"
if [[ ! -d \"$CONDA_BASE\" ]]; then
  echo \"ERROR: could not find miniconda3/anaconda3 in $HOME\" >&2
  exit 1
fi
source \"$CONDA_BASE/etc/profile.d/conda.sh\"
if ! conda env list | awk \"{print \$1}\" | grep -qx \"${REMOTE_CONDA_ENV}\"; then
  conda create -n \"${REMOTE_CONDA_ENV}\" python=3.11 -y
fi
'"

echo "[4/5] Installing package on cluster (editable)"
ssh "${SSH_TARGET}" "bash -lc '
set -euo pipefail
CONDA_BASE=\"$HOME/miniconda3\"
[[ -d \"$CONDA_BASE\" ]] || CONDA_BASE=\"$HOME/anaconda3\"
source \"$CONDA_BASE/etc/profile.d/conda.sh\"
conda activate \"${REMOTE_CONDA_ENV}\"
cd \"${REMOTE_REPO_DIR}\"
python -m pip install -e .
'"

echo "[5/5] Verifying cluster imports"
ssh "${SSH_TARGET}" "bash -lc '
set -euo pipefail
CONDA_BASE=\"$HOME/miniconda3\"
[[ -d \"$CONDA_BASE\" ]] || CONDA_BASE=\"$HOME/anaconda3\"
source \"$CONDA_BASE/etc/profile.d/conda.sh\"
conda activate \"${REMOTE_CONDA_ENV}\"
python -c \"import phy_remote, phylib, zmq, numpy; print(\\\"cluster ok\\\")\"
'"

echo ""
echo "Bootstrap complete."
echo "Set these in phy-launch.env (if not already):"
echo "  REMOTE_REPO_DIR=${REMOTE_REPO_DIR}"
echo "  REMOTE_PYTHON=${REMOTE_PYTHON}"
