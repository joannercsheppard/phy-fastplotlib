#!/usr/bin/env bash
# One-time setup for new users.
# - Verifies local Python
# - Installs package in editable mode
# - Optionally checks cluster connectivity/tools

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CHECK_REMOTE=1
if [[ "${1:-}" == "--no-remote-check" ]]; then
  CHECK_REMOTE=0
fi

LOCAL_PYTHON="${CLIENT_PYTHON:-}"
if [[ -z "${LOCAL_PYTHON}" ]]; then
  if command -v python >/dev/null 2>&1; then
    LOCAL_PYTHON="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    LOCAL_PYTHON="$(command -v python3)"
  else
    echo "ERROR: no python found in PATH."
    echo "Set CLIENT_PYTHON=/path/to/python and rerun."
    exit 1
  fi
fi

echo "Using local python: ${LOCAL_PYTHON}"
"${LOCAL_PYTHON}" --version

echo "Installing package in editable mode..."
"${LOCAL_PYTHON}" -m pip install -e "${REPO_ROOT}"

if [[ ${CHECK_REMOTE} -eq 1 ]]; then
  CLUSTER_USER="${CLUSTER_USER:-${USER}}"
  LOGIN_NODE="${LOGIN_NODE:-login1.int.janelia.org}"
  REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-phy}"
  REMOTE_PYTHON="${REMOTE_PYTHON:-/groups/scicompsoft/home/${CLUSTER_USER}/miniconda3/envs/${REMOTE_CONDA_ENV}/bin/python}"

  if ! command -v ssh >/dev/null 2>&1; then
    echo "ERROR: ssh is not installed or not in PATH."
    exit 1
  fi

  echo "Checking SSH access to ${CLUSTER_USER}@${LOGIN_NODE} ..."
  ssh "${CLUSTER_USER}@${LOGIN_NODE}" "hostname" >/dev/null

  echo "Checking cluster tools (bsub + remote python) ..."
  ssh "${CLUSTER_USER}@${LOGIN_NODE}" "command -v bsub >/dev/null && test -x '${REMOTE_PYTHON}'"
fi

echo ""
echo "Setup complete."
echo ""
echo "Recommended shell settings:"
echo "  export CLUSTER_USER=<your_cluster_username>"
echo "  export LOGIN_NODE=login1.int.janelia.org"
echo "  export REMOTE_PYTHON=/groups/scicompsoft/home/<your_cluster_username>/miniconda3/envs/phy/bin/python"
echo ""
echo "Run launcher:"
echo "  ./phy-launch.sh /path/to/output/a/shank_0/"
