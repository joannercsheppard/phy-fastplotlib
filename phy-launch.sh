#!/usr/bin/env bash
# phy-launch.sh — submit the phy-remote server on the Janelia cluster,
# open the SSH tunnel, and launch the client, all from one command.
#
# Usage:
#   ./phy-launch.sh                          # interactive menu (uses OUTPUT_DIR)
#   ./phy-launch.sh /path/to/dir [port]      # search a specific directory
#   ./phy-launch.sh <probe> <shank> [port]   # e.g. ./phy-launch.sh b 0
#   ./phy-launch.sh --dry-run <mode args...> # print what would run
#   ./phy-launch.sh --help
#
# The default search directory can be overridden via the OUTPUT_DIR env var.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${PHY_LAUNCH_CONFIG:-${SCRIPT_DIR}/phy-launch.env}"

if [[ -f "${CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
  set +a
fi

usage() {
  cat <<'EOF'
Usage:
  ./phy-launch.sh [--dry-run] [--help]
  ./phy-launch.sh [--dry-run] /path/to/dir [port]
  ./phy-launch.sh [--dry-run] <probe_letter> <shank_number> [port]

Modes:
  - No args: interactive dataset menu under OUTPUT_DIR
  - /abs/path: find params.py under that path
  - <probe> <shank>: find params.py under OUTPUT_DIR/<probe>/shank_<shank>

Important env vars:
  CLUSTER_USER     Cluster username (default: $USER)
  LOGIN_NODE       Cluster login node (default: login1.int.janelia.org)
  REMOTE_PYTHON    Remote Python path for server
  CLIENT_PYTHON    Local Python path for GUI client
  OUTPUT_DIR       Default dataset root for interactive/probe-shank modes
EOF
}

DRY_RUN=0
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --*)
      echo "ERROR: unknown option: $arg"
      usage
      exit 1
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

CLUSTER_USER="${CLUSTER_USER:-${USER}}"
LOGIN_NODE="${LOGIN_NODE:-login1.int.janelia.org}"
REMOTE_CONDA_ENV="${REMOTE_CONDA_ENV:-phy}"
PYTHON="${REMOTE_PYTHON:-/groups/scicompsoft/home/${CLUSTER_USER}/miniconda3/envs/${REMOTE_CONDA_ENV}/bin/python}"
CLIENT_PYTHON="${CLIENT_PYTHON:-}"
OUTPUT_DIR="${OUTPUT_DIR:-/groups/voigts/voigtslab/neuropixels_2025/npx11/2026_03_05_npx11_large_maze/output/}"

if ! command -v ssh >/dev/null 2>&1; then
  echo "ERROR: ssh is not installed or not in PATH."
  exit 1
fi

if ! command -v lsof >/dev/null 2>&1; then
  echo "WARNING: lsof not found. Stale tunnel cleanup may be limited."
fi

if [[ -z "${CLIENT_PYTHON}" ]]; then
  if command -v python >/dev/null 2>&1; then
    CLIENT_PYTHON="$(command -v python)"
  elif command -v python3 >/dev/null 2>&1; then
    CLIENT_PYTHON="$(command -v python3)"
  else
    echo "ERROR: no local python interpreter found in PATH."
    echo "Set CLIENT_PYTHON=/path/to/python and rerun."
    exit 1
  fi
fi

if [[ ! -x "${CLIENT_PYTHON}" ]]; then
  echo "ERROR: CLIENT_PYTHON is not executable: ${CLIENT_PYTHON}"
  exit 1
fi

if ! ssh "${CLUSTER_USER}@${LOGIN_NODE}" "command -v bsub >/dev/null"; then
  echo "ERROR: bsub not available via ${CLUSTER_USER}@${LOGIN_NODE}."
  echo "Check cluster access/login node settings and rerun."
  exit 1
fi

if ! ssh "${CLUSTER_USER}@${LOGIN_NODE}" "test -x '${PYTHON}'"; then
  echo "ERROR: remote python not found or not executable: ${PYTHON}"
  echo "Set REMOTE_PYTHON=/path/to/cluster/python and rerun."
  exit 1
fi

# ---------------------------------------------------------------------------
# 1. Determine params path — from args or interactive menu
# ---------------------------------------------------------------------------

if [[ $# -ge 2 && "${1}" =~ ^[a-zA-Z]$ && "${2}" =~ ^[0-9]+$ ]]; then
  # Direct mode: ./phy-launch.sh b 0 [port]
  PROBE="${1}"
  SHANK="${2}"
  PORT="${3:-5557}"

  PARAMS_PATH=$(ssh "${CLUSTER_USER}@${LOGIN_NODE}" \
    "find '${OUTPUT_DIR}/${PROBE}/shank_${SHANK}' -name 'params.py' -maxdepth 2 2>/dev/null | sort | head -1")

  if [[ -z "${PARAMS_PATH}" ]]; then
    echo "ERROR: no params.py found under ${OUTPUT_DIR}/${PROBE}/shank_${SHANK}"
    exit 1
  fi
  echo "Selected: ${PARAMS_PATH}"
  echo ""

elif [[ $# -ge 1 && "${1}" == /* ]]; then
  # Path mode: ./phy-launch.sh /path/to/dir [port]
  SEARCH_DIR="${1}"
  PORT="${2:-5557}"

  echo "Scanning ${SEARCH_DIR} for datasets..."

  DATASET_LIST=$(ssh "${CLUSTER_USER}@${LOGIN_NODE}" \
    "find '${SEARCH_DIR}' -name 'params.py' -maxdepth 6 2>/dev/null | sort")

  if [[ -z "${DATASET_LIST}" ]]; then
    echo "ERROR: no params.py files found under ${SEARCH_DIR}"
    exit 1
  fi

  IFS=$'\n' read -r -d '' -a DATASETS <<< "${DATASET_LIST}" || true

  if [[ ${#DATASETS[@]} -eq 1 ]]; then
    PARAMS_PATH="${DATASETS[0]}"
    echo "Only one dataset found: ${PARAMS_PATH}"
  else
    echo ""
    echo "Available datasets:"
    for i in "${!DATASETS[@]}"; do
      REL="${DATASETS[$i]#${SEARCH_DIR}}"
      REL="${REL#/}"
      printf "  [%d] %s\n" $((i + 1)) "${REL}"
    done
    echo ""

    while true; do
      read -rp "Select dataset [1-${#DATASETS[@]}]: " CHOICE
      if [[ "${CHOICE}" =~ ^[0-9]+$ ]] && \
         [[ "${CHOICE}" -ge 1 ]] && \
         [[ "${CHOICE}" -le ${#DATASETS[@]} ]]; then
        PARAMS_PATH="${DATASETS[$((CHOICE - 1))]}"
        break
      fi
      echo "  Please enter a number between 1 and ${#DATASETS[@]}."
    done
  fi

  echo "Selected: ${PARAMS_PATH}"
  echo ""

else
  # Interactive menu mode (uses OUTPUT_DIR)
  PORT="${1:-5557}"

  echo "Scanning ${OUTPUT_DIR} for datasets..."

  DATASET_LIST=$(ssh "${CLUSTER_USER}@${LOGIN_NODE}" \
    "find '${OUTPUT_DIR}' -name 'params.py' -mindepth 3 -maxdepth 4 2>/dev/null | sort")

  if [[ -z "${DATASET_LIST}" ]]; then
    echo "ERROR: no params.py files found under ${OUTPUT_DIR}"
    exit 1
  fi

  IFS=$'\n' read -r -d '' -a DATASETS <<< "${DATASET_LIST}" || true

  if [[ ${#DATASETS[@]} -eq 1 ]]; then
    PARAMS_PATH="${DATASETS[0]}"
    echo "Only one dataset found: ${PARAMS_PATH}"
  else
    echo ""
    echo "Available datasets:"
    for i in "${!DATASETS[@]}"; do
      REL="${DATASETS[$i]#${OUTPUT_DIR}}"
      REL="${REL#/}"
      printf "  [%d] %s\n" $((i + 1)) "${REL}"
    done
    echo ""

    while true; do
      read -rp "Select dataset [1-${#DATASETS[@]}]: " CHOICE
      if [[ "${CHOICE}" =~ ^[0-9]+$ ]] && \
         [[ "${CHOICE}" -ge 1 ]] && \
         [[ "${CHOICE}" -le ${#DATASETS[@]} ]]; then
        PARAMS_PATH="${DATASETS[$((CHOICE - 1))]}"
        break
      fi
      echo "  Please enter a number between 1 and ${#DATASETS[@]}."
    done
  fi

  echo "Selected: ${PARAMS_PATH}"
  echo ""
fi

  if [[ ${DRY_RUN} -eq 1 ]]; then
    echo "[dry-run] would submit:"
    echo "  ssh ${CLUSTER_USER}@${LOGIN_NODE} \"bsub -J phy-remote -n 1 -gpu 'num=1' -q gpu_l4 -W 480 '${PYTHON} -m phy_remote.server ${PARAMS_PATH} --port ${PORT}'\""
    echo "[dry-run] would poll bpeek until PHY_REMOTE_READY"
    echo "[dry-run] would open tunnel and launch client:"
    echo "  PYTHONPATH=${SCRIPT_DIR}:\$PYTHONPATH ${CLIENT_PYTHON} -m phy_remote.client.app --port ${PORT}"
    exit 0
  fi

# ---------------------------------------------------------------------------
# 2. Submit the bsub job on the cluster
# ---------------------------------------------------------------------------
echo "Submitting server job on ${LOGIN_NODE}..."

BSUB_OUTPUT=$(ssh "${CLUSTER_USER}@${LOGIN_NODE}" \
  "bsub -J phy-remote -n 1 -gpu 'num=1' -q gpu_l4 -W 480 \
    '${PYTHON} -m phy_remote.server ${PARAMS_PATH} --port ${PORT}'")

JOB_ID=$(echo "${BSUB_OUTPUT}" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+')

if [[ -z "${JOB_ID}" ]]; then
  echo "ERROR: could not parse job ID from bsub output:"
  echo "  ${BSUB_OUTPUT}"
  exit 1
fi

echo "Job ${JOB_ID} submitted. Waiting for server to start (this can take 1-2 min)..."

# ---------------------------------------------------------------------------
# 3. Poll bpeek until PHY_REMOTE_READY appears
# ---------------------------------------------------------------------------
COMPUTE_HOST=""
ACTUAL_PORT=""
ATTEMPTS=0
MAX_ATTEMPTS=60  # 5 minutes at 5-second intervals

while [[ $ATTEMPTS -lt $MAX_ATTEMPTS ]]; do
  PEEK=$(ssh "${CLUSTER_USER}@${LOGIN_NODE}" "bpeek ${JOB_ID}" 2>/dev/null || true)
  READY_LINE=$(echo "${PEEK}" | grep "PHY_REMOTE_READY" || true)

  if [[ -n "${READY_LINE}" ]]; then
    READY_LINE=$(echo "${READY_LINE}" | head -1)
    COMPUTE_HOST=$(echo "${READY_LINE}" | grep -oE 'host=\S+' | cut -d= -f2)
    ACTUAL_PORT=$(echo "${READY_LINE}" | grep -oE 'port=[0-9]+' | cut -d= -f2)
    break
  fi

  ATTEMPTS=$((ATTEMPTS + 1))
  printf "\r  waiting... (%ds)" $((ATTEMPTS * 5))
  sleep 5
done
echo ""

if [[ -z "${COMPUTE_HOST}" ]]; then
  echo "ERROR: server did not become ready within $((MAX_ATTEMPTS * 5)) seconds."
  echo "Check job status with: ssh ${CLUSTER_USER}@${LOGIN_NODE} bjobs"
  echo "Check job output with: ssh ${CLUSTER_USER}@${LOGIN_NODE} bpeek ${JOB_ID}"
  exit 1
fi

echo "Server ready: ${COMPUTE_HOST}:${ACTUAL_PORT}"

# ---------------------------------------------------------------------------
# 4. Open the SSH tunnel in the background
# ---------------------------------------------------------------------------
# Kill any stale process already using this port
STALE_PID=$(lsof -ti "tcp:${ACTUAL_PORT}" 2>/dev/null || true)
if [[ -n "${STALE_PID}" ]]; then
  echo "Killing stale process on port ${ACTUAL_PORT} (PID ${STALE_PID})..."
  kill "${STALE_PID}" 2>/dev/null || true
  sleep 1
fi

echo "Opening SSH tunnel: localhost:${ACTUAL_PORT} → ${COMPUTE_HOST}:${ACTUAL_PORT}..."
ssh -N -L "${ACTUAL_PORT}:${COMPUTE_HOST}:${ACTUAL_PORT}" \
  "${CLUSTER_USER}@${LOGIN_NODE}" &
TUNNEL_PID=$!

# Give the tunnel a moment to establish
sleep 1

# Verify tunnel is still running
if ! kill -0 "${TUNNEL_PID}" 2>/dev/null; then
  echo "ERROR: SSH tunnel exited immediately. Check your SSH config."
  exit 1
fi

echo "Tunnel open (PID ${TUNNEL_PID}). Launching client..."

# ---------------------------------------------------------------------------
# 5. Launch the client
# ---------------------------------------------------------------------------
# Run the client in the foreground; when it exits, clean up the tunnel.
PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}" \
  "${CLIENT_PYTHON}" -m phy_remote.client.app --port "${ACTUAL_PORT}" || true

echo "Client closed. Shutting down tunnel..."
kill "${TUNNEL_PID}" 2>/dev/null || true
lsof -ti "tcp:${ACTUAL_PORT}" | xargs kill -9 2>/dev/null || true
echo "Killing cluster job ${JOB_ID}..."
ssh "${CLUSTER_USER}@${LOGIN_NODE}" "bkill ${JOB_ID}" 2>/dev/null || true
echo "Done."
