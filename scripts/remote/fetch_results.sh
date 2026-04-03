#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_ROOT="${REMOTE_ROOT:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
RUN_DIR="${RUN_DIR:-results}"
DEST_DIR="${DEST_DIR:-$ROOT/results/fetched_suite}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: fetch_results.sh [--run-dir PATH] [--dest-dir PATH] [--remote-host HOST] [--remote-user USER] [--remote-root PATH] [--remote-port PORT] [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --dest-dir)
      DEST_DIR="$2"
      shift 2
      ;;
    --remote-host)
      REMOTE_HOST="$2"
      shift 2
      ;;
    --remote-user)
      REMOTE_USER="$2"
      shift 2
      ;;
    --remote-root)
      REMOTE_ROOT="$2"
      shift 2
      ;;
    --remote-port)
      REMOTE_PORT="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$REMOTE_HOST" || -z "$REMOTE_ROOT" ]]; then
  echo "REMOTE_HOST and REMOTE_ROOT are required." >&2
  exit 1
fi

REMOTE_PREFIX="${REMOTE_HOST}:${REMOTE_ROOT}"
if [[ -n "$REMOTE_USER" ]]; then
  REMOTE_PREFIX="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}"
fi

SOURCE="${REMOTE_PREFIX}/${RUN_DIR}"
mkdir -p "$DEST_DIR"

if [[ $DRY_RUN -eq 1 ]]; then
  printf '%s\n' "{" \
    "  \"source\": \"${SOURCE//\"/\\\"}\"," \
    "  \"dest_dir\": \"${DEST_DIR//\"/\\\"}\"," \
    "  \"remote_port\": \"${REMOTE_PORT//\"/\\\"}\"," \
    "  \"command\": \"rsync -av -e \\\"ssh -p ${REMOTE_PORT//\"/\\\"}\\\" ${SOURCE//\"/\\\"}/ ${DEST_DIR//\"/\\\"}/\"" \
    "}"
  exit 0
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -av -e "ssh -p $REMOTE_PORT" --delete "$SOURCE/" "$DEST_DIR/"
else
  rm -rf "$DEST_DIR"
  mkdir -p "$DEST_DIR"
  scp -P "$REMOTE_PORT" -r "$SOURCE/." "$DEST_DIR/"
fi

echo "Fetched results from $SOURCE to $DEST_DIR"
