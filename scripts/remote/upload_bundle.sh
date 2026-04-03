#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_DIR="${BUNDLE_DIR:-$ROOT/results/release_bundle}"
REMOTE_HOST="${REMOTE_HOST:-}"
REMOTE_USER="${REMOTE_USER:-}"
REMOTE_ROOT="${REMOTE_ROOT:-}"
REMOTE_PORT="${REMOTE_PORT:-22}"
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: upload_bundle.sh [--bundle-dir PATH] [--remote-host HOST] [--remote-user USER] [--remote-root PATH] [--remote-port PORT] [--dry-run]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --bundle-dir)
      BUNDLE_DIR="$2"
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

if [[ ! -d "$BUNDLE_DIR" ]]; then
  echo "Missing bundle directory: $BUNDLE_DIR" >&2
  exit 1
fi

if [[ -z "$REMOTE_HOST" || -z "$REMOTE_ROOT" ]]; then
  echo "REMOTE_HOST and REMOTE_ROOT are required." >&2
  exit 1
fi

TARGET="${REMOTE_HOST}:${REMOTE_ROOT}/$(basename "$BUNDLE_DIR")"
SSH_TARGET="${REMOTE_HOST}"
if [[ -n "$REMOTE_USER" ]]; then
  TARGET="${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/$(basename "$BUNDLE_DIR")"
  SSH_TARGET="${REMOTE_USER}@${REMOTE_HOST}"
fi
REMOTE_TARGET_DIR="${REMOTE_ROOT}/$(basename "$BUNDLE_DIR")"

if [[ $DRY_RUN -eq 1 ]]; then
  printf '%s\n' "{" \
    "  \"bundle_dir\": \"${BUNDLE_DIR//\"/\\\"}\"," \
    "  \"target\": \"${TARGET//\"/\\\"}\"," \
    "  \"remote_port\": \"${REMOTE_PORT//\"/\\\"}\"," \
    "  \"command\": \"rsync -av -e \\\"ssh -p ${REMOTE_PORT//\"/\\\"}\\\" --delete ${BUNDLE_DIR//\"/\\\"}/ ${TARGET//\"/\\\"}/\"" \
    "}"
  exit 0
fi

if command -v rsync >/dev/null 2>&1; then
  rsync -av -e "ssh -p $REMOTE_PORT" --delete "$BUNDLE_DIR/" "$TARGET/"
else
  ssh -p "$REMOTE_PORT" "$SSH_TARGET" "rm -rf '$REMOTE_TARGET_DIR' && mkdir -p '$REMOTE_TARGET_DIR'"
  scp -P "$REMOTE_PORT" -r "$BUNDLE_DIR/." "$SSH_TARGET:$REMOTE_TARGET_DIR/"
fi

echo "Uploaded release bundle to $TARGET"
