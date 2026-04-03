#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv/tosem_compact}"
INSTALL=0
DRY_RUN=0
SYSTEM_SITE_PACKAGES=0

usage() {
  cat <<'EOF'
Usage: bootstrap_linux_gpu.sh [--install] [--dry-run] [--system-site-packages] [--python PATH] [--venv PATH]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --install)
      INSTALL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --system-site-packages)
      SYSTEM_SITE_PACKAGES=1
      shift
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
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

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  exit 1
fi

mkdir -p "$ROOT/results/runs" "$ROOT/results/release_bundle" "$ROOT/model_cache"

if [[ $DRY_RUN -eq 1 ]]; then
  printf '%s\n' "{" \
    "  \"root\": \"${ROOT//\"/\\\"}\"," \
    "  \"python\": \"${PYTHON_BIN//\"/\\\"}\"," \
    "  \"venv\": \"${VENV_DIR//\"/\\\"}\"," \
    "  \"install\": ${INSTALL}," \
    "  \"system_site_packages\": ${SYSTEM_SITE_PACKAGES}," \
    "  \"results_dir\": \"${ROOT//\"/\\\"}/results/runs\"," \
    "  \"toolchain_reference\": \"${ROOT//\"/\\\"}/docs/remote_linux_gpu.md\"" \
    "}"
  exit 0
fi

if [[ $INSTALL -eq 1 ]]; then
  if [[ ! -d "$VENV_DIR" ]]; then
    VENV_ARGS=()
    if [[ $SYSTEM_SITE_PACKAGES -eq 1 ]]; then
      VENV_ARGS+=(--system-site-packages)
    fi
    "$PYTHON_BIN" -m venv "${VENV_ARGS[@]}" "$VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip
  python -m pip install -r "$ROOT/requirements.txt"
  if [[ -f "$ROOT/requirements-remote.txt" ]]; then
    python -m pip install -r "$ROOT/requirements-remote.txt"
  fi
  if ! python -c "import torch" >/dev/null 2>&1; then
    if command -v nvidia-smi >/dev/null 2>&1; then
      python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
    else
      python -m pip install torch torchvision torchaudio
    fi
  fi
fi

echo "Bootstrap complete."
echo "Root: $ROOT"
echo "Python: $PYTHON_BIN"
echo "Venv: $VENV_DIR"
echo "Note: bootstrap provisions the Python environment only; run_preflight.sh validates host compilers and runtimes."
