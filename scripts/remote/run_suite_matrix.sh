#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MANIFEST="configs/matrices/suite_all_models_methods.json"
PROFILE="suite_all_models_methods"
OUTPUT_ROOT="$ROOT/results/matrix"
PYTHON_BIN="${PYTHON_BIN:-}"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT/.venv/tosem_compact}"
GPU_SLOTS="${GPU_SLOTS:-8}"
GPU_POOL_MODE="${GPU_POOL_MODE:-shared}"
CPU_WORKERS="${CPU_WORKERS:-12}"
RETRY_COUNT="${RETRY_COUNT:-1}"
DRY_RUN=0
BOOTSTRAP=0
RUN_FULL=0
CLEAN_OUTPUTS=1

usage() {
  cat <<'EOF'
Usage: run_suite_matrix.sh [--manifest PATH] [--profile NAME] [--output-root PATH] [--python PATH] [--venv PATH]
                           [--gpu-slots N] [--gpu-pool-mode split|shared] [--cpu-workers N] [--retry-count N]
                           [--bootstrap] [--run-full] [--dry-run] [--no-clean]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --profile)
      PROFILE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --gpu-slots)
      GPU_SLOTS="$2"
      shift 2
      ;;
    --gpu-pool-mode)
      GPU_POOL_MODE="$2"
      shift 2
      ;;
    --cpu-workers)
      CPU_WORKERS="$2"
      shift 2
      ;;
    --retry-count)
      RETRY_COUNT="$2"
      shift 2
      ;;
    --bootstrap)
      BOOTSTRAP=1
      shift
      ;;
    --run-full)
      RUN_FULL=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --no-clean)
      CLEAN_OUTPUTS=0
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

MANIFEST_PATH="$MANIFEST"
if [[ "$MANIFEST_PATH" != /* ]]; then
  MANIFEST_PATH="$ROOT/$MANIFEST_PATH"
fi

OUTPUT_PATH="$OUTPUT_ROOT"
if [[ "$OUTPUT_PATH" != /* ]]; then
  OUTPUT_PATH="$ROOT/$OUTPUT_PATH"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  STEP_BLOCK='    "build_suite_manifests",
    "suite_preflight",
    "suite_precheck"'
  if [[ $RUN_FULL -eq 1 ]]; then
    STEP_BLOCK="$STEP_BLOCK,
    \"run_full_matrix\""
  fi
  cat <<EOF
{
  "root": "$ROOT",
  "manifest": "$MANIFEST",
  "profile": "$PROFILE",
  "output_root": "$OUTPUT_PATH",
  "venv": "$VENV_DIR",
  "python": "$PYTHON_BIN",
  "bootstrap": $BOOTSTRAP,
  "run_full": $RUN_FULL,
  "clean_outputs": $CLEAN_OUTPUTS,
  "steps": [
$STEP_BLOCK
  ]
}
EOF
  exit 0
fi

if [[ $BOOTSTRAP -eq 1 ]]; then
  bash "$ROOT/scripts/remote/bootstrap_linux_gpu.sh" --install --python "$BOOTSTRAP_PYTHON" --venv "$VENV_DIR"
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  echo "Create the venv first with: bash $ROOT/scripts/remote/bootstrap_linux_gpu.sh --install --venv $VENV_DIR" >&2
  exit 1
fi

if [[ $CLEAN_OUTPUTS -eq 1 ]]; then
  "$PYTHON_BIN" "$ROOT/scripts/clean_suite_outputs.py" --include-full-matrix --include-release-bundle
fi

bash "$ROOT/scripts/remote/run_preflight.sh" \
  --python "$PYTHON_BIN" \
  --venv "$VENV_DIR" \
  --full-manifest "$MANIFEST_PATH" \
  --full-profile "$PROFILE" \
  --output-root "$OUTPUT_PATH"

"$PYTHON_BIN" "$ROOT/scripts/certify_suite_precheck.py" \
  --python-bin "$PYTHON_BIN" \
  --full-manifest "$MANIFEST_PATH" \
  --full-profile "$PROFILE" \
  --output-root "$OUTPUT_PATH" \
  --gpu-slots "$GPU_SLOTS" \
  --gpu-pool-mode "$GPU_POOL_MODE" \
  --cpu-workers "$CPU_WORKERS" \
  --retry-count "$RETRY_COUNT" \
  --fail-fast

if [[ $RUN_FULL -ne 1 ]]; then
  echo "Suite precheck complete. Pass --run-full to start the compact full matrix." >&2
  exit 0
fi

"$PYTHON_BIN" "$ROOT/scripts/run_full_matrix.py" \
  --manifest "$MANIFEST_PATH" \
  --profile "$PROFILE" \
  --output-root "$OUTPUT_PATH" \
  --gpu-slots "$GPU_SLOTS" \
  --gpu-pool-mode "$GPU_POOL_MODE" \
  --cpu-workers "$CPU_WORKERS" \
  --retry-count "$RETRY_COUNT" \
  --fail-fast

echo "Suite matrix run complete."
