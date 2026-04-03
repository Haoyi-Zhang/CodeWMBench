#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
FULL_MANIFEST="configs/matrices/suite_all_models_methods.json"
FULL_PROFILE="suite_all_models_methods"
STAGE_A_MANIFEST="configs/matrices/suite_canary_heavy.json"
STAGE_A_PROFILE="suite_canary_heavy"
STAGE_B_MANIFEST="configs/matrices/model_invocation_smoke.json"
STAGE_B_PROFILE="model_invocation_smoke"
OUTPUT_ROOT="$ROOT/results/matrix"
VENV_DIR="${VENV_DIR:-$ROOT/.venv/tosem_compact}"
PYTHON_BIN="${PYTHON_BIN:-}"
DRY_RUN=0
REQUIRE_HF_TOKEN=0
SKIP_HF_ACCESS=0

usage() {
  cat <<'EOF'
Usage: run_preflight.sh [--full-manifest PATH] [--full-profile NAME] [--stage-a-manifest PATH] [--stage-a-profile NAME]
                        [--stage-b-manifest PATH] [--stage-b-profile NAME] [--output-root PATH]
                        [--venv PATH] [--python PATH] [--dry-run] [--require-hf-token] [--skip-hf-access]
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --full-manifest)
      FULL_MANIFEST="$2"
      shift 2
      ;;
    --full-profile)
      FULL_PROFILE="$2"
      shift 2
      ;;
    --stage-a-manifest)
      STAGE_A_MANIFEST="$2"
      shift 2
      ;;
    --stage-a-profile)
      STAGE_A_PROFILE="$2"
      shift 2
      ;;
    --stage-b-manifest)
      STAGE_B_MANIFEST="$2"
      shift 2
      ;;
    --stage-b-profile)
      STAGE_B_PROFILE="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --venv)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --require-hf-token)
      REQUIRE_HF_TOKEN=1
      shift
      ;;
    --skip-hf-access)
      SKIP_HF_ACCESS=1
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

FULL_MANIFEST_PATH="$FULL_MANIFEST"
if [[ "$FULL_MANIFEST_PATH" != /* ]]; then
  FULL_MANIFEST_PATH="$ROOT/$FULL_MANIFEST_PATH"
fi

STAGE_A_MANIFEST_PATH="$STAGE_A_MANIFEST"
if [[ "$STAGE_A_MANIFEST_PATH" != /* ]]; then
  STAGE_A_MANIFEST_PATH="$ROOT/$STAGE_A_MANIFEST_PATH"
fi

STAGE_B_MANIFEST_PATH="$STAGE_B_MANIFEST"
if [[ "$STAGE_B_MANIFEST_PATH" != /* ]]; then
  STAGE_B_MANIFEST_PATH="$ROOT/$STAGE_B_MANIFEST_PATH"
fi

OUTPUT_PATH="$OUTPUT_ROOT"
if [[ "$OUTPUT_PATH" != /* ]]; then
  OUTPUT_PATH="$ROOT/$OUTPUT_PATH"
fi

if [[ -z "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$VENV_DIR/bin/python"
fi

if [[ $DRY_RUN -eq 1 ]]; then
  cat <<EOF
{
  "root": "$ROOT",
  "full_manifest": "$FULL_MANIFEST",
  "full_profile": "$FULL_PROFILE",
  "stage_a_manifest": "$STAGE_A_MANIFEST",
  "stage_a_profile": "$STAGE_A_PROFILE",
  "stage_b_manifest": "$STAGE_B_MANIFEST",
  "stage_b_profile": "$STAGE_B_PROFILE",
  "output_root": "$OUTPUT_PATH",
  "venv": "$VENV_DIR",
  "python": "$PYTHON_BIN",
  "checks": [
    "build_suite_manifests",
    "audit_benchmarks",
    "audit_suite_matrix",
    "python_version",
    "toolchain",
    "cuda",
    "disk",
    "stage_a_dry_run",
    "stage_b_dry_run"
  ]
}
EOF
  exit 0
fi

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing Python interpreter: $PYTHON_BIN" >&2
  echo "Create the venv first with: bash $ROOT/scripts/remote/bootstrap_linux_gpu.sh --install --venv $VENV_DIR" >&2
  exit 1
fi

if [[ $REQUIRE_HF_TOKEN -eq 1 && -z "${HF_ACCESS_TOKEN:-}" ]]; then
  echo "HF_ACCESS_TOKEN is required for this preflight." >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT/scripts/build_suite_manifests.py"
bash "$ROOT/scripts/fetch_runtime_upstreams.sh" all
"$PYTHON_BIN" "$ROOT/scripts/audit_benchmarks.py" --manifest "$FULL_MANIFEST_PATH" --matrix-profile "$FULL_PROFILE" --profile "$FULL_PROFILE"

AUDIT_ARGS=(
  "$PYTHON_BIN" "$ROOT/scripts/audit_full_matrix.py"
  "--manifest" "$FULL_MANIFEST_PATH"
  "--profile" "$FULL_PROFILE"
  "--strict-hf-cache"
  "--model-load-smoke"
  "--runtime-smoke"
)
if [[ $SKIP_HF_ACCESS -eq 1 ]]; then
  AUDIT_ARGS+=("--skip-hf-access")
fi
"${AUDIT_ARGS[@]}"

"$PYTHON_BIN" -V
command -v g++ >/dev/null 2>&1 || { echo "Missing g++" >&2; exit 1; }
command -v javac >/dev/null 2>&1 || { echo "Missing javac" >&2; exit 1; }
command -v node >/dev/null 2>&1 || { echo "Missing node" >&2; exit 1; }
command -v go >/dev/null 2>&1 || { echo "Missing go" >&2; exit 1; }
command -v nvidia-smi >/dev/null 2>&1 || { echo "Missing nvidia-smi" >&2; exit 1; }
g++ --version | head -n 1
javac -version
node --version
go version
nvidia-smi >/dev/null
df -h "$ROOT"
"$PYTHON_BIN" "$ROOT/scripts/run_full_matrix.py" --manifest "$STAGE_A_MANIFEST_PATH" --profile "$STAGE_A_PROFILE" --output-root "$OUTPUT_PATH" --dry-run
"$PYTHON_BIN" "$ROOT/scripts/run_full_matrix.py" --manifest "$STAGE_B_MANIFEST_PATH" --profile "$STAGE_B_PROFILE" --output-root "$OUTPUT_PATH" --dry-run

echo "Remote preflight passed."
