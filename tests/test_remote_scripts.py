from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_remote_preflight_dry_run_declares_prepare_step():
    script = (ROOT / "scripts" / "remote" / "run_preflight.sh").read_text(encoding="utf-8")
    assert '"build_suite_manifests"' in script
    assert '"audit_benchmarks"' in script
    assert '"audit_suite_matrix"' in script
    assert 'FULL_MANIFEST="configs/matrices/suite_all_models_methods.json"' in script
    assert 'STAGE_A_MANIFEST="configs/matrices/suite_canary_heavy.json"' in script
    assert 'STAGE_B_MANIFEST="configs/matrices/model_invocation_smoke.json"' in script
    assert 'VENV_DIR="${VENV_DIR:-$ROOT/.venv/tosem_compact}"' in script
    assert '--venv PATH' in script


def test_remote_preflight_audits_suite_before_matrix_dry_runs():
    script = (ROOT / "scripts" / "remote" / "run_preflight.sh").read_text(encoding="utf-8")
    assert "build_suite_manifests.py" in script
    assert "audit_benchmarks.py" in script
    assert "audit_full_matrix.py" in script
    assert script.index("build_suite_manifests.py") < script.index("audit_full_matrix.py")
    assert '--manifest "$FULL_MANIFEST_PATH"' in script
    assert '--matrix-profile "$FULL_PROFILE"' in script
    assert '--strict-hf-cache' in script
    assert '--model-load-smoke' in script
    assert '--runtime-smoke' in script
    assert 'PYTHON_BIN="$VENV_DIR/bin/python"' in script
    assert 'Create the venv first with:' in script


def test_remote_transfer_scripts_support_remote_port():
    upload = (ROOT / "scripts" / "remote" / "upload_bundle.sh").read_text(encoding="utf-8")
    fetch = (ROOT / "scripts" / "remote" / "fetch_results.sh").read_text(encoding="utf-8")
    assert 'REMOTE_PORT="${REMOTE_PORT:-22}"' in upload
    assert '--remote-port PORT' in upload
    assert 'rsync -av -e "ssh -p $REMOTE_PORT"' in upload
    assert 'scp -P "$REMOTE_PORT"' in upload
    assert 'REMOTE_PORT="${REMOTE_PORT:-22}"' in fetch
    assert '--remote-port PORT' in fetch
    assert 'rsync -av -e "ssh -p $REMOTE_PORT"' in fetch
    assert 'scp -P "$REMOTE_PORT"' in fetch
    assert 'RUN_DIR="${RUN_DIR:-results}"' in fetch


def test_remote_bootstrap_defaults_to_tosem_compact_venv():
    script = (ROOT / "scripts" / "remote" / "bootstrap_linux_gpu.sh").read_text(encoding="utf-8")
    assert 'VENV_DIR="${VENV_DIR:-$ROOT/.venv/tosem_compact}"' in script


def test_remote_suite_matrix_uses_bootstrapped_venv_python():
    script = (ROOT / "scripts" / "remote" / "run_suite_matrix.sh").read_text(encoding="utf-8")
    assert 'MANIFEST="configs/matrices/suite_all_models_methods.json"' in script
    assert 'OUTPUT_ROOT="$ROOT/results/matrix"' in script
    assert 'VENV_DIR="${VENV_DIR:-$ROOT/.venv/tosem_compact}"' in script
    assert '--venv PATH' in script
    assert '--bootstrap' in script
    assert 'BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-python3}"' in script
    assert 'bash "$ROOT/scripts/remote/run_preflight.sh"' in script
    assert 'scripts/certify_suite_precheck.py' in script
    assert 'scripts/run_full_matrix.py' in script
    assert 'RUN_FULL=0' in script
    assert '--run-full' in script
    assert 'Suite precheck complete. Pass --run-full to start the compact full matrix.' in script
    assert '--fail-fast' in script
    assert 'if [[ ! -x "$PYTHON_BIN" ]]; then' in script
    assert 'Create the venv first with:' in script

