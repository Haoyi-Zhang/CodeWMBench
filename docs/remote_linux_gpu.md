# Linux GPU Workflow

This document describes the Linux GPU rerun path for the active `TOSEM-compact` workflow. The compact full matrix uses four local backbones and four imported official baselines. The recommended control flow is:

1. bootstrap the host
2. clean active outputs
3. run preflight
4. run the two-stage compact precheck
5. inspect the gate result
6. run the compact full matrix explicitly

The precheck and the full matrix are intentionally documented as separate steps.

## Environment

- `REMOTE_HOST`
- `REMOTE_PORT`
- `REMOTE_USER`
- `REMOTE_ROOT`
- `HF_ACCESS_TOKEN`
- `PYTHON_BIN`
- `VENV_DIR`
- `CUDA_VISIBLE_DEVICES`

## Scripts

- `scripts/remote/bootstrap_linux_gpu.sh`: prepares directories and optionally creates a virtual environment
- `scripts/remote/run_preflight.sh`: rebuilds manifests, audits benchmark inputs, checks toolchains, and dry-runs the active precheck manifests
- `scripts/remote/run_suite_matrix.sh`: cleans active outputs, runs preflight, and runs the compact precheck by default; pass `--run-full` to continue into the full matrix. Use `--no-clean` only if you intentionally want to preserve an existing result tree.
- `scripts/remote/upload_bundle.sh`: uploads the release bundle to the remote host
- `scripts/remote/fetch_results.sh`: mirrors remote result directories back to the local machine

## Toolchains

Recommended reference images:

- Python: `python:3.11`
- C++: `gcc:13`
- Java: `openjdk:21`
- JavaScript: `node:20`
- Go: `golang:1.22`

Minimum supported host versions for local validation:

- Python: `3.10+`
- `g++`: `11+`
- `javac` / `java`: `17+`
- `node`: `12.22+`
- `go`: `1.18+`

The active multilingual official-runtime slice is `python`, `cpp`, and `java`.

## Suggested Sequence

1. Run `bash scripts/remote/bootstrap_linux_gpu.sh --install --venv <path>`
2. Run `python scripts/clean_suite_outputs.py --include-full-matrix --include-release-bundle`
3. Run `bash scripts/fetch_runtime_upstreams.sh all`
4. Run `bash scripts/remote/run_preflight.sh`
5. Run `python scripts/certify_suite_precheck.py ...` or `make suite-precheck`
6. Inspect the gate JSON and live logs
7. Run `bash scripts/remote/run_suite_matrix.sh --run-full` only after the compact precheck passes
8. Fetch result directories or raw artifacts back to the local machine

For hosts that do not listen on port `22`, export `REMOTE_PORT` before using the upload or fetch helpers.
