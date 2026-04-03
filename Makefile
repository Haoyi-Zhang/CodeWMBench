.RECIPEPREFIX = >
PYTHON ?= $(shell if command -v python3 >/dev/null 2>&1; then echo python3; elif command -v python >/dev/null 2>&1; then echo python; else echo python3; fi)
CONFIG ?= configs/debug.yaml
RELEASE_BUNDLE_DIR ?= results/release_bundle
SUITE_MATRIX_MANIFEST ?= configs/matrices/suite_all_models_methods.json
SUITE_MATRIX_PROFILE ?= suite_all_models_methods
PRECHECK_STAGE_A_MANIFEST ?= configs/matrices/suite_canary_heavy.json
PRECHECK_STAGE_A_PROFILE ?= suite_canary_heavy
PRECHECK_STAGE_B_MANIFEST ?= configs/matrices/model_invocation_smoke.json
PRECHECK_STAGE_B_PROFILE ?= model_invocation_smoke
PRECHECK_GATE_OUTPUT ?= results/certifications/suite_precheck_gate.json
SUITE_GPU_SLOTS ?= 8
SUITE_CPU_WORKERS ?= 12
SUITE_MONITOR_INDEX ?= results/matrix/$(SUITE_MATRIX_PROFILE)/matrix_index.json

.PHONY: help prepare-fixture prepare-data validate validate-anonymity package-release install-tools suite-clean suite-matrix suite-matrix-dry-run suite-precheck suite-precheck-dry-run suite-monitor export-summaries dataset-stats

help:
> @echo "Targets:"
> @echo "  prepare-fixture Normalize the synthetic debug fixture"
> @echo "  validate        Check public release setup and runtime prerequisites"
> @echo "  validate-anonymity Check anonymous-release hygiene for staged bundles"
> @echo "  package-release Stage a sanitized release bundle"
> @echo "  suite-clean     Delete active suite outputs before a clean rerun"
> @echo "  suite-matrix    Run the active TOSEM-compact full matrix"
> @echo "  suite-matrix-dry-run Preview the active TOSEM-compact full matrix"
> @echo "  suite-precheck  Run the two-stage suite precheck"
> @echo "  suite-precheck-dry-run Preview both precheck manifests"
> @echo "  suite-monitor   Watch the active suite matrix index and GPU state"
> @echo "  export-summaries Refresh report metadata and export final figures/tables from finished full-run results"
> @echo "  dataset-stats   Export repository-tracked dataset statistics figures/tables"
> @echo "  install-tools   Create local directories and verify prerequisites"

prepare-fixture:
> $(PYTHON) scripts/prepare_data.py --config $(CONFIG)

prepare-data: prepare-fixture

validate:
> $(PYTHON) scripts/validate_setup.py --config $(CONFIG)

validate-anonymity:
> $(PYTHON) scripts/validate_setup.py --config $(CONFIG) --check-anonymity

package-release:
> bash scripts/package_zenodo.sh
> $(PYTHON) scripts/validate_release_bundle.py --bundle $(RELEASE_BUNDLE_DIR)

suite-clean:
> $(PYTHON) scripts/clean_suite_outputs.py --include-full-matrix --include-release-bundle

suite-matrix:
> $(PYTHON) scripts/run_full_matrix.py --manifest $(SUITE_MATRIX_MANIFEST) --profile $(SUITE_MATRIX_PROFILE) --gpu-slots $(SUITE_GPU_SLOTS) --gpu-pool-mode shared --cpu-workers $(SUITE_CPU_WORKERS) --fail-fast

suite-matrix-dry-run:
> $(PYTHON) scripts/run_full_matrix.py --manifest $(SUITE_MATRIX_MANIFEST) --profile $(SUITE_MATRIX_PROFILE) --gpu-slots $(SUITE_GPU_SLOTS) --gpu-pool-mode shared --cpu-workers $(SUITE_CPU_WORKERS) --dry-run

suite-precheck:
> $(PYTHON) scripts/certify_suite_precheck.py --full-manifest $(SUITE_MATRIX_MANIFEST) --full-profile $(SUITE_MATRIX_PROFILE) --stage-a-manifest $(PRECHECK_STAGE_A_MANIFEST) --stage-a-profile $(PRECHECK_STAGE_A_PROFILE) --stage-b-manifest $(PRECHECK_STAGE_B_MANIFEST) --stage-b-profile $(PRECHECK_STAGE_B_PROFILE) --output $(PRECHECK_GATE_OUTPUT) --gpu-slots $(SUITE_GPU_SLOTS) --cpu-workers $(SUITE_CPU_WORKERS) --fail-fast

suite-precheck-dry-run:
> $(PYTHON) scripts/run_full_matrix.py --manifest $(PRECHECK_STAGE_A_MANIFEST) --profile $(PRECHECK_STAGE_A_PROFILE) --gpu-slots $(SUITE_GPU_SLOTS) --gpu-pool-mode shared --cpu-workers $(SUITE_CPU_WORKERS) --dry-run
> $(PYTHON) scripts/run_full_matrix.py --manifest $(PRECHECK_STAGE_B_MANIFEST) --profile $(PRECHECK_STAGE_B_PROFILE) --gpu-slots $(SUITE_GPU_SLOTS) --gpu-pool-mode shared --cpu-workers $(SUITE_CPU_WORKERS) --dry-run

suite-monitor:
> $(PYTHON) scripts/monitor_matrix.py --matrix-index $(SUITE_MONITOR_INDEX) --watch-seconds 5

export-summaries:
> $(PYTHON) scripts/refresh_report_metadata.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json
> $(PYTHON) scripts/render_paper_figures.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --suite all --paper-track generation_time --require-times-new-roman --output-dir results/figures/suite_all_models_methods
> $(PYTHON) scripts/export_full_run_tables.py --matrix-index results/matrix/suite_all_models_methods/matrix_index.json --output-dir results/tables/suite_all_models_methods

dataset-stats:
> $(PYTHON) scripts/export_dataset_statistics.py

install-tools:
> bash scripts/install_tools.sh
