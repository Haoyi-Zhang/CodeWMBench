# Archived Configurations

This directory stores historical, diagnostic, or compatibility-only configuration files that are **not** part of the active public workflow.

The active compact workflow uses only:

- `configs/matrices/suite_all_models_methods.json`
- `configs/matrices/suite_canary_heavy.json`
- `configs/matrices/model_invocation_smoke.json`
- the active benchmark source configs and official runtime configs kept directly under `configs/`

Files under `configs/archive/` remain only for fixture coverage or historical reference. They must not be treated as the canonical commands or manifests for the current compact precheck or the current compact full run.
