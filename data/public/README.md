# Public Benchmark Snapshots

This directory stores pinned, normalized snapshots of the public benchmarks used by `CodeWMBench`.

Each snapshot is produced by `scripts/fetch_public_benchmarks.py` or `scripts/prepare_data.py --public-source ...` and includes a companion `.manifest.json` file with:

- source URL
- pinned upstream revision
- archive or source-tree hash
- split
- license note
- task count
- normalization adapter name
- sample-id sidecar path
- extraction rules for repo-backed sources

Registered public sources:

- `human_eval`
- `humaneval_plus`
- `mbpp_plus`
- `humaneval_x`
- `mbxp_5lang`
- `class_eval`

`class_eval` remains an optional normalization path only. It is excluded from the active compact suite because its licensing and evaluation conditions differ from the public executable sources used in the current benchmark slice.

The `_cache/` directory is a local fetch cache and is not part of the public repository interface. Public manifests and normalized rows should only expose scrubbed release-facing path labels, not machine-local `_cache` paths.
