# Baseline Provenance

This repository treats upstream runtime methods as imported official baselines with explicit per-method provenance. Each runtime method is pinned to its own official upstream repository and then adapted into the project-native benchmark runner without modifying algorithm internals:

- `stone_runtime` -> `third_party/STONE-watermarking.UPSTREAM.json`
- `sweet_runtime` -> `third_party/SWEET-watermark.UPSTREAM.json`
- `ewd_runtime` -> `third_party/EWD.UPSTREAM.json`
- `kgw_runtime` -> `third_party/KGW-lm-watermarking.UPSTREAM.json`

## Baseline Matrix

- `stone_runtime`
- `sweet_runtime`
- `ewd_runtime`
- `kgw_runtime`

## Auxiliary Provenance Helpers

- `bash scripts/fetch_runtime_upstreams.sh all` fetches the pinned official checkouts for `stone_runtime`, `sweet_runtime`, `ewd_runtime`, and `kgw_runtime` and refuses remote-url, dirty-tree, or commit mismatches.
- `python scripts/run_runtime_family.py --family runtime_official ...` is a low-level diagnostic helper for adapter debugging; it is not the primary active workflow.
- `python scripts/evaluate_baseline_family.py --input results/runs/<run_id>` is a low-level analysis helper for imported baseline diagnostics; it is not the primary active workflow.

## Primary Active Workflow

- `make suite-precheck` runs the two-stage suite gate on the current active benchmark.
- `make suite-matrix` runs the active TOSEM-compact full matrix.
- Both workflows use the same canonical configs, provenance rules, logging, and figure pipeline.

## Packaging Rule

- `scripts/package_zenodo.sh` always records a per-method `baseline_provenance.json` entry for `stone`, `sweet`, `ewd`, and `kgw`.
- Only redistributable vendored snapshots are bundled into the public release artifact. Non-redistributable or runtime-only external checkouts stay out of the bundle but still appear in the provenance map with sanitized public paths.
- The bundle never includes `.git`, `paper/`, `proposal.md`, cached artifacts, or generated run outputs.

## Public Release Behavior

- The main CLI and configs stay project-native.
- Baseline source and licensing stay visible in `docs/`, `third_party/`, and the generated provenance manifest.
- Runtime baselines require a local Hugging Face model, a GPU, and a valid upstream checkout whose `origin` remote matches the lock manifest, whose `HEAD` commit matches the pinned SHA, and whose worktree is clean.
- `validate_setup.py` fails if any imported runtime method declares a missing or mismatched official checkout.
- Official-baseline comparisons keep the shared runtime generation policy aligned at the benchmark layer: `max_new_tokens=256`, `do_sample=true`, `temperature=0.2`, `top_p=0.95`, and `no_repeat_ngram_size=4`.
