# Baseline Provenance

This repository treats the runtime methods used in the release as pinned upstream baseline implementations with explicit per-method provenance. Each runtime method is pinned to its upstream repository, then executed through project-native adapters that keep the watermark algorithm logic intact while routing orchestration, local model loading, and decoding policy through the shared benchmark layer:

- `stone_runtime` -> `third_party/STONE-watermarking.UPSTREAM.json`
- `sweet_runtime` -> `third_party/SWEET-watermark.UPSTREAM.json`
- `ewd_runtime` -> `third_party/EWD.UPSTREAM.json`
- `kgw_runtime` -> `third_party/KGW-lm-watermarking.UPSTREAM.json`

## Baseline Matrix

| Method | Upstream repository | Pinned commit | Source subpath | License status |
| --- | --- | --- | --- | --- |
| `stone_runtime` | `https://github.com/inistory/STONE-watermarking.git` | `bb5d809c0c494a219411e861f2313cca2b9fd7b4` | `stone_implementation` | `unverified` |
| `sweet_runtime` | `https://github.com/hongcheki/sweet-watermark.git` | `853b47eb064c180beebd383302d09491fc98a565` | `.` | `unverified` |
| `ewd_runtime` | `https://github.com/luyijian3/EWD.git` | `605756acf802528a3df89d95a4661a031eafc79b` | `.` | `unverified` |
| `kgw_runtime` | `https://github.com/jwkirchenbauer/lm-watermarking.git` | `82922516930c02f8aa322765defdb5863d07a00e` | `.` | `redistributable` |

## Auxiliary Provenance Helpers

- `bash scripts/fetch_runtime_upstreams.sh all` fetches the pinned upstream checkouts for `stone_runtime`, `sweet_runtime`, `ewd_runtime`, and `kgw_runtime` and refuses remote-url, dirty-tree, or commit mismatches.
- `python scripts/run_runtime_family.py --family runtime_official ...` is a low-level diagnostic helper for adapter debugging; it is not the primary active workflow.
- `python scripts/evaluate_baseline_family.py --input results/runs/<run_id>` is a low-level analysis helper for imported baseline diagnostics; it is not the primary active workflow.
- `stone_runtime` uses the upstream `stone_implementation` subpath recorded in `third_party/STONE-watermarking.UPSTREAM.json`.
- `kgw_runtime` is pinned to the upstream repository named `lm-watermarking`, recorded in `third_party/KGW-lm-watermarking.UPSTREAM.json`.

## Primary Active Workflow

- `make suite-precheck` runs the two-stage suite gate on the current active benchmark.
- `make suite-matrix` runs the active TOSEM-compact full matrix.
- Both workflows use the same canonical configs, provenance rules, logging, and figure pipeline.

## Packaging Rule

- `scripts/package_zenodo.sh` always records a per-method `baseline_provenance.json` entry for `stone`, `sweet`, `ewd`, and `kgw`.
- Only redistributable vendored snapshots are bundled into the public release artifact. Non-redistributable or runtime-only external checkouts stay out of the bundle but still appear in the provenance map with sanitized public paths.
- The bundle never includes `.git`, `paper/`, `proposal.md`, cached artifacts, or generated run outputs.

## Reviewer-Facing Guarantee

For the active public release:

- the four baseline implementations are pinned to explicit upstream commits with recorded provenance
- the project adapts orchestration, local model loading, and decoding interfaces around them
- the repository does **not** edit the baseline watermark algorithm logic itself
- runtime baseline comparisons use the same benchmark-layer generation policy and the same compact source definitions across methods
- redistribution and license status are tracked per upstream manifest rather than assumed to be uniform across all four baselines
- upstream provenance does not imply uniform redistribution permission across all four methods

## Public Release Behavior

- The main CLI and configs stay project-native.
- Baseline source and licensing stay visible in `docs/`, `third_party/`, and the generated provenance manifest.
- Runtime baselines require a local Hugging Face model, a GPU, and a valid upstream checkout whose `origin` remote matches the lock manifest, whose `HEAD` commit matches the pinned SHA, and whose worktree is clean.
- `validate_setup.py` fails if any imported runtime method declares a missing or mismatched pinned upstream checkout.
- Runtime baseline comparisons keep the shared runtime generation policy aligned at the benchmark layer: `max_new_tokens=256`, `do_sample=true`, `temperature=0.2`, `top_p=0.95`, and `no_repeat_ngram_size=4`.
