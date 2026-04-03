# Local Model Matrix

The active `TOSEM-compact` suite compares the same four imported official watermark baselines on each of the following local backbones:

- `Qwen/Qwen2.5-Coder-14B-Instruct`
- `Qwen/Qwen2.5-Coder-7B-Instruct`
- `bigcode/starcoder2-7b`
- `deepseek-ai/deepseek-coder-6.7b-instruct`

This gives the compact suite three model families plus one within-family scale comparison for Qwen.

## Required Environment Variables

- `HF_ACCESS_TOKEN`
- `HF_ACCESS_TOKEN_FALLBACK`

## Active Baseline Roster

- `stone_runtime`
- `sweet_runtime`
- `ewd_runtime`
- `kgw_runtime`

Within each matched model slice, these four official baselines are compared on the same benchmark rows. Cross-model execution may run in parallel, but ranking and score aggregation remain aligned to matched `model x method x source` slices.

For multilingual suite sources, the active official-runtime execution path uses the common supported slice across the four imported methods: `python`, `cpp`, and `java`.

## Active Suite Manifests

- `configs/matrices/suite_all_models_methods.json`
- `configs/matrices/suite_canary_heavy.json`
- `configs/matrices/model_invocation_smoke.json`

`suite_canary_heavy` is the heavy representative-model precheck. `model_invocation_smoke` verifies that the remaining three backbones can complete the same imported-baseline call path on the short Python smoke slice.

## Runtime Expectations

The imported official baselines load local Hugging Face weights from the local cache. The active public workflow is local-model-only; API-backed execution is not part of the compact suite.

For a 40 GB GPU card, the safe defaults remain:

- `device = cuda`
- `dtype = float16`
- `max_new_tokens = 256`

Before starting a large run, validate model access with:

```bash
python scripts/check_model_access.py --require-all
```
