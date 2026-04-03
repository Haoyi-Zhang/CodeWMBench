# Datasets And Compact Slice

`CodeWMBench` combines public executable benchmarks and crafted multilingual benchmark families under one normalized schema.

The repository distinguishes three layers:

- `data/public/`: pinned public benchmark snapshots
- `data/compact/crafted/`: crafted benchmark snapshots used by the active public release
- `data/compact/collections/`: the current `TOSEM-compact` evaluation slice used by the active suite manifests

## Inventory

The documented suite inventory contains:

- `HumanEval`
- `HumanEval+`
- `MBPP+`
- `HumanEval-X`
- `MBXP-5lang`
- `crafted_original`
- `crafted_translation`
- `crafted_stress`

The current aggregate suite score uses the following seven atomic source groups:

- `HumanEval+`
- `MBPP+`
- `HumanEval-X (py/cpp/java slice)`
- `MBXP-5lang (py/cpp/java slice)`
- `Crafted Original`
- `Crafted Translation`
- `Crafted Stress`

`HumanEval` remains part of the documented inventory but is excluded from aggregate scoring to avoid double-counting against `HumanEval+`.

## Multilingual Slice Policy

The public inventory remains five-language overall. The current official-runtime comparison slice is the common executable support set shared across the four imported baselines:

- `python`
- `cpp`
- `java`

This affects the current compact execution slice for:

- `HumanEval-X`
- `MBXP-5lang`
- `crafted_original`
- `crafted_translation`
- `crafted_stress`

Whenever those datasets appear in active result figures or tables, they should be interpreted as the current `python/cpp/java` comparison slice.

## Crafted Benchmarks

The three crafted benchmark families serve distinct roles:

- `crafted_original`: native multilingual benchmark tasks
- `crafted_translation`: cross-language translation-oriented tasks
- `crafted_stress`: harder or more adversarial task structures intended to stress watermark robustness and utility

The crafted sources are **expert-informed manually curated** and then audited through deterministic build and manifest checks. In this repository, that means:

- manual task-family design and prompt/contract authoring
- manual screening and normalization of benchmark records
- automated manifest, parity, and validation audits over the finalized records

This wording is intentionally factual. The repository does not claim unverifiable multi-expert authorship or review counts.

## Compact Slice

The active `TOSEM-compact` release keeps `HumanEval+` complete and applies deterministic compact slices elsewhere:

- `HumanEval+`: `164`
- `MBPP+`: `240`
- `HumanEval-X (py/cpp/java slice)`: `120`
- `MBXP-5lang (py/cpp/java slice)`: `120`
- `Crafted Original`: `120`
- `Crafted Translation`: `120`
- `Crafted Stress`: `120`

The compact multilingual slices are deterministic and family/category balanced instead of naive head truncation.

## Statistics Files

Dataset statistics are exported under [`results/tables/dataset_statistics`](../results/tables/dataset_statistics) and [`results/figures/dataset_statistics`](../results/figures/dataset_statistics) using:

```bash
python scripts/export_dataset_statistics.py
```

These outputs summarize:

- full inventory counts
- compact slice counts
- language distribution
- task-category distribution
- crafted family coverage
