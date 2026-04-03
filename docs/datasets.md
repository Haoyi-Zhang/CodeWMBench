# Datasets And Compact Slice

`CodeWMBench` combines public executable benchmarks and crafted multilingual benchmark families under one normalized schema.

The repository distinguishes three layers:

- `data/public/`: pinned public benchmark snapshots
- `data/compact/crafted/`: crafted benchmark snapshots used to build the active public release
- `data/compact/collections/`: the current `TOSEM-compact` execution slice used by the active suite manifests

## Inventory vs Active Execution Slice

The documented suite inventory contains:

- `HumanEval`
- `HumanEval+`
- `MBPP+`
- `HumanEval-X`
- `MBXP-5lang`
- `crafted_original`
- `crafted_translation`
- `crafted_stress`

The active aggregate suite score uses the following seven atomic source groups:

- `HumanEval+`
- `MBPP+`
- `HumanEval-X (py/cpp/java slice)`
- `MBXP-5lang (py/cpp/java slice)`
- `Crafted Original`
- `Crafted Translation`
- `Crafted Stress`

`HumanEval` remains part of the documented inventory but is excluded from aggregate scoring to avoid double-counting against `HumanEval+`.

One sentence should be read as normative across the repository:

> `HumanEval-X` and `MBXP-5lang` are five-language inventory datasets, but the active official-runtime comparison scores only the `python/cpp/java` common-support execution slice.

The same interpretation applies to the active compact crafted sources: the documented crafted inventory remains five-language overall, while the active official-runtime slice executes on `python/cpp/java`.

## Canonical Benchmark Definition Table

The machine-readable benchmark-definition table is exported to:

- [`results/tables/dataset_statistics/benchmark_definition_summary.csv`](../results/tables/dataset_statistics/benchmark_definition_summary.csv)
- [`results/tables/dataset_statistics/benchmark_definition_summary.json`](../results/tables/dataset_statistics/benchmark_definition_summary.json)

For quick review, the active public release can be summarized as:

| Source | Inventory size | Active compact size | Scored in aggregate | Execution slice | Languages | Sampling rule | Type |
| --- | ---: | ---: | --- | --- | --- | --- | --- |
| `HumanEval` | `164` | `0` | `No` | inventory only | `python` | inventory only; excluded from aggregate scoring | public |
| `HumanEval+` | `164` | `164` | `Yes` | `python` | `python` | full retained | public |
| `MBPP+` | `378` | `240` | `Yes` | `python` | `python` | deterministic compact sample | public |
| `HumanEval-X (5-lang inv.; py/cpp/java exec.)` | `820` | `120` | `Yes` | `python/cpp/java` | `5 inv. / 3 exec.` | deterministic common-support slice | public |
| `MBXP-5lang (5-lang inv.; py/cpp/java exec.)` | `4693` | `120` | `Yes` | `python/cpp/java` | `5 inv. / 3 exec.` | deterministic common-support slice | public |
| `Crafted Original` | `1500` | `120` | `Yes` | `python/cpp/java` | `5 inv. / 3 exec.` | deterministic family/category-balanced compact sample | crafted |
| `Crafted Translation` | `1000` | `120` | `Yes` | `python/cpp/java` | `5 inv. / 3 exec.` | deterministic family/category-balanced compact sample | crafted |
| `Crafted Stress` | `750` | `120` | `Yes` | `python/cpp/java` | `5 inv. / 3 exec.` | deterministic family/category-balanced compact sample | crafted |

`compact_slice_composition` is the README-facing figure because it shows what the active release actually executes. `inventory_source_counts` is intentionally an inventory-only figure and should not be read as the scored runtime slice. Whenever a figure uses shortened multilingual labels, they always denote the active `HumanEval-X (py/cpp/java slice)` and `MBXP-5lang (py/cpp/java slice)` execution slice rather than the five-language inventories.

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

## Statistics Files

Dataset statistics are exported under [`results/tables/dataset_statistics`](../results/tables/dataset_statistics) and [`results/figures/dataset_statistics`](../results/figures/dataset_statistics) using:

```bash
python scripts/export_dataset_statistics.py
```

Use the outputs with the following scope:

- `compact_slice_composition`: README-facing figure for the active execution slice
- `inventory_source_counts`: inventory-only figure for the full documented source inventory
- `language_distribution`: compact execution-slice language coverage (`python/cpp/java`)
- `task_category_distribution` and `dataset_task_category_breakdown`: **crafted-only** category coverage views. Here, `Other` aggregates remaining compact crafted categories outside the displayed top groups.
- `crafted_family_coverage` and `dataset_family_breakdown`: **crafted-only** family coverage views. Here, `Other` aggregates remaining compact crafted template families outside the displayed top groups.

The exported machine-readable tables include:

- `dataset_inventory_summary.{csv,json}`
- `compact_slice_summary.{csv,json}`
- `benchmark_definition_summary.{csv,json}`
- `dataset_language_breakdown.{csv,json}`
- `dataset_task_category_breakdown.{csv,json}`
- `dataset_family_breakdown.{csv,json}`
