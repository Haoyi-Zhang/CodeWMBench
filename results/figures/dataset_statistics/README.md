This directory contains repository-tracked dataset inventory and compact-slice figures.

They are generated with:

```bash
python scripts/export_dataset_statistics.py
```

All figures are rendered in a TOSEM-friendly single-column style with a Times New Roman-first font configuration.

Figure scope:

- `compact_slice_composition`: active execution slice
- `inventory_source_counts`: inventory-only
- `language_distribution`: active execution slice
- `task_category_distribution`: crafted-only category coverage
- `crafted_family_coverage`: crafted-only family coverage
- `evaluation_dimensions_overview`: metric-framework overview

Abbreviation notes used by the figures:

- `HE+` = `HumanEval+`
- `HE+ (active)` and `MBPP+ (active)` denote active compact execution counts
- `HEX (py/cpp/java)` = `HumanEval-X (py/cpp/java slice)`
- `MBXP-5lang (py/cpp/java)` = `MBXP-5lang (py/cpp/java slice)`
- `HEX (5-lang inv.)` and `MBXP-5lang (5-lang inv.)` denote inventory-only counts
- `Orig. (active)` / `Trans. (active)` / `Stress (active)` denote the three crafted compact execution slices
- `Orig. (inv.)` / `Trans. (inv.)` / `Stress (inv.)` denote the crafted inventory counts
- `Other` in crafted-only category/family views aggregates the remaining compact crafted groups outside the displayed top buckets
- In `evaluation_dimensions_overview`, `Pass retention` denotes watermarked pass preservation, `1 - neg. FPR` denotes one minus the negative-control false-positive rate, and `Neg. support` denotes negative-control support coverage

See [`docs/datasets.md`](../../../docs/datasets.md) for the corresponding table-level definitions and scope notes.
