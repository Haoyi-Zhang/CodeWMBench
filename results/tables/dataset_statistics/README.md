This directory contains repository-tracked dataset inventory and compact-slice tables.

They are generated with:

```bash
python scripts/export_dataset_statistics.py
```

The JSON and CSV files summarize:

- the full documented inventory
- the active TOSEM-compact execution slice
- the canonical benchmark-definition table
- crafted-only category and family coverage views

The canonical benchmark-definition files are:

- `benchmark_definition_summary.csv`
- `benchmark_definition_summary.json`

They are the primary disambiguation table for multilingual sources: `HumanEval-X` and `MBXP-5lang` remain five-language inventory datasets in the repository, but the active official-runtime comparison scores only the `python/cpp/java` common-support execution slice.
