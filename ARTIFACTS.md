# Artifacts

Large raw full-run artifacts are not stored in this GitHub repository.

See [docs/artifacts.md](docs/artifacts.md) for:

- what stays in git versus what is distributed externally
- the recommended Zenodo layout for raw results
- how to restore raw results locally
- how to regenerate summary figures and tables from the released artifacts

Release templates live under [artifacts/](artifacts):

- [artifacts/raw_results_manifest.template.json](artifacts/raw_results_manifest.template.json)
- [artifacts/SHA256SUMS.template.txt](artifacts/SHA256SUMS.template.txt)

These template files intentionally retain `TBD` placeholders until the archival artifact is published. The public GitHub repository should not claim a finished raw-results download URL before that metadata exists.

Until that archival artifact exists, this repository should be read as:

- a complete public code and compact-data release
- a repository-tracked summary and dataset-statistics release
- a documented raw-artifact regeneration workflow

It should not be read as a self-contained host of the raw full-run result tree.
