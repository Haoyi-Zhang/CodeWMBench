# Third-Party Baselines

This directory records the exact upstream provenance for the baseline implementations used in this release: repository URL, pinned commit, source subpath, and public-facing checkout path.

- `STONE-watermarking.UPSTREAM.json` captures the pinned upstream URL and commit for `stone_runtime`.
- `SWEET-watermark.UPSTREAM.json` captures the pinned upstream URL and commit for `sweet_runtime`.
- `EWD.UPSTREAM.json` captures the pinned upstream URL and commit for `ewd_runtime`.
- `KGW-lm-watermarking.UPSTREAM.json` captures the pinned upstream URL and commit for `kgw_runtime`.
- Imported pinned upstream checkouts stay outside the public release artifact unless a redistributable vendored snapshot is intentionally staged.
- The public release path relies on the pinned provenance manifests plus fetch scripts instead of shipping the external checkout contents directly.
