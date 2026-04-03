# Third-Party Baselines

This directory records the pinned provenance for imported baseline implementations.

- `STONE-watermarking.UPSTREAM.json` captures the pinned upstream URL and commit for `stone_runtime`.
- `SWEET-watermark.UPSTREAM.json` captures the pinned upstream URL and commit for `sweet_runtime`.
- `EWD.UPSTREAM.json` captures the pinned upstream URL and commit for `ewd_runtime`.
- `KGW-lm-watermarking.UPSTREAM.json` captures the pinned upstream URL and commit for `kgw_runtime`.
- Imported official checkouts stay outside the anonymous bundle unless a redistributable vendored snapshot is intentionally staged.
- The submission path relies on the pinned provenance manifests plus fetch scripts instead of shipping the external checkout contents directly.
