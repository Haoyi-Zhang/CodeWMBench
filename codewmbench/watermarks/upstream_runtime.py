from __future__ import annotations

from pathlib import Path
from typing import Any

from ..baselines.stone_family import (
    build_stone_family_bundle,
    is_stone_family_baseline,
    runtime_watermark_names as _runtime_watermark_names,
    stone_family_checkout_available as _checkout_available,
    stone_family_checkout_metadata as _checkout_metadata,
)
from ..baselines.stone_family.common import resolve_checkout
from ..baselines.stone_family.runtime import _backend_for as _stone_backend_for
from ..baselines.stone_family.runtime import _load_backend as _stone_load_backend


def runtime_watermark_names() -> tuple[str, ...]:
    return _runtime_watermark_names()


def is_runtime_watermark(name: str) -> bool:
    return is_stone_family_baseline(name)


def resolve_upstream_root(name: str = "stone_runtime") -> Path | None:
    checkout = resolve_checkout(name)
    return checkout.source_root if checkout is not None else None


def upstream_checkout_available(name: str = "stone_runtime") -> bool:
    return _checkout_available(name)


def upstream_checkout_metadata(name: str = "stone_runtime") -> dict[str, Any]:
    return _checkout_metadata(name)


def build_upstream_runtime_bundle(name: str):
    return build_stone_family_bundle(name)


def _backend_for(method, example, spec):
    return _stone_backend_for(method, example, spec)


def _load_backend(*args, **kwargs):
    return _stone_load_backend(*args, **kwargs)
