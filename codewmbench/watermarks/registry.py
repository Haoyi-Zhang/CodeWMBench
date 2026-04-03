from __future__ import annotations

from .base import WatermarkBundle
from .kgw import build_kgw_bundle
from .method_stubs import build_comment_bundle, build_identifier_bundle
from .structural import build_structural_flow_bundle
from .upstream_runtime import build_upstream_runtime_bundle


def available_watermarks() -> tuple[str, ...]:
    return (
        "kgw",
        "comment",
        "identifier",
        "structural_flow",
        "stone_runtime",
        "sweet_runtime",
        "ewd_runtime",
        "kgw_runtime",
    )


def watermark_origin(name: str) -> str:
    normalized = str(name).strip().lower()
    if normalized.endswith("_runtime"):
        return "upstream"
    return "native"


def build_watermark_bundle(name: str) -> WatermarkBundle:
    name = name.lower()
    if name == "kgw":
        return build_kgw_bundle()
    if name == "comment":
        return build_comment_bundle()
    if name == "identifier":
        return build_identifier_bundle()
    if name == "structural_flow":
        return build_structural_flow_bundle()
    if name in {"stone_runtime", "sweet_runtime", "ewd_runtime", "kgw_runtime"}:
        return build_upstream_runtime_bundle(name)
    raise KeyError(f"unknown watermark scheme: {name}")
