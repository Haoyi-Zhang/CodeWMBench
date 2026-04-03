from __future__ import annotations

from scripts.run_runtime_family import _selected_watermarks


def test_runtime_family_runner_defaults_to_runtime_official():
    assert _selected_watermarks(None, "runtime_official") == (
        "stone_runtime",
        "sweet_runtime",
        "ewd_runtime",
        "kgw_runtime",
    )


def test_runtime_family_runner_preserves_stone_family_alias():
    assert _selected_watermarks(None, "stone_family") == (
        "stone_runtime",
        "sweet_runtime",
        "ewd_runtime",
        "kgw_runtime",
    )

def test_runtime_family_runner_preserves_explicit_selection():
    assert _selected_watermarks(["STONE_RUNTIME", "kgw_runtime"], "runtime_official") == (
        "stone_runtime",
        "kgw_runtime",
    )
