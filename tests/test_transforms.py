from __future__ import annotations

from codewmbench.transforms.registry import build_transform_bundle
from codewmbench.utils import normalize_whitespace, strip_comments


def test_strip_comments_transform():
    source = """
# headline
def demo(value):
    return value  # trailing
""".strip()
    transform = build_transform_bundle("strip_comments")
    mutated = transform.apply(source)

    assert "#" not in mutated
    assert "headline" not in mutated
    assert mutated == normalize_whitespace(strip_comments(source))


def test_canonicalize_text_transform():
    source = """
// heading
def Demo(Value):
    return Value
""".strip()
    transform = build_transform_bundle("canonicalize_text")
    mutated = transform.apply(source)

    assert mutated == "def demo(value):\n    return value"
