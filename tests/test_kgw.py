from __future__ import annotations

from codewmbench.attacks.registry import build_attack_bundle
from codewmbench.models import BenchmarkExample, WatermarkSpec
from codewmbench.watermarks.registry import build_watermark_bundle


def test_kgw_embed_and_detect(sample_example, sample_spec):
    bundle = build_watermark_bundle("kgw")
    watermarked = bundle.embed(sample_example, sample_spec)

    clean_result = bundle.detect(watermarked, sample_spec, example_id=sample_example.example_id)

    comment_strip = build_attack_bundle("comment_strip")
    stripped = comment_strip.apply(watermarked.source, seed=11)
    stripped_result = bundle.detect(stripped.source, sample_spec, example_id=sample_example.example_id)

    identifier_rename = build_attack_bundle("identifier_rename")
    renamed = identifier_rename.apply(watermarked.source, seed=11)
    renamed_result = bundle.detect(renamed.source, sample_spec, example_id=sample_example.example_id)

    assert clean_result.detected is True
    assert clean_result.score >= sample_spec.parameters["threshold"]
    assert clean_result.metadata["channels"] == ["comment", "identifier"]
    assert stripped_result.detected is True
    assert stripped_result.metadata["channels"] == ["identifier"]
    assert renamed_result.score <= clean_result.score
    assert watermarked.metadata["marker"]


def _example(example_id: str, language: str, source: str) -> BenchmarkExample:
    return BenchmarkExample(
        example_id=example_id,
        language=language,
        prompt="watermark test",
        reference_solution=source,
        reference_tests=(),
        execution_tests=(),
    )


def _spec() -> WatermarkSpec:
    return WatermarkSpec(
        name="kgw",
        secret="anonymous",
        payload="wm",
        strength=1.0,
        parameters={"threshold": 0.5},
    )


def test_kgw_injects_language_specific_identifiers_and_preserves_legal_positions():
    bundle = build_watermark_bundle("kgw")
    spec = _spec()
    cases = [
        (
            "python",
            _example(
                "py-1",
                "python",
                "#!/usr/bin/env python3\n\ndef factorial(n):\n    return n\n",
            ),
            "_py",
            "# wm:",
            1,
        ),
        (
            "cpp",
            _example(
                "cpp-1",
                "cpp",
                "#include <iostream>\n\nint main() { return 0; }\n",
            ),
            "_cpp",
            "// wm:",
            0,
        ),
        (
            "java",
            _example(
                "java-1",
                "java",
                "package demo;\n\npublic class Demo {\n    public int value() { return 1; }\n}\n",
            ),
            "_java",
            "// wm:",
            0,
        ),
        (
            "javascript",
            _example(
                "js-1",
                "javascript",
                "#!/usr/bin/env node\n'use strict';\nfunction sum(values) {\n  return values.length;\n}\n",
            ),
            "_js",
            "// wm:",
            2,
        ),
        (
            "go",
            _example(
                "go-1",
                "go",
                "package main\n\nfunc main() {}\n",
            ),
            "_go",
            "// wm:",
            0,
        ),
    ]

    for language, example, suffix, comment_marker, comment_index in cases:
        watermarked = bundle.embed(example, spec)
        identifier = watermarked.metadata["identifier"]
        assert identifier.endswith(suffix)
        assert watermarked.metadata["channels"] == ["comment", "identifier"]
        assert identifier in watermarked.source
        detected = bundle.detect(watermarked, spec, example_id=example.example_id)
        assert detected.metadata["channels"] == ["comment", "identifier"]

        lines = watermarked.source.splitlines()
        if language in {"python", "javascript"}:
            assert lines[0].startswith("#!")
            if language == "javascript":
                assert lines[1] == "'use strict';"
            assert lines[comment_index].startswith(comment_marker)
        elif language == "java":
            assert lines[0].startswith(comment_marker)
            assert watermarked.source.index(identifier) > watermarked.source.index("{")
            assert identifier in watermarked.source
        else:
            assert lines[0].startswith(comment_marker)


def test_kgw_uses_legal_java_interface_constants():
    bundle = build_watermark_bundle("kgw")
    spec = _spec()
    example = _example(
        "java-interface",
        "java",
        "package demo;\n\npublic interface Demo {\n    void run();\n}\n",
    )

    watermarked = bundle.embed(example, spec)
    identifier = watermarked.metadata["identifier"]

    assert identifier.endswith("_java")
    assert f'String {identifier} = "wm";' in watermarked.source
    assert "private static final String" not in watermarked.source
    assert watermarked.metadata["channels"] == ["comment", "identifier"]
    assert watermarked.source.index(identifier) > watermarked.source.index("interface Demo")


def test_kgw_preserves_javascript_directive_prologue_without_shebang():
    bundle = build_watermark_bundle("kgw")
    spec = _spec()
    example = _example(
        "js-directive",
        "javascript",
        "'use strict';\nfunction sum(values) {\n  return values.length;\n}\n",
    )

    watermarked = bundle.embed(example, spec)
    detected = bundle.detect(watermarked, spec, example_id=example.example_id)

    lines = watermarked.source.splitlines()
    assert lines[0] == "'use strict';"
    assert lines[1].startswith("// wm:")
    assert "const wm_" in watermarked.source
    assert watermarked.metadata["channels"] == ["comment", "identifier"]
    assert detected.metadata["channels"] == ["comment", "identifier"]


def test_kgw_disables_identifier_channel_when_java_has_no_class_body():
    bundle = build_watermark_bundle("kgw")
    spec = _spec()
    example = _example(
        "java-disabled",
        "java",
        "package demo;\nimport java.util.List;\n",
    )

    watermarked = bundle.embed(example, spec)
    detected = bundle.detect(watermarked, spec, example_id=example.example_id)

    assert watermarked.metadata["identifier"] is None
    assert watermarked.metadata["channels"] == ["comment"]
    assert watermarked.source.startswith("// wm:")
    assert "private static final String" not in watermarked.source
    assert detected.metadata["channels"] == ["comment"]


def test_kgw_disables_identifier_channel_for_java_enum():
    bundle = build_watermark_bundle("kgw")
    spec = _spec()
    example = _example(
        "java-enum",
        "java",
        "package demo;\n\npublic enum Demo {\n    READY,\n    DONE;\n}\n",
    )

    watermarked = bundle.embed(example, spec)
    detected = bundle.detect(watermarked, spec, example_id=example.example_id)

    assert watermarked.metadata["identifier"] is None
    assert watermarked.metadata["channels"] == ["comment"]
    assert "private static final String" not in watermarked.source
    assert 'String wm_' not in watermarked.source
    assert detected.metadata["channels"] == ["comment"]


def test_kgw_java_and_javascript_suffixes_do_not_collide():
    bundle = build_watermark_bundle("kgw")
    spec = _spec()
    java_example = _example("shared-marker", "java", "public class Demo {\n}\n")
    js_example = _example("shared-marker", "javascript", "function demo() { return 1; }\n")

    java_watermarked = bundle.embed(java_example, spec)
    js_watermarked = bundle.embed(js_example, spec)

    assert java_watermarked.metadata["identifier"].endswith("_java")
    assert js_watermarked.metadata["identifier"].endswith("_js")
    assert java_watermarked.metadata["identifier"] != js_watermarked.metadata["identifier"]
