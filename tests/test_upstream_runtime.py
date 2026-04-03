from __future__ import annotations

import json
import sys
from pathlib import Path

from codewmbench.baselines.stone_family import official_runtime
from codewmbench.baselines.stone_family.official_runtime import _resolve_snapshot_path
from codewmbench.models import WatermarkSpec, WatermarkedSnippet
from codewmbench.watermarks.registry import available_watermarks, build_watermark_bundle


def test_runtime_watermarks_are_registered():
    names = set(available_watermarks())
    assert {"stone_runtime", "sweet_runtime", "ewd_runtime", "kgw_runtime"}.issubset(names)


def test_runtime_bundle_uses_internal_generation(monkeypatch, sample_example):
    class FakeBackend:
        def generate_unwatermarked_text(self, prompt: str) -> str:
            return "def generated(x):\n    return x\n"

        def generate_watermarked_text(self, prompt: str) -> str:
            return "def generated(x):\n    marker = 1\n    return x\n"

        def detect_watermark(self, text: str, return_dict: bool = True):
            payload = {"is_watermarked": "marker" in text, "score": 9.7 if "marker" in text else 0.2}
            return payload if return_dict else (payload["is_watermarked"], payload["score"])

    monkeypatch.setattr(
        "codewmbench.baselines.stone_family.official_runtime._backend_for",
        lambda method, example, spec: FakeBackend(),
    )
    monkeypatch.setattr(
        "codewmbench.baselines.stone_family.official_runtime._load_backend",
        lambda *args, **kwargs: FakeBackend(),
    )

    bundle = build_watermark_bundle("stone_runtime")
    spec = WatermarkSpec(
        name="stone_runtime",
        secret="anonymous",
        payload="wm",
        strength=1.0,
        parameters={"model_name": "Qwen/Qwen2.5-Coder-7B-Instruct", "z_threshold": 4.0},
    )

    prepared = bundle.prepare_example(sample_example, spec)
    watermarked = bundle.embed(prepared, spec)
    detected = bundle.detect(watermarked, spec, example_id=sample_example.example_id)

    assert bundle.uses_internal_generation is True
    assert prepared.metadata["provider_mode"] == "watermark_runtime"
    assert "marker = 1" in watermarked.source
    assert detected.detected is True
    assert detected.threshold == 4.0
    assert detected.score >= detected.threshold


def test_runtime_detector_treats_insufficient_token_errors_as_non_detection(monkeypatch, sample_example):
    class FakeBackend:
        def detect_watermark(self, text: str, return_dict: bool = True):
            raise ValueError("Must have at least 1 token to score after filtering.")

    monkeypatch.setattr(
        "codewmbench.baselines.stone_family.official_runtime._load_backend",
        lambda *args, **kwargs: FakeBackend(),
    )

    bundle = build_watermark_bundle("stone_runtime")
    spec = WatermarkSpec(
        name="stone_runtime",
        secret="anonymous",
        payload="wm",
        strength=1.0,
        parameters={"model_name": "bigcode/starcoder2-7b", "z_threshold": 4.0},
    )

    detected = bundle.detect("pass", spec, example_id=sample_example.example_id)

    assert detected.detected is False
    assert detected.score == 0.0
    assert "insufficient_tokens_after_preprocessing" in detected.evidence
    assert detected.metadata["payload"]["runtime_detection_error_kind"] == "insufficient_tokens_after_preprocessing"


def test_runtime_backend_forwards_explicit_dtype(monkeypatch, sample_example):
    captured = {}

    class FakeBackend:
        def generate_unwatermarked_text(self, prompt: str) -> str:
            return "def solve():\n    return 1\n"

    def fake_load_backend(*args):
        captured["dtype_name"] = args[-1]
        return FakeBackend()

    monkeypatch.setattr(
        "codewmbench.baselines.stone_family.official_runtime._load_backend",
        fake_load_backend,
    )

    bundle = build_watermark_bundle("stone_runtime")
    spec = WatermarkSpec(
        name="stone_runtime",
        secret="anonymous",
        payload="wm",
        strength=1.0,
        parameters={
            "model_name": "Qwen/Qwen2.5-Coder-14B-Instruct",
            "dtype": "float16",
        },
    )

    bundle.prepare_example(sample_example, spec)

    assert captured["dtype_name"] == "float16"


def test_runtime_detector_preserves_generation_prompt_whitespace(monkeypatch, sample_example):
    captured = {}

    class FakeBackend:
        def detect_watermark(self, text: str, return_dict: bool = True, *, prompt: str | None = None):
            captured["text"] = text
            captured["prompt"] = prompt
            payload = {"is_watermarked": False, "score": 0.0}
            return payload if return_dict else (payload["is_watermarked"], payload["score"])

    monkeypatch.setattr(
        "codewmbench.baselines.stone_family.official_runtime._load_backend",
        lambda *args, **kwargs: FakeBackend(),
    )

    bundle = build_watermark_bundle("ewd_runtime")
    spec = WatermarkSpec(
        name="ewd_runtime",
        secret="anonymous",
        payload="wm",
        strength=1.0,
        parameters={"model_name": "bigcode/starcoder2-7b"},
    )

    snippet = WatermarkedSnippet(
        example_id=sample_example.example_id,
        language=sample_example.language,
        source="def solve():\n    return 1\n",
        watermark=spec,
        metadata={"generation_prompt": "  keep trailing space \n"},
    )

    bundle.detect(snippet, spec, example_id=sample_example.example_id)

    assert captured["prompt"] == "  keep trailing space \n"


def test_runtime_loader_prefers_snapshot_path_from_cache(tmp_path: Path) -> None:
    entry = tmp_path / "models--bigcode--starcoder2-7b"
    snapshot = entry / "snapshots" / "snapshot-id"
    snapshot.mkdir(parents=True, exist_ok=True)
    (entry / "refs").mkdir(parents=True, exist_ok=True)
    (entry / "refs" / "main").write_text("snapshot-id\n", encoding="utf-8")

    resolved = _resolve_snapshot_path(str(tmp_path), "bigcode/starcoder2-7b")

    assert resolved == str(snapshot)


def test_kgw_runtime_detection_payload_is_json_serializable(monkeypatch, sample_example):
    import torch

    class FakeBackend:
        def detect_watermark(self, text: str, return_dict: bool = True, *, prompt: str | None = None):
            payload = {
                "prediction": torch.tensor(True),
                "score": torch.tensor(4.5),
                "z_score": torch.tensor(4.5),
                "tokens": torch.tensor([1, 2, 3]),
                "nested": {"green_mask": torch.tensor([[1, 0], [0, 1]])},
            }
            return payload if return_dict else (True, 4.5)

    monkeypatch.setattr(
        "codewmbench.baselines.stone_family.official_runtime._load_backend",
        lambda *args, **kwargs: FakeBackend(),
    )

    bundle = build_watermark_bundle("kgw_runtime")
    spec = WatermarkSpec(
        name="kgw_runtime",
        secret="anonymous",
        payload="wm",
        strength=1.0,
        parameters={"model_name": "bigcode/starcoder2-7b", "z_threshold": 4.0},
    )

    detected = bundle.detect("pass", spec, example_id=sample_example.example_id)
    payload = detected.metadata["payload"]

    assert payload["prediction"] is True
    assert payload["tokens"] == [1, 2, 3]
    assert payload["nested"]["green_mask"] == [[1, 0], [0, 1]]
    json.dumps(detected.as_dict())


def test_kgw_runtime_prefers_explicit_seeding_scheme(monkeypatch, sample_example):
    captured = {}

    class FakeBackend:
        def generate_unwatermarked_text(self, prompt: str) -> str:
            return "def solve():\n    return 1\n"

    def fake_load_backend(*args):
        captured["runtime_name"] = args[0]
        captured["seeding_scheme"] = args[10]
        return FakeBackend()

    monkeypatch.setattr(
        "codewmbench.baselines.stone_family.official_runtime._load_backend",
        fake_load_backend,
    )

    bundle = build_watermark_bundle("kgw_runtime")
    spec = WatermarkSpec(
        name="kgw_runtime",
        secret="anonymous",
        payload="wm",
        strength=1.0,
        parameters={
            "model_name": "bigcode/starcoder2-7b",
            "seeding_scheme": "selfhash",
            "f_scheme": "time",
        },
    )

    bundle.prepare_example(sample_example, spec)

    assert captured["runtime_name"] == "kgw_runtime"
    assert captured["seeding_scheme"] == "selfhash"


def test_non_stone_runtime_reuses_backend_across_languages(monkeypatch):
    official_runtime._BACKEND_CACHE.clear()
    official_runtime._SHARED_MODEL_CACHE.clear()
    build_calls: list[tuple[str, str]] = []

    class FakeBackend:
        pass

    def fake_build(*args):
        build_calls.append((args[0], args[2]))
        return FakeBackend()

    monkeypatch.setattr(official_runtime, "_build_backend", fake_build)

    backend_python = official_runtime._load_backend(
        "kgw_runtime",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "python",
        "cuda",
        0.5,
        2.0,
        15485863,
        0,
        4.0,
        0.9,
        "selfhash",
        "left",
        "all_pl",
        "False",
        200,
        1,
        True,
        0.95,
        0.2,
        4,
        True,
        "HF_ACCESS_TOKEN",
        "model_cache/huggingface",
        True,
        "float16",
    )
    backend_java = official_runtime._load_backend(
        "kgw_runtime",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "java",
        "cuda",
        0.5,
        2.0,
        15485863,
        0,
        4.0,
        0.9,
        "selfhash",
        "left",
        "all_pl",
        "False",
        200,
        1,
        True,
        0.95,
        0.2,
        4,
        True,
        "HF_ACCESS_TOKEN",
        "model_cache/huggingface",
        True,
        "float16",
    )

    assert backend_python is backend_java
    assert build_calls == [("kgw_runtime", "python")]


def test_stone_runtime_evicts_previous_language_backend(monkeypatch):
    official_runtime._BACKEND_CACHE.clear()
    official_runtime._SHARED_MODEL_CACHE.clear()
    built: list[str] = []
    released: list[str] = []

    class FakeBackend:
        def __init__(self, label: str):
            self.label = label

    def fake_build(*args):
        language = args[2]
        built.append(language)
        return FakeBackend(language)

    def fake_release(backend):
        released.append(backend.label)

    monkeypatch.setattr(official_runtime, "_build_backend", fake_build)
    monkeypatch.setattr(official_runtime, "_release_backend", fake_release)

    backend_python = official_runtime._load_backend(
        "stone_runtime",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "python",
        "cuda",
        0.5,
        2.0,
        15485863,
        0,
        4.0,
        0.9,
        "selfhash",
        "left",
        "all_pl",
        "False",
        200,
        1,
        True,
        0.95,
        0.2,
        4,
        True,
        "HF_ACCESS_TOKEN",
        "model_cache/huggingface",
        True,
        "float16",
    )
    backend_java = official_runtime._load_backend(
        "stone_runtime",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        "java",
        "cuda",
        0.5,
        2.0,
        15485863,
        0,
        4.0,
        0.9,
        "selfhash",
        "left",
        "all_pl",
        "False",
        200,
        1,
        True,
        0.95,
        0.2,
        4,
        True,
        "HF_ACCESS_TOKEN",
        "model_cache/huggingface",
        True,
        "float16",
    )

    assert backend_python is not backend_java
    assert built == ["python", "java"]
    assert released == ["python"]


def test_shared_model_loader_reuses_cached_model(monkeypatch):
    official_runtime._SHARED_MODEL_CACHE.clear()
    load_calls: list[str] = []

    class FakeModel:
        def __init__(self):
            self.to_calls: list[str] = []
            self.eval_calls = 0

        def to(self, device: str):
            self.to_calls.append(device)
            return self

        def eval(self):
            self.eval_calls += 1

    fake_model = FakeModel()
    fake_tokenizer = object()

    def fake_load_model_and_tokenizer(*args, **kwargs):
        load_calls.append(kwargs["device"])
        return fake_model, fake_tokenizer

    monkeypatch.setattr(official_runtime, "_load_model_and_tokenizer", fake_load_model_and_tokenizer)

    first = official_runtime._load_shared_model_and_tokenizer(
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        device="cuda",
        trust_remote_code=True,
        token_env="HF_ACCESS_TOKEN",
        cache_dir="model_cache/huggingface",
        local_files_only=True,
        dtype_name="float16",
    )
    second = official_runtime._load_shared_model_and_tokenizer(
        "Qwen/Qwen2.5-Coder-14B-Instruct",
        device="cuda",
        trust_remote_code=True,
        token_env="HF_ACCESS_TOKEN",
        cache_dir="model_cache/huggingface",
        local_files_only=True,
        dtype_name="float16",
    )

    assert first == (fake_model, fake_tokenizer)
    assert second == (fake_model, fake_tokenizer)
    assert load_calls == ["cuda"]
    assert fake_model.eval_calls == 1
    assert fake_model.to_calls == ["cuda"]


def test_release_backend_preserves_shared_generation_model():
    class FakeModel:
        def __init__(self):
            self.to_calls: list[str] = []

        def to(self, device: str):
            self.to_calls.append(device)
            return self

    fake_model = FakeModel()
    backend = type(
        "FakeStoneBackend",
        (),
        {
            "_cw_owns_model": False,
            "config": type("Cfg", (), {"generation_model": fake_model})(),
        },
    )()

    official_runtime._release_backend(backend)

    assert fake_model.to_calls == []


def test_release_backend_falls_back_to_stone_generation_model(monkeypatch):
    released: list[str] = []
    emptied: list[bool] = []

    class FakeModel:
        def __init__(self) -> None:
            self.moves: list[str] = []

        def to(self, device: str):
            self.moves.append(device)
            return self

    class FakeConfig:
        def __init__(self, model: FakeModel) -> None:
            self.generation_model = model

    class FakeBackend:
        def __init__(self) -> None:
            self.config = FakeConfig(FakeModel())

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available() -> bool:
                return True

            @staticmethod
            def empty_cache() -> None:
                emptied.append(True)

    backend = FakeBackend()
    model = backend.config.generation_model

    monkeypatch.setitem(sys.modules, "torch", FakeTorch)
    monkeypatch.setattr(official_runtime.gc, "collect", lambda: released.append("gc"))

    official_runtime._release_backend(backend)

    assert model.moves == ["cpu"]
    assert backend.config.generation_model is None
    assert released == ["gc"]
    assert emptied == [True]
