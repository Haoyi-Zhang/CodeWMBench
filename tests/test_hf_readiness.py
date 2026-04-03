from __future__ import annotations

import contextlib
import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "_hf_readiness.py"
    spec = importlib.util.spec_from_file_location("_hf_readiness", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_load_local_hf_model_forces_offline_mode(monkeypatch, tmp_path) -> None:
    module = _load_module()

    fake_torch = ModuleType("torch")
    fake_torch.float16 = "float16"
    fake_torch.float32 = "float32"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.cuda = SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None)

    @contextlib.contextmanager
    def inference_mode():
        yield

    fake_torch.inference_mode = inference_mode

    @dataclass
    class FakeTensor:
        value: str
        device: str | None = None

        def to(self, device: str):
            self.device = device
            return self

    class FakeTokenizer:
        last_prompt = ""

        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            assert kwargs["local_files_only"] is True
            assert "token" not in kwargs
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            return cls()

        def __init__(self):
            self.pad_token_id = None
            self.eos_token_id = 0
            self.eos_token = "<eos>"
            self.pad_token = None

        def __call__(self, prompt: str, return_tensors: str = "pt"):
            FakeTokenizer.last_prompt = prompt
            return {"input_ids": FakeTensor("input_ids"), "attention_mask": FakeTensor("attention_mask")}

        def decode(self, token_ids, skip_special_tokens: bool = True):
            return self.last_prompt + " a + b\n"

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_name: str, **kwargs):
            assert kwargs["local_files_only"] is True
            assert "token" not in kwargs
            assert os.environ.get("HF_HUB_OFFLINE") == "1"
            assert os.environ.get("TRANSFORMERS_OFFLINE") == "1"
            return cls()

        def to(self, device: str):
            self.device = device
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            return [[1, 2, 3]]

    fake_transformers = ModuleType("transformers")
    fake_transformers.AutoTokenizer = SimpleNamespace(from_pretrained=FakeTokenizer.from_pretrained)
    fake_transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=FakeModel.from_pretrained)

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setenv("HF_ACCESS_TOKEN", "secret-token")

    requirement = module.HFModelRequirement(
        model="acme/test-model",
        cache_dir=str(tmp_path / "hf_cache"),
        local_files_only=True,
        token_env="HF_ACCESS_TOKEN",
        device="cuda",
        dtype="float16",
    )

    result = module.smoke_load_local_hf_model(requirement)

    assert result["status"] == "ok"
    assert result["device"] == "cuda"
    assert result["generated_preview"].endswith("a + b\n")
