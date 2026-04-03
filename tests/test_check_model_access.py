from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_module():
    path = ROOT / "scripts" / "check_model_access.py"
    spec = importlib.util.spec_from_file_location("check_model_access", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_model_matrix_is_populated():
    module = _load_module()
    assert "Qwen/Qwen2.5-Coder-14B-Instruct" in module.DEFAULT_MODELS
    assert "Qwen/Qwen2.5-Coder-7B-Instruct" in module.DEFAULT_MODELS
    assert "bigcode/starcoder2-7b" in module.DEFAULT_MODELS
    assert "deepseek-ai/deepseek-coder-6.7b-instruct" in module.DEFAULT_MODELS
    assert len(module.DEFAULT_MODELS) == 4
    assert all("codegemma" not in model.lower() for model in module.DEFAULT_MODELS)


def test_probe_marks_accessible_on_success(monkeypatch):
    module = _load_module()

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(module.urllib.request, "urlopen", lambda request, timeout=30.0: _Response())
    result = module._probe("Qwen/Qwen2.5-Coder-7B-Instruct", "token", 30.0)
    assert result["accessible"] is True
    assert result["status"] == 200
