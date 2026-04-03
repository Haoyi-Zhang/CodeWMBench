from __future__ import annotations

import gc
import importlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping

from ...models import BenchmarkExample, DetectionResult, WatermarkSpec, WatermarkedSnippet
from ...watermarks.base import WatermarkBundle, WatermarkDetector, WatermarkEmbedder
from .common import resolve_checkout, stone_family_checkout_metadata, temporary_sys_path


_METHOD_LABELS = {
    "stone_runtime": "STONE",
    "sweet_runtime": "SWEET",
    "ewd_runtime": "EWD",
    "kgw_runtime": "KGW",
}


_BACKEND_CACHE: dict[tuple[Any, ...], Any] = {}
_SHARED_MODEL_CACHE: dict[tuple[Any, ...], tuple[Any, Any]] = {}


def _bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _device_name(value: Any) -> str:
    if value:
        return str(value)
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _normalize_dtype_name(value: Any, *, device: str | None = None) -> str:
    text = str(value or "").strip().lower()
    if not text or text == "auto":
        return "float16" if str(device or "").strip().lower().startswith("cuda") else "float32"
    if text in {"fp16", "half"}:
        return "float16"
    if text in {"bf16"}:
        return "bfloat16"
    if text in {"fp32"}:
        return "float32"
    return text


def _parse_torch_dtype(torch_module: Any, dtype_name: str, *, device: str) -> Any:
    normalized = _normalize_dtype_name(dtype_name, device=device)
    mapping = {
        "float16": getattr(torch_module, "float16", None),
        "bfloat16": getattr(torch_module, "bfloat16", None),
        "float32": getattr(torch_module, "float32", None),
    }
    if normalized not in mapping or mapping[normalized] is None:
        raise ValueError(
            f"unsupported runtime watermark dtype '{dtype_name}' (expected auto, float16, bfloat16, fp16, bf16, float32, or fp32)"
        )
    return mapping[normalized]


def _model_name(spec: WatermarkSpec) -> str:
    model_name = str(spec.parameters.get("model_name", "")).strip()
    if not model_name:
        raise ValueError("runtime watermark methods require watermark.model_name")
    return model_name


def _strip_prompt_prefix(prompt: str, generated: str) -> str:
    text = str(generated)
    if prompt and text.startswith(prompt):
        return text[len(prompt) :].lstrip("\n\r")
    return text


def _combine_prompt(prompt: str | None, text: str) -> str:
    clean_prompt = str(prompt or "")
    clean_text = str(text or "")
    if clean_prompt and not clean_text.startswith(clean_prompt):
        return clean_prompt + clean_text
    return clean_text


def _prompt_for_detection(snippet_source: WatermarkedSnippet | None) -> str:
    if snippet_source is None:
        return ""
    metadata = dict(snippet_source.metadata)
    return str(metadata.get("generation_prompt", "") or metadata.get("prompt", ""))


def _baseline_metadata(runtime_name: str) -> dict[str, Any]:
    payload = stone_family_checkout_metadata(runtime_name)
    return {
        "baseline_family": "runtime_official",
        "baseline_origin": str(payload.get("origin", "external_checkout")),
        "baseline_upstream_commit": str(payload.get("upstream_commit", "")),
        "baseline_repo_url": str(payload.get("repo_url", "")),
        "baseline_license_path": payload.get("license_path"),
        "baseline_license_status": str(payload.get("license_status", "")),
        "baseline_method": str(payload.get("method_symbol", _METHOD_LABELS.get(runtime_name, runtime_name))),
        "baseline_runtime_name": runtime_name,
    }


def _json_safe_runtime_payload(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe_runtime_payload(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_runtime_payload(item) for item in value]
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        try:
            return _json_safe_runtime_payload(value.detach().cpu().tolist())
        except Exception:
            pass
    if hasattr(value, "tolist"):
        try:
            return _json_safe_runtime_payload(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return _json_safe_runtime_payload(value.item())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        try:
            return _json_safe_runtime_payload(dict(value.__dict__))
        except Exception:
            pass
    return str(value)


def _is_empty_runtime_detection_error(exc: Exception) -> bool:
    message = str(exc).strip().lower()
    return (
        "must have at least 1 token to score" in message
        or "at least 1 token to score" in message
        or "cannot score" in message
    )


def _call_detect_watermark(backend: Any, text: str, *, return_dict: bool, prompt: str | None) -> Any:
    try:
        return backend.detect_watermark(text, return_dict=return_dict, prompt=prompt)
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument 'prompt'" not in message and 'unexpected keyword argument "prompt"' not in message:
            raise
        return backend.detect_watermark(text, return_dict=return_dict)


def _repo_cache_dirname(model_name: str) -> str:
    normalized = str(model_name or "").strip().replace("\\", "/").strip("/")
    if not normalized:
        return ""
    return "models--" + normalized.replace("/", "--")


def _resolve_cache_dir(cache_dir: str, model_name: str, *, local_files_only: bool) -> str:
    repo_dirname = _repo_cache_dirname(model_name)

    def _has_repo_entries(path: Path) -> bool:
        if not path.exists() or not path.is_dir():
            return False
        if repo_dirname and (path / repo_dirname).exists():
            return True
        return any(child.is_dir() and child.name.startswith(("models--", "datasets--", "spaces--")) for child in path.iterdir())

    configured = str(cache_dir or "").strip()
    if configured:
        configured_path = Path(configured)
        if _has_repo_entries(configured_path):
            return str(configured_path)
        if configured_path.exists():
            hub_path = configured_path / "hub"
            if _has_repo_entries(hub_path):
                return str(hub_path)
        if configured_path.exists() or not local_files_only:
            return configured
    if local_files_only:
        hf_home = str(os.environ.get("HF_HOME", "")).strip()
        if hf_home:
            hf_home_path = Path(hf_home)
            if _has_repo_entries(hf_home_path):
                return hf_home
            hub_path = hf_home_path / "hub"
            if _has_repo_entries(hub_path):
                return str(hub_path)
            return hf_home
    return configured


def _resolve_snapshot_path(cache_dir: str, model_name: str) -> str:
    repo_dirname = _repo_cache_dirname(model_name)
    configured = str(cache_dir or "").strip()
    if not configured or not repo_dirname:
        return ""

    base = Path(configured)
    candidates = [base / repo_dirname]
    if base.name != "hub":
        candidates.append(base / "hub" / repo_dirname)
    for entry_path in candidates:
        if not entry_path.exists():
            continue
        refs_main = entry_path / "refs" / "main"
        if refs_main.exists():
            revision = refs_main.read_text(encoding="utf-8").strip()
            if revision:
                candidate = entry_path / "snapshots" / revision
                if candidate.exists():
                    return str(candidate)
        snapshots_dir = entry_path / "snapshots"
        if snapshots_dir.exists():
            snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
            if len(snapshots) == 1:
                return str(snapshots[0])
    return ""


def _model_device(model: Any) -> Any:
    try:
        import torch

        parameter = next(model.parameters())
        return parameter.device
    except Exception:
        return getattr(model, "device", torch.device("cpu"))  # type: ignore[name-defined]


def _move_batch_to_device(batch: Any, device: Any) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if isinstance(batch, dict):
        return {key: value.to(device) if hasattr(value, "to") else value for key, value in batch.items()}
    return batch


class _IsolatedModuleLoader:
    def __init__(self, prefixes: tuple[str, ...], source_root: Path):
        self._prefixes = prefixes
        self._source_root = source_root
        self._saved: dict[str, Any] = {}

    def __enter__(self):
        existing = list(sys.modules)
        for name in existing:
            if any(name == prefix or name.startswith(prefix + ".") for prefix in self._prefixes):
                self._saved[name] = sys.modules[name]
                del sys.modules[name]
        self._ctx = temporary_sys_path(self._source_root)
        self._ctx.__enter__()
        importlib.invalidate_caches()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            current = list(sys.modules)
            for name in current:
                if any(name == prefix or name.startswith(prefix + ".") for prefix in self._prefixes):
                    del sys.modules[name]
            sys.modules.update(self._saved)
        finally:
            self._ctx.__exit__(exc_type, exc_val, exc_tb)


@dataclass(slots=True)
class _HFOfficialBackend:
    runtime_name: str
    model: Any
    tokenizer: Any
    wm_logits_processor: Any
    detect_fn: Callable[[str, str | None], dict[str, Any]]
    max_new_tokens: int
    do_sample: bool
    top_p: float
    temperature: float
    no_repeat_ngram_size: int
    num_beams: int
    owns_model: bool = True

    def _generate(self, prompt: str, *, use_watermark: bool) -> str:
        import torch
        from transformers import LogitsProcessorList

        tokenized = self.tokenizer(prompt, return_tensors="pt")
        tokenized = _move_batch_to_device(tokenized, _model_device(self.model))
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "num_beams": int(self.num_beams),
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None) or getattr(self.tokenizer, "eos_token_id", None),
        }
        if int(self.no_repeat_ngram_size) > 0:
            kwargs["no_repeat_ngram_size"] = int(self.no_repeat_ngram_size)
        if self.do_sample:
            kwargs["do_sample"] = True
            kwargs["top_p"] = float(self.top_p)
            kwargs["temperature"] = float(self.temperature)
        else:
            kwargs["do_sample"] = False
        if use_watermark:
            kwargs["logits_processor"] = LogitsProcessorList([self.wm_logits_processor])
        with torch.no_grad():
            output_tokens = self.model.generate(**tokenized, **kwargs)
        prompt_len = tokenized["input_ids"].shape[-1]
        return self.tokenizer.batch_decode(
            output_tokens[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

    def generate_unwatermarked_text(self, prompt: str) -> str:
        return self._generate(prompt, use_watermark=False)

    def generate_watermarked_text(self, prompt: str) -> str:
        return self._generate(prompt, use_watermark=True)

    def detect_watermark(self, text: str, return_dict: bool = True, *, prompt: str | None = None) -> dict[str, Any] | tuple[bool, float]:
        payload = self.detect_fn(text, prompt)
        if return_dict:
            return payload
        return bool(payload.get("is_watermarked", False)), float(payload.get("score", 0.0))


def _load_model_and_tokenizer(
    model_name: str,
    *,
    device: str,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get(token_env, "")
    tokenizer_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
        "torch_dtype": _parse_torch_dtype(torch, dtype_name, device=device),
    }
    load_target = model_name
    if token:
        tokenizer_kwargs["token"] = token
        model_kwargs["token"] = token
    resolved_cache_dir = _resolve_cache_dir(cache_dir, model_name, local_files_only=local_files_only)
    if local_files_only and resolved_cache_dir:
        snapshot_path = _resolve_snapshot_path(resolved_cache_dir, model_name)
        if snapshot_path:
            load_target = snapshot_path
    if resolved_cache_dir:
        tokenizer_kwargs["cache_dir"] = resolved_cache_dir
        model_kwargs["cache_dir"] = resolved_cache_dir
    if local_files_only:
        tokenizer_kwargs["local_files_only"] = True
        model_kwargs["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(load_target, **tokenizer_kwargs)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token_id", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(load_target, **model_kwargs)
    if device and device != "cpu":
        model = model.to(device)
    model.eval()
    if getattr(model.config, "pad_token_id", None) is None and getattr(tokenizer, "pad_token_id", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer


def _shared_model_cache_key(
    model_name: str,
    *,
    device: str,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
) -> tuple[Any, ...]:
    return (
        str(model_name),
        str(device),
        bool(trust_remote_code),
        str(token_env),
        str(cache_dir),
        bool(local_files_only),
        _normalize_dtype_name(dtype_name, device=device),
    )


def _load_shared_model_and_tokenizer(
    model_name: str,
    *,
    device: str,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
) -> tuple[Any, Any]:
    key = _shared_model_cache_key(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        token_env=token_env,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        dtype_name=dtype_name,
    )
    cached = _SHARED_MODEL_CACHE.get(key)
    if cached is not None:
        model, tokenizer = cached
        if device and device != "cpu":
            try:
                model = model.to(device)
            except Exception:
                pass
        model.eval()
        _SHARED_MODEL_CACHE[key] = (model, tokenizer)
        return model, tokenizer

    model, tokenizer = _load_model_and_tokenizer(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        token_env=token_env,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        dtype_name=dtype_name,
    )
    _SHARED_MODEL_CACHE[key] = (model, tokenizer)
    return model, tokenizer


def _tokenize_with_prefix(tokenizer: Any, model: Any, prompt: str, text: str) -> tuple[Any, Any]:
    import torch

    combined = _combine_prompt(prompt, text)
    device = _model_device(model)
    tokenized_text = tokenizer(combined, return_tensors="pt", add_special_tokens=True)["input_ids"][0].to(device)
    if prompt:
        tokenized_prefix = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0].to(device)
    else:
        tokenized_prefix = torch.empty(0, dtype=tokenized_text.dtype, device=device)
    return tokenized_text, tokenized_prefix


def _sweet_entropy(model: Any, tokenized_text: Any) -> list[float]:
    import torch

    tokenized_text = tokenized_text.to(_model_device(model))
    with torch.no_grad():
        output = model(tokenized_text.unsqueeze(0), return_dict=True)
        probs = torch.softmax(output.logits, dim=-1)
        entropy = -torch.where(probs > 0, probs * probs.log(), probs.new([0.0])).sum(dim=-1)
    values = entropy[0].detach().cpu().tolist()
    return [0.0] + values[:-1]


def _cpu_detection_inputs(tokenized_text: Any, tokenized_prefix: Any) -> tuple[Any, Any]:
    cpu_text = tokenized_text.detach().cpu() if hasattr(tokenized_text, "detach") else tokenized_text
    cpu_prefix = tokenized_prefix.detach().cpu() if hasattr(tokenized_prefix, "detach") else tokenized_prefix
    return cpu_text, cpu_prefix


def _load_stone_backend(
    runtime_name: str,
    model_name: str,
    language: str,
    device: str,
    gamma: float,
    delta: float,
    hash_key: int,
    prefix_length: int,
    z_threshold: float,
    entropy_threshold: float,
    f_scheme: str,
    window_scheme: str,
    skipping_rule: str,
    watermark_on_pl: str,
    max_new_tokens: int,
    min_length: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    no_repeat_ngram_size: int,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
):
    checkout = resolve_checkout(runtime_name)
    if checkout is None:
        raise FileNotFoundError(f"missing official upstream checkout for {runtime_name}")
    model, tokenizer = _load_shared_model_and_tokenizer(
        model_name,
        device=device,
        trust_remote_code=trust_remote_code,
        token_env=token_env,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        dtype_name=dtype_name,
    )
    with _IsolatedModuleLoader(("watermark", "utils"), checkout.source_root):
        from utils.transformers_config import TransformersConfig
        from watermark.auto_watermark import STONEAutoWatermark

        transformers_config = TransformersConfig(
            model=model,
            tokenizer=tokenizer,
            vocab_size=len(tokenizer),
            device=device,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        backend = STONEAutoWatermark.load(
            "STONE",
            transformers_config=transformers_config,
            skipping_rule=skipping_rule,
            watermark_on_pl=watermark_on_pl,
            gamma=gamma,
            delta=delta,
            hash_key=hash_key,
            z_threshold=z_threshold,
            prefix_length=prefix_length,
            language=language,
        )
        setattr(backend, "_cw_owns_model", False)
        return backend


def _load_kgw_backend(
    runtime_name: str,
    model_name: str,
    device: str,
    gamma: float,
    delta: float,
    z_threshold: float,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    no_repeat_ngram_size: int,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
    seeding_scheme: str,
):
    checkout = resolve_checkout(runtime_name)
    if checkout is None:
        raise FileNotFoundError(f"missing official upstream checkout for {runtime_name}")
    with _IsolatedModuleLoader(
        ("extended_watermark_processor", "alternative_prf_schemes", "normalizers", "homoglyphs"),
        checkout.source_root,
    ):
        from extended_watermark_processor import WatermarkDetector, WatermarkLogitsProcessor

        model, tokenizer = _load_model_and_tokenizer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            token_env=token_env,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            dtype_name=dtype_name,
        )
        processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
        )
        detector = WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            seeding_scheme=seeding_scheme,
            device=_model_device(model),
            tokenizer=tokenizer,
            z_threshold=z_threshold,
        )

        def _detect(text: str, prompt: str | None) -> dict[str, Any]:
            payload = detector.detect(
                text=_combine_prompt(prompt, text),
                return_prediction=True,
                return_scores=True,
                z_threshold=z_threshold,
                convert_to_float=True,
            )
            score = float(payload.get("z_score", 0.0))
            detected = bool(payload.get("prediction", score >= z_threshold))
            return {**payload, "is_watermarked": detected, "score": score}

        return _HFOfficialBackend(
            runtime_name=runtime_name,
            model=model,
            tokenizer=tokenizer,
            wm_logits_processor=processor,
            detect_fn=_detect,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=1,
        )


def _load_sweet_backend(
    runtime_name: str,
    model_name: str,
    device: str,
    gamma: float,
    delta: float,
    z_threshold: float,
    entropy_threshold: float,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    no_repeat_ngram_size: int,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
):
    checkout = resolve_checkout(runtime_name)
    if checkout is None:
        raise FileNotFoundError(f"missing official upstream checkout for {runtime_name}")
    with _IsolatedModuleLoader(("watermark", "sweet"), checkout.source_root):
        from sweet import SweetDetector, SweetLogitsProcessor

        model, tokenizer = _load_model_and_tokenizer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            token_env=token_env,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            dtype_name=dtype_name,
        )
        processor = SweetLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            entropy_threshold=entropy_threshold,
        )
        detector = SweetDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            tokenizer=tokenizer,
            z_threshold=z_threshold,
            entropy_threshold=entropy_threshold,
        )

        def _detect(text: str, prompt: str | None) -> dict[str, Any]:
            tokenized_text, tokenized_prefix = _tokenize_with_prefix(tokenizer, model, str(prompt or ""), text)
            tokenized_text, tokenized_prefix = _cpu_detection_inputs(tokenized_text, tokenized_prefix)
            entropy = _sweet_entropy(model, tokenized_text)
            payload = detector.detect(
                tokenized_text=tokenized_text,
                tokenized_prefix=tokenized_prefix,
                entropy=entropy,
            )
            score = float(payload.get("z_score", 0.0))
            detected = score >= z_threshold
            return {**payload, "prediction": detected, "is_watermarked": detected, "score": score}

        return _HFOfficialBackend(
            runtime_name=runtime_name,
            model=model,
            tokenizer=tokenizer,
            wm_logits_processor=processor,
            detect_fn=_detect,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=1,
        )


def _load_ewd_backend(
    runtime_name: str,
    model_name: str,
    device: str,
    gamma: float,
    delta: float,
    z_threshold: float,
    max_new_tokens: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    no_repeat_ngram_size: int,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
    hash_key: int,
    entropy_threshold: float,
):
    checkout = resolve_checkout(runtime_name)
    if checkout is None:
        raise FileNotFoundError(f"missing official upstream checkout for {runtime_name}")
    with _IsolatedModuleLoader(("watermark",), checkout.source_root):
        from watermark import WatermarkDetector, WatermarkLogitsProcessor

        model, tokenizer = _load_model_and_tokenizer(
            model_name,
            device=device,
            trust_remote_code=trust_remote_code,
            token_env=token_env,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            dtype_name=dtype_name,
        )
        processor = WatermarkLogitsProcessor(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            hash_key=hash_key,
            entropy_threshold=entropy_threshold,
        )
        detector = WatermarkDetector(
            vocab=list(tokenizer.get_vocab().values()),
            gamma=gamma,
            delta=delta,
            tokenizer=tokenizer,
            type="ewd",
            model=model,
            acc=SimpleNamespace(device=_model_device(model)),
            hash_key=hash_key,
            entropy_threshold=entropy_threshold,
        )

        def _detect(text: str, prompt: str | None) -> dict[str, Any]:
            tokenized_text, tokenized_prefix = _tokenize_with_prefix(tokenizer, model, str(prompt or ""), text)
            tokenized_text, tokenized_prefix = _cpu_detection_inputs(tokenized_text, tokenized_prefix)
            payload = detector.detect(
                tokenized_text=tokenized_text,
                tokenized_prefix=tokenized_prefix,
            )
            score = float(payload.get("z_score", 0.0))
            detected = score >= z_threshold
            return {**payload, "prediction": detected, "is_watermarked": detected, "score": score}

        return _HFOfficialBackend(
            runtime_name=runtime_name,
            model=model,
            tokenizer=tokenizer,
            wm_logits_processor=processor,
            detect_fn=_detect,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            no_repeat_ngram_size=no_repeat_ngram_size,
            num_beams=1,
        )


def _backend_cache_language(runtime_name: str, language: str) -> str:
    normalized = str(runtime_name).strip().lower()
    if normalized == "stone_runtime":
        return str(language or "python")
    return "__shared__"


def _backend_cache_key(
    runtime_name: str,
    model_name: str,
    language: str,
    device: str,
    gamma: float,
    delta: float,
    hash_key: int,
    prefix_length: int,
    z_threshold: float,
    entropy_threshold: float,
    f_scheme: str,
    window_scheme: str,
    skipping_rule: str,
    watermark_on_pl: str,
    max_new_tokens: int,
    min_length: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    no_repeat_ngram_size: int,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
) -> tuple[Any, ...]:
    return (
        str(runtime_name).strip().lower(),
        str(model_name),
        _backend_cache_language(runtime_name, language),
        str(device),
        float(gamma),
        float(delta),
        int(hash_key),
        int(prefix_length),
        float(z_threshold),
        float(entropy_threshold),
        str(f_scheme),
        str(window_scheme),
        str(skipping_rule),
        str(watermark_on_pl),
        int(max_new_tokens),
        int(min_length),
        bool(do_sample),
        float(top_p),
        float(temperature),
        int(no_repeat_ngram_size),
        bool(trust_remote_code),
        str(token_env),
        str(cache_dir),
        bool(local_files_only),
        _normalize_dtype_name(dtype_name, device=device),
    )


def _backend_eviction_group(key: tuple[Any, ...]) -> tuple[Any, ...]:
    return (key[0], key[1], key[3], key[10], key[14], key[16], key[24])


def _backend_model_for_release(backend: Any) -> Any:
    model = getattr(backend, "model", None)
    if model is not None:
        return model
    config = getattr(backend, "config", None)
    if config is not None:
        generation_model = getattr(config, "generation_model", None)
        if generation_model is not None:
            return generation_model
    return None


def _release_backend(backend: Any) -> None:
    owns_model = bool(getattr(backend, "_cw_owns_model", getattr(backend, "owns_model", True)))
    model = _backend_model_for_release(backend)
    if model is not None and owns_model:
        try:
            model.to("cpu")
        except Exception:
            pass
    config = getattr(backend, "config", None)
    if owns_model and config is not None and getattr(config, "generation_model", None) is model:
        try:
            config.generation_model = None
        except Exception:
            pass
    del backend
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _build_backend(
    runtime_name: str,
    model_name: str,
    language: str,
    device: str,
    gamma: float,
    delta: float,
    hash_key: int,
    prefix_length: int,
    z_threshold: float,
    entropy_threshold: float,
    f_scheme: str,
    window_scheme: str,
    skipping_rule: str,
    watermark_on_pl: str,
    max_new_tokens: int,
    min_length: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    no_repeat_ngram_size: int,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
):
    normalized = str(runtime_name).strip().lower()
    if normalized == "stone_runtime":
        return _load_stone_backend(
            normalized,
            model_name,
            language,
            device,
            gamma,
            delta,
            hash_key,
            prefix_length,
            z_threshold,
            entropy_threshold,
            f_scheme,
            window_scheme,
            skipping_rule,
            watermark_on_pl,
            max_new_tokens,
            min_length,
            do_sample,
            top_p,
            temperature,
            no_repeat_ngram_size,
            trust_remote_code,
            token_env,
            cache_dir,
            local_files_only,
            dtype_name,
        )
    if normalized == "kgw_runtime":
        return _load_kgw_backend(
            normalized,
            model_name,
            device,
            gamma,
            delta,
            z_threshold,
            max_new_tokens,
            do_sample,
            top_p,
            temperature,
            no_repeat_ngram_size,
            trust_remote_code,
            token_env,
            cache_dir,
            local_files_only,
            dtype_name,
            str(f_scheme or "selfhash"),
        )
    if normalized == "sweet_runtime":
        return _load_sweet_backend(
            normalized,
            model_name,
            device,
            gamma,
            delta,
            z_threshold,
            entropy_threshold,
            max_new_tokens,
            do_sample,
            top_p,
            temperature,
            no_repeat_ngram_size,
            trust_remote_code,
            token_env,
            cache_dir,
            local_files_only,
            dtype_name,
        )
    if normalized == "ewd_runtime":
        return _load_ewd_backend(
            normalized,
            model_name,
            device,
            gamma,
            delta,
            z_threshold,
            max_new_tokens,
            do_sample,
            top_p,
            temperature,
            no_repeat_ngram_size,
            trust_remote_code,
            token_env,
            cache_dir,
            local_files_only,
            dtype_name,
            hash_key,
            entropy_threshold,
        )
    raise KeyError(f"unknown runtime watermark method: {runtime_name}")


def _load_backend(
    runtime_name: str,
    model_name: str,
    language: str,
    device: str,
    gamma: float,
    delta: float,
    hash_key: int,
    prefix_length: int,
    z_threshold: float,
    entropy_threshold: float,
    f_scheme: str,
    window_scheme: str,
    skipping_rule: str,
    watermark_on_pl: str,
    max_new_tokens: int,
    min_length: int,
    do_sample: bool,
    top_p: float,
    temperature: float,
    no_repeat_ngram_size: int,
    trust_remote_code: bool,
    token_env: str,
    cache_dir: str,
    local_files_only: bool,
    dtype_name: str,
):
    key = _backend_cache_key(
        runtime_name,
        model_name,
        language,
        device,
        gamma,
        delta,
        hash_key,
        prefix_length,
        z_threshold,
        entropy_threshold,
        f_scheme,
        window_scheme,
        skipping_rule,
        watermark_on_pl,
        max_new_tokens,
        min_length,
        do_sample,
        top_p,
        temperature,
        no_repeat_ngram_size,
        trust_remote_code,
        token_env,
        cache_dir,
        local_files_only,
        dtype_name,
    )
    cached = _BACKEND_CACHE.get(key)
    if cached is not None:
        return cached

    group = _backend_eviction_group(key)
    for stale_key in [candidate for candidate in list(_BACKEND_CACHE) if _backend_eviction_group(candidate) == group and candidate != key]:
        stale_backend = _BACKEND_CACHE.pop(stale_key)
        _release_backend(stale_backend)

    backend = _build_backend(
        runtime_name,
        model_name,
        language,
        device,
        gamma,
        delta,
        hash_key,
        prefix_length,
        z_threshold,
        entropy_threshold,
        f_scheme,
        window_scheme,
        skipping_rule,
        watermark_on_pl,
        max_new_tokens,
        min_length,
        do_sample,
        top_p,
        temperature,
        no_repeat_ngram_size,
        trust_remote_code,
        token_env,
        cache_dir,
        local_files_only,
        dtype_name,
    )
    _BACKEND_CACHE[key] = backend
    return backend


def _backend_for(runtime_name: str, example: BenchmarkExample, spec: WatermarkSpec):
    kgw_seeding_scheme = str(
        spec.parameters.get("seeding_scheme")
        or spec.parameters.get("f_scheme")
        or "selfhash"
    )
    return _load_backend(
        runtime_name,
        _model_name(spec),
        str(spec.parameters.get("language") or example.language or "python"),
        _device_name(spec.parameters.get("device")),
        float(spec.parameters.get("gamma", 0.5)),
        float(spec.parameters.get("delta", 0.5)),
        int(spec.parameters.get("hash_key", 15485863)),
        int(spec.parameters.get("prefix_length", 0)),
        float(spec.parameters.get("z_threshold", 10.0)),
        float(spec.parameters.get("entropy_threshold", 0.9)),
        kgw_seeding_scheme,
        str(spec.parameters.get("window_scheme", "left")),
        str(spec.parameters.get("skipping_rule", "all_pl")),
        str(spec.parameters.get("watermark_on_pl", "False")),
        int(spec.parameters.get("max_new_tokens", 200)),
        int(spec.parameters.get("min_length", 1)),
        _bool(spec.parameters.get("do_sample"), True),
        float(spec.parameters.get("top_p", 0.95)),
        float(spec.parameters.get("temperature", 0.2)),
        int(spec.parameters.get("no_repeat_ngram_size", 4)),
        _bool(spec.parameters.get("trust_remote_code"), True),
        str(spec.parameters.get("token_env", "HF_ACCESS_TOKEN")),
        str(spec.parameters.get("cache_dir", "")),
        _bool(spec.parameters.get("local_files_only"), False),
        str(spec.parameters.get("dtype", "auto")),
    )


@dataclass(slots=True)
class RuntimePreparer:
    name: str

    def prepare(self, example: BenchmarkExample, spec: WatermarkSpec) -> BenchmarkExample:
        backend = _backend_for(self.name, example, spec)
        baseline = _strip_prompt_prefix(example.prompt, backend.generate_unwatermarked_text(example.prompt))
        metadata = {
            **dict(example.metadata),
            **_baseline_metadata(self.name),
            "provider_mode": "watermark_runtime",
            "runtime_watermark_method": _METHOD_LABELS[self.name],
            "runtime_model_name": _model_name(spec),
            "generation_prompt": example.prompt,
        }
        return BenchmarkExample(
            example_id=example.example_id,
            language=example.language,
            prompt=example.prompt,
            reference_solution=baseline,
            reference_tests=tuple(example.reference_tests),
            execution_tests=tuple(example.execution_tests),
            metadata=metadata,
        )


@dataclass(slots=True)
class RuntimeEmbedder(WatermarkEmbedder):
    name: str

    def embed(self, example: BenchmarkExample, spec: WatermarkSpec) -> WatermarkedSnippet:
        backend = _backend_for(self.name, example, spec)
        watermarked = _strip_prompt_prefix(example.prompt, backend.generate_watermarked_text(example.prompt))
        return WatermarkedSnippet(
            example_id=example.example_id,
            language=example.language,
            source=watermarked,
            watermark=spec,
            metadata={
                "engine": "official-runtime",
                "generation_prompt": example.prompt,
                **_baseline_metadata(self.name),
            },
        )


@dataclass(slots=True)
class RuntimeDetector(WatermarkDetector):
    name: str

    def detect(self, source: str | WatermarkedSnippet, spec: WatermarkSpec, *, example_id: str = "") -> DetectionResult:
        snippet_source = source if isinstance(source, WatermarkedSnippet) else None
        snippet = snippet_source.source if snippet_source is not None else str(source)
        prompt = _prompt_for_detection(snippet_source)
        kgw_seeding_scheme = str(
            spec.parameters.get("seeding_scheme")
            or spec.parameters.get("f_scheme")
            or "selfhash"
        )
        backend = _load_backend(
            self.name,
            _model_name(spec),
            str((snippet_source.language if snippet_source is not None else spec.parameters.get("language", "python"))),
            _device_name(spec.parameters.get("device")),
            float(spec.parameters.get("gamma", 0.5)),
            float(spec.parameters.get("delta", 0.5)),
            int(spec.parameters.get("hash_key", 15485863)),
            int(spec.parameters.get("prefix_length", 0)),
            float(spec.parameters.get("z_threshold", 10.0)),
            float(spec.parameters.get("entropy_threshold", 0.9)),
            kgw_seeding_scheme,
            str(spec.parameters.get("window_scheme", "left")),
            str(spec.parameters.get("skipping_rule", "all_pl")),
            str(spec.parameters.get("watermark_on_pl", "False")),
            int(spec.parameters.get("max_new_tokens", 200)),
            int(spec.parameters.get("min_length", 1)),
            _bool(spec.parameters.get("do_sample"), True),
            float(spec.parameters.get("top_p", 0.95)),
            float(spec.parameters.get("temperature", 0.2)),
            int(spec.parameters.get("no_repeat_ngram_size", 4)),
            _bool(spec.parameters.get("trust_remote_code"), True),
            str(spec.parameters.get("token_env", "HF_ACCESS_TOKEN")),
            str(spec.parameters.get("cache_dir", "")),
            _bool(spec.parameters.get("local_files_only"), False),
            str(spec.parameters.get("dtype", "auto")),
        )
        try:
            payload = _call_detect_watermark(backend, snippet, return_dict=True, prompt=prompt)
        except ValueError as exc:
            if not _is_empty_runtime_detection_error(exc):
                raise
            payload = {
                "is_watermarked": False,
                "score": 0.0,
                "runtime_detection_error": str(exc),
                "runtime_detection_error_kind": "insufficient_tokens_after_preprocessing",
            }
        payload = _json_safe_runtime_payload(payload)
        threshold = float(spec.parameters.get("z_threshold", spec.parameters.get("threshold", 0.5)))
        score = float(payload.get("score", 0.0))
        detected = bool(payload.get("is_watermarked", payload.get("prediction", score >= threshold)))
        return DetectionResult(
            example_id=example_id,
            method=self.name,
            score=score,
            detected=detected,
            threshold=threshold,
            evidence=tuple(
                item
                for item in (
                    str(payload.get("tokens", [])),
                    str(payload.get("runtime_detection_error_kind", "")),
                )
                if item
            ),
            metadata={
                "engine": "official-runtime",
                "prompt_available": bool(prompt),
                **_baseline_metadata(self.name),
                "payload": payload,
            },
        )


def build_runtime_bundle(name: str) -> WatermarkBundle:
    normalized = str(name).lower()
    if normalized not in _METHOD_LABELS:
        raise KeyError(f"unknown runtime watermark scheme: {name}")
    return WatermarkBundle(
        name=normalized,
        embedder=RuntimeEmbedder(name=normalized),
        detector=RuntimeDetector(name=normalized),
        preparer=RuntimePreparer(name=normalized),
    )
