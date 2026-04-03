from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from _shared import dump_json, markdown_table, read_jsonl
from codewmbench.baselines.stone_family.evaluation import binary_auroc, calculate_stem


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate imported official runtime baseline runs with AUROC, PPL, and STEM.")
    parser.add_argument("--input", type=Path, required=True, help="Run directory or report.json path.")
    parser.add_argument("--records", type=Path, default=None, help="Optional baseline_eval_records.jsonl path.")
    parser.add_argument("--payloads", type=Path, default=None, help="Optional private payload JSONL with raw texts for perplexity.")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--ppl-model", type=str, default="bigcode/starcoder2-7b", help="Reference model for perplexity.")
    parser.add_argument("--device", type=str, default="", help="Torch device for perplexity evaluation.")
    parser.add_argument("--skip-perplexity", action="store_true", help="Skip perplexity and STEM naturalness terms.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Optional cap for PPL samples.")
    parser.add_argument("--token-env", type=str, default="HF_ACCESS_TOKEN", help="Credential env for gated HF models.")
    parser.add_argument("--cache-dir", type=str, default="", help="Optional Hugging Face cache root for perplexity evaluation.")
    parser.add_argument("--local-files-only", action="store_true", help="Require local-only HF loading for perplexity evaluation.")
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code for perplexity evaluation.")
    return parser.parse_args()


def resolve_report_path(path: Path) -> Path:
    if path.is_dir():
        candidate = path / "report.json"
        if candidate.exists():
            return candidate
        raise FileNotFoundError(f"{path} does not contain report.json")
    return path


def resolve_records_path(report_path: Path, explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit
    candidate = report_path.with_name("baseline_eval_records.jsonl")
    if not candidate.exists():
        raise FileNotFoundError(f"Missing baseline eval records: {candidate}")
    return candidate


def resolve_payloads_path(report_path: Path, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    candidate = report_path.with_name("baseline_eval_payloads.private.jsonl")
    return candidate if candidate.exists() else None


def load_report(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _payload_texts(
    records: list[dict[str, Any]],
    payloads: list[dict[str, Any]] | None,
    *,
    field: str,
) -> list[str]:
    by_example: dict[str, dict[str, Any]] = {}
    for payload in payloads or []:
        example_id = str(payload.get("example_id", "")).strip()
        if example_id:
            by_example[example_id] = payload
    values: list[str] = []
    for record in records:
        example_id = str(record.get("example_id", "")).strip()
        payload = by_example.get(example_id, {})
        text = str(payload.get(field, "")).strip()
        if not text:
            text = str(record.get(field, "")).strip()
        if text:
            values.append(text)
    return values


def _device_name(value: str) -> str:
    if value.strip():
        return value.strip()
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _average_perplexity(
    texts: list[str],
    *,
    model_name: str,
    token_env: str,
    device: str,
    cache_dir: str,
    local_files_only: bool,
    trust_remote_code: bool,
) -> float:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    token = os.environ.get(token_env, "")
    kwargs: dict[str, Any] = {}
    if token:
        kwargs["token"] = token
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if local_files_only:
        kwargs["local_files_only"] = True
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code, **kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust_remote_code, **kwargs).to(device)
    values: list[float] = []
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
            values.append(float(torch.exp(outputs.loss).item()))
    return round(mean(values), 4) if values else 0.0


def evaluate_records(
    report: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    payloads: list[dict[str, Any]] | None = None,
    ppl_model: str,
    device: str,
    skip_perplexity: bool,
    sample_limit: int | None,
    token_env: str,
    cache_dir: str = "",
    local_files_only: bool = False,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    if sample_limit is not None and sample_limit > 0:
        records = records[:sample_limit]
    summary = dict(report.get("summary", {}))
    baseline_family = ""
    origins = sorted({str(record.get("baseline_origin", "")).strip() for record in records if str(record.get("baseline_origin", "")).strip()})
    commits = sorted({str(record.get("baseline_upstream_commit", "")).strip() for record in records if str(record.get("baseline_upstream_commit", "")).strip()})
    datasets = sorted({str(record.get("dataset", "")).strip() for record in records if str(record.get("dataset", "")).strip()})
    source_groups = sorted({str(record.get("source_group", "")).strip() for record in records if str(record.get("source_group", "")).strip()})
    model_labels = sorted({str(record.get("model_label", "")).strip() for record in records if str(record.get("model_label", "")).strip()})
    evaluation_tracks = sorted({str(record.get("evaluation_track", "")).strip() for record in records if str(record.get("evaluation_track", "")).strip()})
    if records:
        baseline_family = str(records[0].get("baseline_family", ""))
    human_scores = [float(record.get("human_detect_score", 0.0)) for record in records if record.get("human_detect_score") is not None]
    watermarked_scores = [float(record.get("watermarked_detect_score", 0.0)) for record in records if record.get("watermarked_detect_score") is not None]
    clean_reference_scores = [
        float(record.get("clean_reference_detect_score", 0.0))
        for record in records
        if record.get("clean_reference_detect_score") is not None
    ]
    watermarked_validations = [record.get("watermarked_validation", {}) for record in records if isinstance(record.get("watermarked_validation"), dict)]
    watermarked_passes = [1.0 if validation.get("passed") else 0.0 for validation in watermarked_validations if validation.get("available")]
    watermarked_pass_rate = round(mean(watermarked_passes), 4) if watermarked_passes else 0.0
    human_ppl: float | None = None
    clean_reference_ppl: float | None = None
    watermarked_ppl: float | None = None
    perplexity_available = False
    if not skip_perplexity:
        human_texts = _payload_texts(records, payloads, field="human_reference_solution")
        clean_reference_texts = _payload_texts(records, payloads, field="clean_reference_solution")
        watermarked_texts = _payload_texts(records, payloads, field="watermarked_source")
        if human_texts and clean_reference_texts and watermarked_texts:
            perplexity_available = True
            human_ppl = _average_perplexity(
                human_texts,
                model_name=ppl_model,
                token_env=token_env,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            clean_reference_ppl = _average_perplexity(
                clean_reference_texts,
                model_name=ppl_model,
                token_env=token_env,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
            watermarked_ppl = _average_perplexity(
                watermarked_texts,
                model_name=ppl_model,
                token_env=token_env,
                device=device,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                trust_remote_code=trust_remote_code,
            )
    human_auroc = round(binary_auroc(human_scores, watermarked_scores), 4)
    clean_reference_auroc = round(binary_auroc(clean_reference_scores, watermarked_scores), 4)
    result = {
        "baseline_family": baseline_family,
        "record_count": len(records),
        "baseline_origins": origins,
        "baseline_upstream_commits": commits,
        "datasets": datasets,
        "source_groups": source_groups,
        "model_labels": model_labels,
        "evaluation_tracks": evaluation_tracks,
        "watermark_schemes": sorted({str(record.get("watermark_scheme", "")).strip() for record in records if str(record.get("watermark_scheme", "")).strip()}),
        "human_vs_watermarked_auroc": human_auroc,
        "clean_reference_vs_watermarked_auroc": clean_reference_auroc,
        "watermarked_pass_rate": watermarked_pass_rate,
        "perplexity_available": perplexity_available,
        "summary_watermarked_functional_metrics": summary.get("watermarked_functional_metrics", {}),
        "average_perplexity_human": human_ppl,
        "average_perplexity_clean_reference": clean_reference_ppl,
        "average_perplexity_watermarked": watermarked_ppl,
    }
    if not skip_perplexity and perplexity_available and human_ppl is not None and clean_reference_ppl is not None and watermarked_ppl is not None:
        result["stem_human_reference"] = calculate_stem(watermarked_pass_rate, human_auroc, human_ppl, watermarked_ppl)
        result["stem_clean_reference"] = calculate_stem(watermarked_pass_rate, clean_reference_auroc, clean_reference_ppl, watermarked_ppl)
    elif not skip_perplexity:
        result["perplexity_unavailable_reason"] = "missing_private_payloads"
    return result


def main() -> int:
    args = parse_args()
    try:
        report_path = resolve_report_path(args.input)
        records_path = resolve_records_path(report_path, args.records)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    payloads_path = resolve_payloads_path(report_path, args.payloads)
    report = load_report(report_path)
    records = read_jsonl(records_path)
    payloads = read_jsonl(payloads_path) if payloads_path is not None and payloads_path.exists() else []
    config = dict(report.get("config", {}))
    metadata = dict(config.get("metadata", {}))
    watermark_metadata = dict(metadata.get("watermark", {}))
    provider_metadata = dict(metadata.get("provider", {}))
    derived_cache_dir = str(args.cache_dir or watermark_metadata.get("cache_dir") or provider_metadata.get("cache_dir") or "").strip()
    derived_local_files_only = bool(
        args.local_files_only
        or watermark_metadata.get("local_files_only")
        or provider_metadata.get("local_files_only")
    )
    derived_trust_remote_code = bool(
        args.trust_remote_code
        or watermark_metadata.get("trust_remote_code")
        or provider_metadata.get("trust_remote_code")
    )
    derived_token_env = str(args.token_env or watermark_metadata.get("token_env") or provider_metadata.get("token_env") or "HF_ACCESS_TOKEN")
    evaluation = evaluate_records(
        report,
        records,
        payloads=payloads,
        ppl_model=args.ppl_model,
        device=_device_name(args.device),
        skip_perplexity=args.skip_perplexity,
        sample_limit=args.sample_limit,
        token_env=derived_token_env,
        cache_dir=derived_cache_dir,
        local_files_only=derived_local_files_only,
        trust_remote_code=derived_trust_remote_code,
    )
    print(f"Evaluated {report_path}")
    table_rows = [[key, value] for key, value in evaluation.items() if key not in {"summary_watermarked_functional_metrics"}]
    print(markdown_table(["metric", "value"], table_rows))
    if args.output is not None:
        dump_json(args.output, evaluation)
        print(f"Wrote baseline evaluation to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
