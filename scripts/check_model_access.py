from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_MODELS = (
    "Qwen/Qwen2.5-Coder-14B-Instruct",
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    "bigcode/starcoder2-7b",
    "deepseek-ai/deepseek-coder-6.7b-instruct",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Hugging Face model access with the current token.")
    parser.add_argument("--token-env", default="HF_ACCESS_TOKEN", help="Environment variable containing the Hugging Face token.")
    parser.add_argument("--model", action="append", dest="models", default=None, help="Model id to probe. Repeat to override the default set.")
    parser.add_argument("--timeout", type=float, default=30.0, help="Network timeout in seconds.")
    parser.add_argument("--require-all", action="store_true", help="Exit with code 1 if any requested model is inaccessible.")
    return parser.parse_args()


def _probe(model_id: str, token: str, timeout: float) -> dict[str, object]:
    url = f"https://huggingface.co/{model_id}/resolve/main/config.json"
    request = urllib.request.Request(url, method="HEAD")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return {
                "model": model_id,
                "accessible": True,
                "status": getattr(response, "status", 200),
                "reason": "ok",
            }
    except urllib.error.HTTPError as exc:
        return {
            "model": model_id,
            "accessible": False,
            "status": exc.code,
            "reason": exc.reason,
        }
    except urllib.error.URLError as exc:
        return {
            "model": model_id,
            "accessible": False,
            "status": None,
            "reason": str(exc.reason),
        }


def main() -> int:
    args = parse_args()
    token = os.environ.get(args.token_env, "")
    if not token:
        raise SystemExit(f"Missing {args.token_env}")

    models = tuple(args.models or DEFAULT_MODELS)
    results = [_probe(model_id, token, args.timeout) for model_id in models]
    accessible = [item["model"] for item in results if item["accessible"]]
    blocked = [item["model"] for item in results if not item["accessible"]]
    payload = {
        "token_env": args.token_env,
        "requested_models": list(models),
        "accessible_models": accessible,
        "blocked_models": blocked,
        "results": results,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.require_all and blocked:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
