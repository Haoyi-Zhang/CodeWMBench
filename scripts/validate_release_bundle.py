from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_setup


REQUIRED_BASELINES = ("stone", "sweet", "ewd", "kgw")
FORBIDDEN_BUNDLE_PREFIXES = (
    ".coordination",
    "configs/archive",
    "results/audits",
    "results/archive",
    "results/runs",
    "results/matrix",
    "results/figures",
    "results/certifications",
    "results/release_bundle",
    "results/fetched_suite",
    "results/test_release_bundle",
    "results/tmp",
    "results/submission_preflight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a staged release bundle.")
    parser.add_argument("--bundle", type=Path, required=True, help="Bundle root produced by scripts/package_zenodo.sh")
    parser.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_files(bundle: Path) -> list[Path]:
    return sorted(path for path in bundle.rglob("*") if path.is_file())


def _load_checksums(path: Path) -> dict[str, str]:
    payload: dict[str, str] = {}
    if not path.exists():
        return payload
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = str(line).strip()
        if not stripped:
            continue
        parts = stripped.split(None, 1)
        if len(parts) != 2:
            continue
        digest, relpath = parts
        payload[relpath.strip()] = digest.strip()
    return payload


def _provenance_issues(name: str, payload: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    origin = str(payload.get("origin", "")).strip()
    source_path = str(payload.get("source_path", "")).strip()
    checkout_valid = bool(payload.get("checkout_valid", False))
    verification_issues = [str(item) for item in payload.get("verification_issues", [])]

    for key in ("source_path", "vendored_path", "external_path", "selected_checkout_path"):
        value = str(payload.get(key, "")).strip()
        if ".coordination/" in value or "\\.coordination\\" in value:
            issues.append(f"{name}: {key} leaks internal checkout path: {value}")

    if origin in {"vendored_snapshot", "external_checkout"} and not checkout_valid:
        issues.append(f"{name}: {origin} selected without a valid checkout")
    if origin in {"vendored_unverified", "external_unverified"}:
        issues.append(f"{name}: unverified provenance origin '{origin}' is not allowed in a release bundle")
    if origin == "vendored_snapshot" and not bool(payload.get("bundle_eligible", False)):
        issues.append(f"{name}: vendored snapshot selected but bundle_eligible is false")
    if origin == "missing":
        issues.append(f"{name}: baseline provenance is missing from the staged bundle")
    if origin == "external_checkout" and not source_path:
        issues.append(f"{name}: external checkout selected without sanitized source_path")
    if origin == "vendored_snapshot" and not source_path:
        issues.append(f"{name}: vendored snapshot selected without publishable source_path")
    if origin == "missing" and verification_issues:
        issues.extend(f"{name}: {item}" for item in verification_issues)
    if origin not in {"missing", "vendored_snapshot", "external_checkout", "vendored_unverified", "external_unverified"}:
        issues.append(f"{name}: unexpected provenance origin '{origin}'")
    if str(payload.get("repo_url", "")).strip() == "":
        issues.append(f"{name}: missing repo_url in provenance payload")
    return issues


def main() -> int:
    args = parse_args()
    bundle = args.bundle.resolve()
    issues: list[str] = []

    if not bundle.exists():
        issues.append(f"missing bundle root: {bundle}")
    baseline_path = bundle / "baseline_provenance.json"
    manifest_path = bundle / "bundle.manifest.json"
    checksums_path = bundle / "SHA256SUMS.txt"
    if not baseline_path.exists():
        issues.append(f"missing bundle artifact: {baseline_path}")
    if not manifest_path.exists():
        issues.append(f"missing bundle artifact: {manifest_path}")
    if not checksums_path.exists():
        issues.append(f"missing bundle artifact: {checksums_path}")

    anonymity_findings: list[str] = []
    provenance_report: dict[str, Any] = {}
    if not issues:
        baseline_payload = _load_json(baseline_path)
        manifest_payload = _load_json(manifest_path)
        observed_keys = set(baseline_payload)
        if observed_keys != set(REQUIRED_BASELINES):
            missing = sorted(set(REQUIRED_BASELINES) - observed_keys)
            extras = sorted(observed_keys - set(REQUIRED_BASELINES))
            if missing:
                issues.append(f"baseline_provenance.json missing required entries: {missing}")
            if extras:
                issues.append(f"baseline_provenance.json contains unexpected entries: {extras}")
        for required_key in REQUIRED_BASELINES:
            if required_key not in baseline_payload:
                continue
            payload = baseline_payload[required_key]
            if not isinstance(payload, dict):
                issues.append(f"{required_key}: provenance entry must be a JSON object")
                continue
            issues.extend(_provenance_issues(required_key, payload))
            provenance_report[required_key] = {
                "origin": payload.get("origin"),
                "checkout_valid": payload.get("checkout_valid"),
                "bundle_eligible": payload.get("bundle_eligible"),
                "verification_issues": payload.get("verification_issues", []),
            }
        included = set(manifest_payload.get("included", []))
        bundle_files = _bundle_files(bundle)
        actual_files = {file.relative_to(bundle).as_posix() for file in bundle_files}
        for required_path in ("baseline_provenance.json", "bundle.manifest.json"):
            if required_path not in included:
                issues.append(f"bundle.manifest.json missing included entry for {required_path}")
        if set(manifest_payload.get("baseline_provenance_map", {})) != set(REQUIRED_BASELINES):
            issues.append("bundle.manifest.json baseline_provenance_map does not match the required four-baseline roster")
        missing_from_manifest = sorted(actual_files - included)
        missing_from_bundle = sorted(included - actual_files)
        issues.extend(f"bundle.manifest.json is missing actual file: {entry}" for entry in missing_from_manifest)
        issues.extend(f"bundle.manifest.json references missing file: {entry}" for entry in missing_from_bundle)
        checksums = _load_checksums(checksums_path)
        checksum_files = set(checksums)
        issues.extend(f"SHA256SUMS.txt is missing actual file: {entry}" for entry in sorted(actual_files - checksum_files))
        issues.extend(f"SHA256SUMS.txt references missing file: {entry}" for entry in sorted(checksum_files - actual_files))
        forbidden_entries = sorted(
            {
                file.relative_to(bundle).as_posix()
                for file in bundle_files
                if any(file.relative_to(bundle).as_posix().startswith(prefix) for prefix in FORBIDDEN_BUNDLE_PREFIXES)
            }
        )
        issues.extend(f"bundle contains forbidden path: {entry}" for entry in forbidden_entries)
        anonymity_findings = validate_setup.scan_for_identity_markers(
            bundle_files,
            labels=set(validate_setup.IDENTITY_PATTERNS),
        )
        issues.extend(anonymity_findings)

    report = {
        "bundle": str(bundle),
        "status": "failed" if issues else "passed",
        "issues": issues,
        "anonymity_findings": anonymity_findings,
        "provenance": provenance_report,
    }
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if issues:
        print("Bundle validation failed:")
        for issue in issues:
            print(f"- {issue}")
        return 1

    print("Bundle validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
