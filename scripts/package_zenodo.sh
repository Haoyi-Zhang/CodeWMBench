#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${1:-$ROOT/results/release_bundle}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "Missing Python interpreter: $PYTHON_BIN" >&2
    exit 1
  fi
fi

"$PYTHON_BIN" - "$ROOT" "$OUT_DIR" <<'PY'
from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2])
if not out.is_absolute():
    out = (root / out).resolve()

if str(root) not in sys.path:
    sys.path.insert(0, str(root))

from codewmbench.baselines.stone_family.common import load_upstream_manifest, stone_family_checkout_metadata

allowed = [
    "README.md",
    "ARTIFACTS.md",
    "docs",
    "Makefile",
    ".env.example",
    ".gitignore",
    "pyproject.toml",
    "requirements.txt",
    "requirements-remote.txt",
    "codewmbench",
    "configs",
    "scripts",
    "artifacts",
    "data/compact",
    "data/fixtures",
    "data/public",
    "results/schema.json",
    "model_cache/README.md",
    "third_party/README.md",
    "third_party/STONE-watermarking.UPSTREAM.json",
    "third_party/SWEET-watermark.UPSTREAM.json",
    "third_party/EWD.UPSTREAM.json",
    "third_party/KGW-lm-watermarking.UPSTREAM.json",
]

BASELINES = {
    "stone": "stone_runtime",
    "sweet": "sweet_runtime",
    "ewd": "ewd_runtime",
    "kgw": "kgw_runtime",
}

excluded_dirs = {
    root / ".git",
    root / "paper",
    root / "proposal.md",
    root / "configs" / "archive",
    root / "data" / "interim",
    root / "results" / "runs",
}
policy_exclusions = [
    root / ".git",
    root / "paper",
    root / "proposal.md",
    root / "configs" / "archive",
    root / "data" / "interim",
    root / "results" / "runs",
    root / "results" / "submission_preflight",
    root / "scripts" / "archive_suite_outputs.py",
]
ignored_patterns = ("__pycache__", "*.pyc", ".pytest_cache", "*.log", "_cache", "*.source.jsonl.gz", ".git")
forbidden_bundle_roots = {
    Path("configs/archive"),
    Path("results/archive"),
    Path("results/matrix"),
    Path("results/figures"),
    Path("results/certifications"),
    Path("results/release_bundle"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_excluded(path: Path) -> str | None:
    for excluded in excluded_dirs:
        try:
            path.relative_to(excluded)
            return str(excluded.relative_to(root))
        except ValueError:
            if path == excluded:
                return str(excluded.relative_to(root))
    for forbidden in forbidden_bundle_roots:
        absolute = root / forbidden
        try:
            path.relative_to(absolute)
            return str(forbidden.as_posix())
        except ValueError:
            if path == absolute:
                return str(forbidden.as_posix())
    return None


def _copytree_ignore(current_dir: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    current_path = Path(current_dir)
    for name in names:
        candidate = current_path / name
        if is_excluded(candidate) is not None:
            ignored.add(name)
            continue
        for pattern in ignored_patterns:
            if candidate.match(pattern) or candidate.name == pattern:
                ignored.add(name)
                break
    return ignored


def _sanitize_upstream_manifest_for_bundle(path: Path) -> None:
    if path.parent.name != "third_party" or not path.name.endswith(".UPSTREAM.json"):
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return
    external_root = str(payload.get("external_root", "")).strip()
    public_external_root = str(payload.get("public_external_root", "")).strip()
    if ".coordination/" in external_root or "\\.coordination\\" in external_root:
        payload["external_root"] = public_external_root or ""
    payload["bundle_sanitized"] = True
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def remove_tree(path: Path) -> None:
    last_error: Exception | None = None
    for _ in range(5):
        try:
            shutil.rmtree(path)
            return
        except FileNotFoundError:
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2)
    if last_error is not None:
        raise last_error


def _public_baseline_payload(method: str) -> dict[str, object]:
    manifest = load_upstream_manifest(method)
    metadata = dict(stone_family_checkout_metadata(method))
    checkout_root = str(manifest.get("checkout_root", "")).strip()
    external_root = str(manifest.get("external_root", "")).strip()
    public_external_root = str(manifest.get("public_external_root", "")).strip()
    external_path = public_external_root or external_root.replace(".coordination/external/", "external_checkout/")
    vendored_exists = bool(checkout_root) and (root / checkout_root).exists()
    external_exists = bool(external_root) and (root / external_root).exists()
    metadata_issues = [str(item) for item in metadata.get("checkout_issues", [])]
    verification_issues = [
        item
        for item in metadata_issues
        if not item.startswith("missing local checkout for ")
    ]
    repo_url = str(metadata.get("repo_url", "") or manifest.get("repo_url", "")).strip()
    pinned_commit = str(metadata.get("pinned_commit", "") or manifest.get("pinned_commit", "")).strip()
    upstream_commit = str(metadata.get("upstream_commit", "") or manifest.get("pinned_commit", "")).strip()
    license_status = str(metadata.get("license_status", "") or manifest.get("license_status", "")).strip()
    checkout_valid = bool(repo_url) and bool(pinned_commit) and not verification_issues
    redistributable = bool(metadata.get("redistributable", False)) or license_status.lower() == "redistributable"
    if vendored_exists:
        if redistributable and checkout_valid:
            origin = "vendored_snapshot"
            bundle_eligible = True
            source_path = checkout_root
        else:
            origin = "vendored_unverified"
            bundle_eligible = False
            source_path = ""
    else:
        origin = "external_checkout"
        bundle_eligible = False
        source_path = external_path if repo_url and pinned_commit else ""
    return {
        "origin": origin,
        "source_path": source_path,
        "vendored_path": checkout_root,
        "external_path": external_path,
        "selected_checkout_path": source_path,
        "checkout_valid": checkout_valid,
        "bundle_eligible": bundle_eligible,
        "repo_url": repo_url,
        "pinned_commit": pinned_commit,
        "upstream_commit": upstream_commit,
        "license_status": license_status,
        "verification_issues": verification_issues,
        "vendored_exists": vendored_exists,
        "external_exists": external_exists,
        "redistributable": redistributable,
    }


if out.exists():
    remove_tree(out)
out.mkdir(parents=True, exist_ok=True)

included_files: list[str] = []
excluded: list[str] = []
checksums: list[str] = []
baseline_provenance_map: dict[str, dict[str, object]] = {}

fail_closed_issues: list[str] = []
for public_name, method in BASELINES.items():
    payload = _public_baseline_payload(method)
    baseline_provenance_map[public_name] = payload
    origin = str(payload.get("origin", "")).strip()
    if origin in {"vendored_unverified", "external_unverified"}:
        details = [str(item) for item in payload.get("verification_issues", [])]
        suffix = f" ({'; '.join(details)})" if details else ""
        fail_closed_issues.append(
            f"{public_name}: refusing to stage a release bundle with provenance origin '{origin}'{suffix}"
        )
    if payload["origin"] == "vendored_snapshot" and bool(payload.get("bundle_eligible", False)):
        allowed.append(str(payload["vendored_path"]))

if fail_closed_issues:
    for issue in fail_closed_issues:
        print(issue, file=sys.stderr)
    raise SystemExit(1)

for path in policy_exclusions:
    if not path.exists():
        continue
    excluded.append(f"policy\t{path.relative_to(root).as_posix()}")

for public_name, payload in baseline_provenance_map.items():
    vendored_path = str(payload.get("vendored_path", "")).strip()
    external_path = str(payload.get("external_path", "")).strip()
    if bool(payload.get("vendored_exists")) and not bool(payload.get("bundle_eligible", False)):
        excluded.append(
            f"policy\t{vendored_path}\t{public_name} vendored checkout is not bundled without a verified git checkout and redistributable license"
        )
    if bool(payload.get("external_exists")):
        excluded.append(
            f"policy\t{external_path}\t{public_name} external checkout is runtime-only and not bundled"
        )

for rel in allowed:
    source = root / rel
    if not source.exists():
        excluded.append(f"missing\t{rel}")
        continue
    reason = is_excluded(source)
    if reason is not None:
        excluded.append(f"policy\t{rel}\t{reason}")
        continue
    destination = out / rel
    if source.is_dir():
        shutil.copytree(source, destination, ignore=_copytree_ignore)
        for file in destination.rglob(".gitkeep"):
            file.unlink()
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    if destination.is_file():
        _sanitize_upstream_manifest_for_bundle(destination)
    else:
        for file in destination.rglob("*.UPSTREAM.json"):
            _sanitize_upstream_manifest_for_bundle(file)
    files = [destination] if destination.is_file() else [path for path in destination.rglob("*") if path.is_file()]
    for file in files:
        relative = file.relative_to(out).as_posix()
        included_files.append(relative)
        checksums.append(f"{sha256(file)}  {relative}")

provenance_path = out / "baseline_provenance.json"
provenance_path.write_text(
    json.dumps(baseline_provenance_map, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    encoding="utf-8",
    newline="\n",
)
checksums.append(f"{sha256(provenance_path)}  baseline_provenance.json")
included_files.append("baseline_provenance.json")

try:
    bundle_root = out.relative_to(root).as_posix()
except ValueError:
    bundle_root = out.name

generated_entries = {
    "baseline_provenance.json",
    "MANIFEST.txt",
    "SHA256SUMS.txt",
    "EXCLUDED.txt",
    "bundle.manifest.json",
}
included = sorted(set(included_files).union(generated_entries))

(out / "MANIFEST.txt").write_text("\n".join(included) + "\n", encoding="utf-8", newline="\n")
(out / "EXCLUDED.txt").write_text("\n".join(sorted(excluded)) + "\n", encoding="utf-8", newline="\n")
checksums.append(f"{sha256(out / 'MANIFEST.txt')}  MANIFEST.txt")
checksums.append(f"{sha256(out / 'EXCLUDED.txt')}  EXCLUDED.txt")

final_file_count = len(checksums) + 1
bundle_manifest = {
    "bundle_root": bundle_root,
    "included": included,
    "excluded": sorted(excluded),
    "file_count": final_file_count,
    "included_count": len(included),
    "baseline_provenance_map": baseline_provenance_map,
}
(out / "bundle.manifest.json").write_text(
    json.dumps(bundle_manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
    encoding="utf-8",
    newline="\n",
)
checksums.append(f"{sha256(out / 'bundle.manifest.json')}  bundle.manifest.json")
checksums.append(f"{'0' * 64}  SHA256SUMS.txt")
(out / "SHA256SUMS.txt").write_text("\n".join(sorted(checksums)) + "\n", encoding="utf-8", newline="\n")
print(f"Anonymous release bundle staged at {out}")
print("Before publishing, verify that paper/, proposal.md, caches, and local run outputs are absent.")
print(f"Included entries: {len(included)}; files hashed: {len(checksums)}")
PY
