from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path

from scripts import validate_release_bundle


BASELINES = ("stone", "sweet", "ewd", "kgw")


def _entry(name: str, *, origin: str = "external_checkout", valid: bool = True) -> dict[str, object]:
    return {
        "origin": origin,
        "source_path": f"external_checkout/{name}",
        "vendored_path": f"third_party/{name}",
        "external_path": f"external_checkout/{name}",
        "selected_checkout_path": f"external_checkout/{name}",
        "checkout_valid": valid,
        "bundle_eligible": False,
        "repo_url": f"https://example.com/{name}.git",
        "verification_issues": [] if valid else ["origin remote mismatch"],
    }


def _write_bundle(tmp_path: Path) -> Path:
    bundle = tmp_path / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "README.md").write_text("bundle\n", encoding="utf-8")
    provenance = {name: _entry(name) for name in BASELINES}
    (bundle / "baseline_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    _refresh_bundle_metadata(bundle)
    return bundle


def _refresh_bundle_metadata(bundle: Path) -> None:
    for metadata_path in (bundle / "bundle.manifest.json", bundle / "SHA256SUMS.txt"):
        if metadata_path.exists():
            metadata_path.unlink()
    provenance = json.loads((bundle / "baseline_provenance.json").read_text(encoding="utf-8"))
    payload_files = sorted(
        path.relative_to(bundle).as_posix()
        for path in bundle.rglob("*")
        if path.is_file() and path.name not in {"bundle.manifest.json", "SHA256SUMS.txt"}
    )
    manifest = {
        "included": sorted([*payload_files, "bundle.manifest.json", "SHA256SUMS.txt"]),
        "baseline_provenance_map": provenance,
    }
    (bundle / "bundle.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    checksum_lines: list[str] = []
    for relative in [*payload_files, "bundle.manifest.json"]:
        path = bundle / relative
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checksum_lines.append(f"{digest}  {relative}")
    checksum_lines.append(f"{'0' * 64}  SHA256SUMS.txt")
    (bundle / "SHA256SUMS.txt").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")


def test_validate_release_bundle_passes_for_four_valid_baselines(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path)
    report = tmp_path / "report.json"

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_release_bundle.py", "--bundle", str(bundle), "--report", str(report)],
    )

    assert validate_release_bundle.main() == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"


def test_validate_release_bundle_rejects_missing_required_baseline(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path)
    report = tmp_path / "report.json"
    baseline = json.loads((bundle / "baseline_provenance.json").read_text(encoding="utf-8"))
    del baseline["ewd"]
    (bundle / "baseline_provenance.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    _refresh_bundle_metadata(bundle)

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_release_bundle.py", "--bundle", str(bundle), "--report", str(report)],
    )

    assert validate_release_bundle.main() == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert any("missing required entries" in issue for issue in payload["issues"])


def test_validate_release_bundle_rejects_invalid_runtime_provenance(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path)
    report = tmp_path / "report.json"
    baseline = json.loads((bundle / "baseline_provenance.json").read_text(encoding="utf-8"))
    baseline["sweet"] = _entry("sweet", valid=False)
    (bundle / "baseline_provenance.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    _refresh_bundle_metadata(bundle)

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_release_bundle.py", "--bundle", str(bundle), "--report", str(report)],
    )

    assert validate_release_bundle.main() == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert any("selected without a valid checkout" in issue for issue in payload["issues"])


def test_validate_release_bundle_rejects_extra_legacy_entry(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path)
    report = tmp_path / "report.json"
    baseline = json.loads((bundle / "baseline_provenance.json").read_text(encoding="utf-8"))
    baseline["unexpected_runtime"] = _entry("unexpected_runtime")
    (bundle / "baseline_provenance.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    _refresh_bundle_metadata(bundle)

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_release_bundle.py", "--bundle", str(bundle), "--report", str(report)],
    )

    assert validate_release_bundle.main() == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert any("unexpected entries" in issue for issue in payload["issues"])


def test_validate_release_bundle_scans_all_bundle_files_for_identity_markers(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path)
    report = tmp_path / "report.json"
    notes = bundle / "notes.txt"
    notes.write_text("contact: author@example.com\n", encoding="utf-8")
    _refresh_bundle_metadata(bundle)

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_release_bundle.py", "--bundle", str(bundle), "--report", str(report)],
    )

    assert validate_release_bundle.main() == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert any("matched email" in issue for issue in payload["issues"])


def test_validate_release_bundle_rejects_forbidden_bundle_paths(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path)
    report = tmp_path / "report.json"
    forbidden = bundle / "configs" / "archive" / "legacy.yaml"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("legacy: true\n", encoding="utf-8")
    _refresh_bundle_metadata(bundle)

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_release_bundle.py", "--bundle", str(bundle), "--report", str(report)],
    )

    assert validate_release_bundle.main() == 1
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert any("forbidden path" in issue for issue in payload["issues"])


def test_validate_release_bundle_does_not_flag_generic_coordination_literals_in_code(tmp_path: Path, monkeypatch) -> None:
    bundle = _write_bundle(tmp_path)
    report = tmp_path / "report.json"
    script_path = bundle / "scripts" / "sample.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text('EXTERNAL_ROOT = ".coordination/external/runtime.gitcheckout"\\n', encoding="utf-8")
    _refresh_bundle_metadata(bundle)

    monkeypatch.setattr(
        sys,
        "argv",
        ["validate_release_bundle.py", "--bundle", str(bundle), "--report", str(report)],
    )

    assert validate_release_bundle.main() == 0
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["status"] == "passed"
