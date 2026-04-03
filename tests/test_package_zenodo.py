from __future__ import annotations

import json
import os
import shutil
import stat
import sys
from pathlib import Path

import pytest

from tests._stone_test_helpers import create_runtime_checkout, update_manifest
from scripts import validate_release_bundle


ROOT = Path(__file__).resolve().parents[1]
BASH_RELEASE_TESTS_ENABLED = shutil.which("bash") is not None and os.name != "nt"
_BASELINE_ENV = {
    "stone_runtime": ("CODEWMBENCH_STONE_UPSTREAM_ROOT", "CODEWMBENCH_STONE_UPSTREAM_MANIFEST"),
    "sweet_runtime": ("CODEWMBENCH_SWEET_UPSTREAM_ROOT", "CODEWMBENCH_SWEET_UPSTREAM_MANIFEST"),
    "ewd_runtime": ("CODEWMBENCH_EWD_UPSTREAM_ROOT", "CODEWMBENCH_EWD_UPSTREAM_MANIFEST"),
    "kgw_runtime": ("CODEWMBENCH_KGW_UPSTREAM_ROOT", "CODEWMBENCH_KGW_UPSTREAM_MANIFEST"),
}


def _on_rm_error(func, path, exc_info):
    target = Path(path)
    if target.exists():
        target.chmod(stat.S_IWRITE)
        func(path)


def _remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_on_rm_error)


def _run_bundle_script(output_relative: Path) -> Path:
    output_dir = ROOT / output_relative
    if output_dir.exists():
        _remove_tree(output_dir)
    previous = os.getcwd()
    os.chdir(ROOT)
    try:
        os.environ["PYTHON_BIN"] = sys.executable
        bash = shutil.which("bash") or "bash"
        null_device = "NUL" if os.name == "nt" else "/dev/null"
        status = os.system(f'"{bash}" scripts/package_zenodo.sh {output_relative.as_posix()} > {null_device} 2>&1')
        assert status == 0
        return output_dir
    finally:
        os.chdir(previous)


def _prepare_external_runtime_suite(tmp_path: Path, monkeypatch) -> None:
    for method, (root_env, manifest_env) in _BASELINE_ENV.items():
        checkout, manifest, _ = create_runtime_checkout(
            ROOT,
            method,
            relative_path=f".coordination/external/{method}.pytest-external",
            manifest_root=tmp_path / "manifests",
            include_license=(method == "kgw_runtime"),
            license_status="redistributable" if method == "kgw_runtime" else "unverified",
        )
        update_manifest(
            manifest,
            checkout_root=f"third_party/{Path(checkout).name}",
            external_root=f".coordination/external/{method}.pytest-external",
            public_external_root=f"external_checkout/{method}.pytest-external",
        )
        monkeypatch.setenv(root_env, str(checkout))
        monkeypatch.setenv(manifest_env, str(manifest))


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_records_policy_exclusions(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    output_relative = Path("results/test_release_bundle_pytest")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        output_dir = _run_bundle_script(output_relative)

        manifest_text = (output_dir / "MANIFEST.txt").read_text(encoding="utf-8")
        excluded_text = (output_dir / "EXCLUDED.txt").read_text(encoding="utf-8")
        bundle_manifest = (output_dir / "bundle.manifest.json").read_text(encoding="utf-8")
        provenance = json.loads((output_dir / "baseline_provenance.json").read_text(encoding="utf-8"))

        assert str(ROOT).replace("\\", "/") not in bundle_manifest
        assert '"bundle_root": "results/test_release_bundle_pytest"' in bundle_manifest
        assert set(provenance) == {"stone", "sweet", "ewd", "kgw"}
        for relative_path in (
            ".git",
            "paper",
            "proposal.md",
            "configs/archive",
            "data/interim",
            "results/runs",
            "results/submission_preflight",
        ):
            if (ROOT / relative_path).exists():
                assert f"policy\t{relative_path}" in excluded_text
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_excludes_archived_configs_from_bundle(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    output_relative = Path("results/test_release_bundle_archive_filter")
    output_dir = ROOT / output_relative
    archive_root = ROOT / "configs" / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    legacy_config = archive_root / "legacy-suite-config.yaml"
    legacy_config.write_text("legacy: true\n", encoding="utf-8")
    try:
        _remove_tree(output_dir)
        output_dir = _run_bundle_script(output_relative)
        manifest_entries = set((output_dir / "MANIFEST.txt").read_text(encoding="utf-8").splitlines())
        excluded_text = (output_dir / "EXCLUDED.txt").read_text(encoding="utf-8")

        assert "configs/archive" not in manifest_entries
        assert "configs/archive/legacy-suite-config.yaml" not in manifest_entries
        assert "policy\tconfigs/archive" in excluded_text
        assert not (output_dir / "configs" / "archive").exists()
        assert (output_dir / "scripts" / "clean_suite_outputs.py").exists()
        assert not (output_dir / "scripts" / "archive_suite_outputs.py").exists()
    finally:
        if legacy_config.exists():
            legacy_config.unlink()
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_writes_four_baseline_provenance_entries(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    output_relative = Path("results/test_release_bundle_four_baselines")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        output_dir = _run_bundle_script(output_relative)
        bundle_manifest = json.loads((output_dir / "bundle.manifest.json").read_text(encoding="utf-8"))
        provenance = json.loads((output_dir / "baseline_provenance.json").read_text(encoding="utf-8"))

        assert set(provenance) == {"stone", "sweet", "ewd", "kgw"}
        assert set(bundle_manifest["baseline_provenance_map"]) == {"stone", "sweet", "ewd", "kgw"}
        assert all(entry["origin"] == "external_checkout" for entry in provenance.values())
        assert all(entry["checkout_valid"] is True for entry in provenance.values())
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_bundle_passes_release_validator(tmp_path: Path, monkeypatch):
    _prepare_external_runtime_suite(tmp_path, monkeypatch)
    output_relative = Path("results/test_release_bundle_validator")
    output_dir = ROOT / output_relative
    try:
        _remove_tree(output_dir)
        output_dir = _run_bundle_script(output_relative)
        previous_argv = sys.argv[:]
        try:
            sys.argv = ["validate_release_bundle.py", "--bundle", str(output_dir)]
            assert validate_release_bundle.main() == 0
        finally:
            sys.argv = previous_argv
    finally:
        _remove_tree(output_dir)


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_includes_redistributable_vendored_checkout(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_vendored_kgw")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "lm-watermarking"
    vendored_backup = vendored_root.with_name("lm-watermarking.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        checkout, manifest, commit = create_runtime_checkout(
            ROOT,
            "kgw_runtime",
            relative_path="third_party/lm-watermarking",
            manifest_root=tmp_path / "manifests",
            include_license=True,
            license_status="redistributable",
        )
        update_manifest(
            manifest,
            checkout_root="third_party/lm-watermarking",
            external_root=".coordination/external/kgw_runtime.pytest-external",
        )
        monkeypatch.setenv("CODEWMBENCH_KGW_UPSTREAM_ROOT", str(checkout))
        monkeypatch.setenv("CODEWMBENCH_KGW_UPSTREAM_MANIFEST", str(manifest))

        output_dir = _run_bundle_script(output_relative)
        manifest_entries = set((output_dir / "MANIFEST.txt").read_text(encoding="utf-8").splitlines())
        provenance = json.loads((output_dir / "baseline_provenance.json").read_text(encoding="utf-8"))

        assert any(entry.startswith("third_party/lm-watermarking/") for entry in manifest_entries)
        assert provenance["kgw"]["origin"] == "vendored_snapshot"
        assert provenance["kgw"]["bundle_eligible"] is True
        assert provenance["kgw"]["upstream_commit"] == commit
    finally:
        _remove_tree(output_dir)
        _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))


@pytest.mark.skipif(not BASH_RELEASE_TESTS_ENABLED, reason="bash release-bundle tests require a non-Windows bash environment")
def test_package_zenodo_rejects_unverified_vendored_checkout(tmp_path: Path, monkeypatch):
    output_relative = Path("results/test_release_bundle_unverified_vendored")
    output_dir = ROOT / output_relative
    vendored_root = ROOT / "third_party" / "sweet-watermark"
    vendored_backup = vendored_root.with_name("sweet-watermark.pytest-backup")
    if vendored_backup.exists():
        _remove_tree(vendored_backup)
    if vendored_root.exists():
        shutil.move(str(vendored_root), str(vendored_backup))
    try:
        _prepare_external_runtime_suite(tmp_path, monkeypatch)
        checkout, manifest, _ = create_runtime_checkout(
            ROOT,
            "sweet_runtime",
            relative_path="third_party/sweet-watermark",
            manifest_root=tmp_path / "manifests",
            include_license=False,
            license_status="unverified",
        )
        update_manifest(manifest, checkout_root="third_party/sweet-watermark")
        monkeypatch.setenv("CODEWMBENCH_SWEET_UPSTREAM_ROOT", str(checkout))
        monkeypatch.setenv("CODEWMBENCH_SWEET_UPSTREAM_MANIFEST", str(manifest))

        previous = os.getcwd()
        os.chdir(ROOT)
        try:
            os.environ["PYTHON_BIN"] = sys.executable
            bash = shutil.which("bash") or "bash"
            null_device = "NUL" if os.name == "nt" else "/dev/null"
            status = os.system(f'"{bash}" scripts/package_zenodo.sh {output_relative.as_posix()} > {null_device} 2>&1')
            assert status != 0
        finally:
            os.chdir(previous)
    finally:
        _remove_tree(output_dir)
        _remove_tree(vendored_root)
        if vendored_backup.exists():
            shutil.move(str(vendored_backup), str(vendored_root))
