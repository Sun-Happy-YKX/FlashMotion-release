"""Stdlib tests for the environment reporter's import and execution boundaries."""
from __future__ import annotations

import importlib.util
import difflib
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("environment_reporter", ROOT / "tools/check_environment.py")
ENV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENV)
PATCH_SPEC = importlib.util.spec_from_file_location("mamba_interface_patch", ROOT / "tools/patch_mamba_interface.py")
PATCH = importlib.util.module_from_spec(PATCH_SPEC)
PATCH_SPEC.loader.exec_module(PATCH)


class EnvironmentTests(unittest.TestCase):
    def test_default_never_imports_native_or_queries_GPU(self):
        with patch.object(ENV.importlib, "import_module", side_effect=AssertionError("Unexpected import")), \
             patch.object(ENV.metadata, "version", side_effect=metadata.PackageNotFoundError), \
             patch.object(ENV.shutil, "which", return_value=None), \
             patch.object(ENV.subprocess, "run", side_effect=AssertionError("Unexpected subprocess")):
            report = ENV.report_environment("s2g")
        self.assertEqual(report["status"], "NEEDS_REVIEW")
        self.assertEqual(report["native_imports"], {})
        self.assertFalse(report["cuda"]["requested"])
        self.assertFalse(report["model_or_dataset_inference"])
        self.assertFalse(report["installation_performed"])

    def test_version_commands_are_fixed_argv_without_shell(self):
        with patch.object(ENV.shutil, "which", return_value="/tool/nvcc"), \
             patch.object(ENV.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout="V12.8.61", stderr="")) as run:
            result = ENV.command_version("nvcc", "--version")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(run.call_args.args[0], ["/tool/nvcc", "--version"])
        self.assertNotIn("shell", run.call_args.kwargs)
        self.assertEqual(run.call_args.kwargs["timeout"], 15)

    def test_native_failure_is_reported_not_hidden_or_installed(self):
        with patch.object(ENV.importlib, "import_module", side_effect=ImportError("missing provider")):
            imports, sources, cuda = ENV.native_probe("s2g", False)
        self.assertTrue(all(not row["imported"] for row in imports.values()))
        self.assertTrue(all(not row["match"] for row in sources.values()))
        self.assertIsNone(cuda["tensor_probe_passed"])

    def test_source_origin_reports_only_valid_commit_not_URL_credentials(self):
        distribution = SimpleNamespace(read_text=lambda name:
            '{"url":"https://user:fixture-secret@example.invalid/repo?token=fixture",'
            '"vcs_info":{"vcs":"git","commit_id":"10b5d6358f27966f6a40e4bf0baa17a460688128"}}')
        with patch.object(ENV.metadata, "distribution", return_value=distribution):
            result = ENV.source_origin("mamba-ssm")
        self.assertTrue(result["git_commit_matches"])
        self.assertNotIn("fixture", str(result))
        self.assertNotIn("url", result)

    def test_cuda_flag_is_the_only_tensor_probe_boundary(self):
        tensor = SimpleNamespace(cpu=lambda: SimpleNamespace(item=lambda: 1))
        torch = SimpleNamespace(ones=lambda *args, **kwargs: tensor,
                                cuda=SimpleNamespace(synchronize=lambda: None, get_device_name=lambda: "test device"),
                                version=SimpleNamespace(cuda="12.8"))
        modules = {name: SimpleNamespace() for name in ("selective_scan_cuda", "causal_conv1d_cuda", "mamba_ssm", "causal_conv1d")}
        modules["torch"] = torch
        with patch.object(ENV.importlib, "import_module", side_effect=lambda name: modules[name]):
            _, _, cuda = ENV.native_probe("t2m", True)
        self.assertTrue(cuda["tensor_probe_passed"])

    def test_runtime_pins_match_recorded_lock(self):
        for task in ("t2m", "s2g", "m2d"):
            canonical = lambda name: name.lower().replace("_", "-").replace(".", "-")
            lock = ROOT / "requirements" / f"{task}.observed.20261010.lock.txt"
            installed = {canonical(name): version for name, version in ENV.read_pins(lock).items()}
            for name, version in ENV.read_pins(ROOT / "requirements" / f"{task}.runtime.txt").items():
                self.assertEqual(installed[canonical(name)], version, (task, name))


class ProviderPatchTests(unittest.TestCase):
    def upstream_bytes(self):
        recorded = PATCH.ASSET.read_bytes()
        self.assertEqual(PATCH.sha256(recorded), PATCH.OBSERVED_SHA)
        self.assertEqual(recorded.count(PATCH.NEW), 1)
        original = recorded.replace(PATCH.NEW, PATCH.OLD, 1)
        self.assertEqual(PATCH.sha256(original), PATCH.UPSTREAM_SHA)
        return original

    def fixture(self, prefix):
        path = prefix / "lib/python3.10/site-packages/mamba_ssm/ops/selective_scan_interface.py"
        path.parent.mkdir(parents=True)
        path.write_bytes(self.upstream_bytes())
        return path

    def test_minimal_patch_reproduces_exact_recorded_bytes(self):
        self.assertEqual(PATCH.restored_bytes(self.upstream_bytes()), PATCH.ASSET.read_bytes())

    def test_unified_diff_and_upstream_license_are_exact(self):
        generated = "".join(difflib.unified_diff(
            self.upstream_bytes().decode("utf-8").splitlines(keepends=True),
            PATCH.ASSET.read_text(encoding="utf-8").splitlines(keepends=True),
            fromfile="a/mamba_ssm/ops/selective_scan_interface.py",
            tofile="b/mamba_ssm/ops/selective_scan_interface.py"))
        self.assertEqual(generated, (PATCH.ASSET.parent / "mamba_selective_scan_interface.patch").read_text(encoding="utf-8"))
        license_bytes = (PATCH.ASSET.parent / "LICENSE.mamba.Apache-2.0.txt").read_bytes()
        self.assertEqual(len(license_bytes), 11348)
        self.assertEqual(PATCH.sha256(license_bytes), "760939b000194d04548ede6a857bbe735d1695d8422ec85955c8e2bd7f4b95c5")

    def test_default_patch_inspection_never_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            target = self.fixture(prefix)
            result = PATCH.patch_file(target, prefix)
            self.assertEqual(result["status"], "READY")
            self.assertFalse(result["applied"])
            self.assertEqual(target.read_bytes(), self.upstream_bytes())
            self.assertFalse(target.with_name(target.name + ".flashmotion-upstream.bak").exists())

    def test_explicit_patch_is_exact_recoverable_and_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            target = self.fixture(prefix)
            result = PATCH.patch_file(target, prefix, True)
            self.assertTrue(result["applied"])
            self.assertEqual(target.read_bytes(), PATCH.ASSET.read_bytes())
            self.assertEqual(Path(result["backup"]).read_bytes(), self.upstream_bytes())
            self.assertEqual(PATCH.patch_file(target, prefix, True)["status"], "ALREADY_RECORDED")

    def test_unknown_provider_bytes_are_refused_without_writes(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            target = self.fixture(prefix)
            target.write_bytes(b"unknown provider\n")
            with self.assertRaises(ValueError):
                PATCH.patch_file(target, prefix, True)
            self.assertEqual(target.read_bytes(), b"unknown provider\n")
            self.assertFalse(target.with_name(target.name + ".flashmotion-upstream.bak").exists())

    def test_target_outside_environment_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory)
            target = self.fixture(prefix)
            selected = prefix / "different-env"
            selected.mkdir()
            with self.assertRaises(ValueError):
                PATCH.patch_file(target, selected, True)
            self.assertEqual(target.read_bytes(), self.upstream_bytes())

    def test_runtime_patch_requires_activated_linux_conda_env(self):
        with patch.dict(PATCH.os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                PATCH.installed_target("t2m")

    def test_installed_target_refuses_wrong_git_revision_before_file_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "flashmotion-t2m"
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "conda-meta/history").touch()
            distribution = SimpleNamespace(version="2.2.6.post3", read_text=lambda name:
                '{"vcs_info":{"vcs":"git","commit_id":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}}',
                locate_file=lambda name: self.fail("Unknown Git revision must not select a patch target"))
            with patch.dict(PATCH.os.environ, {"CONDA_PREFIX": str(prefix)}), \
                 patch.object(PATCH.sys, "prefix", str(prefix)), \
                 patch.object(PATCH.sys, "platform", "linux"), \
                 patch.object(PATCH.metadata, "distribution", return_value=distribution):
                with self.assertRaises(ValueError):
                    PATCH.installed_target("t2m")

    def test_installed_target_selects_only_pinned_new_task_distribution(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "flashmotion-s2g"
            (prefix / "conda-meta").mkdir(parents=True)
            (prefix / "conda-meta/history").touch()
            distribution = SimpleNamespace(version="2.2.6.post3", read_text=lambda name:
                '{"vcs_info":{"vcs":"git","commit_id":"' + PATCH.COMMIT + '"}}',
                locate_file=lambda name: prefix / "lib/python3.10/site-packages" / name)
            with patch.dict(PATCH.os.environ, {"CONDA_PREFIX": str(prefix)}), \
                 patch.object(PATCH.sys, "prefix", str(prefix)), \
                 patch.object(PATCH.sys, "platform", "linux"), \
                 patch.object(PATCH.metadata, "distribution", return_value=distribution):
                target, selected = PATCH.installed_target("s2g")
                self.assertEqual(selected, prefix.resolve())
                self.assertTrue(target.is_relative_to(selected))
                with self.assertRaises(ValueError):
                    PATCH.installed_target("m2d")


if __name__ == "__main__":
    unittest.main()
