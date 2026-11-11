"""Stdlib-only guards; these tests do not emulate or execute ML inference."""
import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("s2g_infer_under_test", ROOT / "tasks/s2g/infer.py")
infer = importlib.util.module_from_spec(SPEC)
MODULES_BEFORE = set(sys.modules)
SPEC.loader.exec_module(infer)


class S2GInferenceTests(unittest.TestCase):
    def argv(self):
        args = []
        for name in ("checkpoint", "config", "normalizer", "manifest", "source-proof", "cache-integrity"):
            args.extend(["--" + name, str(ROOT / (name + ".fixture")), "--" + name + "-sha256", "a" * 64])
        for name in ("source-dir", "cache-root", "raw-root", "metric-assets", "evaluator-assets", "output"):
            args.extend(["--" + name, str(ROOT / (name + ".fixture"))])
        return args + ["--seed", "42", "--steps", "4", "--expected-epoch", "1500"]

    def test_import_and_default_plan_have_no_native_imports_or_execution(self):
        native = ("torch", "numpy", "scipy", "mamba_ssm", "causal_conv1d", "model", "loss", "data", "evaluation")
        self.assertFalse(any(name.split(".")[0] in native for name in set(sys.modules) - MODULES_BEFORE))
        for extra in ([], ["--dry-run"]):
            with mock.patch.object(infer, "make_plan", return_value={"status": "fixture_plan_only"}), \
                    mock.patch.object(infer, "execute", side_effect=AssertionError("execution forbidden")), \
                    contextlib.redirect_stdout(io.StringIO()) as output:
                self.assertEqual(infer.main(self.argv() + extra), 0)
                self.assertEqual(json.loads(output.getvalue())["status"], "fixture_plan_only")
        self.assertFalse(any(name.split(".")[0] in native for name in set(sys.modules) - MODULES_BEFORE))

    def test_execute_is_explicit_and_conflicts_with_dry_run(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            infer.parser().parse_args(self.argv() + ["--execute", "--dry-run"])
        with mock.patch.object(infer, "make_plan", return_value={"status": "fixture"}), \
                mock.patch.object(infer, "execute", return_value={"status": "mock_boundary_only"}) as execute, \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(infer.main(self.argv() + ["--execute"]), 0)
            execute.assert_called_once()

    def test_only_seed42_and_paper_step_counts(self):
        for flag, value in (("--seed", "43"), ("--steps", "2")):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                infer.parser().parse_args(self.argv() + [flag, value])
        for value in ("0", "-1", "0.081", "nan", "inf"):
            with self.assertRaises(argparse.ArgumentTypeError):
                infer.fraction_arg(value)
        self.assertEqual(infer.fraction_arg("0.08"), 0.08)
        self.assertEqual(infer.fraction_arg("0.04"), 0.04)

    def test_hash_and_seed_exactness(self):
        ident = "1_wayne_0_10_10"
        self.assertEqual(infer.clip_seed(ident), 42 + int(hashlib.sha256(ident.encode()).hexdigest()[:8], 16))
        self.assertEqual(infer.digest_arg("A" * 64), "a" * 64)
        for value in ("a" * 63, "g" * 64):
            with self.assertRaises(argparse.ArgumentTypeError):
                infer.digest_arg(value)
        value = {"b": 2, "a": 1}
        self.assertEqual(infer.canonical_hash(value), hashlib.sha256(b'{"a":1,"b":2}').hexdigest())

    def test_frozen_source_receipt_and_exact_function_ast(self):
        proof = ROOT / "manifests/frozen_source_collection.json"
        source = ROOT / infer.SOURCE_PREFIX
        verified = infer.verify_sources(source, proof, infer.sha256(proof))
        self.assertEqual(set(verified), set(infer.SOURCE_FILES))
        tree = ast.parse((source / "train.py").read_text(encoding="utf-8"))
        expected = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        selected = infer.native_ast(source / "train.py")
        self.assertEqual([node.name for node in selected.body], list(infer.FUNCTIONS))
        for node in selected.body:
            self.assertEqual(ast.dump(node), ast.dump(expected[node.name]))
        compile(selected, str(source / "train.py"), "exec")  # compile, never execute
        with self.assertRaises(ValueError):
            infer.verify_sources(source, proof, "0" * 64)
        with tempfile.TemporaryDirectory(prefix="s2g-proof-fixture-") as directory:
            changed = json.loads(proof.read_text(encoding="utf-8"))
            changed["files"] = [record for record in changed["files"] if record["path"] != infer.SOURCE_PREFIX + "model.py"]
            target = Path(directory) / "fixture-proof.json"
            target.write_text(json.dumps(changed), encoding="utf-8")
            with self.assertRaises(ValueError):
                infer.verify_sources(source, target, infer.sha256(target))

    def test_fresh_output_refuses_overwrite_and_input_overlap(self):
        with tempfile.TemporaryDirectory(prefix="s2g-output-fixture-") as directory:
            root = Path(directory)
            protected = root / "cache"
            protected.mkdir()
            self.assertEqual(infer.fresh_output(root / "new-test", [protected]), root / "new-test")
            for path in (root, protected, protected / "new", root / "missing-parent" / "new"):
                with self.assertRaises(ValueError):
                    infer.fresh_output(path, [protected])
        with self.assertRaises(ValueError):
            infer.absolute("relative/path")


if __name__ == "__main__":
    unittest.main()
