"""Stdlib-only T2M static/plan tests, not a GPU reproducibility test.

Run with: python -B -m unittest discover -s tests -p test_t2m_entrypoints.py -v
No model, checkpoint pickle, torch/numpy, network, or native inference is loaded.
No output directory, launch receipt, or bytecode is written by these tests.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
from pathlib import Path
import runpy
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "tasks/t2m/run.py"
CONFIG = ROOT / "configs/t2m.paths.example.json"
SNAPSHOTS = ROOT / "tasks/t2m/snapshots"


def read_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def named_function(tree: ast.Module, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name)


def calls(tree: ast.AST, text: str) -> list[ast.Call]:
    return [node for node in ast.walk(tree) if isinstance(node, ast.Call) and ast.unparse(node.func) == text]


def assert_keyword(case: unittest.TestCase, call: ast.Call, name: str, value: str) -> None:
    matches = [keyword for keyword in call.keywords if keyword.arg == name]
    case.assertEqual(len(matches), 1)
    case.assertEqual(ast.unparse(matches[0].value), value)


class T2MStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # run_name differs from __main__: do not call the CLI execution boundary.
        cls.entry = runpy.run_path(str(ENTRY), run_name="t2m_static_test_module")
        cls.snapshots = sorted({row[0] for row in cls.entry["PROFILES"].values()})

    def args(self, profile="full", mode="eval", **changes):
        values = dict(profile=profile, mode=mode, paths=str(CONFIG),
                      output=f"../../outputs/t2m_new_runs/static_test_{profile}_{mode}",
                      checkpoint="/external/trusted/selected.pt" if mode == "eval" else None,
                      python=None, gpus=None, port=None, steps=None,
                      eval_strategy=None, dry_run=False, execute=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_full_train_and_eval_plans_pin_seed_without_subprocess_or_output(self):
        with patch("subprocess.run", side_effect=AssertionError("No native execution in static tests")) as child:
            for profile in self.entry["PROFILES"]:
                for mode in ("train", "eval"):
                    with self.subTest(profile=profile, mode=mode):
                        plan = self.entry["build_plan"](self.args(profile, mode))
                        self.assertTrue(plan["dry_run"])
                        self.assertFalse(plan["executed"])
                        self.assertFalse(plan["resume"] or plan["retry"])
                        self.assertEqual(plan["entry_args"][plan["entry_args"].index("--seed") + 1], "42")
                        self.assertFalse(Path(plan["output"]).exists())
                        self.assertGreater(plan["source"]["verified_code_files"], 0)
                        if mode == "eval":
                            self.assertEqual(plan["entry_args"][plan["entry_args"].index("--limit") + 1], "0")
                            self.assertEqual(plan["entry_args"][plan["entry_args"].index("--embedding-batch-size") + 1], "32")
            child.assert_not_called()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_default_cli_is_dry_run_even_without_action_flag(self):
        stdout = io.StringIO()
        with patch("subprocess.run", side_effect=AssertionError("Unexpected subprocess")) as child:
            with contextlib.redirect_stdout(stdout):
                result = self.entry["main"](["--profile", "full", "--mode", "train"])
        self.assertEqual(result, 0)
        self.assertIn('"dry_run": true', stdout.getvalue())
        child.assert_not_called()

    def test_only_full_profile_is_exposed_and_excluded_profiles_refuse(self):
        self.assertEqual(set(self.entry["PROFILES"]), {"full"})
        self.assertEqual(set(self.entry["MANIFEST_SHA"]), {"restart_20260922"})
        excluded = ("no_fk", "uncond_r3", "mamba", "add", "adain", "add_scan",
                    "scan_only", "no_dynamic", "transformer_nonar",
                    "transformer_prefix", "flow_matching", "diffusion")
        with patch("subprocess.run", side_effect=AssertionError("Unexpected subprocess")) as child:
            for profile in excluded:
                with self.subTest(profile=profile):
                    with self.assertRaisesRegex(ValueError, "only the Full main-model"):
                        self.entry["build_plan"](self.args(profile))
                    with self.assertRaisesRegex(ValueError, "only the Full main-model"):
                        self.entry["frozen_snapshot"](profile)
                    with contextlib.redirect_stderr(io.StringIO()):
                        with self.assertRaises(SystemExit) as refusal:
                            self.entry["main"](["--profile", profile, "--mode", "eval"])
                    self.assertEqual(refusal.exception.code, 2)
            child.assert_not_called()
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_full_original_training_configuration_is_unchanged(self):
        plan = self.entry["build_plan"](self.args(mode="train"))
        args = plan["entry_args"]
        expected = {"--x0-variant": "tr_only", "--cross-modal-fusion": "adain",
                    "--direction-fusion": "sum", "--batch-size": "288",
                    "--max-length": "200", "--max-steps": "112000",
                    "--fm-weight": "1", "--rec-weight": "0.636",
                    "--vel-weight": "0.323", "--acc-weight": "0.2",
                    "--fk-weight": "0.646", "--floor-weight": "0",
                    "--mixed-precision": "bf16", "--ema-decay": ".9999",
                    "--seed": "42"}
        for option, value in expected.items():
            self.assertEqual(args[args.index(option) + 1], value)
        self.assertIn("--condition-selection", args)
        self.assertNotIn("--cfg-dropout-prob", args)
        self.assertEqual(plan["argv"][plan["argv"].index("--num_processes") + 1], "8")

    def test_frozen_train_seeding_precedes_random_decoder_initialization(self):
        for snapshot in self.snapshots:
            with self.subTest(snapshot=snapshot):
                path = SNAPSHOTS / snapshot / "code/train_t2m_long.py"
                tree = read_tree(path)
                _, defaults = self.entry["parser_options"](path)
                self.assertEqual(defaults["--seed"], 42)
                self.assertEqual(defaults["--ema-decay"], .9999)
                main = named_function(tree, "main")
                seeds = calls(main, "set_seed")
                self.assertEqual(len(seeds), 1)
                self.assertEqual(ast.unparse(seeds[0].args[0]), "args.seed")
                assert_keyword(self, seeds[0], "device_specific", "True")
                self.assertLess(seeds[0].lineno, calls(main, "OneShotT2MDecoder")[0].lineno)
                loader = calls(main, "DataLoader")[0]
                assert_keyword(self, loader, "worker_init_fn", "seed_worker")
                assert_keyword(self, loader, "generator", "loader_generator")
                assert_keyword(self, loader, "shuffle", "True")
                self.assertIn("torch.Generator().manual_seed(args.seed)", ast.unparse(main))
                worker = ast.unparse(named_function(tree, "seed_worker"))
                self.assertIn("torch.initial_seed() % 2 ** 32", worker)
                self.assertIn("np.random.seed(worker_seed)", worker)
                self.assertIn("random.seed(worker_seed)", worker)

    def test_frozen_eval_seed_chain_and_ema_selection(self):
        for snapshot in self.snapshots:
            with self.subTest(snapshot=snapshot):
                path = SNAPSHOTS / snapshot / "code/evaluate_t2m_ablation.py"
                tree = read_tree(path)
                _, defaults = self.entry["parser_options"](path)
                self.assertEqual(defaults["--seed"], 42)
                sample = ast.unparse(named_function(tree, "sample_episode"))
                self.assertIn("torch.Generator(device=device).manual_seed(seed)", sample)
                self.assertIn("objective.sample(condition, mask, steps, generator)", sample)
                main = ast.unparse(named_function(tree, "main"))
                self.assertIn("stable_seed(args.seed, variant, name, step)", main)
                self.assertIn("stable_seed(args.seed, variant, name, segment_index, 'single', step)", main)
                load = ast.unparse(named_function(tree, "load_objective"))
                self.assertIn("objective.load_state_dict(checkpoint['model'])", load)
                self.assertIn("if 'ema_decoder' in checkpoint:", load)
                self.assertIn("objective.decoder.load_state_dict(checkpoint['ema_decoder'])", load)
                self.assertIn("objective.eval().requires_grad_(False)", load)
                # EMA is conditional; this is NOT a runtime assertion that every input has EMA.

    def test_stable_seed_sha_algorithm_without_loading_benchmark_dependencies(self):
        expected_payloads = ((42, "tr_only", "episode", 4),
                             (42, "tr_only", "episode", 0, "single", 4),
                             (42, "tr_only", "episode", 1))
        values = []
        for snapshot in self.snapshots:
            with self.subTest(snapshot=snapshot):
                function = named_function(read_tree(SNAPSHOTS / snapshot / "code/t2m_long/babel_long_benchmark.py"), "stable_seed")
                isolated = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
                scope = {"hashlib": hashlib}
                # Execute only this audited stdlib SHA helper, not the source module/imports.
                exec(compile(isolated, "<isolated_stable_seed>", "exec"), scope)
                actual = []
                for payload in expected_payloads:
                    reference = int.from_bytes(hashlib.sha256("::".join(map(str, payload)).encode("utf-8")).digest()[:8], "little") % 2**31
                    actual.append(scope["stable_seed"](*payload))
                    self.assertEqual(actual[-1], reference)
                    self.assertEqual(scope["stable_seed"](*payload), actual[-1])
                self.assertEqual(len(set(actual)), len(actual))
                values.append(actual)
        self.assertTrue(all(value == values[0] for value in values))

    def test_metric_seed_and_exact_split_contract_are_static(self):
        for snapshot in self.snapshots:
            with self.subTest(snapshot=snapshot):
                code = SNAPSHOTS / snapshot / "code"
                benchmark = (code / "t2m_long/babel_long_benchmark.py").read_text(encoding="utf-8")
                self.assertIn("np.random.default_rng(seed)", benchmark)
                self.assertIn("torch.Generator().manual_seed(seed + repeat)", benchmark)
                data = (code / "t2m_long/data.py").read_text(encoding="utf-8")
                self.assertIn('"train": "train_processed.txt", "val": "val_processed.txt"', data)
                self.assertIn('"humanml3d", "train", max_length', data)
                self.assertIn('"babel", "train", max_length', data)
                evaluation = (code / "evaluate_t2m_ablation.py").read_text(encoding="utf-8")
                self.assertIn('"babel", "val", max_length=None', evaluation)
                self.assertIn("sorted(cohort_ids)", evaluation)
                main = ast.unparse(named_function(read_tree(code / "evaluate_t2m_ablation.py"), "main"))
                self.assertNotIn("set_seed(", main)  # explicit sampling/metric RNG, not global eval reseeding

    def test_sampling_randn_uses_explicit_generator(self):
        for snapshot in self.snapshots:
            with self.subTest(snapshot=snapshot):
                tree = read_tree(SNAPSHOTS / snapshot / "code/t2m_long/meanflow.py")
                for name in ("sample", "_ddim_sample"):
                    noises = calls(named_function(tree, name), "torch.randn")
                    self.assertEqual(len(noises), 1)
                    assert_keyword(self, noises[0], "generator", "generator")

    def test_full_1_and_4_are_separate_native_views_not_main_reencoding(self):
        for step in (1, 4):
            plan = self.entry["build_plan"](self.args("full", steps=[step]))
            index = plan["entry_args"].index("--steps")
            self.assertEqual(plan["entry_args"][index + 1], str(step))
            self.assertTrue(plan["argv"][1].endswith("evaluate_t2m_ablation.py"))
            self.assertNotIn("_ours_reencode_20260925.py", plan["display_command"])
            self.assertIn("full_source_binding", plan["source"])

    def test_safety_rejections_do_not_start_children(self):
        invalid = (self.args(output="/efs/original/live/output"),
                   self.args(mode="train", checkpoint="/external/selected.pt"),
                   self.args(mode="train", port=0),
                   self.args(gpus="0,0"), self.args(eval_strategy="streaming"),
                   self.args(port=29999), self.args(mode="train", steps=[4]),
                   self.args(steps=[0]), self.args(steps=[2]), self.args(steps=[4, 4]),
                   self.args(output=str(ROOT / "docs")))
        with patch("subprocess.run", side_effect=AssertionError("Unexpected subprocess")) as child:
            for args in invalid:
                with self.subTest(args=vars(args)):
                    with self.assertRaises(ValueError):
                        self.entry["build_plan"](args)
            plan = self.entry["build_plan"](self.args(execute=True))
            self.assertTrue(plan["missing_inputs"])
            with self.assertRaises(ValueError):
                self.entry["execute_plan"](plan)
            child.assert_not_called()


class MainReproductionStaticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runner = runpy.run_path(str(ROOT / "tasks/t2m/reproduce_main.py"), run_name="main_reproduction_static_test")

    def args(self, nfe=1, **changes):
        values = dict(paths=str(CONFIG), nfe=nfe, checkpoint=f"/external/trusted/step-{self.runner['SELECTED'][nfe]['step']:07d}.pt",
                      checkpoint_sha256=self.runner["SELECTED"][nfe]["checkpoint_sha256"],
                      canonical="/external/authenticated/canonical_gt.npz",
                      canonical_sha256=self.runner["CANONICAL_SHA"],
                      historical_source=str(ROOT / "tasks/t2m/reference/ours_reencode_20260925.py"),
                      code_root=None, output=f"../../outputs/t2m_new_runs/main_static_nfe{nfe}",
                      gpu="7", cap=.08, seed=42, dry_run=False, execute=False)
        values.update(changes)
        return argparse.Namespace(**values)

    def test_two_main_plans_are_fresh_and_do_not_import_native_dependencies(self):
        for nfe, step, epoch in ((1, 100100, 7150), (4, 81900, 5850)):
            with self.subTest(nfe=nfe):
                plan = self.runner["build_plan"](self.args(nfe))
                self.assertTrue(plan["dry_run"])
                self.assertFalse(plan["executed"])
                self.assertFalse(plan["reuse_raw"] or plan["resume"] or plan["retry"])
                self.assertEqual(plan["selected"]["step"], step)
                self.assertEqual(plan["selected"]["epoch"], epoch)
                self.assertEqual(plan["gpu"], "7")
                self.assertEqual(plan["seed"], 42)
                self.assertEqual(plan["precision"], {"generation_matmul": "high", "embedding_matmul": "highest", "embedding_batch": 32})
                self.assertFalse(Path(plan["output"]).exists())
                self.assertTrue(plan["missing_inputs"])
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_main_safety_seed_checkpoint_canonical_cap_and_fresh_output(self):
        invalid = (self.args(seed=43), self.args(cap=0), self.args(cap=.08001),
                   self.args(cap=float("nan")), self.args(cap=float("inf")),
                   self.args(checkpoint_sha256="0" * 64), self.args(canonical_sha256="0" * 64),
                   self.args(gpu="0,7"), self.args(output="/efs/yangkaixing/FlowerDance_T2M_Long/eval/new"),
                   self.args(output=str(ROOT / "docs")))
        for args in invalid:
            with self.subTest(args=vars(args)):
                with self.assertRaises(ValueError):
                    self.runner["build_plan"](args)
        plan = self.runner["build_plan"](self.args(execute=True))
        with self.assertRaises(ValueError):
            self.runner["execute"](plan)
        self.assertNotIn("torch", sys.modules)

    def test_main_source_uses_authentic_kernels_not_historical_controller(self):
        path = ROOT / "tasks/t2m/reproduce_main.py"
        tree = read_tree(path)
        execution = named_function(tree, "execute")
        source = ast.unparse(execution)
        self.assertIn("kernels.stable_seed(42, variant, name, plan['nfe'])", source)
        self.assertIn("kernels.stable_seed(42, variant, name, segment_index, 'single', plan['nfe'])", source)
        self.assertIn("torch.set_float32_matmul_precision('high')", source)
        self.assertIn("torch.set_float32_matmul_precision('highest')", source)
        self.assertIn("MultiPredictionEmbeddingCollector(evaluator, ['single', 'multi'], 32)", source)
        self.assertIn("np.array_equal(embeddings[key], reference[key])", source)
        self.assertIn("torch.device('cpu'), seed=42", source)
        self.assertNotIn("subprocess", source)
        self.assertNotIn("nvidia-smi", source)
        self.assertNotIn("raw.exists", source)
        self.assertNotIn("metrics.json').exists", source)
        self.assertEqual(len(calls(execution, "np.load")), 1)  # canonical only; not cached generation
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


if __name__ == "__main__":
    unittest.main()
