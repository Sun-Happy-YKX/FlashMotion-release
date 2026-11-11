"""Stdlib-only M2D sampler/source checks; no native model replay or timing.

Run: python -B -m unittest discover -s tests -p test_m2d_direct_sampling.py -v
The frozen scorer and model are parsed as data, never imported or executed.
"""
from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("m2d_adapter_under_test", ROOT / "tasks/m2d/adapter.py")
adapter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(adapter)
SNAPSHOT = ROOT / adapter.SNAPSHOT
SCORER = SNAPSHOT / "provenance/music_corrected_eval_20260930.py"
MODEL = SNAPSHOT / "code/model/flowmatching.py"
OLD_UPDATE = "y,_=FlowMatching._x_r_and_velocity(y,y0,t[:,None,None],r[:,None,None])"
DIRECT_UPDATE = (
    "ratio=(r[:,None,None]/t[:,None,None].clamp_min(1e-5)).clamp(0.0,1.0);"
    "y=ratio*y+(1.0-ratio)*y0"
)
SAMPLER_REASON = "inference-only omit unused velocity; same direct interpolation"


def function(tree: ast.AST, name: str) -> ast.FunctionDef:
    return next(node for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef) and node.name == name)


def calls(tree: ast.AST, name: str) -> list[ast.Call]:
    return [node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == name]


def replay_spans(original: bytes, records: list[dict]) -> bytes:
    """Independent byte splice; records refer to original, not shifted offsets."""
    generated = original
    for record in reversed(records):
        start, end = record["start_byte"], record["end_byte"]
        if original[start:end].decode("utf-8") != record["before"]:
            raise AssertionError("Span does not describe the original bytes")
        generated = generated[:start] + record["after"].encode("utf-8") + generated[end:]
    return generated


class DirectM2DSamplingTests(unittest.TestCase):
    def setUp(self):
        self.original = SCORER.read_bytes()
        self.selection = {"version": "genre", "epoch": 2250, "n": 1,
                          "checkpoint": "/models/main/train-2250.pt", "sha256": "a" * 64}

    def adapt(self, selection=None):
        return adapter.adapt_eval(self.original, "/work/independent-eval", "/datasets/m2d",
                                  self.selection if selection is None else selection)

    def test_only_one_additional_exact_sampler_span_for_both_main_nfe(self):
        for epoch, nfe in ((2250, 1), (2100, 4)):
            with self.subTest(epoch=epoch, nfe=nfe):
                selection = dict(self.selection, epoch=epoch, n=nfe,
                                 checkpoint=f"/models/main/train-{epoch}.pt")
                generated, records = self.adapt(selection)
                self.assertEqual(len(records), 8)  # Seven existing relocations/seed/scope spans.
                sampler = [item for item in records if item["reason"] == SAMPLER_REASON]
                self.assertEqual(len(sampler), 1)
                self.assertEqual(sampler[0]["before"], OLD_UPDATE)
                self.assertEqual(sampler[0]["after"], DIRECT_UPDATE)
                self.assertEqual(replay_spans(self.original, records), generated)
                previous = replay_spans(self.original, [item for item in records
                                                       if item["reason"] != SAMPLER_REASON])
                self.assertEqual(previous.count(OLD_UPDATE.encode()), 1)
                self.assertEqual(previous.replace(OLD_UPDATE.encode(), DIRECT_UPDATE.encode(), 1), generated)
                self.assertEqual(SCORER.read_bytes(), self.original)

    def test_direct_arithmetic_is_identical_to_native_x_r_operation_order(self):
        generated, _ = self.adapt()
        main = function(ast.parse(generated), "main")
        loop = next(node for node in ast.walk(main) if isinstance(node, ast.For)
                    and isinstance(node.target, ast.Name) and node.target.id == "step")
        expected = ast.parse(DIRECT_UPDATE).body
        self.assertEqual([ast.dump(node) for node in loop.body[-2:]],
                         [ast.dump(node) for node in expected])
        self.assertEqual(calls(main, "FlowMatching._x_r_and_velocity"), [])
        self.assertEqual(calls(main, "torch.lerp"), [])
        self.assertEqual(ast.unparse(loop.iter), "range(n)")
        native = function(ast.parse(MODEL.read_bytes()), "_x_r_and_velocity")
        expected_native = ast.parse(
            "t_safe=t.clamp_min(eps)\n"
            "ratio=(r/t_safe).clamp(0.0,1.0)\n"
            "x_r=ratio*x_t+(1.0-ratio)*x0_pred"
        ).body
        assignments = [node for node in native.body if isinstance(node, ast.Assign)]
        self.assertEqual([ast.dump(node) for node in assignments[:3]],
                         [ast.dump(node) for node in expected_native])
        self.assertEqual(native.args.defaults[0].value, 1e-5)
        self.assertEqual([target.id for node in assignments for target in node.targets],
                         ["t_safe", "ratio", "x_r", "interval", "interval_velocity"])
        self.assertEqual(ast.unparse(native.body[-1]), "return (x_r, interval_velocity)")

    def test_noise_model_time_grid_postprocessing_and_assertions_unchanged(self):
        generated, _ = self.adapt()
        before, after = function(ast.parse(self.original), "main"), function(ast.parse(generated), "main")
        # These cover the same RNG draw, batch/grid/endpoints, conditioning,
        # native clamp/normalizer and skeletal conversion after all NFE steps.
        for name in ("torch.randn", "torch.linspace", "torch.full", "torch.full_like",
                     "model.infer_pred", "normalizer.unnormalize", "pose_to_root_and_rotations",
                     "rotate_pred_like_gt", "smpl.forward", "model.load_state_dict",
                     "model.cuda", "torch.cuda.set_per_process_memory_fraction", "fcntl.flock"):
            with self.subTest(call=name):
                self.assertEqual([ast.dump(node) for node in calls(before, name)],
                                 [ast.dump(node) for node in calls(after, name)])
        self.assertEqual([ast.dump(node) for node in ast.walk(before) if isinstance(node, ast.Assert)],
                         [ast.dump(node) for node in ast.walk(after) if isinstance(node, ast.Assert)])
        self.assertIn(b"y0=model.infer_pred(y,cond,genre,t,h);" + DIRECT_UPDATE.encode(), generated)
        self.assertIn(b"y=y.cpu();assert torch.isfinite(y).all();clips.append(float((y.abs()>1).float().mean()))", generated)

    def test_existing_paths_seed_scope_and_metric_transforms_remain(self):
        generated, records = self.adapt()
        before, after = ast.parse(self.original), ast.parse(generated)
        for name in ("kinetic", "geo_chunk", "extract", "fid", "diversity", "pfc", "rank_all"):
            self.assertEqual(ast.dump(function(before, name)), ast.dump(function(after, name)))
        self.assertIn(b"C=Path('/work/independent-eval/code')", generated)
        self.assertIn(b"S=Path('/datasets/m2d')", generated)
        self.assertIn(b"OUT=R/'results'", generated)
        self.assertIn(b"plan=[{'version': 'genre', 'epoch': 2250, 'n': 1, 'checkpoint': '/models/main/train-2250.pt'}]", generated)
        self.assertNotIn(b"glob('train-*.pt')", generated)
        self.assertEqual(calls(after, "rank_all"), [])
        for name, expected_count in (("random.seed", 1), ("np.random.seed", 1),
                                     ("torch.manual_seed", 2), ("torch.cuda.manual_seed_all", 1),
                                     ("torch.cuda.manual_seed", 1)):
            matched = calls(function(after, "main"), name)
            self.assertEqual(len(matched), expected_count, name)
            self.assertTrue(all(ast.dump(node.args[0]) == ast.dump(ast.Constant(42)) for node in matched), name)
        self.assertEqual(sum(item["reason"].startswith("Explicit independent Python/NumPy/Torch seed42")
                             for item in records), 1)
        self.assertIn(b"free>=30*1024**3", generated)
        self.assertIn(b"fcntl.LOCK_EX|fcntl.LOCK_NB", generated)

    def test_all_formal_frozen_model_and_training_sources_keep_pinned_bytes(self):
        formal_payload = (SNAPSHOT / "provenance/formal_code_sha256.json").read_bytes()
        self.assertEqual(adapter.sha(formal_payload), adapter.FORMAL_SHA)
        before = {relative: (SNAPSHOT / "code" / relative).read_bytes()
                  for relative in json.loads(formal_payload)}
        self.adapt()
        for relative, expected in json.loads(formal_payload).items():
            with self.subTest(source=relative):
                after = (SNAPSHOT / "code" / relative).read_bytes()
                self.assertEqual(before[relative], after)
                self.assertEqual(adapter.sha(after), expected)
        self.assertEqual(adapter.sha(self.original), adapter.EVAL_SHA)
        train, records = adapter.adapt_train(before["train_pair.py"], "/datasets/m2d")
        self.assertEqual(len(records), 1)
        self.assertEqual(train, before["train_pair.py"].replace(adapter.OLD_PROJECT.encode(),
                                                             b"/datasets/m2d", 1))
        self.assertNotIn("torch", sys.modules)
        self.assertNotIn("numpy", sys.modules)

    def test_wrong_source_and_invalid_selections_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "repaired scorer"):
            adapter.adapt_eval(self.original + b"\n", "/work/new", "/datasets/m2d", self.selection)
        for field, value in (("n", 2), ("n", True), ("version", "other"), ("epoch", 0),
                             ("sha256", "not-a-sha")):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.adapt(dict(self.selection, **{field: value}))

    def test_sampler_shape_guard_rejects_missing_duplicate_changed_loop_and_collision(self):
        cases = {
            "missing": self.original.replace(OLD_UPDATE.encode(), b"y=y0", 1),
            "duplicate": self.original.replace(OLD_UPDATE.encode(), (OLD_UPDATE + ";" + OLD_UPDATE).encode(), 1),
            "changed-loop": self.original.replace(b"for step in range(n):", b"for step in range(n+1):", 1),
            "ratio-collision": self.original.replace(b"pred=[];clips=[];", b"ratio=1;pred=[];clips=[];", 1),
        }
        for label, payload in cases.items():
            with self.subTest(label=label), patch.object(adapter, "EVAL_SHA", adapter.sha(payload)):
                with self.assertRaisesRegex(ValueError, "sampling update"):
                    adapter.adapt_eval(payload, "/work/new", "/datasets/m2d", self.selection)


if __name__ == "__main__":
    unittest.main()
