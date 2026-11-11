"""Stdlib AST/installation checks, not a substitute for native model replay."""
import __future__
import ast
import builtins
import copy
import functools
import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("direct_sampling_under_test", ROOT / "tasks/t2m/sampling.py")
sampling = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(sampling)
FROZEN = ROOT / "tasks/t2m/snapshots/restart_20260922/code/t2m_long/meanflow.py"


def no_grad():
    def decorate(function):
        @functools.wraps(function)
        def wrapped(*args, **kwargs):
            return function(*args, **kwargs)
        return wrapped
    return decorate


def objective_fixture():
    tree = ast.parse(FROZEN.read_text(encoding="utf-8"))
    sampler = next(node for cls in tree.body if isinstance(cls, ast.ClassDef)
                   for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "sample")
    module = ast.Module(body=[sampler], type_ignores=[])
    native_imports = SimpleNamespace(no_grad=no_grad, Tensor=object, Generator=object)
    namespace = {"torch": native_imports}
    # Annotation strings only; do not import or execute the frozen native module.
    import __future__
    exec(compile(module, str(FROZEN), "exec", flags=__future__.annotations.compiler_flag), namespace)

    class Objective:
        x0_variant = "tr_only"
        sample = namespace["sample"]

        def x_r_and_velocity(self, *args):
            raise AssertionError("Training helper must not execute in this AST test")

    return Objective(), sampler


class DirectT2MSamplingTests(unittest.TestCase):
    def test_exact_single_inference_assignment_replaced_training_untouched(self):
        before = hashlib.sha256(FROZEN.read_bytes()).hexdigest()
        objective, original_ast = objective_fixture()
        training_helper = type(objective).x_r_and_velocity
        original_class_sampler = type(objective).sample
        evidence = sampling.install_direct_sampler(objective)
        self.assertEqual(evidence["replaced_assignments"], 1)
        self.assertTrue(evidence["interpolation_operation_order_unchanged"])
        self.assertIs(type(objective).x_r_and_velocity, training_helper)
        self.assertIs(type(objective).sample, original_class_sampler)
        self.assertNotIn("x_r_and_velocity", objective.sample.__wrapped__.__code__.co_names)
        self.assertIn("clamp_min", objective.sample.__wrapped__.__code__.co_names)
        self.assertEqual(before, hashlib.sha256(FROZEN.read_bytes()).hexdigest())
        matches = [node for node in ast.walk(original_ast) if isinstance(node, ast.Assign)
                   and ast.dump(node) == ast.dump(sampling.EXPECTED_CALL)]
        self.assertEqual(len(matches), 1)

    def test_refuse_non_main_variant_and_repeat_install(self):
        objective, _ = objective_fixture()
        objective.x0_variant = "diffusion"
        with self.assertRaises(ValueError):
            sampling.install_direct_sampler(objective)
        objective.x0_variant = "tr_only"
        sampling.install_direct_sampler(objective)
        with self.assertRaises(ValueError):
            sampling.install_direct_sampler(objective)

    def test_future_annotations_and_original_no_grad_decorator_retained(self):
        objective, original_ast = objective_fixture()
        original_annotations = objective.sample.__wrapped__.__annotations__.copy()
        sampling.install_direct_sampler(objective)
        generated = objective.sample.__wrapped__
        self.assertTrue(generated.__code__.co_flags & __future__.annotations.compiler_flag)
        self.assertEqual(generated.__annotations__, original_annotations)
        self.assertEqual(generated.__annotations__, {
            "condition": "torch.Tensor", "mask": "torch.Tensor", "steps": "int",
            "generator": "torch.Generator | None", "return": "torch.Tensor",
        })
        self.assertTrue(all(isinstance(value, str) for value in generated.__annotations__.values()))
        self.assertEqual(ast.unparse(original_ast.decorator_list[0]), "torch.no_grad()")
        self.assertIs(objective.sample.__self__, objective)

    def test_whole_sampler_ast_diff_is_only_the_requested_two_assignments(self):
        objective, original_ast = objective_fixture()
        expected = copy.deepcopy(original_ast)
        loop = next(node for node in ast.walk(expected) if isinstance(node, ast.For)
                    and ast.unparse(node.iter) == "range(steps)")
        positions = [index for index, node in enumerate(loop.body)
                     if isinstance(node, ast.Assign) and ast.dump(node) == ast.dump(sampling.EXPECTED_CALL)]
        self.assertEqual(len(positions), 1)
        # Construct the expected delta independently, rather than reusing the
        # installer's replacement AST. Every other node, including no_grad,
        # RNG/model calls, masks, branches and endpoints, must remain identical.
        replacement = ast.parse(
            "ratio = (r[:, None, None] / t[:, None, None].clamp_min(1e-5)).clamp(0.0, 1.0)\n"
            "x = ratio * x + (1.0 - ratio) * x0"
        ).body
        offset = positions[0]
        loop.body[offset:offset + 1] = replacement
        with patch("builtins.compile", wraps=builtins.compile) as compiled:
            sampling.install_direct_sampler(objective)
        generated = [call.args[0] for call in compiled.call_args_list
                     if call.args and isinstance(call.args[0], ast.Module)
                     and len(call.args) > 2 and call.args[2] == "exec"]
        self.assertEqual(len(generated), 1)
        self.assertEqual(len(generated[0].body), 1)
        self.assertEqual(ast.dump(generated[0].body[0]), ast.dump(expected))

    def test_algebra_and_additive_epsilon_are_distinct(self):
        xt, x0, t, r = 3.0, -2.0, 0.5, 0.25
        direct = (r / t) * xt + (1.0 - r / t) * x0
        euler = xt - (t - r) * ((xt - x0) / t)
        self.assertEqual(direct, euler)
        eps = 0.01
        regularized = xt - (t - r) * ((xt - x0) / (t + eps))
        equivalent = ((r + eps) / (t + eps)) * xt + ((t - r) / (t + eps)) * x0
        self.assertAlmostEqual(regularized, equivalent)
        self.assertNotEqual(direct, regularized)


if __name__ == "__main__":
    unittest.main()
