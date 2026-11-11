"""Inference-only removal of an unused velocity from the frozen Full sampler.

The original training/model source stays byte-preserved. We replace precisely
one assignment in a copy of its sampling AST, keeping RNG, masks, grid, model
calls and the original interpolation operation order. No Torch import here.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from types import MethodType


EXPECTED_CALL = ast.parse(
    "x, _ = self.x_r_and_velocity(x, x0, t[:, None, None], r[:, None, None])"
).body[0]
DIRECT_UPDATE = ast.parse(
    "ratio = (r[:, None, None] / t[:, None, None].clamp_min(1e-5)).clamp(0.0, 1.0)\n"
    "x = ratio * x + (1.0 - ratio) * x0"
).body


def install_direct_sampler(objective) -> dict:
    """Install one Full inference sampler, never replace the training helper."""
    if objective.x0_variant != "tr_only":
        raise ValueError("Only the two selected Full tr_only main models are supported")
    if "sample" in vars(objective):
        raise ValueError("Sampler already overridden; refuse repeated installation")
    wrapped = type(objective).sample
    original = wrapped.__wrapped__
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != "sample":
        raise ValueError("Unexpected frozen sampler structure")
    matches = 0

    class ReplaceUnusedVelocity(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal matches
            if ast.dump(node) == ast.dump(EXPECTED_CALL):
                matches += 1
                return [ast.copy_location(ast.parse(ast.unparse(item)).body[0], node)
                        for item in DIRECT_UPDATE]
            return self.generic_visit(node)

    tree = ReplaceUnusedVelocity().visit(tree)
    if matches != 1:
        raise ValueError("Expected exactly one frozen x_r/unused-velocity assignment")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    exec(compile(tree, inspect.getsourcefile(original), "exec"), namespace)
    objective.sample = MethodType(namespace["sample"], objective)
    return {
        "sampler": "direct_x_r_without_unused_velocity",
        "original_sample_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "generated_sample_ast_sha256": hashlib.sha256(ast.dump(tree).encode()).hexdigest(),
        "replaced_assignments": matches,
        "training_helper_unchanged": True,
        "rng_grid_masks_model_calls_unchanged": True,
        "interpolation_operation_order_unchanged": True,
    }
