"""Fail-closed real-CUDA diagnostic for Music2Dance MeanFlow derivatives.

This is NOT a trainer or a replacement production backend. Native standard
(fused) and externally conditioned (unfused) Mamba blocks are compared against
the *same weights* evaluated with differentiable torch depthwise causal conv
and the original selective-scan recurrence. Central finite differences are an
independent DIAGNOSTIC ONLY, never substituted into the MeanFlow objective.

Example: python native_jvp_check.py --device cuda:0 --task-name <task>
         --output /absolute/evidence/native_jvp_check.json

CPU/stub/finite-loss checks cannot produce ``passed: true`` in this script.
An evidence consumer must also verify the Python and imported extension
binary hashes match the deployed runtime, even when package versions match.
"""
from __future__ import annotations

import argparse
import copy
import gc
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from importlib.machinery import EXTENSION_SUFFIXES
import json
from pathlib import Path
import socket
import sys
import traceback
import types

import torch
from torch import nn
from torch.nn import functional as F

_MODEL_IMPORT_ERROR = None
try:
    from model import ConditionalSelectiveMamba, GestureM2D
except Exception as error:
    # An unknown native runtime must still produce failed provenance rather
    # than crashing before the diagnostic can save its evidence artifact.
    ConditionalSelectiveMamba = GestureM2D = None
    _MODEL_IMPORT_ERROR = (error, traceback.format_exc())


EXTENDED_NATIVE_CHECKS = (
    "adjoint_linearity_six_cases",
    "softplus_nonzero_state_zero_limit",
    "full_width512_frames240",
    "fixed_dropout_meanflow_target_parameter_gradients",
)


def require_extended_native_checks(proof):
    """Shared read-only fail-closed consumer, not a replacement GPU check."""
    checks = proof.get("extended_checks", {})
    if set(checks) != set(EXTENDED_NATIVE_CHECKS):
        raise RuntimeError("Native evidence lacks the exact mandatory extended-check set")
    if proof.get("kind") != "real_gpu" or proof.get("cuda_available") is not True:
        raise RuntimeError("Extended native checks require actual CUDA evidence")
    for name in EXTENDED_NATIVE_CHECKS:
        if checks[name].get("passed") is not True or checks[name].get("kind") != "real_gpu":
            raise RuntimeError(f"Actual extended native check has not passed: {name}")
    expected = ("standard_mamba_fused", "conditional_mamba_external_delta_B_C",
                "selective_scan_kernel", "causal_conv_kernel",
                "full55_4condition_8motion_meanflow_velocity", "body22_4condition_8motion_meanflow_velocity")
    cases = proof.get("cases", [])
    if tuple(case.get("name") for case in cases) != expected or any(case.get("passed") is not True for case in cases):
        raise RuntimeError("All original six native cases must independently pass")
    invariant = checks[EXTENDED_NATIVE_CHECKS[0]]
    if invariant.get("case_names") != list(expected) or any(case.get("derivative_invariants", {}).get("passed") is not True for case in cases):
        raise RuntimeError("Native adjoint/linearity does not cover all six original cases")
    mixed = checks[EXTENDED_NATIVE_CHECKS[1]]
    if mixed.get("nonzero_history_prefix_frames", 0) < 2 or mixed.get("delta_arguments") != [-200., -90., -80.]:
        raise RuntimeError("Zero/subnormal step audit must follow nonzero state history")
    if mixed.get("case", {}).get("passed") is not True:
        raise RuntimeError("Zero/subnormal native scan case did not pass")
    for name in EXTENDED_NATIVE_CHECKS[2:]:
        group = checks[name]
        if group.get("dimensions") != {"width": 512, "frames": 240, "batch": 1}:
            raise RuntimeError("Extended native audit must use actual width512/T240/batch1")
        if set(group.get("modes", {})) != {"full55", "body22"}:
            raise RuntimeError("Extended native audit must cover both pose modes")
        if any(item.get("passed") is not True for item in group["modes"].values()):
            raise RuntimeError("A full-size native pose-mode case failed")
        if group.get("actual_train_cache") is not True:
            raise RuntimeError("Full-size native audit must bind actual prepared train inputs")
    objective = checks[EXTENDED_NATIVE_CHECKS[3]]
    if objective.get("dropout") != .1 or objective.get("objective") != "original_loss_py_tr_only_unmodified":
        raise RuntimeError("Dropout/MeanFlow audit changed the actual production objective")
    return checks


def source_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _causal_extension_from_interface(interface):
    """Follow imported cpp_functions wrappers to their real extension object."""
    wrapper_names = ("causal_conv1d_fwd_function", "causal_conv1d_bwd_function",
                     "causal_conv1d_update_function")
    wrappers = {name: getattr(interface, name) for name in wrapper_names
                if hasattr(interface, name)}
    extensions = []
    if wrappers:
        # New causal-conv1d interfaces import Python wrappers, not the .so.
        # The wrappers share cpp_functions globals with the decorated custom
        # ops whose bodies call this extension. Require both exercised paths.
        for required in wrapper_names[:2]:
            if required not in wrappers:
                raise RuntimeError(f"Imported causal interface is missing wrapper: {required}")
        for name, wrapper in wrappers.items():
            if not isinstance(wrapper, types.FunctionType):
                raise RuntimeError(f"Imported causal wrapper is not a Python function: {name}")
            extension = wrapper.__globals__.get("causal_conv1d_cuda")
            if not isinstance(extension, types.ModuleType):
                raise RuntimeError(f"Imported causal wrapper has no extension in its globals: {name}")
            extensions.append(extension)
    # Older interfaces call a directly imported extension. If a direct alias
    # also exists in a wrapper-based interface it must point to the same .so.
    if hasattr(interface, "causal_conv1d_cuda"):
        extension = interface.causal_conv1d_cuda
        if not isinstance(extension, types.ModuleType):
            raise RuntimeError("Imported causal interface has no extension module: causal_conv1d_cuda")
        extensions.append(extension)
    if not extensions:
        raise RuntimeError("Imported causal interface has no actual causal_conv1d_cuda call path")
    if any(extension is not extensions[0] for extension in extensions[1:]):
        raise RuntimeError("Imported causal wrapper/direct paths use mixed extension objects")
    return extensions[0]


def kernel_binary_bindings(scan_module, conv_module):
    """Hash the extension objects actually used by the imported interfaces.

    Do not search wheels/site-packages or import a guessed alternative: that
    could bind an unused binary rather than the kernels exercised below.
    Consumers can compare this complete dictionary with the saved evidence.
    This is a read-only inventory helper, never a GPU correctness check.
    """
    scan_causal = _causal_extension_from_interface(scan_module)
    conv_causal = _causal_extension_from_interface(conv_module)
    if scan_causal is not conv_causal:
        raise RuntimeError("Scan and causal interfaces use different causal_conv1d_cuda objects")
    bindings = {}
    for role, extension in (("selective_scan_cuda", getattr(scan_module, "selective_scan_cuda", None)),
                            ("causal_conv1d_cuda", conv_causal)):
        if not isinstance(extension, types.ModuleType):
            raise RuntimeError(f"Imported kernel interface has no extension module: {role}")
        loaded_path = getattr(extension, "__file__", None)
        if not loaded_path or not str(loaded_path).endswith(tuple(EXTENSION_SUFFIXES)):
            raise RuntimeError(f"Actual kernel module is not a compiled extension: {role}")
        binary = Path(loaded_path).resolve(strict=True)
        if not binary.is_file():
            raise RuntimeError(f"Actual kernel extension is not a file: {role}")
        bindings[role] = dict(module=extension.__name__, path=str(binary),
                              sha256=source_hash(binary))
    return bindings


def selective_scan_torch(u, delta, a, b, c, d=None, z=None,
                         delta_bias=None, delta_softplus=True):
    """Original real-valued Mamba recurrence, with frame-variable B/C.

    Supports exactly the representation this backbone uses: u/delta [B,D,T],
    A [D,N], B/C [B,N,T]. Deliberately no custom CUDA/autograd Functions.
    """
    if a.is_complex() or b.ndim != 3 or c.ndim != 3:
        raise ValueError("This diagnostic covers real A and variable [B,N,T] B/C only")
    if u.ndim != 3 or u.shape != delta.shape or a.ndim != 2:
        raise ValueError("Invalid selective scan diagnostic shapes")
    dtype = u.dtype
    u, delta, b, c = u.float(), delta.float(), b.float(), c.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    state = a.new_zeros(u.shape[0], a.shape[0], a.shape[1])
    outputs = []
    for frame in range(u.shape[-1]):
        step = delta[:, :, frame]
        transition = torch.exp(step[:, :, None] * a[None])
        injection = step[:, :, None] * b[:, None, :, frame] * u[:, :, None, frame]
        state = transition * state + injection
        outputs.append((state * c[:, None, :, frame]).sum(-1))
    result = torch.stack(outputs, dim=-1)
    if d is not None:
        result = result + u * d[None, :, None]
    if z is not None:
        result = result * F.silu(z)
    return result.to(dtype)


def reference_mamba_forward(self, hidden_states, selection_condition=None):
    """Same Mamba parameters/operators, only scan/conv evaluated in torch.

    A full-sequence diagnostic, not inference-cache/step emulation.
    """
    batch, frames, _ = hidden_states.shape
    x, z = self.in_proj(hidden_states).transpose(1, 2).chunk(2, dim=1)
    # Native causal-conv is depthwise Conv1d with left-causal kernel alignment
    # and SiLU. nn.Conv1d configured in Mamba pads both sides, then cuts [:T].
    if self.activation not in ("silu", "swish"):
        raise ValueError("Reference backbone requires SiLU causal convolution")
    x = self.act(self.conv1d(x)[..., :frames])
    selected = self.x_proj(x.transpose(1, 2).reshape(batch * frames, -1))
    if hasattr(self, "condition_selection_proj"):
        if selection_condition is None or selection_condition.shape != hidden_states.shape:
            raise ValueError("Conditional reference requires aligned external selection input")
        selected = selected + self.condition_selection_proj(
            selection_condition.reshape(batch * frames, -1))
    elif selection_condition is not None:
        raise ValueError("Standard reference Mamba has no external selection input")
    dt, b, c = selected.split((self.dt_rank, self.d_state, self.d_state), dim=-1)
    delta = F.linear(dt, self.dt_proj.weight).reshape(batch, frames, -1).transpose(1, 2)
    b = b.reshape(batch, frames, self.d_state).transpose(1, 2).contiguous()
    c = c.reshape(batch, frames, self.d_state).transpose(1, 2).contiguous()
    result = selective_scan_torch(x, delta, -self.A_log.float().exp(), b, c,
                                 self.D.float(), z, self.dt_proj.bias.float())
    return self.out_proj(result.transpose(1, 2))


def reference_clone(native):
    """Clone weights, replacing forward methods inside this diagnostic only."""
    from mamba_ssm import Mamba
    reference = copy.deepcopy(native)
    count = 0
    for layer in reference.modules():
        if isinstance(layer, Mamba):
            layer.forward = types.MethodType(reference_mamba_forward, layer)
            count += 1
    if count < 1:
        raise RuntimeError("No genuine Mamba modules found; stubs cannot pass this gate")
    return reference, count


def _flat(values):
    if isinstance(values, torch.Tensor):
        return values.detach().float().reshape(-1)
    return torch.cat([value.detach().float().reshape(-1) for value in values])


def error_metrics(actual, expected):
    actual, expected = _flat(actual), _flat(expected)
    if actual.shape != expected.shape:
        raise ValueError("Diagnostic tensors must have identical shapes")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise ValueError("Nonfinite native/reference tensor; finite values alone do not constitute a pass")
    difference = actual - expected
    reference_norm = torch.linalg.vector_norm(expected)
    return dict(relative_l2=(torch.linalg.vector_norm(difference) / reference_norm.clamp_min(1e-12)).item(),
                max_absolute=difference.abs().max().item(),
                rms=difference.square().mean().sqrt().item(),
                expected_l2=reference_norm.item())


def _fresh(inputs):
    return tuple(value.detach().clone().requires_grad_(True) for value in inputs)


def _vjp(function, inputs, cotangent):
    fresh = _fresh(inputs)
    value = function(*fresh)
    gradients = torch.autograd.grad(value, fresh, grad_outputs=cotangent, allow_unused=True)
    if any(gradient is None for gradient in gradients):
        raise RuntimeError("VJP input unexpectedly disconnected; no silent zero gradients")
    return gradients


def _jvp(function, inputs, tangents):
    return torch.autograd.functional.jvp(function, inputs, tangents,
                                         create_graph=False, strict=False)[1]


def _central_difference(function, inputs, tangents, epsilon):
    with torch.no_grad():
        plus = function(*(value + epsilon * tangent for value, tangent in zip(inputs, tangents)))
        minus = function(*(value - epsilon * tangent for value, tangent in zip(inputs, tangents)))
        return (plus - minus) / (2 * epsilon)


def audit_case(name, native, reference, inputs, tangents, *, epsilons,
               forward_tolerance=3e-4, vjp_tolerance=2e-3,
               jvp_tolerance=5e-3, finite_difference_tolerance=3e-2):
    """All four checks are mandatory; collect failures rather than early-pass."""
    result = dict(name=name, passed=False, failures=[], errors={}, exception_tracebacks={},
                  tolerances=dict(forward=forward_tolerance, vjp=vjp_tolerance,
                                  jvp=jvp_tolerance, central_difference=finite_difference_tolerance))
    native_jvp = reference_jvp = native_vjp = cotangent = None
    try:
        with torch.no_grad():
            native_value, reference_value = native(*inputs), reference(*inputs)
        metrics = error_metrics(native_value, reference_value)
        result["errors"]["forward"] = metrics
        if metrics["relative_l2"] > forward_tolerance:
            result["failures"].append("Native/reference forward disagreement")
        cotangent = torch.randn_like(reference_value)
        native_vjp = _vjp(native, inputs, cotangent)
        metrics = error_metrics(native_vjp, _vjp(reference, inputs, cotangent))
        result["errors"]["vjp"] = metrics
        if metrics["relative_l2"] > vjp_tolerance:
            result["failures"].append("Native/reference VJP disagreement")
    except Exception as error:
        result["failures"].append(f"Forward/VJP exception: {type(error).__name__}: {error}")
        result["exception_tracebacks"]["forward_vjp"] = traceback.format_exc()
    try:
        reference_jvp = _jvp(reference, inputs, tangents)
        if not torch.isfinite(reference_jvp).all() or _flat(reference_jvp).norm().item() < 1e-8:
            raise RuntimeError("Reference JVP must be finite and nontrivial")
    except Exception as error:
        result["failures"].append(f"Reference JVP exception: {type(error).__name__}: {error}")
        result["exception_tracebacks"]["reference_jvp"] = traceback.format_exc()
    try:
        native_jvp = _jvp(native, inputs, tangents)
        if reference_jvp is not None:
            metrics = error_metrics(native_jvp, reference_jvp)
            result["errors"]["jvp"] = metrics
            if metrics["relative_l2"] > jvp_tolerance:
                result["failures"].append("Native/reference JVP disagreement (finite loss is NOT a pass)")
    except Exception as error:
        result["failures"].append(f"Native JVP exception: {type(error).__name__}: {error}")
        result["exception_tracebacks"]["native_jvp"] = traceback.format_exc()

    invariants = dict(passed=False, adjoint_tolerance=5e-4, linearity_tolerance=jvp_tolerance)
    try:
        if native_jvp is None or native_vjp is None or cotangent is None:
            raise RuntimeError("Adjoint/linearity requires the actual native JVP and VJP")
        jv, q = _flat(native_jvp).double(), _flat(cotangent).double()
        gradients, directions = _flat(native_vjp).double(), _flat(tangents).double()
        forward_dot, backward_dot = (jv*q).sum(), (gradients*directions).sum()
        scale = (jv.norm()*q.norm() + gradients.norm()*directions.norm()).clamp_min(1e-12)
        adjoint_error = ((forward_dot-backward_dot).abs()/scale).item()
        invariants["adjoint"] = dict(normalized_absolute=adjoint_error,
                                     jvp_dot_cotangent=forward_dot.item(), vjp_dot_direction=backward_dot.item())
        other = tuple(.5*direction + .01*torch.randn_like(value) for value, direction in zip(inputs, tangents))
        second = _jvp(native, inputs, other)
        combined = _jvp(native, inputs, tuple(2*v-3*w for v, w in zip(tangents, other)))
        linearity = error_metrics(combined, 2*native_jvp-3*second)
        invariants["linearity"] = linearity
        invariants["passed"] = adjoint_error <= invariants["adjoint_tolerance"] and linearity["relative_l2"] <= jvp_tolerance
        if not invariants["passed"]:
            result["failures"].append("Native JVP violates VJP adjoint or tangent linearity")
    except Exception as error:
        invariants["exception"] = f"{type(error).__name__}: {error}"
        result["exception_tracebacks"]["adjoint_linearity"] = traceback.format_exc()
        result["failures"].append("Native adjoint/linearity audit incomplete")
    result["derivative_invariants"] = invariants

    result["errors"]["finite_difference"] = []
    for epsilon in epsilons:
        entry = dict(epsilon=epsilon)
        try:
            native_fd = _central_difference(native, inputs, tangents, epsilon)
            reference_fd = _central_difference(reference, inputs, tangents, epsilon)
            if reference_jvp is not None:
                entry["reference_jvp_vs_reference_fd"] = error_metrics(reference_jvp, reference_fd)
                entry["reference_jvp_vs_native_fd"] = error_metrics(reference_jvp, native_fd)
            if native_jvp is not None:
                entry["native_jvp_vs_native_fd"] = error_metrics(native_jvp, native_fd)
        except Exception as error:
            entry["exception"] = f"{type(error).__name__}: {error}"
        result["errors"]["finite_difference"].append(entry)
    for key in ("reference_jvp_vs_reference_fd", "reference_jvp_vs_native_fd", "native_jvp_vs_native_fd"):
        errors = [entry[key]["relative_l2"] for entry in result["errors"]["finite_difference"] if key in entry]
        # Several prespecified epsilons diagnose cancellation/truncation, not
        # benchmark-score selection. Native/reference JVP must independently pass.
        if not errors or min(errors) > finite_difference_tolerance:
            result["failures"].append(f"No prespecified central-difference epsilon validates {key}")
    result["passed"] = not result["failures"]
    return result


def _nonzero_selection(model):
    """Diagnostic perturbation ensures delta/B/C conditional path is exercised."""
    with torch.no_grad():
        for layer in model.modules():
            if isinstance(layer, ConditionalSelectiveMamba):
                layer.condition_selection_proj.weight.normal_(std=.01)


def run_checks(device, width, frames, batch, *, cases=None):
    from mamba_ssm import Mamba
    # The caller's report owns the list so later construction/runtime failures
    # do not discard numerical evidence from already completed cases.
    cases = [] if cases is None else cases
    hidden = torch.randn(batch, frames, width, device=device) * .3
    tangent = torch.randn_like(hidden) * .2
    standard = Mamba(d_model=width, d_state=16, d_conv=4, expand=2,
                     use_fast_path=True).to(device).eval()
    if not standard.use_fast_path:
        raise RuntimeError("Standard Mamba must test the actual fused production path")
    reference, _ = reference_clone(standard)
    cases.append(audit_case("standard_mamba_fused", standard, reference,
        (hidden,), (tangent,), epsilons=(1e-2, 3e-3, 1e-3, 3e-4)))

    conditional = ConditionalSelectiveMamba(width).to(device).eval()
    _nonzero_selection(conditional)
    reference, _ = reference_clone(conditional)
    condition = torch.randn_like(hidden) * .3
    condition_tangent = torch.randn_like(condition) * .2
    cases.append(audit_case("conditional_mamba_external_delta_B_C", conditional, reference,
        (hidden, condition), (tangent, condition_tangent),
        epsilons=(1e-2, 3e-3, 1e-3, 3e-4)))

    # Test raw scan as well, separating scan and convolution kernel behavior.
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    inner, state = 2 * width, 16
    u = torch.randn(batch, inner, frames, device=device) * .3
    delta = torch.randn_like(u) * .1
    a = -torch.rand(inner, state, device=device) - .5
    b = torch.randn(batch, state, frames, device=device) * .1
    c = torch.randn_like(b) * .1
    d = torch.randn(inner, device=device) * .1
    z = torch.randn_like(u) * .2
    bias = torch.randn(inner, device=device) * .1 - 2
    scan_inputs = (u, delta, a, b, c, d, z, bias)
    scan_tangents = tuple(torch.randn_like(value) * .05 for value in scan_inputs)
    native_scan = lambda *args: selective_scan_fn(*args, delta_softplus=True)
    reference_scan = lambda *args: selective_scan_torch(*args, delta_softplus=True)
    cases.append(audit_case("selective_scan_kernel", native_scan, reference_scan,
        scan_inputs, scan_tangents, epsilons=(1e-2, 3e-3, 1e-3, 3e-4)))

    from causal_conv1d import causal_conv1d_fn
    conv_weight = torch.randn(inner, 4, device=device) * .1
    conv_bias = torch.randn(inner, device=device) * .1
    conv_inputs = (u, conv_weight, conv_bias)
    conv_tangents = tuple(torch.randn_like(value) * .05 for value in conv_inputs)
    native_conv = lambda value, weight, bias: causal_conv1d_fn(
        value, weight, bias, activation="silu")
    reference_conv = lambda value, weight, bias: F.silu(F.conv1d(
        value, weight[:, None], bias, padding=3, groups=inner)[..., :frames])
    cases.append(audit_case("causal_conv_kernel", native_conv, reference_conv,
        conv_inputs, conv_tangents, epsilons=(1e-2, 3e-3, 1e-3, 3e-4)))

    for mode in ("full55", "body22"):
        model = GestureM2D(mode, hidden_size=width, condition_layers=4, motion_layers=8,
                           dropout=0.).to(device).eval()
        if model.test_backend:
            raise RuntimeError("CPU/test backend cannot be used by the native gate")
        _nonzero_selection(model)
        reference, scan_count = reference_clone(model)
        if scan_count != 24:
            raise RuntimeError("Expected 4+8 bidirectional layers = 24 real Mamba blocks")
        mask = torch.arange(frames, device=device)[None] < (
            frames - torch.arange(batch, device=device).clamp_max(frames - 1))[:, None]
        condition = dict(audio=torch.randn(batch, frames, 1024, device=device) * .2,
                         text=torch.randn(batch, frames, 300, device=device) * .2,
                         person_id=torch.tensor([1, 30], device=device).repeat((batch + 1)//2)[:batch],
                         padding_mask=mask)
        motion = torch.randn(batch, frames, model.motion_dim, device=device) * .3
        driving_velocity = torch.randn_like(motion) * .2
        t = torch.full((batch, 1, 1), .73, device=device)
        r = torch.full_like(t, .21)

        def interval_velocity(module):
            def function(current_x, current_t, current_r):
                prediction = module(current_x, current_t.reshape(-1),
                                    (current_t-current_r).reshape(-1), **condition)
                ratio = current_r/current_t
                backproject = ratio * current_x + (1-ratio) * prediction
                return (current_x-backproject)/(current_t-current_r)
            return function

        case = audit_case(f"{mode}_4condition_8motion_meanflow_velocity",
            interval_velocity(model), interval_velocity(reference), (motion, t, r),
            (driving_velocity, torch.ones_like(t), torch.zeros_like(r)),
            epsilons=(3e-5, 1e-5, 3e-6), finite_difference_tolerance=4e-2)
        case["architecture"] = model.architecture_config()
        case["reference_scan_count"] = scan_count
        cases.append(case)
    return cases


def _actual_train_fixture(config_path, task_name, mode):
    """Read one fixed prepared train window; no preprocessing or evaluation."""
    import numpy as np
    from data import encode_motion
    config_path = Path(config_path).resolve(strict=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = {**config, **config.get("runtime", {})}
    if config.get("task_name") != task_name or config.get("expected_hostname") != socket.gethostname():
        raise RuntimeError("Native diagnostic runtime configuration belongs to another task/host")
    manifest_path = Path(config["manifest"]).resolve(strict=True)
    cache_root = Path(config["cache_root"]).resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    first = manifest["splits"]["train"][0]
    ident = first["id"] if isinstance(first, dict) else first
    clip_path = cache_root/(ident+".npz")
    stats_path = cache_root/f"stats_{mode}.npz"
    with np.load(clip_path, allow_pickle=False) as archive:
        usable = min(240, len(archive["poses"]))
        if usable != 240 or str(archive["source_id"].item()) != ident:
            raise RuntimeError("Actual native diagnostic clip cache is incomplete/incompatible")
        if not str(archive["geometry_schema"].item()).startswith("unposed-shaped-rest-v2"):
            raise RuntimeError("Native diagnostic requires true unposed rest-joint metadata")
        physical = encode_motion(archive["poses"][:usable], archive["trans"][:usable], mode)
        audio = np.asarray(archive["audio"][:usable], dtype=np.float32)
        text = np.asarray(archive["text"][:usable], dtype=np.float32)
        rest = np.asarray(archive["rest_joints"], dtype=np.float32)
        pose_mean = np.asarray(archive["pose_mean"], dtype=np.float32).reshape(55, 3)
    with np.load(stats_path, allow_pickle=False) as archive:
        if str(archive["split"].item()) != "train" or str(archive["mode"].item()) != mode:
            raise RuntimeError("Native diagnostic normalization must use mode-specific train-only statistics")
        mean, std = np.asarray(archive["mean"], dtype=np.float32), np.asarray(archive["std"], dtype=np.float32)
    if mean.shape != (physical.shape[1],) or std.shape != mean.shape or not (std > 0).all():
        raise RuntimeError("Native diagnostic statistics do not match actual motion")
    arrays = ((physical-mean)/std, audio, text)
    arrays = tuple(np.pad(value, ((0, 240-usable), (0, 0)), mode="edge").astype(np.float32) for value in arrays)
    if arrays[1].shape != (240, 1024) or arrays[2].shape != (240, 300) or rest.shape != (55, 3):
        raise RuntimeError("Actual full-size audio/text/rest input shapes are incorrect")
    if any(not np.isfinite(value).all() for value in (*arrays, mean, std, rest, pose_mean)):
        raise RuntimeError("Native diagnostic actual inputs contain nonfinite values")
    fixture = dict(clean=torch.from_numpy(arrays[0][None].copy()),
                   rest=torch.from_numpy(rest[None].copy()), mean=torch.from_numpy(mean.copy()),
                   std=torch.from_numpy(std.copy()), pose_mean=torch.from_numpy(pose_mean.copy()),
                   condition=dict(audio=torch.from_numpy(arrays[1][None].copy()),
                                  text=torch.from_numpy(arrays[2][None].copy()),
                                  person_id=torch.tensor([int(ident.split("_", 1)[0])]),
                                  padding_mask=(torch.arange(240)[None] < usable)))
    provenance = dict(clip_id=ident, usable_frames=usable, window_start=0,
                      config={"path": str(config_path), "sha256": source_hash(config_path)},
                      manifest={"path": str(manifest_path), "sha256": source_hash(manifest_path)},
                      clip_cache={"path": str(clip_path.resolve()), "sha256": source_hash(clip_path)},
                      train_statistics={"path": str(stats_path.resolve()), "sha256": source_hash(stats_path)},
                      pose_mean_sha256=hashlib.sha256(pose_mean.tobytes()).hexdigest())
    return fixture, provenance


def _device_fixture(fixture, device):
    return {key: ({name: value.to(device) for name, value in item.items()} if key == "condition" else item.to(device))
            for key, item in fixture.items()}


def _rng_signature(device):
    return dict(cpu=hashlib.sha256(bytes(torch.get_rng_state().tolist())).hexdigest(),
                cuda=hashlib.sha256(bytes(torch.cuda.get_rng_state(device).cpu().tolist())).hexdigest())


def _capture_actual_meanflow(objective, fixture, *, seed, device):
    """Capture the untouched loss's real locals/RNG; no reconstructed target.

    Profiling retains only detached CPU copies, never frames or graph locals.
    No replacement JVP engine, dropout masks or altered loss is introduced.
    """
    from loss import MeanFlowLoss
    captured, dropout_events, model_events, handles = {}, [], [], []
    code = MeanFlowLoss.forward.__code__
    velocity_code = next((item for item in code.co_consts if isinstance(item, types.CodeType)
                          and item.co_name == "velocity_function"), None)
    if velocity_code is None:
        raise RuntimeError("Cannot identify the original MeanFlow velocity function")
    old_profile = sys.getprofile()
    names = ("clean", "noise", "t", "r", "x_t", "driving_velocity", "prediction",
             "predicted_velocity", "derivative", "target_velocity", "error", "raw", "weighted", "total")
    def detached(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        return {key: detached(item) for key, item in value.items()}
    def profiler(frame, event, arg):
        if old_profile is not None:
            old_profile(frame, event, arg)
        if event == "return" and arg is None and frame.f_code in (code, velocity_code):
            # Exceptional frame exits also generate a return event. Preserve
            # the original error rather than masking it with a capture error.
            return
        if event == "return" and frame.f_code is code:
            if "total" in captured:
                raise RuntimeError("MeanFlow capture unexpectedly returned more than once")
            local = frame.f_locals
            if any(name not in local for name in names):
                raise RuntimeError("Original MeanFlow capture is missing a mandatory local")
            captured.update({name: detached(local[name]) for name in names})
        if event == "return" and frame.f_code is velocity_code:
            if "jvp_prediction" in captured or "predicted" not in frame.f_locals:
                raise RuntimeError("Original MeanFlow JVP prediction capture is missing/duplicated")
            captured["jvp_prediction"] = detached(frame.f_locals["predicted"])
    model = objective.model
    handles.append(model.register_forward_pre_hook(lambda module, args: model_events.append(_rng_signature(device))))
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Dropout):
            def dropout_hook(module, args, name=name):
                value = args[0]
                dropout_events.append(dict(name=name, p=module.p, training=module.training,
                                           inplace=module.inplace, shape=list(value.shape), stride=list(value.stride()),
                                           dtype=str(value.dtype), rng=_rng_signature(device)))
            handles.append(layer.register_forward_pre_hook(dropout_hook))
    try:
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            sys.setprofile(profiler)
            try:
                total, _ = objective(fixture["clean"], fixture["rest"], fixture["condition"])
            finally:
                sys.setprofile(old_profile)
            total.backward()
        gradients = {}
        for name, parameter in model.named_parameters():
            gradients[name] = None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            parameter.grad = None
        if any(name not in captured for name in (*names, "jvp_prediction")):
            raise RuntimeError("Actual original-loss target/derivative capture is incomplete")
        captured["dropout_events"], captured["model_rng_events"] = dropout_events, model_events
        return captured, gradients
    finally:
        sys.setprofile(old_profile)
        for handle in handles:
            handle.remove()


def _compare_actual_meanflow(native, reference, native_gradients, reference_gradients):
    """Independent target and each-parameter gates; no global-average-only pass."""
    failures, errors = [], {}
    for name in ("clean", "noise", "t", "r", "x_t", "driving_velocity"):
        if not torch.equal(native[name], reference[name]):
            failures.append(f"Actual loss RNG/input differs: {name}")
    for name in ("prediction", "jvp_prediction", "predicted_velocity", "derivative", "target_velocity", "error", "total"):
        metrics = error_metrics(native[name], reference[name])
        errors[name] = metrics
        tolerance = 5e-3 if name in ("derivative", "target_velocity") else 2e-3
        if metrics["relative_l2"] > tolerance:
            failures.append(f"Original MeanFlow actual {name} differs")
    for group in ("raw", "weighted"):
        if set(native[group]) != {"fm", "rec", "vel", "acc", "fk"} or set(reference[group]) != set(native[group]):
            raise RuntimeError("Original MeanFlow does not contain the exact five terms")
        for name in native[group]:
            metrics = error_metrics(native[group][name], reference[group][name])
            errors[f"{group}/{name}"] = metrics
            if metrics["relative_l2"] > 2e-3:
                failures.append(f"Original MeanFlow actual {group}/{name} differs")
    for label, captured in (("native", native), ("reference", reference)):
        events, rng = captured["dropout_events"], captured["model_rng_events"]
        if len(rng) != 2 or rng[0] != rng[1] or len(events) != 24 or events[:12] != events[12:]:
            failures.append(f"{label} prediction/JVP do not use the same actual dropout RNG")
        if any(item["p"] != .1 or item["training"] is not True for item in events):
            failures.append(f"{label} did not use the real training dropout=.1")
        if error_metrics(captured["prediction"], captured["jvp_prediction"])["max_absolute"] > 1e-6:
            failures.append(f"{label} original prediction/JVP dropout realization differs")
    if native["dropout_events"] != reference["dropout_events"] or native["model_rng_events"] != reference["model_rng_events"]:
        failures.append("Native/reference dropout execution RNG/layout differs")
    if set(native_gradients) != set(reference_gradients):
        raise RuntimeError("Native/reference parameter-name sets differ")
    numerator = denominator = 0.
    gradient_errors = {}
    for name in native_gradients:
        actual, expected = native_gradients[name], reference_gradients[name]
        if actual is None or expected is None:
            failures.append(f"Missing/disconnected actual parameter gradient: {name}")
            gradient_errors[name] = {"native_none": actual is None, "reference_none": expected is None}
            continue
        if actual.shape != expected.shape or actual.dtype != expected.dtype:
            raise RuntimeError(f"Parameter gradient shape/dtype differs: {name}")
        metrics = error_metrics(actual, expected)
        gradient_errors[name] = metrics
        difference, value = (actual.double()-expected.double()).reshape(-1), expected.double().reshape(-1)
        numerator += difference.square().sum().item()
        denominator += value.square().sum().item()
        if metrics["expected_l2"] > 1e-6:
            if metrics["relative_l2"] > 2e-3:
                failures.append(f"Actual per-parameter MeanFlow gradient differs: {name}")
        elif metrics["max_absolute"] > 5e-6:
            failures.append(f"Near-zero per-parameter gradient exceeds absolute tolerance: {name}")
    global_relative_l2 = (numerator/max(denominator, 1e-24))**.5
    if denominator < 1e-16 or global_relative_l2 > 2e-3:
        failures.append("Original MeanFlow global parameter gradient is zero or differs")
    return dict(passed=not failures, failures=failures, errors=errors,
                parameter_gradients=dict(parameter_count=len(native_gradients), relative_l2=global_relative_l2,
                                         reference_l2=denominator**.5, per_parameter=gradient_errors),
                dropout_execution=dict(native_events=native["dropout_events"],
                                       reference_events=reference["dropout_events"],
                                       native_model_rng=native["model_rng_events"], reference_model_rng=reference["model_rng_events"]))


def run_extended_checks(device, config_path, task_name, seed, cases, *, checks=None):
    """Actual CUDA validation only; never reduce dimensions or substitute CPU."""
    if device.type != "cuda" or not torch.cuda.is_available() or config_path is None:
        raise RuntimeError("Extended native audit requires actual CUDA and --runtime-config")
    from loss import MeanFlowLoss
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    checks = {} if checks is None else checks
    checks[EXTENDED_NATIVE_CHECKS[0]] = dict(kind="real_gpu", passed=len(cases) == 6 and all(
        case.get("derivative_invariants", {}).get("passed") is True for case in cases),
        case_names=[case["name"] for case in cases])
    # Exercise underflow/subnormals AFTER nonzero state, not a vacuous all-zero state.
    batch, channels, frames, state = 1, 32, 16, 16
    u = torch.randn(batch, channels, frames, device=device)*.3
    delta = torch.full_like(u, -2.)
    delta[..., 4:8], delta[..., 8:12], delta[..., 12:] = -200., -90., -80.
    a = -torch.rand(channels, state, device=device)-.5
    b = torch.randn(batch, state, frames, device=device)*.1
    c = torch.randn_like(b)*.1
    inputs = (u, delta, a, b, c)
    tangents = tuple(torch.randn_like(value)*.05 for value in inputs)
    mixed = dict(kind="real_gpu", passed=False, nonzero_history_prefix_frames=4,
                 delta_arguments=[-200., -90., -80.])
    checks[EXTENDED_NATIVE_CHECKS[1]] = mixed
    mixed["case"] = audit_case("selective_scan_nonzero_history_underflow_subnormal",
        lambda *args: selective_scan_fn(*args, delta_softplus=True),
        lambda *args: selective_scan_torch(*args, delta_softplus=True), inputs, tangents,
        epsilons=(1e-2, 3e-3, 1e-3, 3e-4))
    mixed["passed"] = mixed["case"]["passed"]
    del inputs, tangents, u, delta, a, b, c
    dimensions = dict(width=512, frames=240, batch=1)
    full = dict(kind="real_gpu", passed=False, dimensions=dimensions, actual_train_cache=True, modes={})
    objective_group = dict(kind="real_gpu", passed=False, dimensions=dimensions, actual_train_cache=True,
                           dropout=.1, objective="original_loss_py_tr_only_unmodified", modes={})
    checks[EXTENDED_NATIVE_CHECKS[2]], checks[EXTENDED_NATIVE_CHECKS[3]] = full, objective_group
    for mode in ("full55", "body22"):
        cpu_fixture, provenance = _actual_train_fixture(config_path, task_name, mode)
        fixture = _device_fixture(cpu_fixture, device)
        model = GestureM2D(mode, hidden_size=512, condition_layers=4, motion_layers=8, dropout=0.).to(device).eval()
        if model.test_backend:
            raise RuntimeError("Fake Mamba cannot pass full-size native derivative checks")
        _nonzero_selection(model)
        reference, count = reference_clone(model)
        if count != 24:
            raise RuntimeError("Full-size derivative reference must contain all 24 genuine Mamba blocks")
        motion = fixture["clean"]
        tangent = torch.randn_like(motion)*.2
        t, r = torch.full((1, 1, 1), .73, device=device), torch.full((1, 1, 1), .21, device=device)
        def velocity(module):
            def function(current_x, current_t, current_r):
                prediction = module(current_x, current_t.reshape(-1), (current_t-current_r).reshape(-1), **fixture["condition"])
                return MeanFlowLoss.x_r_and_velocity(current_x, prediction, current_t, current_r)[1]
            return function
        full["modes"][mode] = audit_case(f"{mode}_full_width512_T240_meanflow_velocity",
            velocity(model), velocity(reference), (motion, t, r),
            (tangent, torch.ones_like(t), torch.zeros_like(r)),
            epsilons=(1e-3, 3e-4, 1e-4, 3e-5), finite_difference_tolerance=4e-2)
        full["modes"][mode].update(input_provenance=provenance, architecture=model.architecture_config(),
                                   reference_scan_count=count)
        del model, reference, motion, tangent, t, r, velocity
        gc.collect()
        torch.cuda.empty_cache()

        model = GestureM2D(mode, hidden_size=512, condition_layers=4, motion_layers=8, dropout=.1).to(device).train()
        _nonzero_selection(model)
        objective = MeanFlowLoss(model, fixture["mean"], fixture["std"], mode=mode, pose_mean=fixture["pose_mean"]).to(device)
        native_capture, native_gradients = _capture_actual_meanflow(objective, fixture, seed=seed+991, device=device)
        protocol = objective.protocol_config()
        del objective
        model.cpu()
        gc.collect()
        torch.cuda.empty_cache()
        # Clone on CPU after native graph/gradients have been released. Never
        # retain two complete full-size GPU objectives or graphs concurrently.
        reference, count = reference_clone(model)
        del model
        reference = reference.to(device).train()
        objective = MeanFlowLoss(reference, fixture["mean"], fixture["std"], mode=mode, pose_mean=fixture["pose_mean"]).to(device)
        reference_capture, reference_gradients = _capture_actual_meanflow(objective, fixture, seed=seed+991, device=device)
        result = _compare_actual_meanflow(native_capture, reference_capture, native_gradients, reference_gradients)
        result.update(input_provenance=provenance, architecture=reference.architecture_config(),
                      original_loss_protocol=protocol, reference_scan_count=count)
        objective_group["modes"][mode] = result
        del reference, objective, fixture, cpu_fixture, native_capture, reference_capture, native_gradients, reference_gradients
        gc.collect()
        torch.cuda.empty_cache()
    full["passed"] = len(full["modes"]) == 2 and all(item["passed"] for item in full["modes"].values())
    objective_group["passed"] = len(objective_group["modes"]) == 2 and all(item["passed"] for item in objective_group["modes"].values())
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--runtime-config", type=Path,
                        help="Actual config.target.json; required for full-size original-loss checks")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parent
    report = dict(schema=1, kind="real_gpu", passed=False, hostname=socket.gethostname(),
                  task_name=args.task_name, timestamp_utc=datetime.now(timezone.utc).isoformat(),
                  source_hashes={name: source_hash(directory/name) for name in
                                 ("model.py", "loss.py", "native_jvp_check.py", "native_higherorder_jvp.py")},
                  device=args.device, seed=args.seed,
                  dimensions=dict(width=args.width, frames=args.frames, batch=args.batch),
                  torch_version=torch.__version__, torch_cuda_version=torch.version.cuda,
                  cuda_available=torch.cuda.is_available(), cases=[], failures=[],
                  native_derivative_repair=None, extended_checks={},
                  training=False, finite_difference_used_for_training=False,
                  reference_backend_used_for_training=False)
    try:
        if not args.task_name.strip():
            raise RuntimeError("Task name is mandatory for evidence provenance")
        device = torch.device(args.device)
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("A genuine CUDA GPU is mandatory; CPU cannot pass this diagnostic")
        if _MODEL_IMPORT_ERROR is not None:
            report["model_import_traceback"] = _MODEL_IMPORT_ERROR[1]
            raise RuntimeError("Pinned native model/derivative repair import failed") from _MODEL_IMPORT_ERROR[0]
        if args.width < 16 or args.width % 8 or args.frames < 4 or args.batch < 1:
            raise ValueError("Use width >=16 divisible by 8, frames >=4, batch >=1")
        torch.cuda.set_device(device)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["gpu_name"] = torch.cuda.get_device_name(device)
        report["gpu_capability"] = list(torch.cuda.get_device_capability(device))
        report["mamba_ssm_version"] = metadata.version("mamba-ssm")
        report["causal_conv1d_version"] = metadata.version("causal-conv1d")
        import mamba_ssm.ops.selective_scan_interface as scan_module
        import causal_conv1d.causal_conv1d_interface as conv_module
        from native_higherorder_jvp import install
        report["native_derivative_repair"] = install()
        if report["native_derivative_repair"].get("installed") is not True:
            raise RuntimeError("Exact native derivative repair was not installed")
        report["kernel_python_source_hashes"] = {
            "selective_scan_interface.py": source_hash(scan_module.__file__),
            "causal_conv1d_interface.py": source_hash(conv_module.__file__),
        }
        report["kernel_binary_sources"] = kernel_binary_bindings(scan_module, conv_module)
        run_checks(device, args.width, args.frames, args.batch, cases=report["cases"])
        report["failures"] = [case["name"] for case in report["cases"] if not case["passed"]]
        if len(report["cases"]) != 6 or report["failures"]:
            raise RuntimeError("Original six native cases failed; full-size gate not attempted")
        run_extended_checks(device, args.runtime_config, args.task_name, args.seed, report["cases"],
                            checks=report["extended_checks"])
        require_extended_native_checks(report)
        # Recheck in-memory callable identities AND all source/binary hashes
        # after the numerical checks, not just before them.
        if report["native_derivative_repair"] != install():
            raise RuntimeError("Native derivative repair provenance changed during audit")
        report["passed"] = True
    except Exception as error:
        report["failures"].extend(case["name"] for case in report["cases"] if not case["passed"])
        report["failures"].extend(name for name, check in report["extended_checks"].items() if check.get("passed") is not True)
        report["failures"].append(f"{type(error).__name__}: {error}")
        report["exception_traceback"] = traceback.format_exc()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"passed": report["passed"], "kind": report["kind"],
                      "evidence": str(args.output.resolve()), "failures": report["failures"]}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
