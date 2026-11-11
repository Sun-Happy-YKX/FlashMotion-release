"""Exact native-kernel cotangent double backward for this pinned Mamba runtime.

The original CUDA forward and first-order VJP are NOT replaced.  A custom
autograd node around their backward outputs supplies only the derivative with
respect to ``dout``.  PyTorch's original reverse-over-reverse functional.jvp
therefore obtains the analytic JVP, without finite differences or a Python
selective-scan recurrence.  Two state-expanded *native CUDA* scans integrate
the primal state and its tangent; native convolutions supply convolution
tangents.  Other higher derivatives fail explicitly.

Importing/installing this module is not GPU correctness evidence.  The real
six-case diagnostic, adjoint/linearity, full-size/dropout/MeanFlow gradient
checks and memory probe remain mandatory before training clearance.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import functools
import hashlib
import importlib
from importlib.machinery import EXTENSION_SUFFIXES
import inspect
from pathlib import Path
import threading
import types

import torch
from torch.nn import functional as F


class UnsupportedNativeDerivative(RuntimeError):
    """The requested derivative lies outside the verified repair domain."""


# CUDA autograd executes Python backwards on engine worker threads, which do
# not inherit ContextVars from the Python thread that called functional.jvp.
# Authorization therefore belongs to the engine GraphTask, never a thread or
# process-wide Boolean. Only an identity node appended AFTER the user primal
# returns may authorize the first engine pass.
_AUTH_LOCK = threading.RLock()
_GRAPH_AUTHORIZATIONS = {}
_STATE = None
_EXPECTED_SOURCES = {
    "selective_scan_interface.py":
        "2260a1486cea11fff63ba69d584175864f2687354b722fa9c16820de6a45158f",
    "causal_conv1d_interface.py":
        "221f41aa963a8f84683d9c6b2b4bf448692feb963fefcd778fa1b4ed2688c859",
    "cpp_functions.py":
        "80e75c8f761c88ede2ff73d244b0e9c0d2a974a780e2d3d8befeec1bd8aac29e",
}
_SUPPORTED = [
    "FP32 real CUDA tensors; full-sequence Mamba, state size 1..256",
    "frame-variable B/C [batch,state,time] or [batch,1,state,time]",
    "delta_softplus=True, optional D/z/delta_bias",
    "depthwise causal conv width 4, optional bias and SiLU",
    "fused Mamba with derived B/C, no B/C bias or RMS normalization",
    "original functional.jvp engine, create_graph=False, cotangent-only double backward",
]
_UNSUPPORTED = [
    "primal/parameter Hessians and higher-order derivatives outside functional.jvp",
    "functional.jvp(create_graph=True), forward_ad/jacfwd/vmap derivative engines",
    "FP16/BF16/autocast, complex A, non-softplus delta, other B/C groups",
    "inference/initial/final states, sequence indices, last-state derivatives",
    "fused external B/C, B/C projection bias, RMS normalization, fused output bias",
]


def _graph_task_id():
    graph_id = torch._C._current_graph_task_id()
    if not isinstance(graph_id, int) or graph_id < 0:
        raise UnsupportedNativeDerivative(
            f"Native derivative requires an autograd GraphTask; "
            f"thread={threading.get_ident()} graph_task={graph_id}")
    return graph_id


def _authorization_error(message):
    # Actual worker-thread/GraphTask evidence is useful if a pinned engine ever
    # changes. These process-local IDs disclose no data, credentials or paths.
    return UnsupportedNativeDerivative(
        f"{message}; thread={threading.get_ident()} "
        f"graph_task={torch._C._current_graph_task_id()}")


def _queue_graph_completion(callback):
    torch.autograd.Variable._execution_engine.queue_callback(callback)


class _JVPAuthorization:
    """One synchronous original-engine JVP, with exactly two graph passes."""

    def __init__(self):
        self.active = True
        self.first_graph = None
        self.first_complete = False
        self.second_graph = None
        self.second_complete = False

    def authorize_first(self):
        graph_id = _graph_task_id()
        with _AUTH_LOCK:
            if (not self.active or self.first_complete
                    or self.first_graph not in (None, graph_id)):
                raise _authorization_error("Expired or mismatched native JVP first-pass authorization")
            existing = _GRAPH_AUTHORIZATIONS.get(graph_id)
            if existing is not None and existing is not self:
                raise _authorization_error("GraphTask already belongs to a different native JVP")
            if self.first_graph is None:
                self.first_graph = graph_id
                _GRAPH_AUTHORIZATIONS[graph_id] = self
                _queue_graph_completion(self._finish_first)

    def _finish_first(self):
        with _AUTH_LOCK:
            self.first_complete = True
            if _GRAPH_AUTHORIZATIONS.get(self.first_graph) is self:
                del _GRAPH_AUTHORIZATIONS[self.first_graph]

    def authorize_second(self):
        graph_id = _graph_task_id()
        with _AUTH_LOCK:
            if (not self.active or not self.first_complete or self.second_complete
                    or graph_id == self.first_graph
                    or self.second_graph not in (None, graph_id)):
                raise _authorization_error("Expired or mismatched native JVP cotangent-pass authorization")
            if self.second_graph is None:
                self.second_graph = graph_id
                _queue_graph_completion(self._finish_second)

    def _finish_second(self):
        with _AUTH_LOCK:
            self.second_complete = True

    def invalidate(self):
        with _AUTH_LOCK:
            self.active = False
            if _GRAPH_AUTHORIZATIONS.get(self.first_graph) is self:
                del _GRAPH_AUTHORIZATIONS[self.first_graph]


def _first_pass_authorization():
    graph_id = _graph_task_id()
    with _AUTH_LOCK:
        authorization = _GRAPH_AUTHORIZATIONS.get(graph_id)
        if (authorization is None or not authorization.active
                or authorization.first_complete or authorization.first_graph != graph_id):
            raise _authorization_error(
                "Native primal/parameter Hessians are unsupported; use scoped functional.jvp")
        return authorization


class _JVPOutputAnchor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, authorization, *outputs):
        ctx.authorization = authorization
        ctx.set_materialize_grads(False)
        return outputs

    @staticmethod
    def backward(ctx, *cotangents):
        if not torch.is_grad_enabled():
            raise _authorization_error("Native JVP anchor requires the original create-graph cotangent pass")
        ctx.authorization.authorize_first()
        return (None,) + cotangents


def _anchor_outputs(outputs, authorization):
    is_tuple = isinstance(outputs, tuple)
    values = outputs if is_tuple else (outputs,)
    selected = tuple(index for index, value in enumerate(values)
                     if isinstance(value, torch.Tensor) and value.requires_grad)
    if not selected:
        return outputs
    anchored = _JVPOutputAnchor.apply(authorization, *(values[index] for index in selected))
    result = list(values)
    for index, value in zip(selected, anchored):
        result[index] = value
    return tuple(result) if is_tuple else result[0]


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file():
        raise RuntimeError(f"Required runtime source is not a file: {path}")
    return {"path": str(path), "sha256": _sha(path)}


def _causal_extension(interface):
    """Follow actual imported Python wrapper globals, never guess-import a .so."""
    names = ("causal_conv1d_fwd_function", "causal_conv1d_bwd_function",
             "causal_conv1d_update_function")
    wrappers = {name: getattr(interface, name) for name in names
                if hasattr(interface, name)}
    extensions = []
    if wrappers:
        if any(name not in wrappers for name in names[:2]):
            raise RuntimeError("Causal interface is missing an exercised fwd/bwd wrapper")
        for name, wrapper in wrappers.items():
            if not isinstance(wrapper, types.FunctionType):
                raise RuntimeError(f"Actual causal wrapper is not a Python function: {name}")
            extension = wrapper.__globals__.get("causal_conv1d_cuda")
            if not isinstance(extension, types.ModuleType):
                raise RuntimeError(f"Actual causal wrapper has no extension global: {name}")
            extensions.append(extension)
    if hasattr(interface, "causal_conv1d_cuda"):
        extensions.append(interface.causal_conv1d_cuda)
    if not extensions or any(not isinstance(value, types.ModuleType) for value in extensions):
        raise RuntimeError("Actual causal interface has no compiled-extension call path")
    if any(value is not extensions[0] for value in extensions):
        raise RuntimeError("Causal interface wrapper paths use different extension objects")
    return extensions[0]


def _binary_records(scan_module, conv_module):
    scan_causal = _causal_extension(scan_module)
    conv_causal = _causal_extension(conv_module)
    if scan_causal is not conv_causal:
        raise RuntimeError("Scan and causal interfaces use different compiled extensions")
    records = {}
    for role, extension in (("selective_scan_cuda", scan_module.selective_scan_cuda),
                            ("causal_conv1d_cuda", conv_causal)):
        if not isinstance(extension, types.ModuleType):
            raise RuntimeError(f"Actual native module missing: {role}")
        path = getattr(extension, "__file__", None)
        if not path or not str(path).endswith(tuple(EXTENSION_SUFFIXES)):
            raise RuntimeError(f"Actual native module is not a compiled extension: {role}")
        records[role] = {"module": extension.__name__, **_file_record(path)}
    return records


def _require_fp32_cuda(*values):
    device = None
    for value in values:
        if value is None:
            continue
        if not isinstance(value, torch.Tensor) or value.dtype != torch.float32 or not value.is_cuda:
            raise UnsupportedNativeDerivative("Repair requires real FP32 CUDA tensors")
        if device is None:
            device = value.device
        elif value.device != device:
            raise UnsupportedNativeDerivative("Repair tensors must share the same CUDA device")
    if torch.is_autocast_enabled():
        raise UnsupportedNativeDerivative("Autocast is outside the exact FP32 repair domain")


def _direction(value, tangent):
    if tangent is None:
        return 0.0
    if value is None or tangent.shape != value.shape:
        raise UnsupportedNativeDerivative("Native tangent shape does not match its primal")
    _require_fp32_cuda(value, tangent)
    return tangent


def _variable(value, batch, channels, state, frames):
    if value.shape == (batch, state, frames):
        return value[:, None].expand(batch, channels, state, frames)
    if value.shape == (batch, 1, state, frames):
        return value.expand(batch, channels, state, frames)
    raise UnsupportedNativeDerivative("Only single-group frame-variable B/C are supported")


def _variable_direction(value, tangent, batch, channels, state, frames):
    if tangent is None:
        return 0.0
    _direction(value, tangent)
    return _variable(tangent, batch, channels, state, frames)


@dataclass(frozen=True)
class _NativeMath:
    scan_fwd: object
    conv_fwd: object

    def _state_scan(self, u, delta, a, b):
        """Independent 1-state channels; no time loop or reference recurrence."""
        batch, channels, state, frames = u.shape
        flat = lambda value: value.reshape(batch, channels * state, frames).contiguous()
        one = a.new_ones((channels * state, 1))
        # The public domain is softplus=True. These INTERNAL scans receive the
        # already-positive softplus step and deliberately disable another softplus.
        result = self.scan_fwd(flat(u), flat(delta), a.reshape(channels * state, 1).contiguous(),
                               b.reshape(batch, channels * state, 1, frames).contiguous(),
                               one, None, None, None, False)[0]
        return result.reshape(batch, channels, state, frames)

    def scan_value_and_jvp(self, u, delta, a, b, c, d, z, bias, tangents,
                           *, delta_softplus):
        if not delta_softplus:
            raise UnsupportedNativeDerivative("Non-softplus delta is not supported (zero-step forcing)")
        _require_fp32_cuda(u, delta, a, b, c, d, z, bias, *tangents)
        if u.ndim != 3 or delta.shape != u.shape or a.ndim != 2:
            raise UnsupportedNativeDerivative("Invalid native scan shapes")
        batch, channels, frames = u.shape
        if a.shape[0] != channels or not 1 <= a.shape[1] <= 256 or frames < 1:
            raise UnsupportedNativeDerivative("Unsupported native scan dimensions")
        state = a.shape[1]
        if d is not None and d.shape != (channels,):
            raise UnsupportedNativeDerivative("D must have one value per channel")
        if bias is not None and bias.shape != (channels,):
            raise UnsupportedNativeDerivative("Delta bias must have one value per channel")
        if z is not None and z.shape != u.shape:
            raise UnsupportedNativeDerivative("z must match the scan input")
        if len(tangents) != 8:
            raise UnsupportedNativeDerivative("Scan requires eight tangent slots")
        vu, vdelta, va, vb, vc, vd, vz, vbias = tangents
        vu = _direction(u, vu)
        vdelta = _direction(delta, vdelta)
        va = _direction(a, va)
        vd = _direction(d, vd)
        vz = _direction(z, vz)
        vbias = _direction(bias, vbias)
        b_all = _variable(b, batch, channels, state, frames)
        c_all = _variable(c, batch, channels, state, frames)
        vb_all = _variable_direction(b, vb, batch, channels, state, frames)
        vc_all = _variable_direction(c, vc, batch, channels, state, frames)
        step_arg = delta if bias is None else delta + bias[None, :, None]
        step = F.softplus(step_arg)
        step_all = step[:, :, None].expand(batch, channels, state, frames)
        u_all = u[:, :, None].expand(batch, channels, state, frames)
        h = self._state_scan(u_all, step_all, a, b_all)
        h_previous = F.pad(h[..., :-1], (1, 0))
        va_all = va if isinstance(va, float) else va[None, :, :, None]
        vu_all = vu if isinstance(vu, float) else vu[:, :, None]
        vstep_arg = vdelta + (vbias if isinstance(vbias, float) else vbias[None, :, None])
        # sigmoid(x)/softplus(x) = (1-exp(-s))/s. Use its s=0 limit
        # for FP32 softplus underflow. The outer native scan multiplies by s,
        # so forcing is correctly zero in that representational limit.
        divisor = torch.where(step > 0, step, torch.ones_like(step))
        ratio = torch.where(step > 0, -torch.expm1(-step) / divisor, torch.ones_like(step))
        ratio = (ratio * vstep_arg)[:, :, None]
        transition = torch.exp(step_all * a[None, :, :, None])
        forcing_over_step = (
            transition * (ratio * a[None, :, :, None] + va_all) * h_previous
            + (ratio * b_all + vb_all) * u_all + b_all * vu_all
        )
        one_b = u.new_ones((batch, channels, state, frames))
        h_dot = self._state_scan(forcing_over_step, step_all, a, one_b)
        value = (c_all * h).sum(2)
        tangent = (vc_all * h + c_all * h_dot).sum(2)
        if d is not None:
            value = value + d[None, :, None] * u
            tangent = tangent + d[None, :, None] * vu
            if not isinstance(vd, float):
                tangent = tangent + vd[None, :, None] * u
        if z is not None:
            sigmoid = torch.sigmoid(z)
            silu_prime = sigmoid + z * sigmoid * (1 - sigmoid)
            tangent = tangent * F.silu(z) + value * silu_prime * vz
            value = value * F.silu(z)
        return value, tangent

    def conv_jvp(self, x, weight, bias, tangents, *, activation):
        _require_fp32_cuda(x, weight, bias, *tangents)
        if x.ndim != 3 or weight.shape != (x.shape[1], 4):
            raise UnsupportedNativeDerivative("Repair supports causal conv width 4 only")
        if bias is not None and bias.shape != (x.shape[1],):
            raise UnsupportedNativeDerivative("Conv bias shape does not match channels")
        if len(tangents) != 3:
            raise UnsupportedNativeDerivative("Conv requires three tangent slots")
        vx, vw, vb = (_direction(value, tangent)
                      for value, tangent in zip((x, weight, bias), tangents))
        def conv(value, kernel, offset=None):
            return self.conv_fwd(value.contiguous(), kernel.contiguous(),
                                 offset.contiguous() if offset is not None else None,
                                 None, None, None, False)
        tangent = torch.zeros_like(x)
        if not isinstance(vx, float):
            tangent = tangent + conv(vx, weight)
        if not isinstance(vw, float):
            tangent = tangent + conv(x, vw)
        if not isinstance(vb, float):
            tangent = tangent + vb[None, :, None]
        if activation:
            preactivation = conv(x, weight, bias)
            sigmoid = torch.sigmoid(preactivation)
            tangent = tangent * (sigmoid + preactivation * sigmoid * (1 - sigmoid))
        return tangent

    def from_ctx(self, kind, ctx, directions):
        if kind == "scan":
            if ctx.has_z:
                u, delta, a, b, c, d, z, bias, _, _ = ctx.saved_tensors
            else:
                u, delta, a, b, c, d, bias, _ = ctx.saved_tensors
                z = None
            tangents = list(directions[:8])
            # Saved B/C were promoted to rank 4, original backward returns
            # rank-3 gradients for rank-3 inputs.
            for index in (3, 4):
                if tangents[index] is not None and tangents[index].ndim == 3:
                    tangents[index] = tangents[index][:, None]
            return self.scan_value_and_jvp(u, delta, a, b, c, d, z, bias, tangents,
                                            delta_softplus=ctx.delta_softplus)[1]
        if kind == "conv":
            x, weight, bias, seq_idx, initial = ctx.saved_tensors
            if seq_idx is not None or initial is not None or ctx.return_final_states:
                raise UnsupportedNativeDerivative("Conv state/sequence-index derivatives are unsupported")
            if any(value is not None for value in directions[3:]):
                raise UnsupportedNativeDerivative("Conv tangent supplied for an unsupported state input")
            return self.conv_jvp(x, weight, bias, directions[:3], activation=ctx.activation)
        if kind == "fused":
            return self.fused_jvp(ctx, directions)
        raise UnsupportedNativeDerivative(f"Unknown native operation: {kind}")

    def fused_jvp(self, ctx, directions):
        (xz, conv_weight, conv_bias, selected, x_proj, dt_proj, out_proj,
         saved_conv, saved_delta, a, b, c, d, bias, _, b_rms, c_rms, dt_rms, _) = ctx.saved_tensors
        if (not ctx.is_variable_B or not ctx.is_variable_C
                or not ctx.B_proj_bias_is_None or not ctx.C_proj_bias_is_None
                or not ctx.out_proj_bias_is_None
                or any(value is not None for value in (b_rms, c_rms, dt_rms))):
            raise UnsupportedNativeDerivative("Fused external B/C, biases or RMS path is outside repair domain")
        if len(directions) != 20 or any(value is not None for value in directions[12:]):
            raise UnsupportedNativeDerivative("Fused tangent supplied for unsupported metadata/RMS input")
        if directions[6] is not None or directions[8] is not None or directions[9] is not None:
            raise UnsupportedNativeDerivative("Fused out-bias/external B/C tangents are unsupported")
        _require_fp32_cuda(xz, conv_weight, conv_bias, selected, x_proj, dt_proj,
                          out_proj, a, b, c, d, bias, *directions)
        batch, two_channels, frames = xz.shape
        channels, state, rank = two_channels // 2, a.shape[1], dt_proj.shape[1]
        if two_channels % 2 or a.shape[0] != channels or selected.shape != (batch * frames, rank + 2 * state):
            raise UnsupportedNativeDerivative("Unsupported fused Mamba shape")
        x, z = xz.chunk(2, dim=1)
        vxz = _direction(xz, directions[0])
        vx, vz = (None, None) if isinstance(vxz, float) else vxz.chunk(2, dim=1)
        vconv_weight = directions[1]
        if vconv_weight is not None:
            if vconv_weight.shape != (channels, 1, 4):
                raise UnsupportedNativeDerivative("Fused conv weight tangent has unexpected shape")
            vconv_weight = vconv_weight[:, 0]
        conv_value = saved_conv
        if conv_value is None:
            conv_value = self.conv_fwd(x, conv_weight, conv_bias, None, None, None, True)
        conv_dot = self.conv_jvp(x, conv_weight, conv_bias,
                                 (vx, vconv_weight, directions[2]), activation=True)
        flatten = lambda value: value.transpose(1, 2).reshape(batch * frames, channels)
        selected_dot = F.linear(flatten(conv_dot), x_proj)
        vx_proj = _direction(x_proj, directions[3])
        if not isinstance(vx_proj, float):
            selected_dot = selected_dot + F.linear(flatten(conv_value), vx_proj)
        dt_value = saved_delta
        if dt_value is None:
            dt_value = F.linear(selected[:, :rank], dt_proj).reshape(batch, frames, channels).transpose(1, 2).contiguous()
        dt_dot = F.linear(selected_dot[:, :rank], dt_proj)
        vdt_proj = _direction(dt_proj, directions[4])
        if not isinstance(vdt_proj, float):
            dt_dot = dt_dot + F.linear(selected[:, :rank], vdt_proj)
        dt_dot = dt_dot.reshape(batch, frames, channels).transpose(1, 2).contiguous()
        variable = lambda value: value.reshape(batch, frames, state).transpose(1, 2)[:, None].contiguous()
        b_dot = variable(selected_dot[:, rank:rank + state])
        c_dot = variable(selected_dot[:, -state:])
        scan_value, scan_dot = self.scan_value_and_jvp(
            conv_value, dt_value, a, b, c, d, z, bias,
            (conv_dot, dt_dot, directions[7], b_dot, c_dot, directions[10], vz, directions[11]),
            delta_softplus=ctx.delta_softplus,
        )
        result = F.linear(scan_dot.transpose(1, 2), out_proj)
        vout_proj = _direction(out_proj, directions[5])
        if not isinstance(vout_proj, float):
            result = result + F.linear(scan_value.transpose(1, 2), vout_proj)
        return result


@dataclass(frozen=True)
class _Payload:
    kind: str
    original_ctx: object
    original_backward: object
    extra_grad_outputs: tuple
    native_math: _NativeMath
    authorization: _JVPAuthorization


class _CotangentVJP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dout, payload):
        ctx.set_materialize_grads(False)
        ctx.payload = payload
        # Exactly the vendor VJP, including preallocated fused dxz buffers.
        # Wrapping its entire returned tuple prevents the buffer alias path
        # from accidentally losing the cotangent graph.
        return payload.original_backward(payload.original_ctx, dout, *payload.extra_grad_outputs)

    @staticmethod
    def backward(ctx, *directions):
        if torch.is_grad_enabled():
            raise _authorization_error("Only create_graph=False cotangent double backward is implemented")
        payload = ctx.payload
        payload.authorization.authorize_second()
        return payload.native_math.from_ctx(payload.kind, payload.original_ctx, directions), None


def _backward_wrapper(kind, original, native_math):
    @functools.wraps(original)
    def backward(ctx, dout, *extra):
        if not torch.is_grad_enabled():
            return original(ctx, dout, *extra)
        authorization = _first_pass_authorization()
        if dout is None or any(value is not None for value in extra):
            raise UnsupportedNativeDerivative("Native last/final-state output derivatives are unsupported")
        return _CotangentVJP.apply(dout, _Payload(kind, ctx, original, extra, native_math, authorization))
    return backward


def _jvp_scope_wrapper(original):
    @functools.wraps(original)
    def jvp(func, inputs, v=None, create_graph=False, strict=False):
        if create_graph:
            raise UnsupportedNativeDerivative("create_graph=True would request unsupported primal Hessians")
        authorization = _JVPAuthorization()
        def primal_only(*arguments):
            # User code executed by the primal function must not use this
            # permission to request a parameter Hessian inside a JVP. Only
            # the anchor added AFTER that code returns can register a GraphTask.
            return _anchor_outputs(func(*arguments), authorization)
        try:
            # Same engine, signature, inputs and numerical objective. This
            # adapter changes only the permitted cotangent derivative scope.
            return original(primal_only, inputs, v=v, create_graph=False, strict=strict)
        finally:
            authorization.invalidate()
    return jvp


def _assert_hooks(state):
    if torch.autograd.functional.jvp is not state["jvp_hook"]:
        raise RuntimeError("Native JVP scope hook was replaced after installation")
    for cls, hook in state["class_hooks"]:
        if cls.backward is not hook:
            raise RuntimeError(f"Native backward hook was replaced: {cls.__name__}")
    for label, getter, original in state["runtime_bindings"]:
        if getter() is not original:
            raise RuntimeError(f"Installed native runtime callable/global was replaced: {label}")


def install():
    """Idempotently install on the exact pinned imported runtime; no GPU launch.

    Unknown Python sources, absent/mixed/noncompiled native modules or an
    overwritten installed hook fail closed before mutation. The extension
    binary SHA values are recorded, not mistaken for numerical validation.
    """
    global _STATE
    if _STATE is not None:
        _assert_hooks(_STATE)
        return metadata()
    scan_module = importlib.import_module("mamba_ssm.ops.selective_scan_interface")
    conv_module = importlib.import_module("causal_conv1d.causal_conv1d_interface")
    cpp_module = importlib.import_module("causal_conv1d.cpp_functions")
    modules = {"selective_scan_interface.py": scan_module,
               "causal_conv1d_interface.py": conv_module, "cpp_functions.py": cpp_module}
    runtime_sources = {name: _file_record(module.__file__) for name, module in modules.items()}
    for name, record in runtime_sources.items():
        if record["sha256"] != _EXPECTED_SOURCES[name]:
            raise RuntimeError(f"Unrecognized actual runtime source SHA256: {name}")
    binaries = _binary_records(scan_module, conv_module)
    for name in ("causal_conv1d_fwd_function", "causal_conv1d_bwd_function", "causal_conv1d_update_function"):
        if getattr(scan_module, name) is not getattr(conv_module, name) or getattr(scan_module, name) is not getattr(cpp_module, name):
            raise RuntimeError(f"Actual imported causal wrappers do not share identity: {name}")
    native_math = _NativeMath(scan_module.selective_scan_cuda.fwd,
                              scan_module.causal_conv1d_fwd_function)
    originals, hooks, class_sources, bindings = {}, [], {}, []
    def bind(label, getter):
        bindings.append((label, getter, getter()))
    for kind, cls in (("scan", scan_module.SelectiveScanFn), ("conv", conv_module.CausalConv1dFn),
                      ("fused", scan_module.MambaInnerFn)):
        original = cls.backward
        originals[kind] = original
        class_sources[cls.__name__] = {
            "module": cls.__module__,
            "source_sha256": hashlib.sha256(inspect.getsource(cls).encode("utf-8")).hexdigest(),
            "original_backward_sha256": hashlib.sha256(inspect.getsource(original).encode("utf-8")).hexdigest(),
            "original_forward_sha256": hashlib.sha256(inspect.getsource(cls.forward).encode("utf-8")).hexdigest(),
        }
        bind(f"{cls.__name__}.forward", lambda cls=cls: cls.forward)
        hooks.append((cls, _backward_wrapper(kind, original, native_math)))
    scan_extension = scan_module.selective_scan_cuda
    conv_extension = _causal_extension(conv_module)
    bind("scan.selective_scan_cuda", lambda: scan_module.selective_scan_cuda)
    for name in ("fwd", "bwd"):
        bind(f"selective_scan_cuda.{name}", lambda name=name: getattr(scan_extension, name))
    for name in ("causal_conv1d_fwd", "causal_conv1d_bwd", "causal_conv1d_update"):
        bind(f"causal_conv1d_cuda.{name}", lambda name=name: getattr(conv_extension, name))
    for name in ("causal_conv1d_fwd_function", "causal_conv1d_bwd_function", "causal_conv1d_update_function"):
        wrapper = getattr(cpp_module, name)
        for label, module in (("scan", scan_module), ("conv", conv_module), ("cpp", cpp_module)):
            bind(f"{label}.{name}", lambda module=module, name=name: getattr(module, name))
        bind(f"{name}.globals.causal_conv1d_cuda", lambda wrapper=wrapper: wrapper.__globals__.get("causal_conv1d_cuda"))
        # The actual Python wrappers call these decorated custom-op globals.
        op_name = {"causal_conv1d_fwd_function": "_causal_conv1d_fwd_cpp",
                   "causal_conv1d_bwd_function": "_causal_conv1d_bwd_cpp",
                   "causal_conv1d_update_function": "_causal_conv1d_update_cpp"}[name]
        if wrapper.__globals__.get(op_name) is None:
            raise RuntimeError(f"Actual causal wrapper is missing its custom-op global: {op_name}")
        bind(f"{name}.globals.{op_name}", lambda wrapper=wrapper, op_name=op_name: wrapper.__globals__.get(op_name))
    original_jvp = torch.autograd.functional.jvp
    jvp_hook = _jvp_scope_wrapper(original_jvp)
    # All source/binary/identity checks precede any modification.
    for cls, hook in hooks:
        cls.backward = staticmethod(hook)
    torch.autograd.functional.jvp = jvp_hook
    _STATE = {"originals": originals, "class_hooks": hooks, "jvp_hook": jvp_hook,
              "original_jvp": original_jvp, "runtime_sources": runtime_sources,
              "binary_sources": binaries, "original_classes": class_sources,
              "installed_helper_sha256": _sha(__file__), "runtime_bindings": bindings}
    return metadata()


def metadata():
    """Serializable provenance only; never grants GPU/training clearance."""
    result = {
        "schema": 1, "kind": "native_cotangent_double_backward", "installed": _STATE is not None,
        "helper_path": str(Path(__file__).resolve()), "helper_sha256": _sha(__file__),
        "jvp_engine": "original_torch.autograd.functional.jvp_reverse_over_reverse",
        "scope": "cotangent_only_create_graph_false",
        "scope_authority": "output_anchor_graph_task_id_two_pass_expiring_authorization",
        "supported": list(_SUPPORTED),
        "unsupported_derivatives": list(_UNSUPPORTED),
        "finite_difference_used_for_training": False, "reference_backend_used_for_training": False,
        "gpu_correctness_verified": False,
    }
    if _STATE is not None:
        _assert_hooks(_STATE)
        if result["helper_sha256"] != _STATE["installed_helper_sha256"]:
            raise RuntimeError("Native helper source changed after hook installation")
        for record in (*_STATE["runtime_sources"].values(), *_STATE["binary_sources"].values()):
            if _sha(record["path"]) != record["sha256"]:
                raise RuntimeError("Installed runtime source/binary changed after native hook installation")
        result.update(runtime_python_sources=copy.deepcopy(_STATE["runtime_sources"]),
                      kernel_binary_sources=copy.deepcopy(_STATE["binary_sources"]),
                      original_classes=copy.deepcopy(_STATE["original_classes"]),
                      runtime_identity_checks=[label for label, _, _ in _STATE["runtime_bindings"]])
    else:
        result.update(runtime_python_sources={}, kernel_binary_sources={}, original_classes={},
                      runtime_identity_checks=[])
    return result
