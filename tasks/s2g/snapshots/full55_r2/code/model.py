"""Music2Dance conditional backbone adapted to SemTalk speech gestures.

The reference is ``_adain_patch/m2d/model/model.py``: four condition blocks,
eight motion blocks, GenreGate -> PersonGate, AdaIN, conditional delta/B/C,
and bidirectional sum. No clean-motion, seed-pose, or expression input exists.
Real runs require mamba_ssm; an explicitly supplied factory is only for CPU
interface tests and is recorded in architecture metadata.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
from torch import nn
from torch.nn import functional as F

try:
    from mamba_ssm import Mamba as _Mamba
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except ImportError:
    _Mamba = None
    selective_scan_fn = None

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

# The exact native repair changes neither the model operators nor MeanFlow's
# loss. Unknown/missing native runtimes fail closed; injected CPU factories
# do not install it and cannot grant GPU clearance.
if _Mamba is not None:
    if causal_conv1d_fn is None:
        raise RuntimeError("Real Mamba requires the pinned native causal convolution runtime")
    from native_higherorder_jvp import install as _install_native_derivative_repair
    _install_native_derivative_repair()


SEMTALK_RAW_PERSON_IDS = (
    1, 2, 3, 4, 5, 6, 7, 30, 9, 10, 11, 12, 13,
    28, 15, 16, 17, 18, 27, 20, 21, 22, 23, 24, 25,
)
SMPLX_PARENTS = (
    -1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14,
    16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32,
    20, 34, 35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47,
    21, 49, 50, 21, 52, 53,
)
MOTION_DIMENSIONS = {"full55": 333, "body22": 135}


def valid_lengths(mask: torch.Tensor | None, batch: int, frames: int,
                  device: torch.device) -> torch.Tensor:
    """Validate suffix padding; True denotes valid frames, not padding."""
    if mask is None:
        return torch.full((batch,), frames, dtype=torch.long, device=device)
    if mask.shape != (batch, frames) or mask.dtype != torch.bool:
        raise ValueError("padding_mask must be bool [B,T], with True = valid")
    mask = mask.to(device)
    lengths = mask.long().sum(1)
    if (lengths <= 0).any():
        raise ValueError("all sequences must contain at least one valid frame")
    expected = torch.arange(frames, device=device)[None] < lengths[:, None]
    if not torch.equal(mask, expected):
        raise ValueError("only contiguous valid prefixes followed by padding are supported")
    return lengths


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 4 or dim % 2:
            raise ValueError("time embedding width must be even and at least four")
        self.dim = dim

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        frequencies = torch.exp(torch.arange(half, device=value.device,
            dtype=value.dtype) * (-math.log(10000.0) / (half - 1)))
        angles = value.reshape(-1, 1) * frequencies[None]
        return torch.cat((angles.sin(), angles.cos()), dim=-1)


class DenseFiLM(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.Mish(), nn.Linear(width, 2 * width))

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.block(value).unsqueeze(1).chunk(2, dim=-1)


def affine(value: torch.Tensor, scale_shift: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    scale, shift = scale_shift
    return (1 + scale) * value + shift


class CrossModalAdaIN(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(width, width * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, value: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        mean = value.mean(1, keepdim=True)
        variance = (value - mean).square().mean(1, keepdim=True)
        std = (variance + self.eps).sqrt()
        scale, shift = self.modulation(condition).tanh().chunk(2, dim=-1)
        return mean + std * ((1 + scale) * ((value - mean) / std) + shift)


class PersonGate(nn.Module):
    """Exact GenreGate topology, with the official 25-person raw-ID mapping."""
    def __init__(self, width: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(25, width)
        self.proj_g = nn.Linear(width, width)
        self.delta = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                   nn.Linear(width, width))
        self.alpha = nn.Sequential(nn.Linear(2 * width, width), nn.GELU(),
                                   nn.Linear(width, width))
        nn.init.constant_(self.alpha[-1].bias, -4.0)
        mapping = torch.full((31,), -1, dtype=torch.long)
        for contiguous, raw in enumerate(SEMTALK_RAW_PERSON_IDS):
            mapping[raw] = contiguous
        self.register_buffer("raw_to_contiguous", mapping)

    def forward(self, value: torch.Tensor, raw_person_id: torch.Tensor) -> torch.Tensor:
        raw_person_id = raw_person_id.reshape(-1).long()
        if raw_person_id.numel() != value.shape[0]:
            raise ValueError("one raw BEAT2 person ID is required per sequence")
        if (raw_person_id < 0).any() or (raw_person_id >= 31).any():
            raise ValueError("raw BEAT2 person ID is outside the mapping")
        mapped = self.raw_to_contiguous.index_select(0, raw_person_id)
        if (mapped < 0).any():
            raise ValueError("person ID is not one of SemTalk's 25 training speakers")
        style = self.proj_g(self.emb(mapped))[:, None].expand(-1, value.shape[1], -1)
        joined = torch.cat((value, style), dim=-1)
        return value + self.alpha(joined).sigmoid() * self.delta(joined)


class ConditionalSelectiveMamba(_Mamba if _Mamba is not None else nn.Module):
    """Reference external residual on input-dependent delta, B and C maps."""
    def __init__(self, width: int) -> None:
        if _Mamba is None:
            raise ImportError("mamba_ssm with selective_scan_fn is required for real training")
        super().__init__(d_model=width, d_state=16, d_conv=4, expand=2,
                         use_fast_path=False)
        self.condition_selection_proj = nn.Linear(width,
            self.dt_rank + 2 * self.d_state, bias=False)
        nn.init.zeros_(self.condition_selection_proj.weight)

    def forward(self, hidden_states: torch.Tensor,
                selection_condition: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape != selection_condition.shape:
            raise ValueError("selection condition must match Mamba input")
        batch, frames, _ = hidden_states.shape
        xz = self.in_proj(hidden_states).transpose(1, 2)
        x, z = xz.chunk(2, dim=1)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :frames])
        else:
            x = causal_conv1d_fn(x=x, weight=self.conv1d.weight[:, 0],
                bias=self.conv1d.bias, activation=self.activation)
        x_flat = x.transpose(1, 2).reshape(batch * frames, -1)
        condition_flat = selection_condition.reshape(batch * frames, -1)
        selected = self.x_proj(x_flat) + self.condition_selection_proj(condition_flat)
        dt, b_state, c_state = torch.split(selected,
            [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight).reshape(batch, frames, -1).transpose(1, 2)
        b_state = b_state.reshape(batch, frames, self.d_state).transpose(1, 2).contiguous()
        c_state = c_state.reshape(batch, frames, self.d_state).transpose(1, 2).contiguous()
        out = selective_scan_fn(x, dt, -self.A_log.float().exp(), b_state, c_state,
            self.D.float(), z=z, delta_bias=self.dt_proj.bias.float(), delta_softplus=True)
        return self.out_proj(out.transpose(1, 2))


MambaFactory = Callable[[int, bool], nn.Module]


def _real_mamba_factory(width: int, conditional: bool) -> nn.Module:
    if _Mamba is None:
        raise ImportError("mamba_ssm is missing; do not substitute a fake scan in a real run")
    return ConditionalSelectiveMamba(width) if conditional else _Mamba(
        d_model=width, d_state=16, d_conv=4, expand=2)


class BidirectionalMamba(nn.Module):
    def __init__(self, width: int, dropout: float, conditional: bool,
                 factory: MambaFactory) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width, eps=1e-5)
        self.forward_scan = factory(width, conditional)
        self.backward_scan = factory(width, conditional)
        self.dropout = nn.Dropout(dropout)
        self.conditional = conditional

    def forward(self, value: torch.Tensor,
                condition: torch.Tensor | None = None) -> torch.Tensor:
        value = self.norm(value)
        if self.conditional:
            if condition is None:
                raise ValueError("motion selective scan requires a condition")
            forward = self.forward_scan(value, condition)
            backward = self.backward_scan(value.flip(1), condition.flip(1)).flip(1)
        else:
            forward = self.forward_scan(value)
            backward = self.backward_scan(value.flip(1)).flip(1)
        return self.dropout(forward + backward)


class ConditionBlock(nn.Module):
    def __init__(self, width: int, dropout: float, factory: MambaFactory) -> None:
        super().__init__()
        self.person_gate = PersonGate(width)
        self.mamba = BidirectionalMamba(width, dropout, False, factory)

    def forward(self, value: torch.Tensor, person_id: torch.Tensor) -> torch.Tensor:
        return self.mamba(self.person_gate(value, person_id))


class MotionBlock(nn.Module):
    def __init__(self, width: int, dropout: float, factory: MambaFactory) -> None:
        super().__init__()
        self.mamba = BidirectionalMamba(width, dropout, True, factory)
        self.film1 = DenseFiLM(width)
        self.film2 = DenseFiLM(width)
        self.adain = CrossModalAdaIN(width)

    def forward(self, value: torch.Tensor, condition: torch.Tensor,
                time: torch.Tensor) -> torch.Tensor:
        conditioned = affine(condition, self.film2(time))
        value = value + affine(self.mamba(value, conditioned), self.film1(time))
        return self.adain(value, conditioned)


class GestureM2D(nn.Module):
    """Variable-horizon x0 predictor: local rotations plus Y-up [dx,dz,y]."""
    def __init__(self, mode: str = "full55", hidden_size: int = 512,
                 condition_layers: int = 4, motion_layers: int = 8,
                 dropout: float = 0.1, audio_dim: int = 1024,
                 text_dim: int = 300, *,
                 test_mamba_factory: MambaFactory | None = None) -> None:
        super().__init__()
        if mode not in MOTION_DIMENSIONS:
            raise ValueError("mode must be full55 or body22")
        if min(condition_layers, motion_layers) <= 0:
            raise ValueError("both stacks require at least one layer")
        self.mode = mode
        self.motion_dim = MOTION_DIMENSIONS[mode]
        self.hidden_size = hidden_size
        self.audio_dim, self.text_dim = audio_dim, text_dim
        self.condition_layer_count, self.motion_layer_count = condition_layers, motion_layers
        self.dropout_probability = dropout
        self.test_backend = test_mamba_factory is not None
        factory = test_mamba_factory or _real_mamba_factory

        def time_mlp() -> nn.Sequential:
            return nn.Sequential(SinusoidalPosEmb(hidden_size),
                nn.Linear(hidden_size, 4 * hidden_size), nn.Mish(),
                nn.Linear(4 * hidden_size, hidden_size))
        self.time_mlp, self.interval_mlp = time_mlp(), time_mlp()
        self.input_projection = nn.Linear(self.motion_dim, hidden_size)
        # Frame-aligned speech/text are concatenated before the reference's
        # single condition projection. No motion-derived semantic feature.
        self.condition_projection = nn.Linear(audio_dim + text_dim, hidden_size)
        self.condition_stack = nn.ModuleList(ConditionBlock(hidden_size, dropout, factory)
            for _ in range(condition_layers))
        self.motion_stack = nn.ModuleList(MotionBlock(hidden_size, dropout, factory)
            for _ in range(motion_layers))
        self.output_projection = nn.Linear(hidden_size, self.motion_dim)

    def architecture_config(self) -> dict:
        return dict(name=type(self).__name__, mode=self.mode, motion_dim=self.motion_dim,
            hidden_size=self.hidden_size, condition_layers=self.condition_layer_count,
            motion_layers=self.motion_layer_count, dropout=self.dropout_probability,
            audio_dim=self.audio_dim, text_dim=self.text_dim, person_count=25,
            raw_person_ids=list(SEMTALK_RAW_PERSON_IDS), person_injection="GenreGate-homologous",
            cross_modal_fusion="adain", condition_selection="delta/B/C", direction_fusion="sum",
            rotation_representation="local_rot6d", root_representation="dx_dz_y",
            height_axis="y", seed_pose_input=False, expression_input=False,
            test_backend=self.test_backend)

    def _forward_unpadded(self, x: torch.Tensor, times: torch.Tensor,
                          intervals: torch.Tensor, audio: torch.Tensor,
                          text: torch.Tensor, person_id: torch.Tensor) -> torch.Tensor:
        # Preserve the reference's projection/time/condition execution order.
        value = self.input_projection(x)
        time = self.time_mlp(times * 1000) + self.interval_mlp(intervals * 1000)
        condition = self.condition_projection(torch.cat((audio, text), dim=-1))
        for layer in self.condition_stack:
            condition = layer(condition, person_id)
        for layer in self.motion_stack:
            value = layer(value, condition, time)
        return self.output_projection(value)

    def forward(self, x: torch.Tensor, times: torch.Tensor, intervals: torch.Tensor,
                *, audio: torch.Tensor, text: torch.Tensor,
                person_id: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.motion_dim:
            raise ValueError(f"expected motion [B,T,{self.motion_dim}]")
        batch, frames = x.shape[:2]
        if audio.shape != (batch, frames, self.audio_dim):
            raise ValueError("audio feature shape does not match motion")
        if text.shape != (batch, frames, self.text_dim):
            raise ValueError("text feature shape does not match motion")
        if times.numel() != batch or intervals.numel() != batch:
            raise ValueError("one time and interval are required per sample")
        times, intervals = times.reshape(batch), intervals.reshape(batch)
        lengths = valid_lengths(padding_mask, batch, frames, x.device)
        if torch.all(lengths == frames):
            return self._forward_unpadded(x, times, intervals, audio, text, person_id)
        # Slice by true length instead of running backward scans over padding.
        # This makes valid outputs identical to individual unpadded sequences.
        output = x.new_zeros(x.shape)
        for length in lengths.unique().tolist():
            indices = (lengths == length).nonzero(as_tuple=False).flatten()
            predicted = self._forward_unpadded(x.index_select(0, indices)[:, :length],
                times.index_select(0, indices), intervals.index_select(0, indices),
                audio.index_select(0, indices)[:, :length], text.index_select(0, indices)[:, :length],
                person_id.index_select(0, indices))
            output = output.index_copy(0, indices, F.pad(predicted, (0, 0, 0, frames - length)))
        return output

    @torch.no_grad()
    def sample(self, noise: torch.Tensor, *, steps: int = 4,
               project_func: Callable | None = None, **condition: torch.Tensor) -> torch.Tensor:
        if self.training:
            raise RuntimeError("call model.eval() before sampling")
        if steps < 1:
            raise ValueError("steps must be positive")
        grid = torch.linspace(1, 0, steps + 1, device=noise.device, dtype=noise.dtype)
        current = noise
        mask = condition.get("padding_mask")
        if mask is not None:
            current = current * mask[..., None]
        for index in range(steps):
            t = grid[index].expand(noise.shape[0])
            r = grid[index + 1].expand(noise.shape[0])
            prediction = self(current, t, t - r, **condition)
            if project_func is not None:
                prediction = project_func(prediction)
            ratio = (r / t.clamp_min(1e-5)).reshape(-1, 1, 1)
            current = ratio * current + (1 - ratio) * prediction
            if mask is not None:
                current = current * mask[..., None]
        return current


__all__ = ["GestureM2D", "PersonGate", "CrossModalAdaIN", "ConditionalSelectiveMamba",
           "SEMTALK_RAW_PERSON_IDS", "SMPLX_PARENTS", "MOTION_DIMENSIONS", "valid_lengths"]
