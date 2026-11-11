from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
from torch import Tensor
from torch.nn import functional as F
from model.utils import SinusoidalPosEmb
from model.rotary_embedding_torch import RotaryEmbedding
from mamba_ssm import Mamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn
except ImportError:
    causal_conv1d_fn = None

class DenseFiLM(nn.Module):
    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(nn.Mish(), nn.Linear(embed_channels, embed_channels * 2))

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift

def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


class CrossModalAdaIN(nn.Module):
    """Identity-initialized AdaIN modulation driven by frame-wise conditions."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(d_model, d_model * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, condition, mask=None):
        if mask is None:
            mean = x.mean(dim=1, keepdim=True)
            variance = (x - mean).square().mean(dim=1, keepdim=True)
            valid = None
        else:
            valid = mask[..., None].to(dtype=x.dtype)
            count = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
            mean = (x * valid).sum(dim=1, keepdim=True) / count
            variance = ((x - mean).square() * valid).sum(dim=1, keepdim=True) / count

        std = torch.sqrt(variance + self.eps)
        normalized = (x - mean) / std
        scale_delta, shift_delta = torch.tanh(self.modulation(condition)).chunk(2, dim=-1)
        output = mean + std * ((1.0 + scale_delta) * normalized + shift_delta)
        if valid is not None:
            output = output * valid
        return output


class ConditionalSelectiveMamba(Mamba):
    """Mamba whose input-dependent delta/B/C also depend on a frame-wise condition."""

    def __init__(self, d_model, **kwargs):
        # The fused Mamba kernel cannot accept an external delta/B/C residual.
        super().__init__(d_model=d_model, use_fast_path=False, **kwargs)
        self.condition_selection_proj = nn.Linear(
            d_model, self.dt_rank + 2 * self.d_state, bias=False
        )
        nn.init.zeros_(self.condition_selection_proj.weight)

    def forward(self, hidden_states, selection_condition, inference_params=None):
        if inference_params is not None:
            raise NotImplementedError("Conditional selective Mamba only supports full-sequence inference")
        if selection_condition.shape != hidden_states.shape:
            raise ValueError(
                f"selection condition {selection_condition.shape} must match input {hidden_states.shape}"
            )

        batch, seqlen, _ = hidden_states.shape
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )

        x_flat = rearrange(x, "b d l -> (b l) d")
        condition_flat = rearrange(selection_condition, "b l d -> (b l) d")
        # This is the only condition injection in the SSM: it directly changes
        # the input-dependent discretization delta and the B/C selection maps.
        x_dbl = self.x_proj(x_flat) + self.condition_selection_proj(condition_flat)
        dt, B, C = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        A = -torch.exp(self.A_log.float())
        y = selective_scan_fn(
            x,
            dt,
            A,
            B,
            C,
            self.D.float(),
            z=z,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
        )
        return self.out_proj(rearrange(y, "b d l -> b l d"))


class IntraModalMamba(nn.Module):
    def __init__(
        self,
        d_model,
        dropout,
        layer_norm_eps,
        condition_selection=False,
        direction_fusion="sum",
    ):
        super().__init__()
        if direction_fusion not in {"sum", "cond_gate"}:
            raise ValueError(f"Unsupported direction fusion: {direction_fusion}")
        self.condition_selection = condition_selection
        self.direction_fusion = direction_fusion
        mamba_cls = ConditionalSelectiveMamba if condition_selection else Mamba
        self.mamba_forward = mamba_cls(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.mamba_backward = mamba_cls(d_model=d_model, d_state=16, d_conv=4, expand=2)
        self.direction_gate = (
            nn.Linear(d_model, d_model) if direction_fusion == "cond_gate" else None
        )
        if self.direction_gate is not None:
            nn.init.zeros_(self.direction_gate.weight)
            nn.init.zeros_(self.direction_gate.bias)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, condition=None):
        x = self.norm(x)
        if self.condition_selection:
            if condition is None:
                raise ValueError("condition is required when condition_selection=True")
            x1 = self.mamba_forward(x, condition)
            x2 = self.mamba_backward(x.flip(1), condition.flip(1)).flip(1)
        else:
            x1 = self.mamba_forward(x)
            x2 = self.mamba_backward(x.flip(1)).flip(1)

        if self.direction_gate is None:
            fused = x1 + x2
        else:
            if condition is None:
                raise ValueError("condition is required for conditional direction gating")
            balance = torch.tanh(self.direction_gate(condition))
            # balance=0 exactly reproduces the original bidirectional sum.
            fused = (1.0 + balance) * x1 + (1.0 - balance) * x2
        x = self.dropout(fused)
        return x


class DanceFiLMLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=True, cross_modal_fusion="add", condition_selection=False, direction_fusion="sum"):
        super().__init__()

        if cross_modal_fusion not in {"add", "adain"}:
            raise ValueError(f"Unsupported cross-modal fusion: {cross_modal_fusion}")
        self.cross_modal_fusion = cross_modal_fusion

        self.intra_mamba = IntraModalMamba(
            d_model,
            dropout,
            layer_norm_eps,
            condition_selection=condition_selection,
            direction_fusion=direction_fusion,
        )

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.cross_adain = CrossModalAdaIN(d_model, layer_norm_eps) if cross_modal_fusion == "adain" else None

    # x, cond, t
    def forward(self, x, cond, genre, t):
        conditioned = featurewise_affine(cond, self.film2(t))
        # intra-modal -> film -> residual
        x_1 = self.intra_mamba(x, condition=conditioned)
        x = x + featurewise_affine(x_1, self.film1(t))

        # Cross-modal ablation: the original residual add or condition-driven AdaIN.
        if self.cross_modal_fusion == "add":
            x = x + conditioned
        else:
            x = self.cross_adain(x, conditioned)
        return x


class GenreGate(nn.Module):
    def __init__(self, num_genres: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(num_genres, d_model)
        self.proj_g = nn.Linear(d_model, d_model)
        self.delta = nn.Sequential(
            nn.Linear(2*d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.alpha = nn.Sequential(
            nn.Linear(2*d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model)
        )

        nn.init.constant_(self.alpha[-1].bias, -4.0)  

    def forward(self, x, genre):
        # x: [B,T,C], genre: [B]
        g = self.proj_g(self.emb(genre))[:, None, :]         # [B,1,C]
        z = torch.cat([x, g.expand(-1, x.size(1), -1)], -1)  # [B,T,2C]
        delta = self.delta(z)                                # [B,T,C]
        alpha = torch.sigmoid(self.alpha(z))                 # [B,T,C]
        return x + alpha * delta


class MusicFiLMLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=F.relu, layer_norm_eps=1e-5, batch_first=True, num_genres=16, use_genre=True):
        super().__init__()

        self.intra_mamba = IntraModalMamba(d_model, dropout, layer_norm_eps)
        self.genre_gate = GenreGate(num_genres, d_model) if use_genre else None
        self.film = DenseFiLM(d_model)

    def forward(self, x, genre):
        if self.genre_gate is not None:
            x = self.genre_gate(x, genre)
        x = self.intra_mamba(x)
        return x


class DanceDecoder(nn.Module):
    def __init__(self, nfeats, seq_len=150, latent_dim=256, ff_size=1024, num_layers=4, num_heads=4, dropout=0.2, cond_feature_dim=4800, activation=F.gelu, use_rotary=True, num_genres=16, cross_modal_fusion="add", condition_selection=False, direction_fusion="sum", use_genre=True, **kwargs):
        super().__init__()

        self.use_genre = bool(use_genre)
        self.num_genres = int(num_genres)
        self.null_genre = self.num_genres - 1
        self.cross_modal_fusion = cross_modal_fusion
        self.condition_selection = condition_selection
        self.direction_fusion = direction_fusion

        # time embedding processing
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(latent_dim), nn.Linear(latent_dim, latent_dim * 4), nn.Mish(), nn.Linear(latent_dim * 4, latent_dim),)
        self.intv_mlp = nn.Sequential(SinusoidalPosEmb(latent_dim), nn.Linear(latent_dim, latent_dim * 4), nn.Mish(), nn.Linear(latent_dim * 4, latent_dim),)


        # input and condition projection
        self.input_projection = nn.Linear(nfeats, latent_dim)
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.final_layer = nn.Linear(latent_dim, nfeats)

        # Dance decoder
        self.dance_num_layers = 8
        self.dancelayerstack = nn.ModuleList([])
        for _ in range(self.dance_num_layers):
            self.dancelayerstack.append(DanceFiLMLayer(latent_dim, num_heads, ff_size, dropout, activation, batch_first=True, cross_modal_fusion=cross_modal_fusion, condition_selection=condition_selection, direction_fusion=direction_fusion))

        # Music encoder
        self.music_num_layers = 4
        self.musiclayerstack = nn.ModuleList([])
        for _ in range(self.music_num_layers):
            self.musiclayerstack.append(MusicFiLMLayer(latent_dim, num_heads, ff_size, dropout, activation, batch_first=True, num_genres=self.num_genres, use_genre=self.use_genre))
    
        self.null_cond_embed = nn.Parameter(torch.zeros(1, 1, latent_dim))
        self.guidance_weight = 4


    def forward(
        self,
        x,
        cond_embed,
        genre,
        times,
        interval,
        cond_drop_prob=0.0,
        cond_drop_mask=None,
    ):
        batch_size, device = x.shape[0], x.device

        # project to latent space
        x = self.input_projection(x)
        t = self.time_mlp(times * 1000)
        intv = self.intv_mlp(interval * 1000)
        t = t + intv
        cond = self.cond_projection(cond_embed)

        if cond_drop_mask is None and cond_drop_prob > 0.0:
            cond_drop_mask = torch.rand(batch_size, device=device) < cond_drop_prob
        if cond_drop_mask is not None:
            cond_drop_mask = cond_drop_mask.to(device=device, dtype=torch.bool)
            if cond_drop_mask.ndim != 1 or cond_drop_mask.shape[0] != batch_size:
                raise ValueError(
                    f"Expected cond_drop_mask [{batch_size}], got {cond_drop_mask.shape}"
                )
            if self.num_genres < 17:
                raise RuntimeError(
                    "Null conditioning requires a decoder with 17 genre embeddings"
                )
            cond = torch.where(
                cond_drop_mask[:, None, None],
                self.null_cond_embed.to(cond.dtype).expand(batch_size, cond.shape[1], -1),
                cond,
            )
            genre = torch.where(
                cond_drop_mask,
                torch.full_like(genre, self.null_genre),
                genre,
            )

        # Pass through the transformer decoder
        for i in range(self.music_num_layers):
            cond = self.musiclayerstack[i](cond, genre)

        # Pass through the transformer decoder
        for i in range(self.dance_num_layers):
            x = self.dancelayerstack[i](x, cond, genre, t)

        # project to SMPL space
        output = self.final_layer(x)
        return output

    def infer_pred(self, x, cond_embed, genre, times, interval):
        # uncondition_v = self.forward(x, cond_embed, genre, times, interval, cond_drop_prob=1.0)
        # condition_v = self.forward(x, cond_embed, genre, times, interval, cond_drop_prob=0.0)
        # return uncondition_v + (condition_v - uncondition_v) * self.guidance_weight
        return self.forward(x, cond_embed, genre, times, interval, cond_drop_prob=0.0)
