from __future__ import annotations

import torch
from torch import nn

from model.model import DanceDecoder


class OneShotT2MDecoder(nn.Module):
    """FlowerDance's bidirectional Mamba backbone with frame-wise text features."""

    def __init__(
        self,
        motion_dim: int = 263,
        text_dim: int = 4096,
        latent_dim: int = 512,
        dropout: float = 0.1,
        enable_unconditional: bool = False,
        cross_modal_fusion: str = "add",
        condition_selection: bool = False,
        direction_fusion: str = "sum",
    ):
        super().__init__()
        self.enable_unconditional = bool(enable_unconditional)
        self.decoder = DanceDecoder(
            nfeats=motion_dim,
            seq_len=200,
            latent_dim=latent_dim,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=dropout,
            cond_feature_dim=text_dim,
            # DanceDecoder reserves genre index 16 for its learned null branch.
            num_genres=17 if self.enable_unconditional else 1,
            cross_modal_fusion=cross_modal_fusion,
            condition_selection=condition_selection,
            direction_fusion=direction_fusion,
        )

        # The learned null token is trainable only for the CFG ablation.
        self.decoder.null_cond_embed.requires_grad_(self.enable_unconditional)
        # MusicFiLMLayer keeps a legacy FiLM module that DanceDecoder never
        # calls. Freeze it so DDP does not treat it as a broken training graph.
        for layer in self.decoder.musiclayerstack:
            layer.film.requires_grad_(False)

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
        time: torch.Tensor,
        interval: torch.Tensor,
        cond_drop_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        valid = mask[..., None].to(dtype=x.dtype)
        x = x * valid
        condition = condition.to(dtype=x.dtype) * valid
        genre = torch.zeros(x.shape[0], device=x.device, dtype=torch.long)
        output = self.decoder(
            x,
            condition,
            genre,
            time,
            interval,
            cond_drop_mask=cond_drop_mask,
            mask=mask,
        )
        return output * valid

    def infer_pred(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
        time: torch.Tensor,
        interval: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(x, condition, mask, time, interval)
