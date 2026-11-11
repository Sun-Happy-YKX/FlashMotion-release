from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd.functional import jvp

from .representation import recover_joints_from_ric


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    expanded = mask
    while expanded.ndim < value.ndim:
        expanded = expanded.unsqueeze(-1)
    expanded = expanded.to(dtype=value.dtype)
    feature_count = value[0, 0].numel() if value.ndim > 2 else 1
    denominator = expanded.sum().clamp_min(1.0) * feature_count
    return (value * expanded).sum() / denominator


def _cosine_alphas_cumprod(timesteps: int, offset: float = 0.008) -> torch.Tensor:
    """Nichol-Dhariwal cosine VP schedule, returned at the noisy states."""

    grid = torch.arange(timesteps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(
        ((grid / timesteps + offset) / (1.0 + offset)) * torch.pi / 2.0
    ).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = (1.0 - alpha_bar[1:] / alpha_bar[:-1]).clamp(max=0.999)
    return torch.cumprod(1.0 - betas, dim=0).float()


class T2MMeanFlow(nn.Module):
    """x0-parameterized MeanFlow ablations for the 263D T2M representation."""

    def __init__(
        self,
        decoder: nn.Module,
        mean: torch.Tensor,
        std: torch.Tensor,
        min_t: float = 0.05,
        min_interval: float = 0.05,
        fm_time_weight_power: float = 2.0,
        loss_weights: dict[str, float] | None = None,
        x0_variant: str = "tr_t0_full",
        cfg_dropout_prob: float = 0.1,
        cfg_omega: float = 2.0,
    ):
        super().__init__()
        self.decoder = decoder
        self.register_buffer("motion_mean", mean.float().reshape(1, 1, -1))
        self.register_buffer("motion_std", std.float().reshape(1, 1, -1))
        self.min_t = float(min_t)
        self.min_interval = float(min_interval)
        self.fm_time_weight_power = float(fm_time_weight_power)
        valid_variants = {
            "x0_flow",
            "diffusion",
            "t0_only",
            "tr_only",
            "tr_t0_aux",
            "tr_t0_full",
            "tr_t0_full_avg",
            "tr_cfg_baked",
            "tr_t0_aux_cfg_baked",
            "tr_t0_full_fm_only",
            "tr_t0_full_rec_only",
            "tr_t0_aux_rec_only",
        }
        if x0_variant not in valid_variants:
            raise ValueError(f"Unknown x0 variant: {x0_variant}")
        self.x0_variant = x0_variant
        self.diffusion_timesteps = 1000
        if self.x0_variant == "diffusion":
            self.register_buffer(
                "diffusion_alphas_cumprod",
                _cosine_alphas_cumprod(self.diffusion_timesteps),
            )
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.cfg_omega = float(cfg_omega)
        self.loss_weights = {
            "fm": 1.0,
            "rec": 0.636,
            "vel": 0.323,
            "acc": 0.2,
            "fk": 0.646,
            "floor": 0.0,
        }
        if loss_weights:
            self.loss_weights.update(loss_weights)
        if self.x0_variant == "tr_t0_full_fm_only":
            self.loss_weights.update(
                {"rec": 0.0, "vel": 0.0, "acc": 0.0, "fk": 0.0, "floor": 0.0}
            )
        elif self.x0_variant in {"tr_t0_full_rec_only", "tr_t0_aux_rec_only"}:
            self.loss_weights.update({"vel": 0.0, "acc": 0.0, "fk": 0.0, "floor": 0.0})
        if not 0.0 <= self.cfg_dropout_prob <= 1.0:
            raise ValueError("cfg_dropout_prob must be in [0, 1]")
        if self.cfg_omega < 0.0:
            raise ValueError("cfg_omega must be non-negative")

    def sample_two_timesteps(self, batch: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        lower_t = max(self.min_t, self.min_interval)
        t = lower_t + (1.0 - lower_t) * torch.rand((batch, 1, 1), device=device)
        r = torch.rand((batch, 1, 1), device=device) * (t - self.min_interval)
        return t, r

    @staticmethod
    def x_r_and_velocity(
        x_t: torch.Tensor, x0_pred: torch.Tensor, t: torch.Tensor, r: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = (r / t.clamp_min(1e-5)).clamp(0.0, 1.0)
        x_r = ratio * x_t + (1.0 - ratio) * x0_pred
        velocity = (x_t - x_r) / (t - r).clamp_min(1e-5)
        return x_r, velocity

    def _alignment_losses(
        self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        loss_names = ("rec", "vel", "acc", "fk", "floor")
        if not any(self.loss_weights[name] for name in loss_names):
            zero = prediction.new_zeros(())
            return {name: zero for name in loss_names}
        rec = _masked_mean(F.mse_loss(prediction, target, reduction="none"), mask)
        if not any(self.loss_weights[name] for name in ("vel", "acc", "fk", "floor")):
            zero = prediction.new_zeros(())
            return {"rec": rec, "vel": zero, "acc": zero, "fk": zero, "floor": zero}
        prediction_raw = prediction * self.motion_std + self.motion_mean
        target_raw = target.detach() * self.motion_std + self.motion_mean
        prediction_joints = recover_joints_from_ric(prediction_raw)
        target_joints = recover_joints_from_ric(target_raw)
        fk = _masked_mean(
            F.mse_loss(prediction_joints, target_joints, reduction="none"), mask
        )

        velocity_mask = mask[:, 1:] & mask[:, :-1]
        prediction_velocity = prediction_joints[:, 1:] - prediction_joints[:, :-1]
        target_velocity = target_joints[:, 1:] - target_joints[:, :-1]
        vel = _masked_mean(
            F.mse_loss(prediction_velocity, target_velocity, reduction="none"),
            velocity_mask,
        )

        acceleration_mask = velocity_mask[:, 1:] & velocity_mask[:, :-1]
        prediction_acceleration = prediction_velocity[:, 1:] - prediction_velocity[:, :-1]
        target_acceleration = target_velocity[:, 1:] - target_velocity[:, :-1]
        acc = _masked_mean(
            F.mse_loss(prediction_acceleration, target_acceleration, reduction="none"),
            acceleration_mask,
        )

        # HumanML3D uses Y as the absolute height axis and XZ as the horizontal
        # plane.  The final four representation channels are the official foot
        # contacts for left/right ankle and toe joints.  This loss is computed
        # only after the standard 263D coordinate recovery, never in normalized
        # feature space.
        foot_indices = (7, 10, 8, 11)
        prediction_feet = prediction_joints[:, :, foot_indices]
        foot_height = prediction_feet[..., 1]
        valid_foot_mask = mask.unsqueeze(-1).expand_as(foot_height)
        lowest_foot = foot_height.masked_fill(~valid_foot_mask, torch.inf).amin((1, 2))
        floor_anchor = lowest_foot.square().mean()
        floor_penetration = _masked_mean(F.relu(-foot_height).square(), mask)

        target_contact = target_raw[..., -4:].detach().clamp(0.0, 1.0)
        foot_velocity = prediction_feet[:, 1:] - prediction_feet[:, :-1]
        contact_skate = foot_velocity.square().sum(-1) * target_contact[:, 1:]
        contact_skate = _masked_mean(contact_skate, velocity_mask)
        floor = floor_anchor + floor_penetration + contact_skate
        return {"rec": rec, "vel": vel, "acc": acc, "fk": fk, "floor": floor}

    def _branch(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        path_velocity: torch.Tensor,
        target: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
        cond_drop_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        interval = t - r
        fork_devices = [x_t.device.index] if x_t.device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            x0_pred = self.decoder(
                x_t,
                condition,
                mask,
                t.reshape(-1),
                interval.reshape(-1),
                cond_drop_mask=cond_drop_mask,
            )
            _, predicted_velocity = self.x_r_and_velocity(x_t, x0_pred, t, r)

        def velocity_function(
            current_x_t: torch.Tensor,
            current_t: torch.Tensor,
            current_r: torch.Tensor,
        ) -> torch.Tensor:
            current_x0 = self.decoder(
                current_x_t,
                condition,
                mask,
                current_t.reshape(-1),
                (current_t - current_r).reshape(-1),
                cond_drop_mask=cond_drop_mask,
            )
            return self.x_r_and_velocity(current_x_t, current_x0, current_t, current_r)[1]

        _, derivative = jvp(
            velocity_function,
            (x_t, t, r),
            (path_velocity, torch.ones_like(t), torch.zeros_like(r)),
        )
        target_velocity = (path_velocity - interval * derivative).detach()
        per_frame = F.mse_loss(predicted_velocity, target_velocity, reduction="none").mean(-1)
        per_sample = (per_frame * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        per_sample = per_sample * t.reshape(-1).pow(self.fm_time_weight_power)
        fm = (per_sample / (per_sample.detach().add(1e-3).pow(0.75))).mean()
        return {"fm": fm, **self._alignment_losses(x0_pred, target, mask)}

    def _direct_x0_flow_branch(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        path_velocity: torch.Tensor,
        target: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Ordinary x0 Flow Matching: no endpoint r and no MeanFlow JVP."""

        x0_pred = self.decoder(
            x_t,
            condition,
            mask,
            t.reshape(-1),
            torch.zeros_like(t).reshape(-1),
        )
        predicted_velocity = (x_t - x0_pred) / t.clamp_min(1e-5)
        per_frame = F.mse_loss(
            predicted_velocity, path_velocity, reduction="none"
        ).mean(-1)
        per_sample = (per_frame * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        per_sample = per_sample * t.reshape(-1).pow(self.fm_time_weight_power)
        fm = (per_sample / per_sample.detach().add(1e-3).pow(0.75)).mean()
        return {"fm": fm, **self._alignment_losses(x0_pred, target, mask)}

    def _diffusion_branch(
        self,
        motion: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Cosine VP diffusion with direct x0 prediction."""

        batch = motion.shape[0]
        timestep = torch.randint(
            0, self.diffusion_timesteps, (batch,), device=motion.device
        )
        alpha = self.diffusion_alphas_cumprod[timestep].reshape(batch, 1, 1)
        valid = mask[..., None].to(dtype=motion.dtype)
        noise = torch.randn_like(motion) * valid
        x_t = (alpha.sqrt() * motion + (1.0 - alpha).sqrt() * noise) * valid
        time = timestep.to(motion.dtype) / (self.diffusion_timesteps - 1)
        x0_pred = self.decoder(
            x_t,
            condition,
            mask,
            time,
            torch.zeros_like(time),
        )
        per_frame = F.mse_loss(x0_pred, motion, reduction="none").mean(-1)
        per_sample = (per_frame * mask).sum(-1) / mask.sum(-1).clamp_min(1)
        fm = (per_sample / per_sample.detach().add(1e-3).pow(0.75)).mean()
        return {"fm": fm, **self._alignment_losses(x0_pred, motion, mask)}

    def _baked_cfg_targets(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        r: torch.Tensor,
        path_velocity: torch.Tensor,
        condition: torch.Tensor,
        mask: torch.Tensor,
        drop_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build stopped baked-CFG velocity and x0 targets for one branch."""

        all_uncond = torch.ones(
            (x_t.shape[0],), device=x_t.device, dtype=torch.bool
        )
        with torch.no_grad():
            x0_uncond = self.decoder(
                x_t,
                condition,
                mask,
                t.reshape(-1),
                (t - r).reshape(-1),
                cond_drop_mask=all_uncond,
            )
            velocity_uncond = (x_t - x0_uncond) / t.clamp_min(1e-5)
            guided_velocity = velocity_uncond + self.cfg_omega * (
                path_velocity - velocity_uncond
            )
            driving_velocity = torch.where(
                drop_mask[:, None, None], path_velocity, guided_velocity
            )
            guided_x0 = x_t - t * driving_velocity
        return driving_velocity, guided_x0

    def forward(
        self, motion: torch.Tensor, condition: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        valid = mask[..., None].to(dtype=motion.dtype)
        batch = motion.shape[0]
        noise = torch.randn_like(motion) * valid
        if self.x0_variant == "diffusion":
            raw = self._diffusion_branch(motion, condition, mask)
        elif self.x0_variant == "x0_flow":
            t = self.min_t + (1.0 - self.min_t) * torch.rand(
                (batch, 1, 1), device=motion.device
            )
            x_t = ((1.0 - t) * motion + t * noise) * valid
            path_velocity = (noise - motion) * valid
            raw = self._direct_x0_flow_branch(
                x_t, t, path_velocity, motion, condition, mask
            )
        elif self.x0_variant == "t0_only":
            t = self.min_t + (1.0 - self.min_t) * torch.rand(
                (batch, 1, 1), device=motion.device
            )
            r = torch.zeros_like(t)
            x_t = ((1.0 - t) * motion + t * noise) * valid
            path_velocity = (noise - motion) * valid
            raw = self._branch(x_t, t, r, path_velocity, motion, condition, mask)
        else:
            t, r = self.sample_two_timesteps(batch, motion.device)
            x_t = ((1.0 - t) * motion + t * noise) * valid
            path_velocity = (noise - motion) * valid
            if self.x0_variant == "tr_only":
                raw = self._branch(x_t, t, r, path_velocity, motion, condition, mask)
            elif self.x0_variant in {"tr_t0_aux", "tr_t0_aux_rec_only"}:
                tr = self._branch(x_t, t, r, path_velocity, motion, condition, mask)
                t0 = self.min_t + (1.0 - self.min_t) * torch.rand(
                    (batch, 1, 1), device=motion.device
                )
                noise_t0 = torch.randn_like(motion) * valid
                x_t0 = ((1.0 - t0) * motion + t0 * noise_t0) * valid
                x0_t0 = self.decoder(
                    x_t0, condition, mask, t0.reshape(-1), t0.reshape(-1)
                )
                t0_alignment = self._alignment_losses(x0_t0, motion, mask)
                raw = {
                    "fm": tr["fm"],
                    **{name: tr[name] + t0_alignment[name] for name in t0_alignment},
                }
            elif self.x0_variant in {
                "tr_t0_full",
                "tr_t0_full_avg",
                "tr_t0_full_fm_only",
                "tr_t0_full_rec_only",
            }:
                tr = self._branch(x_t, t, r, path_velocity, motion, condition, mask)
                t0 = self._branch(
                    x_t,
                    t,
                    torch.zeros_like(t),
                    path_velocity,
                    motion,
                    condition,
                    mask,
                )
                scale = 0.5 if self.x0_variant == "tr_t0_full_avg" else 1.0
                raw = {name: scale * (tr[name] + t0[name]) for name in tr}
            elif self.x0_variant in {"tr_cfg_baked", "tr_t0_aux_cfg_baked"}:
                drop_mask = torch.rand((batch,), device=motion.device) < self.cfg_dropout_prob
                driving_velocity, guided_x0 = self._baked_cfg_targets(
                    x_t, t, r, path_velocity, condition, mask, drop_mask
                )
                tr = self._branch(
                    x_t,
                    t,
                    r,
                    driving_velocity,
                    guided_x0,
                    condition,
                    mask,
                    cond_drop_mask=drop_mask,
                )
                if self.x0_variant == "tr_cfg_baked":
                    raw = tr
                else:
                    t0 = self.min_t + (1.0 - self.min_t) * torch.rand(
                        (batch, 1, 1), device=motion.device
                    )
                    noise_t0 = torch.randn_like(motion) * valid
                    x_t0 = ((1.0 - t0) * motion + t0 * noise_t0) * valid
                    x0_t0 = self.decoder(
                        x_t0,
                        condition,
                        mask,
                        t0.reshape(-1),
                        t0.reshape(-1),
                        cond_drop_mask=drop_mask,
                    )
                    # Keep the winning aux branch as a true clean-motion
                    # reconstruction anchor. Only the main MeanFlow branch is
                    # baked-guided; the shared drop mask trains the null token.
                    t0_alignment = self._alignment_losses(x0_t0, motion, mask)
                    raw = {
                        "fm": tr["fm"],
                        **{
                            name: tr[name] + t0_alignment[name]
                            for name in t0_alignment
                        },
                    }
            else:
                raise AssertionError(f"Unhandled x0 variant: {self.x0_variant}")
        weighted = {name: raw[name] * self.loss_weights[name] for name in raw}
        total = sum(weighted.values())
        return total, {f"raw/{name}": value for name, value in raw.items()} | {
            f"weighted/{name}": value for name, value in weighted.items()
        }

    @torch.no_grad()
    def _ddim_sample(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Deterministic DDIM (eta=0) with exactly ``steps`` network calls."""

        shape = (condition.shape[0], condition.shape[1], self.motion_mean.shape[-1])
        x = torch.randn(shape, device=condition.device, generator=generator)
        valid = mask[..., None].to(dtype=x.dtype)
        x = x * valid
        timesteps = torch.linspace(
            self.diffusion_timesteps - 1,
            0,
            steps,
            device=condition.device,
        ).round().long().tolist()
        next_timesteps = timesteps[1:] + [-1]
        for timestep, next_timestep in zip(timesteps, next_timesteps):
            index = torch.full(
                (shape[0],), timestep, device=condition.device, dtype=torch.long
            )
            time = index.to(x.dtype) / (self.diffusion_timesteps - 1)
            x0 = self.decoder.infer_pred(
                x, condition, mask, time, torch.zeros_like(time)
            )
            if next_timestep < 0:
                x = x0
                continue
            alpha = self.diffusion_alphas_cumprod[timestep].to(x.dtype)
            alpha_next = self.diffusion_alphas_cumprod[next_timestep].to(x.dtype)
            predicted_noise = (
                x - alpha.sqrt() * x0
            ) / (1.0 - alpha).sqrt().clamp_min(1e-5)
            x = (
                alpha_next.sqrt() * x0
                + (1.0 - alpha_next).sqrt() * predicted_noise
            ) * valid
        return x * valid

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        mask: torch.Tensor,
        steps: int,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if self.x0_variant == "diffusion":
            return self._ddim_sample(condition, mask, steps, generator)
        shape = (condition.shape[0], condition.shape[1], self.motion_mean.shape[-1])
        x = torch.randn(shape, device=condition.device, generator=generator)
        x = x * mask[..., None]
        grid = torch.linspace(1.0, 0.0, steps + 1, device=condition.device)
        for index in range(steps):
            t = torch.full((shape[0],), grid[index], device=condition.device)
            r = torch.full((shape[0],), grid[index + 1], device=condition.device)
            if self.x0_variant == "x0_flow":
                interval = torch.zeros_like(t)
            else:
                interval = t if self.x0_variant == "t0_only" else t - r
            x0 = self.decoder.infer_pred(x, condition, mask, t, interval)
            x, _ = self.x_r_and_velocity(
                x, x0, t[:, None, None], r[:, None, None]
            )
            x = x * mask[..., None]
        return x
