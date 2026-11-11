import copy
import os
import pickle
from pathlib import Path
from functools import partial
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from einops import rearrange, reduce, repeat
from p_tqdm import p_map
from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)
from tqdm import tqdm

from dataset.quaternion import ax_from_6v, quat_slerp
from dataset.motion_representation import (
    CONTACT_DIM,
    FOOT_JOINTS,
    HEIGHT_AXIS,
    pose_to_joints,
    root_velocity_to_position,
)
from vis import skeleton_render

from .utils import extract, make_beta_schedule

def identity(t, *args, **kwargs):
    return t

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
            current_model.parameters(), ma_model.parameters()
        ):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

from torch.autograd.functional import jvp
class FlowMatching(nn.Module):
    def __init__(
        self,
        model,
        horizon,
        repr_dim,
        smpl,
        sigma=0.1,
        loss_type="l2",
        x0_variant="tr_only",
        loss_weights=None,
        min_t=0.05,
        min_interval=0.05,
        fm_time_weight_power=2.0,
        cfg_dropout_prob=0.1,
        cfg_omega=2.0,
    ):
        super().__init__()
        self.horizon = horizon
        self.repr_dim = repr_dim
        self.model = model
        self.smpl = smpl
        self.sigma = sigma
        self.loss_fn = F.mse_loss if loss_type == "l2" else F.l1_loss

        # EMA
        self.ema = EMA(0.9999)
        self.master_model = copy.deepcopy(self.model)

        # p2 weighting (continuous version)
        self.alpha = 2
        self.jvp_fn = jvp
        valid_variants = {
            "t0_only",
            "tr_only",
            "tr_t0",
            "tr_t0_aux",
            "tr_t0_full",
            "tr_cfg_baked",
        }
        if x0_variant not in valid_variants:
            raise ValueError(f"Unknown x0 variant: {x0_variant}")
        # Keep the old name loadable, but record its precise semantics in new
        # checkpoints: random t->r MeanFlow plus an alignment-only t->0 branch.
        self.x0_variant = "tr_t0_aux" if x0_variant == "tr_t0" else x0_variant
        self.loss_weights = {
            "fm": 1.0,
            "rec": 0.636,
            "vel": 0.323,
            "acc": 0.2,
            "fk": 0.646,
            "floor": 1.0,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)
        self.normalizer = None
        self.min_t = float(min_t)
        self.min_interval = float(min_interval)
        self.fm_time_weight_power = float(fm_time_weight_power)
        self.cfg_dropout_prob = float(cfg_dropout_prob)
        self.cfg_omega = float(cfg_omega)
        if not 0.0 < self.min_t <= 1.0:
            raise ValueError(f"min_t must be in (0, 1], got {self.min_t}")
        if not 0.0 < self.min_interval <= 1.0:
            raise ValueError(
                f"min_interval must be in (0, 1], got {self.min_interval}"
            )
        if self.fm_time_weight_power < 0.0:
            raise ValueError("fm_time_weight_power must be non-negative")
        if not 0.0 <= self.cfg_dropout_prob <= 1.0:
            raise ValueError("cfg_dropout_prob must be in [0, 1]")
        if self.cfg_omega < 0.0:
            raise ValueError("cfg_omega must be non-negative")

    def sample_two_timesteps(self, b, device):
        # Match few-step inference: never train at a nearly clean t and never
        # ask the model to distinguish two almost identical time points.
        lower_t = max(self.min_t, self.min_interval)
        t = lower_t + (1.0 - lower_t) * torch.rand(
            (b, 1, 1), device=device
        )
        max_r = t - self.min_interval
        r = torch.rand((b, 1, 1), device=device) * max_r
        return t, r

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def _inverse_normalize(self, pose):
        if self.normalizer is None:
            raise RuntimeError("Motion normalizer must be set before computing losses")
        scaler = self.normalizer.scaler
        offset = scaler.min_.to(device=pose.device, dtype=pose.dtype)
        scale = scaler.scale_.to(device=pose.device, dtype=pose.dtype)
        return (pose - offset) / scale

    @staticmethod
    def _x_r_and_velocity(x_t, x0_pred, t, r, eps=1e-5):
        """Back-project x0 to x_r, then obtain the mean x_t->x_r velocity."""

        t_safe = t.clamp_min(eps)
        ratio = (r / t_safe).clamp(0.0, 1.0)
        x_r = ratio * x_t + (1.0 - ratio) * x0_pred
        interval = (t - r).clamp_min(eps)
        interval_velocity = (x_t - x_r) / interval
        return x_r, interval_velocity

    def _alignment_losses(self, x0_pred, x0):
        rec_loss = self.loss_fn(x0_pred, x0, reduction="none")
        rec_loss = reduce(rec_loss, "b ... -> b", "mean").mean()

        pred_pose = self._inverse_normalize(x0_pred)
        target_pose = self._inverse_normalize(x0.detach())
        pred_joints = pose_to_joints(pred_pose, self.smpl)
        target_joints = pose_to_joints(target_pose, self.smpl)

        fk_loss = self.loss_fn(pred_joints, target_joints, reduction="none")
        fk_loss = reduce(fk_loss, "b ... -> b", "mean").mean()

        pred_velocity = pred_joints[:, 1:] - pred_joints[:, :-1]
        target_velocity = target_joints[:, 1:] - target_joints[:, :-1]
        vel_loss = self.loss_fn(
            pred_velocity, target_velocity, reduction="none"
        )
        vel_loss = reduce(vel_loss, "b ... -> b", "mean").mean()

        pred_acc = pred_velocity[:, 1:] - pred_velocity[:, :-1]
        target_acc = target_velocity[:, 1:] - target_velocity[:, :-1]
        acc_loss = self.loss_fn(pred_acc, target_acc, reduction="none")
        acc_loss = reduce(acc_loss, "b ... -> b", "mean").mean()

        pred_feet = pred_joints[..., list(FOOT_JOINTS), :]
        pred_foot_z = pred_feet[..., HEIGHT_AXIS]
        predicted_floor = pred_foot_z.amin(dim=(1, 2))
        floor_anchor = predicted_floor.square()
        penetration = F.relu(-pred_foot_z).square().mean(dim=(1, 2))

        target_contact = target_pose[..., :CONTACT_DIM].clamp(0.0, 1.0)
        pred_foot_velocity = pred_feet[:, 1:] - pred_feet[:, :-1]
        contact_skate = pred_foot_velocity.square().sum(dim=-1)
        contact_skate = (
            contact_skate * target_contact[:, 1:]
        ).mean(dim=(1, 2))
        floor_loss = (floor_anchor + penetration + contact_skate).mean()
        return rec_loss, vel_loss, acc_loss, fk_loss, floor_loss

    def _meanflow_branch(
        self,
        x_t,
        t,
        r,
        driving_velocity,
        cond,
        genre,
        alignment_target,
        cond_drop_mask=None,
    ):
        """One complete x0-parameterized MeanFlow/JVP training branch."""

        b, device = x_t.shape[0], x_t.device
        interval = t - r
        fork_devices = [device.index] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices):
            x0_pred = self.model(
                x_t,
                cond,
                genre,
                t.view(-1),
                interval.view(-1),
                cond_drop_mask=cond_drop_mask,
            )
            _, u_pred = self._x_r_and_velocity(x_t, x0_pred, t, r)

        def u_func(current_x_t, current_t, current_r):
            clean = self.model(
                current_x_t,
                cond,
                genre,
                current_t.view(-1),
                (current_t - current_r).view(-1),
                cond_drop_mask=cond_drop_mask,
            )
            return self._x_r_and_velocity(
                current_x_t, clean, current_t, current_r
            )[1]

        _, dudt = self.jvp_fn(
            u_func,
            (x_t, t, r),
            (driving_velocity, torch.ones_like(t), torch.zeros_like(r)),
        )
        u_target = (driving_velocity - interval * dudt).detach()
        fm_loss = self.loss_fn(u_pred, u_target, reduction="none")
        fm_loss = reduce(fm_loss, "b ... -> b (...)", "mean")
        # Through the linear back-projection, the interval velocity has the
        # same 1/t sensitivity to a fixed x0 error.
        # Weighting by t^2 brings the leading error scale back to x0 space.
        time_weight = t.reshape(b, 1).pow(self.fm_time_weight_power)
        fm_loss = fm_loss * time_weight
        fm_loss = (fm_loss / ((fm_loss.detach() + 1e-3) ** 0.75)).mean()

        return (fm_loss,) + self._alignment_losses(
            x0_pred, alignment_target.detach()
        )

    @staticmethod
    def _sum_raw_losses(*branches):
        return tuple(sum(values) for values in zip(*branches))

    def fm_losses(self, x0, cond, genre, mode="Uniform"):
        b, device = x0.shape[0], x0.device

        if self.x0_variant == "t0_only":
            t = self.min_t + (1.0 - self.min_t) * torch.rand(
                (b, 1, 1), device=device
            )
            r = torch.zeros_like(t)
            noise = torch.randn_like(x0)
            x_t = (1 - t) * x0 + t * noise
            raw_values = self._meanflow_branch(
                x_t, t, r, noise - x0, cond, genre, x0
            )
        else:
            noise = torch.randn_like(x0)
            t, r = self.sample_two_timesteps(b, device)
            x_t = (1 - t) * x0 + t * noise
            path_velocity = noise - x0

            if self.x0_variant == "tr_only":
                raw_values = self._meanflow_branch(
                    x_t, t, r, path_velocity, cond, genre, x0
                )
            elif self.x0_variant == "tr_t0_aux":
                tr_values = self._meanflow_branch(
                    x_t, t, r, path_velocity, cond, genre, x0
                )
                t0 = self.min_t + (1.0 - self.min_t) * torch.rand(
                    (b, 1, 1), device=device
                )
                x_t0 = (1 - t0) * x0 + t0 * torch.randn_like(x0)
                x0_pred_t0 = self.model(
                    x_t0, cond, genre, t0.view(-1), t0.view(-1)
                )
                t0_alignment = self._alignment_losses(x0_pred_t0, x0)
                raw_values = (tr_values[0],) + self._sum_raw_losses(
                    tr_values[1:], t0_alignment
                )
            elif self.x0_variant == "tr_t0_full":
                # Paired ablation: identical x_t/noise/t for both complete
                # objectives, with only the queried endpoint changed to zero.
                tr_values = self._meanflow_branch(
                    x_t, t, r, path_velocity, cond, genre, x0
                )
                t0_values = self._meanflow_branch(
                    x_t,
                    t,
                    torch.zeros_like(t),
                    path_velocity,
                    cond,
                    genre,
                    x0,
                )
                raw_values = self._sum_raw_losses(tr_values, t0_values)
            elif self.x0_variant == "tr_cfg_baked":
                drop_mask = (
                    torch.rand((b,), device=device) < self.cfg_dropout_prob
                )
                all_uncond = torch.ones((b,), device=device, dtype=torch.bool)
                # The stopped unconditional estimate defines the baked guided
                # vector field. The trained conditional branch still makes one
                # network call at inference; this is not online two-pass CFG.
                with torch.no_grad():
                    x0_uncond = self._inference_model()(
                        x_t,
                        cond,
                        genre,
                        t.view(-1),
                        (t - r).view(-1),
                        cond_drop_mask=all_uncond,
                    )
                    v_uncond = (x_t - x0_uncond) / t.clamp_min(1e-5)
                    v_guided = v_uncond + self.cfg_omega * (
                        path_velocity - v_uncond
                    )
                    driving_velocity = torch.where(
                        drop_mask[:, None, None], path_velocity, v_guided
                    )
                    # In x0 parameterization this is the clean endpoint implied
                    # by the guided target; aligning conditional output to raw x0
                    # here would directly contradict omega != 1 MeanFlow targets.
                    guided_x0 = x_t - t * driving_velocity
                raw_values = self._meanflow_branch(
                    x_t,
                    t,
                    r,
                    driving_velocity,
                    cond,
                    genre,
                    guided_x0,
                    cond_drop_mask=drop_mask,
                )
            else:
                raise AssertionError(f"Unhandled x0 variant: {self.x0_variant}")

        fm_loss, rec_loss, vel_loss, acc_loss, fk_loss, floor_loss = raw_values
        raw_losses = {
            "fm": fm_loss,
            "rec": rec_loss,
            "vel": vel_loss,
            "acc": acc_loss,
            "fk": fk_loss,
            "floor": floor_loss,
        }
        losses = tuple(
            self.loss_weights[name] * raw_losses[name]
            for name in ("fm", "rec", "vel", "acc", "fk", "floor")
        )
        return sum(losses), losses

    def forward(self, x, cond, genre, t_override=None):
        return self.fm_losses(x, cond, genre)

    def _inference_model(self):
        """Return the underlying decoder when training wrapped it with DDP."""

        return self.model.module if hasattr(self.model, "module") else self.model


    # === sample ===
    @torch.no_grad()
    def euler_sample(self, shape, cond, genre, n_steps=21):
        b, device = shape[0], cond.device
        x = torch.randn(shape, device=device)  

        t_values = torch.linspace(1, 0, n_steps, device=device)
        for i in tqdm(range(n_steps - 1)):
            t = torch.full((b,), t_values[i], device=device)
            r = torch.full((b,), t_values[i + 1], device=device)
            # A t0-only model was trained with r=0 (interval=t). We still
            # back-project its x0 prediction to the next ODE grid point here.
            h = t if self.x0_variant == "t0_only" else t - r
            x0_pred = self._inference_model().infer_pred(x, cond, genre, t, h)
            x, _ = self._x_r_and_velocity(
                x, x0_pred, t[:, None, None], r[:, None, None]
            )
        return x

    @torch.no_grad()
    def render_sample(self, shape, cond, genre, normalizer, epoch, render_out, fk_out=None, name=None, sound=True, n_steps=21):
        start = time.perf_counter()
        samples = self.euler_sample(shape, cond, genre, n_steps=n_steps).detach().cpu()
        print('gt', cond.shape, shape, 'pred', samples.shape)
        end = time.perf_counter()
        print(f"Time: {(end - start):.2f} s")

        # Audit the only nonlinear part of prediction post-processing.  The
        # normalizer clips normalized samples to [-1, 1] before applying the
        # inverse transform, so report how much of a generated batch is
        # affected without changing the generated values or saved format.
        channel_groups = {
            "contacts": samples[..., :4],
            "root_vxy": samples[..., 4:6],
            "root_z": samples[..., 6:7],
            "rot6d": samples[..., 7:],
        }
        clip_audit = {}
        for group_name, values in channel_groups.items():
            outside = values.abs() > 1.0
            clip_audit[group_name] = {
                "outside_percent": 100.0 * float(outside.float().mean()),
                "clips_affected": int(
                    outside.reshape(outside.shape[0], -1).any(dim=1).sum()
                ),
                "max_abs": float(values.abs().max()),
            }
        print("Normalized sample clip audit:", clip_audit)

        # unnormalize 
        samples = normalizer.unnormalize(samples)
        sample_contact, samples = torch.split(samples, (4, samples.shape[2] - 4), dim=2)

        # FK 
        b, s, c = samples.shape
        pos = root_velocity_to_position(samples[:, :, :3].to(cond.device))
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        q = ax_from_6v(q).to(cond.device)

        poses = self.smpl.forward(q, pos).detach().cpu().numpy()
        sample_contact = sample_contact.detach().cpu().numpy() if sample_contact is not None else None


        Path(fk_out).mkdir(parents=True, exist_ok=True)
        for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
            path = os.path.normpath(filename)
            pathparts = path.split(os.sep)
            pathparts[-1] = pathparts[-1].replace("npy", "wav")
            audioname = os.path.join(*pathparts)
            os.makedirs(os.path.join(fk_out, str(epoch)), exist_ok=True)
            outname = f"{epoch}/{pathparts[-1][:-4]}.pkl"
            pickle.dump({"smpl_poses": qq.reshape((-1, 72)).cpu().numpy(), "smpl_trans": pos_.cpu().numpy(), "full_pose": pose,}, open(f"{fk_out}/{outname}", "wb"))


    @torch.no_grad()
    def inpaint(self, x, mask, cond, genre, n_steps=101):
        """
        Inpainting for Flow Matching using ODE integration (reverse time: 1 -> 0)
        x: (b, seq, dim) ground-truth sequence
        mask: (b, seq, dim) 
        """
        b, device = x.shape[0], x.device
        self.model.eval()

        e = torch.randn_like(x)
        x_t = e.clone()

        #  (1 → 0)
        t_values = torch.linspace(1.0, 0, n_steps, device=device)
        for i in range(len(t_values) - 1):
            t_cur = t_values[i]
            t_next = t_values[i + 1]

            # batch 
            t_batch = torch.full((b,), t_cur, device=device)
            h_batch = torch.full((b,), t_cur - t_next, device=device)  # interval
            if self.x0_variant == "t0_only":
                h_batch.fill_(t_cur)

            # 
            x0_pred = self._inference_model().infer_pred(
                x_t, cond, genre, t_batch, h_batch
            )
            r_batch = torch.full((b,), t_next, device=device)
            x_t, _ = self._x_r_and_velocity(
                x_t,
                x0_pred,
                t_batch[:, None, None],
                r_batch[:, None, None],
            )


            mask1 = mask * t_next
            # mask1 = mask
            x_t = mask1 * ((1 - t_next) * x + t_next * e) + (1 - mask1) * x_t

        return x_t


    @torch.no_grad()
    def inpaint_sample(self, x, cond, genre, normalizer, epoch, render_out, fk_out=None, name=None, sound=True, n_steps=101, mode="middle"):

        b, seq, dim = x.shape
        device = x.device
        keep = 300  # 

        # ===  mask ===
        mask = torch.ones((b, seq, dim), device=device)
        if mode == "front":
            mask[:, keep:, :] = 0
        elif mode == "back":
            mask[:, :seq-keep, :] = 0
        elif mode == "middle":
            mask[:, keep:seq-keep, :] = 0
        else:
            raise ValueError(f"Unknown inpaint mode {mode}")

        # ===  inpaint ===
        start = time.perf_counter()
        samples = self.inpaint(x, mask, cond, genre, n_steps=n_steps).detach().cpu()
        print(f"Inpaint mode={mode}, shape={samples.shape}")
        end = time.perf_counter()
        print(f"Inpaint Time: {(end - start):.2f} s")

        # unnormalize 
        samples = normalizer.unnormalize(samples)
        sample_contact, samples = torch.split(samples, (4, samples.shape[2] - 4), dim=2)

        # FK 
        b, s, c = samples.shape
        pos = root_velocity_to_position(samples[:, :, :3].to(cond.device))
        q = samples[:, :, 3:].reshape(b, s, 24, 6)
        q = ax_from_6v(q).to(cond.device)

        poses = self.smpl.forward(q, pos).detach().cpu().numpy()
        sample_contact = sample_contact.detach().cpu().numpy() if sample_contact is not None else None


        Path(fk_out).mkdir(parents=True, exist_ok=True)
        for num, (qq, pos_, filename, pose) in enumerate(zip(q, pos, name, poses)):
            path = os.path.normpath(filename)
            pathparts = path.split(os.sep)
            pathparts[-1] = pathparts[-1].replace("npy", "wav")
            audioname = os.path.join(*pathparts)
            os.makedirs(os.path.join(fk_out, str(epoch)), exist_ok=True)
            outname = f"{epoch}/{pathparts[-1][:-4]}.pkl"
            pickle.dump({"smpl_poses": qq.reshape((-1, 72)).cpu().numpy(), "smpl_trans": pos_.cpu().numpy(), "full_pose": pose,}, open(f"{fk_out}/{outname}", "wb"))

