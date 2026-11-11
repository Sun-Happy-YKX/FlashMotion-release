"""Single-branch t->r MeanFlow and SMPL-X local-pose geometry.

Motion is train-stat standardized local row-convention rot6D + [dx,dz,y].
Geometry is computed only after inverse standardization, in meters. Velocity
and acceleration are frame differences, not absolute positions and not /FPS.
No GT pose, expression, hand rotation, or root origin is used at inference.
Official SMPL-X default hand means are added in axis-angle space for FK;
the network's raw pose representation and AESK inputs remain unchanged.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

try:
    from .model import MOTION_DIMENSIONS, SMPLX_PARENTS, valid_lengths
except ImportError:
    from model import MOTION_DIMENSIONS, SMPLX_PARENTS, valid_lengths


@dataclass(frozen=True)
class LossWeights:
    fm: float = 1.0
    rec: float = 0.636
    vel: float = 0.323
    acc: float = 0.2
    fk: float = 0.646


def rotation_6d_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """PyTorch3D/Common.Util convention: the first two matrix rows."""
    if value.shape[-1] != 6:
        raise ValueError("rotation representation must have six channels")
    first, second = value[..., :3], value[..., 3:]
    row1 = F.normalize(first, dim=-1)
    row2 = F.normalize(second - (row1 * second).sum(-1, keepdim=True) * row1, dim=-1)
    row3 = torch.cross(row1, row2, dim=-1)
    return torch.stack((row1, row2, row3), dim=-2)


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must have shape [...,3,3]")
    return matrix[..., :2, :].reshape(*matrix.shape[:-2], 6)


def axis_angle_to_matrix(value: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues rotation, including exactly zero angles."""
    if value.shape[-1] != 3:
        raise ValueError("axis-angle must have three channels")
    x, y, z = value.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1)
    skew = skew.reshape(*value.shape[:-1], 3, 3)
    angle = torch.linalg.vector_norm(value, dim=-1, keepdim=True)[..., None]
    identity = torch.eye(3, device=value.device, dtype=value.dtype)
    # sinc formulas avoid unstable divisions and 0/0 branches at identity.
    return (identity + torch.sinc(angle / torch.pi) * skew
            + .5 * torch.sinc(angle / (2 * torch.pi)).square() * (skew @ skew))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Canonical differentiable AA through best-conditioned quaternion.

    Algebra is equivalent to PyTorch3D's matrix -> quaternion -> AA; candidate
    selection avoids small denominators near pi, and w >= 0 fixes AA branch.
    """
    if matrix.shape[-2:] != (3, 3):
        raise ValueError("rotation matrix must have shape [...,3,3]")
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = matrix.reshape(
        *matrix.shape[:-2], 9).unbind(-1)
    terms = torch.stack((1+m00+m11+m22, 1+m00-m11-m22,
                         1-m00+m11-m22, 1-m00-m11+m22), -1)
    positive = terms > 0
    roots = torch.where(positive, torch.where(positive, terms,
                         torch.ones_like(terms)).sqrt(), torch.zeros_like(terms))
    candidates = torch.stack((
        torch.stack((roots[..., 0].square(), m21-m12, m02-m20, m10-m01), -1),
        torch.stack((m21-m12, roots[..., 1].square(), m10+m01, m02+m20), -1),
        torch.stack((m02-m20, m10+m01, roots[..., 2].square(), m12+m21), -1),
        torch.stack((m10-m01, m20+m02, m21+m12, roots[..., 3].square()), -1)), -2)
    candidates = candidates / (2 * roots.clamp_min(.1)[..., :, None])
    indices = roots.argmax(-1)[..., None, None].expand(*matrix.shape[:-2], 1, 4)
    quaternion = candidates.gather(-2, indices).squeeze(-2)
    quaternion = torch.where(quaternion[..., :1] < 0, -quaternion, quaternion)
    vector = quaternion[..., 1:]
    half_angle = torch.atan2(torch.linalg.vector_norm(vector, dim=-1, keepdim=True),
                             quaternion[..., :1])
    return vector / (.5 * torch.sinc(half_angle / torch.pi))


def encode_root(translation: torch.Tensor) -> torch.Tensor:
    """Y-up xyz -> [delta x, delta z, absolute y], first delta is zero."""
    if translation.ndim < 2 or translation.shape[-1] != 3:
        raise ValueError("translation must contain a time axis and xyz channels")
    first = torch.zeros_like(translation[..., :1, :1])
    dx = torch.cat((first, translation[..., 1:, :1] - translation[..., :-1, :1]), dim=-2)
    dz = torch.cat((first, translation[..., 1:, 2:3] - translation[..., :-1, 2:3]), dim=-2)
    return torch.cat((dx, dz, translation[..., 1:2]), dim=-1)


def decode_root(root: torch.Tensor) -> torch.Tensor:
    """[dx,dz,y] -> canonical xyz; first horizontal delta is ignored.

    This matches the Music2Dance reference decoder and fixes x/z at frame zero
    to zero even if the generated first delta is imperfect. No GT origin is
    added. All subsequent frame deltas, in meters, remain predicted.
    """
    if root.ndim < 2 or root.shape[-1] != 3:
        raise ValueError("root representation requires a time axis and three channels")
    horizontal = torch.cat((torch.zeros_like(root[..., :1, :2]),
                            root[..., 1:, :2]), dim=-2).cumsum(dim=-2)
    return torch.stack((horizontal[..., 0], root[..., 2],
                        horizontal[..., 1]), dim=-1)


def assemble_full_pose(motion: torch.Tensor, mode: str) -> torch.Tensor:
    """Return [..,55,6] local rotations; unmodeled joints use identity only."""
    if mode not in MOTION_DIMENSIONS or motion.shape[-1] != MOTION_DIMENSIONS[mode]:
        raise ValueError("motion dimension does not match requested mode")
    count = 55 if mode == "full55" else 22
    predicted = motion[..., :count * 6].reshape(*motion.shape[:-1], count, 6)
    if count == 55:
        return predicted
    identity = motion.new_tensor([1, 0, 0, 0, 1, 0]).expand(*motion.shape[:-1], 33, 6)
    return torch.cat((predicted, identity), dim=-2)


def local_forward_kinematics(local_rotations: torch.Tensor,
                             rest_joints: torch.Tensor) -> torch.Tensor:
    """Common.LFK equivalent, relative to pelvis; rest joints are absolute xyz.

    local_rotations: [B,T,55,3,3], rest_joints: [B,55,3]. Rest skeletons
    come from static SMPL-X betas, never from posed GT motion.
    """
    if local_rotations.ndim != 5 or local_rotations.shape[-3:] != (55, 3, 3):
        raise ValueError("expected local rotations [B,T,55,3,3]")
    batch, frames = local_rotations.shape[:2]
    if rest_joints.shape != (batch, 55, 3):
        raise ValueError("expected rest_joints [B,55,3]")
    global_rotations = [local_rotations[..., 0, :, :]]
    positions = [local_rotations.new_zeros(batch, frames, 3)]
    for joint in range(1, 55):
        parent = SMPLX_PARENTS[joint]
        bone = (rest_joints[:, joint] - rest_joints[:, parent])[:, None, :, None]
        displacement = (global_rotations[parent] @ bone).squeeze(-1)
        positions.append(positions[parent] + displacement)
        global_rotations.append(global_rotations[parent] @ local_rotations[..., joint, :, :])
    return torch.stack(positions, dim=-2)


def motion_to_joints(physical_motion: torch.Tensor, rest_joints: torch.Tensor,
                     mode: str, padding_mask: torch.Tensor | None = None,
                     *, pose_mean: torch.Tensor | None = None) -> torch.Tensor:
    """Raw physical pose -> SMPL-X full55 joints, with canonical root.

    Rest joints MUST be unposed, not ``smpl(betas).joints`` with the default
    hand mean already applied. Official SMPL-X adds its static pose_mean to
    raw axis-angle, not by composing rotations. Full55 requires this metadata.
    """
    local = rotation_6d_to_matrix(assemble_full_pose(physical_motion, mode))
    if pose_mean is None:
        if mode == "full55":
            raise ValueError("Full55 FK requires explicit official SMPL-X pose_mean metadata")
    else:
        pose_mean = torch.as_tensor(pose_mean, device=local.device, dtype=local.dtype)
        if pose_mean.numel() != 165 or not torch.isfinite(pose_mean).all():
            raise ValueError("SMPL-X pose_mean must contain finite 55 x 3 axis-angles")
        raw_axis_angle = matrix_to_axis_angle(local)
        local = axis_angle_to_matrix(raw_axis_angle + pose_mean.reshape(55, 3))
    joints = local_forward_kinematics(local, rest_joints)
    root = physical_motion[..., -3:]
    if padding_mask is not None:
        root = root * padding_mask[..., None]
    result = joints + decode_root(root).unsqueeze(-2)
    if padding_mask is not None:
        result = result * padding_mask[..., None, None]
    return result


def _per_sample_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Equal weight per sequence, excluding padding from temporal reductions."""
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    expanded = valid.expand_as(value).to(value.dtype)
    sums = (value * expanded).flatten(1).sum(1)
    counts = expanded.flatten(1).sum(1)
    # For a 1/2-frame sequence, velocity/acceleration have no valid pairs.
    return sums / counts.clamp_min(1)


class MeanFlowLoss(nn.Module):
    """Reference tr_only objective with train-only normalization buffers."""
    loss_order = ("fm", "rec", "vel", "acc", "fk")

    def __init__(self, model: nn.Module, motion_mean: torch.Tensor,
                 motion_std: torch.Tensor, *, mode: str | None = None,
                 pose_mean: torch.Tensor | None = None,
                 weights: LossWeights | dict | None = None, min_t: float = 0.05,
                 min_interval: float = 0.05, fm_time_weight_power: float = 2.0) -> None:
        super().__init__()
        unwrapped = getattr(model, "module", model)
        self.mode = mode or unwrapped.mode
        if self.mode not in MOTION_DIMENSIONS:
            raise ValueError("mode must be full55 or body22")
        self.motion_dim = MOTION_DIMENSIONS[self.mode]
        mean = torch.as_tensor(motion_mean, dtype=torch.float32).reshape(-1)
        std = torch.as_tensor(motion_std, dtype=torch.float32).reshape(-1)
        if mean.numel() != self.motion_dim or std.numel() != self.motion_dim:
            raise ValueError("statistics must match the selected local-rotation/root dimensions")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("motion statistics must be finite with strictly positive std")
        if not 0 < min_t < 1 or not 0 < min_interval < 1:
            raise ValueError("minimum time and interval must be in (0,1)")
        if fm_time_weight_power < 0:
            raise ValueError("time weight power must be nonnegative")
        self.model = model
        self.register_buffer("motion_mean", mean)
        self.register_buffer("motion_std", std)
        if pose_mean is None:
            if self.mode == "full55" and not getattr(unwrapped, "test_backend", False):
                raise ValueError("Real full55 MeanFlow requires official SMPL-X pose_mean")
            pose_mean = torch.zeros(55, 3)
        pose_mean = torch.as_tensor(pose_mean, dtype=torch.float32)
        if pose_mean.numel() != 165 or not torch.isfinite(pose_mean).all():
            raise ValueError("pose_mean metadata must contain finite 55 x 3 axis-angles")
        self.register_buffer("pose_mean", pose_mean.reshape(55, 3))
        self.weights = weights if isinstance(weights, LossWeights) else LossWeights(**(weights or {}))
        if any(value < 0 for value in asdict(self.weights).values()):
            raise ValueError("loss weights must be nonnegative")
        self.min_t, self.min_interval = min_t, min_interval
        self.fm_time_weight_power = fm_time_weight_power

    def normalize(self, physical_motion: torch.Tensor) -> torch.Tensor:
        if physical_motion.shape[-1] != self.motion_dim:
            raise ValueError("physical motion dimension does not match normalization statistics")
        return (physical_motion - self.motion_mean.to(physical_motion)) / self.motion_std.to(physical_motion)

    def denormalize(self, standardized_motion: torch.Tensor) -> torch.Tensor:
        return standardized_motion * self.motion_std.to(standardized_motion) + self.motion_mean.to(standardized_motion)

    def protocol_config(self) -> dict:
        return dict(objective="tr_only", weights=asdict(self.weights), min_t=self.min_t,
            min_interval=self.min_interval, fm_time_weight_power=self.fm_time_weight_power,
            normalization="train-only mean/std; no clipping", rotation="local row-rot6d",
            root="frame dx,dz + absolute y; first delta zero", geometry="meters after inverse normalization",
            velocity="joint[t+1]-joint[t], no fps factor", acceleration="velocity[t+1]-velocity[t]",
            geometry_joints=55 if self.mode == "full55" else 22,
            rest_skeleton="true unposed SMPL-X J_regressor(betas-shaped vertices)",
            hand_mean="official SMPL-X AA addition, not rotation composition",
            oracle_motion_input=False)

    def sample_two_timesteps(self, batch: int, device: torch.device,
                             dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        lower = max(self.min_t, self.min_interval)
        t = lower + (1 - lower) * torch.rand(batch, 1, 1, device=device, dtype=dtype)
        r = torch.rand(batch, 1, 1, device=device, dtype=dtype) * (t - self.min_interval)
        return t, r

    @staticmethod
    def x_r_and_velocity(x_t: torch.Tensor, prediction: torch.Tensor,
                         t: torch.Tensor, r: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ratio = (r / t.clamp_min(1e-5)).clamp(0, 1)
        x_r = ratio * x_t + (1 - ratio) * prediction
        return x_r, (x_t - x_r) / (t - r).clamp_min(1e-5)

    def _alignment_losses(self, prediction: torch.Tensor, clean: torch.Tensor,
                          rest_joints: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
        rec = _per_sample_mean((prediction - clean).square(), valid).mean()
        zero = prediction.sum() * 0
        if not (self.weights.vel or self.weights.acc or self.weights.fk):
            return dict(rec=rec, vel=zero, acc=zero, fk=zero)
        predicted_joints = motion_to_joints(self.denormalize(prediction), rest_joints, self.mode, valid,
                                           pose_mean=self.pose_mean)
        with torch.no_grad():
            target_joints = motion_to_joints(self.denormalize(clean), rest_joints, self.mode, valid,
                                            pose_mean=self.pose_mean)
        if self.mode == "body22":
            predicted_joints = predicted_joints[..., :22, :]
            target_joints = target_joints[..., :22, :]
        fk = _per_sample_mean((predicted_joints - target_joints).square(), valid).mean()
        predicted_velocity = predicted_joints[:, 1:] - predicted_joints[:, :-1]
        target_velocity = target_joints[:, 1:] - target_joints[:, :-1]
        valid_velocity = valid[:, 1:] & valid[:, :-1]
        vel = _per_sample_mean((predicted_velocity - target_velocity).square(), valid_velocity).mean()
        predicted_acc = predicted_velocity[:, 1:] - predicted_velocity[:, :-1]
        target_acc = target_velocity[:, 1:] - target_velocity[:, :-1]
        valid_acc = valid_velocity[:, 1:] & valid_velocity[:, :-1]
        acc = _per_sample_mean((predicted_acc - target_acc).square(), valid_acc).mean()
        return dict(rec=rec, vel=vel, acc=acc, fk=fk)

    def forward(self, clean: torch.Tensor, rest_joints: torch.Tensor,
                condition: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if clean.ndim != 3 or clean.shape[-1] != self.motion_dim:
            raise ValueError(f"expected standardized clean [B,T,{self.motion_dim}]")
        batch, frames = clean.shape[:2]
        lengths = valid_lengths(condition.get("padding_mask"), batch, frames, clean.device)
        valid = torch.arange(frames, device=clean.device)[None] < lengths[:, None]
        condition = {**condition, "padding_mask": valid}
        clean = clean * valid[..., None]
        noise = torch.randn_like(clean) * valid[..., None]
        t, r = self.sample_two_timesteps(batch, clean.device, clean.dtype)
        x_t = (1 - t) * clean + t * noise
        driving_velocity = noise - clean
        fork_devices = [clean.device.index] if clean.device.type == "cuda" else []
        # Reference RNG fork makes prediction/JVP share the same dropout draw.
        with torch.random.fork_rng(devices=fork_devices):
            prediction = self.model(x_t, t.reshape(-1), (t - r).reshape(-1), **condition)
            _, predicted_velocity = self.x_r_and_velocity(x_t, prediction, t, r)

        def velocity_function(state: torch.Tensor, time: torch.Tensor,
                              endpoint: torch.Tensor) -> torch.Tensor:
            predicted = self.model(state, time.reshape(-1),
                (time - endpoint).reshape(-1), **condition)
            return self.x_r_and_velocity(state, predicted, time, endpoint)[1]

        _, derivative = torch.autograd.functional.jvp(velocity_function,
            (x_t, t, r), (driving_velocity, torch.ones_like(t), torch.zeros_like(r)),
            create_graph=False, strict=False)
        target_velocity = (driving_velocity - (t - r) * derivative).detach()
        error = _per_sample_mean((predicted_velocity - target_velocity).square(), valid)
        error = error * t.reshape(-1).pow(self.fm_time_weight_power)
        fm = (error / (error.detach() + 1e-3).pow(0.75)).mean()
        raw = {"fm": fm, **self._alignment_losses(prediction, clean.detach(), rest_joints, valid)}
        weighted = {name: raw[name] * getattr(self.weights, name) for name in self.loss_order}
        total = sum(weighted.values())
        return total, {**{f"raw/{name}": raw[name] for name in self.loss_order},
            **{f"weighted/{name}": weighted[name] for name in self.loss_order},
            "sample/t_mean": t.mean(), "sample/r_mean": r.mean()}


__all__ = ["MeanFlowLoss", "LossWeights", "rotation_6d_to_matrix", "matrix_to_rotation_6d",
    "axis_angle_to_matrix", "matrix_to_axis_angle", "encode_root", "decode_root",
    "assemble_full_pose", "local_forward_kinematics", "motion_to_joints"]
