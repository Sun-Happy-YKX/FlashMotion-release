"""Canonical FlowerDance motion representation utilities.

The released FlowerDance preprocessing rotates AIST++ from Y-up to Z-up.
This ablation keeps Z as an absolute height above one shared floor, while the
root X/Y channels store frame-to-frame velocity instead of absolute position.
"""

from __future__ import annotations

import torch

from dataset.quaternion import ax_from_6v


CONTACT_DIM = 4
ROOT_DIM = 3
HEIGHT_AXIS = 2
FOOT_JOINTS = (7, 8, 10, 11)


def root_velocity_to_position(root_motion: torch.Tensor) -> torch.Tensor:
    """Decode ``[vx, vy, z]`` into canonical absolute ``[x, y, z]``.

    The first X/Y velocity is deliberately ignored, fixing every generated
    sequence to the same X/Y origin. Subsequent velocities are cumulatively
    integrated, so the global trajectory remains recoverable.
    """

    if root_motion.ndim != 3 or root_motion.shape[-1] != ROOT_DIM:
        raise ValueError(f"Expected [B, T, 3] root motion, got {root_motion.shape}")
    xy_velocity = root_motion[..., :2]
    zero = torch.zeros_like(xy_velocity[:, :1])
    xy_position = torch.cumsum(
        torch.cat((zero, xy_velocity[:, 1:]), dim=1), dim=1
    )
    return torch.cat((xy_position, root_motion[..., 2:3]), dim=-1)


def pose_to_root_and_rotations(pose: torch.Tensor):
    """Decode an unnormalized canonical pose into root positions and rotations."""

    if pose.ndim != 3 or pose.shape[-1] < CONTACT_DIM + ROOT_DIM:
        raise ValueError(f"Expected [B, T, C] pose tensor, got {pose.shape}")
    body = pose[..., CONTACT_DIM:]
    root = root_velocity_to_position(body[..., :ROOT_DIM])
    rotations_6d = body[..., ROOT_DIM:].reshape(
        pose.shape[0], pose.shape[1], -1, 6
    )
    rotations = ax_from_6v(rotations_6d)
    return root, rotations


def pose_to_joints(pose: torch.Tensor, smpl) -> torch.Tensor:
    """Run FK after decoding the canonical root-velocity representation."""

    root, rotations = pose_to_root_and_rotations(pose)
    return smpl.forward(rotations, root)


@torch.no_grad()
def canonicalize_pose_batch(pose: torch.Tensor, smpl):
    """Convert absolute-root poses to shared-floor, XY-velocity poses.

    The floor of each clip is the minimum Z coordinate among its four SMPL
    ankle/toe joints. Subtracting that scalar from root Z puts every clip on
    exactly the same Z=0 floor without altering joint rotations.
    """

    if pose.ndim != 3:
        raise ValueError(f"Expected [B, T, C] pose tensor, got {pose.shape}")
    result = pose.clone()
    body = result[..., CONTACT_DIM:]
    root = body[..., :ROOT_DIM].clone()
    rotations = ax_from_6v(
        body[..., ROOT_DIM:].reshape(pose.shape[0], pose.shape[1], -1, 6)
    )
    joints = smpl.forward(rotations, root)
    foot_height = joints[..., list(FOOT_JOINTS), HEIGHT_AXIS]
    floor = foot_height.amin(dim=(1, 2))
    root[..., HEIGHT_AXIS] -= floor[:, None]

    xy_velocity = torch.zeros_like(root[..., :2])
    xy_velocity[:, 1:] = root[:, 1:, :2] - root[:, :-1, :2]
    body[..., :ROOT_DIM] = torch.cat(
        (xy_velocity, root[..., HEIGHT_AXIS : HEIGHT_AXIS + 1]), dim=-1
    )
    return result, floor
