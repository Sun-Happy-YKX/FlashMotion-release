from __future__ import annotations

import torch


HML3D_DIM = 263
JOINTS_NUM = 22


def quaternion_inverse(q: torch.Tensor) -> torch.Tensor:
    result = q.clone()
    result[..., 1:] = -result[..., 1:]
    return result


def quaternion_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    qvec = q[..., 1:]
    uv = torch.cross(qvec, v, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return v + 2.0 * (q[..., :1] * uv + uuv)


def recover_root_rotation_position(features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact differentiable HumanML3D root recovery used by FloodDiffusion."""

    rotation_velocity = features[..., 0]
    rotation_angle = torch.zeros_like(rotation_velocity)
    rotation_angle[..., 1:] = rotation_velocity[..., :-1]
    rotation_angle = torch.cumsum(rotation_angle, dim=-1)

    root_quaternion = torch.zeros(
        features.shape[:-1] + (4,), device=features.device, dtype=features.dtype
    )
    root_quaternion[..., 0] = torch.cos(rotation_angle)
    root_quaternion[..., 2] = torch.sin(rotation_angle)

    root_position = torch.zeros(
        features.shape[:-1] + (3,), device=features.device, dtype=features.dtype
    )
    root_position[..., 1:, 0] = features[..., :-1, 1]
    root_position[..., 1:, 2] = features[..., :-1, 2]
    root_position = quaternion_rotate(quaternion_inverse(root_quaternion), root_position)
    root_position = torch.cumsum(root_position, dim=-2)
    root_position[..., 1] = features[..., 3]
    return root_quaternion, root_position


def recover_joints_from_ric(features: torch.Tensor, joints_num: int = JOINTS_NUM) -> torch.Tensor:
    """Recover global joint XYZ from an unnormalized HumanML3D 263D sequence."""

    if features.shape[-1] != HML3D_DIM:
        raise ValueError(f"Expected {HML3D_DIM}D HumanML3D features, got {features.shape[-1]}")
    root_quaternion, root_position = recover_root_rotation_position(features)
    end = 4 + (joints_num - 1) * 3
    positions = features[..., 4:end].reshape(features.shape[:-1] + (joints_num - 1, 3))
    inverse_root = quaternion_inverse(root_quaternion)[..., None, :].expand(
        positions.shape[:-1] + (4,)
    )
    positions = quaternion_rotate(inverse_root, positions)
    positions = positions.clone()
    positions[..., 0] += root_position[..., None, 0]
    positions[..., 2] += root_position[..., None, 2]
    return torch.cat([root_position[..., None, :], positions], dim=-2)

