from pathlib import Path

import numpy as np
import torch
from torch import nn

from t2m_long.data import MotionTextDataset, collate_motion_text
from t2m_long.meanflow import T2MMeanFlow
from t2m_long.representation import recover_joints_from_ric


class UnusedDecoder(nn.Module):
    def forward(self, *args, **kwargs):
        raise AssertionError("decoder is not used by this alignment-loss test")


root = Path("third_party/FloodDiffusion/raw_data/HumanML3D")
mean = torch.from_numpy(np.load(root / "Mean.npy"))
std = torch.from_numpy(np.load(root / "Std.npy"))
objective = T2MMeanFlow(
    UnusedDecoder(),
    mean,
    std,
    loss_weights={"floor": 1.0},
    x0_variant="tr_only",
)

target = torch.randn(2, 12, 263) * 0.1
prediction = (target + torch.randn_like(target) * 0.02).requires_grad_(True)
mask = torch.zeros(2, 12, dtype=torch.bool)
mask[0, :12] = True
mask[1, :7] = True

losses = objective._alignment_losses(prediction, target, mask)
assert set(losses) == {"rec", "vel", "acc", "fk", "floor"}
assert all(value.ndim == 0 and torch.isfinite(value) for value in losses.values()), losses
total = sum(losses[name] * objective.loss_weights[name] for name in losses)
total.backward()
assert prediction.grad is not None and torch.isfinite(prediction.grad).all()
print({name: float(value.detach()) for name, value in losses.items()})
print("FLOOR_LOSS_GRADIENT_OK")

dataset = MotionTextDataset(
    "third_party/FloodDiffusion/raw_data",
    "data/umt5_xxl_sentence_features",
    "babel",
    "train",
    max_length=200,
)
batch = collate_motion_text([dataset[0], dataset[1]])
real_raw = batch["motion"] * std + mean
real_joints = recover_joints_from_ric(real_raw)
real_feet = real_joints[:, :, (7, 10, 8, 11)]
valid = batch["mask"].unsqueeze(-1).expand_as(real_feet[..., 1])
minimum_y = real_feet[..., 1].masked_fill(~valid, torch.inf).amin((1, 2))
all_joint_valid = batch["mask"].unsqueeze(-1).expand_as(real_joints[..., 1])
all_joint_minimum_y = real_joints[..., 1].masked_fill(~all_joint_valid, torch.inf).amin((1, 2))
real_losses = objective._alignment_losses(batch["motion"], batch["motion"], batch["mask"])
print("REAL_SAMPLE_MINIMUM_FOOT_Y", minimum_y.tolist())
print("REAL_SAMPLE_MINIMUM_ALL_JOINT_Y", all_joint_minimum_y.tolist())
print("REAL_SAMPLE_SELF_FLOOR", float(real_losses["floor"]))
# BABEL's converted HumanML3D representation is ground-normalized but small
# retargeting offsets remain (the checked sample is 2.77 cm above zero).
assert all_joint_minimum_y.abs().max() < 0.1, all_joint_minimum_y
assert minimum_y.abs().max() < 0.1, minimum_y
print("REAL_BABEL_COORDINATE_CHECK_OK")
