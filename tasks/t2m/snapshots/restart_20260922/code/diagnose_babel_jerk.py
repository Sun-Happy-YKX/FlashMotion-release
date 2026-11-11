from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from t2m_long.babel_protocol import effective_segments
from t2m_long.representation import recover_joints_from_ric


def jerk_joint_l1(values: np.ndarray) -> np.ndarray:
    jerk = np.diff(values, n=3, axis=0)
    return np.abs(jerk).sum(axis=-1).max(axis=-1)


def jerk_joint_l2(values: np.ndarray) -> np.ndarray:
    jerk = np.diff(values, n=3, axis=0)
    return np.linalg.norm(jerk, axis=-1).max(axis=-1)


def jerk_feature_max(values: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(values, n=3, axis=0)).max(axis=-1)


def jerk_feature_l1(values: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(values, n=3, axis=0)).sum(axis=-1)


def summarize(curves: list[np.ndarray], baseline: float) -> dict[str, float]:
    array = np.stack(curves)
    mean_curve = array.mean(axis=0)
    return {
        "count": len(array),
        "peak_of_mean": float(mean_curve.max()),
        "mean_of_peak": float(array.max(axis=1).mean()),
        "auj_of_mean": float(np.abs(mean_curve - baseline).sum()),
        "mean_of_auj": float(np.abs(array - baseline).sum(axis=1).mean()),
        "mean_raw_area": float(np.abs(array).sum(axis=1).mean()),
        "global_peak": float(array.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--split", default="val_processed.txt")
    args = parser.parse_args()

    data = args.root / "raw_data" / "BABEL_streamed"
    hml = args.root / "raw_data" / "HumanML3D"
    mean = np.load(hml / "Mean.npy")
    std = np.load(hml / "Std.npy")
    ids = [line.strip() for line in (data / args.split).read_text().splitlines() if line.strip()]

    for window in (30, 60):
        curves: dict[str, list[np.ndarray]] = {
            "joints_global": [],
            "joints_root_relative": [],
            "joints_from_normalized_features": [],
            "ric_local": [],
            "feature_raw_max": [],
            "feature_raw_l1": [],
            "feature_normalized_max": [],
            "feature_normalized_l1": [],
            "raw_local_velocity_xyz_l1": [],
            "raw_local_velocity_xyz_l2": [],
            "norm_local_velocity_xyz_l1": [],
            "norm_local_velocity_xyz_l2": [],
            "norm_root_max": [],
            "norm_ric_max": [],
            "norm_rot6d_max": [],
            "norm_local_velocity_max": [],
            "norm_pose_channels_max": [],
        }
        left = window // 2
        right = window - left
        for name in ids:
            motion = np.load(data / "motions" / f"{name}.npy")[:200]
            lines = (data / "texts" / f"{name}.txt").read_text().splitlines()
            ends = [int(float(line.split("#")[3]) * 20 + 0.5) for line in lines]
            segments = effective_segments(ends, len(motion))
            if len(segments) < 2:
                continue
            joints = recover_joints_from_ric(torch.from_numpy(motion)[None]).squeeze(0).numpy()
            joints_rr = joints - joints[:, :1]
            ric = motion[:, 4:67].reshape(len(motion), 21, 3)
            normalized = (motion - mean) / std
            normalized_joints = recover_joints_from_ric(
                torch.from_numpy(normalized)[None].float()
            ).squeeze(0).numpy()
            for segment in segments[1:]:
                boundary = segment.start
                if boundary < left or boundary + right > len(motion):
                    continue
                sl = slice(boundary - left, boundary + right)
                curves["joints_global"].append(jerk_joint_l1(joints[sl]))
                curves["joints_root_relative"].append(jerk_joint_l1(joints_rr[sl]))
                curves["joints_from_normalized_features"].append(
                    jerk_joint_l1(normalized_joints[sl])
                )
                curves["ric_local"].append(jerk_joint_l1(ric[sl]))
                curves["feature_raw_max"].append(jerk_feature_max(motion[sl]))
                curves["feature_raw_l1"].append(jerk_feature_l1(motion[sl]))
                curves["feature_normalized_max"].append(jerk_feature_max(normalized[sl]))
                curves["feature_normalized_l1"].append(jerk_feature_l1(normalized[sl]))
                raw_velocity = motion[sl, 193:259].reshape(-1, 22, 3)
                norm_velocity = normalized[sl, 193:259].reshape(-1, 22, 3)
                curves["raw_local_velocity_xyz_l1"].append(jerk_joint_l1(raw_velocity))
                curves["raw_local_velocity_xyz_l2"].append(jerk_joint_l2(raw_velocity))
                curves["norm_local_velocity_xyz_l1"].append(jerk_joint_l1(norm_velocity))
                curves["norm_local_velocity_xyz_l2"].append(jerk_joint_l2(norm_velocity))
                curves["norm_root_max"].append(jerk_feature_max(normalized[sl, :4]))
                curves["norm_ric_max"].append(jerk_feature_max(normalized[sl, 4:67]))
                curves["norm_rot6d_max"].append(jerk_feature_max(normalized[sl, 67:193]))
                curves["norm_local_velocity_max"].append(
                    jerk_feature_max(normalized[sl, 193:259])
                )
                curves["norm_pose_channels_max"].append(
                    jerk_feature_max(normalized[sl, 4:259])
                )

        print(f"\nWINDOW={window}")
        for key, value in curves.items():
            summary = summarize(value, 0.033031363)
            summary["raw_AUJ"] = summarize(value, 0.0)["auj_of_mean"]
            print(key, summary)


if __name__ == "__main__":
    main()
