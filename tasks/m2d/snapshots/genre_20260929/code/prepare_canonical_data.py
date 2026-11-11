"""Convert released FlowerDance caches to the canonical x0 representation."""

from __future__ import annotations

import argparse
import gc
import json
import pickle
from pathlib import Path

import torch

from dataset.motion_representation import (
    FOOT_JOINTS,
    HEIGHT_AXIS,
    canonicalize_pose_batch,
    pose_to_joints,
)
from dataset.preprocess import Normalizer
from vis import SMPLSkeleton


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-backups",
        type=Path,
        default=Path("data_source/dataset_backups"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("data/dataset_backups")
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def convert_pose_tensor(pose, old_normalizer, smpl, batch_size, device):
    converted = torch.empty_like(pose, device="cpu")
    floors = []
    for start in range(0, len(pose), batch_size):
        end = min(start + batch_size, len(pose))
        normalized = pose[start:end].to(device, non_blocking=True)
        physical = old_normalizer.unnormalize(normalized.clone())
        canonical, floor = canonicalize_pose_batch(physical, smpl)
        converted[start:end].copy_(canonical.cpu())
        floors.append(floor.cpu())
        print(f"converted {end}/{len(pose)}", flush=True)
    return converted, torch.cat(floors)


def normalize_in_chunks(pose, normalizer, batch_size):
    for start in range(0, len(pose), batch_size):
        end = min(start + batch_size, len(pose))
        pose[start:end] = normalizer.normalize(pose[start:end])


def validate_floor(dataset, batch_size, device):
    smpl = SMPLSkeleton(device)
    maxima = []
    for start in range(0, len(dataset), batch_size):
        end = min(start + batch_size, len(dataset))
        pose = dataset.normalizer.unnormalize(
            dataset.data["pose"][start:end].to(device).clone()
        )
        joints = pose_to_joints(pose, smpl)
        floor = joints[..., list(FOOT_JOINTS), HEIGHT_AXIS].amin(dim=(1, 2))
        maxima.append(floor.abs().cpu())
    return float(torch.cat(maxima).max())


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    smpl = SMPLSkeleton(device)

    converted = {}
    floor_stats = {}
    for split in ("train", "test"):
        source = args.source_backups / f"{split}_tensor_dataset.pkl"
        print(f"loading {source}", flush=True)
        with source.open("rb") as handle:
            dataset = pickle.load(handle)
        canonical, floors = convert_pose_tensor(
            dataset.data["pose"],
            dataset.normalizer,
            smpl,
            args.batch_size,
            device,
        )
        dataset.data["pose"] = canonical
        dataset.root_representation = "vx_vy_z"
        dataset.height_axis = "z"
        dataset.floor_height = 0.0
        converted[split] = dataset
        floor_stats[split] = {
            "source_floor_min": float(floors.min()),
            "source_floor_max": float(floors.max()),
            "source_floor_mean": float(floors.mean()),
        }
        del floors
        gc.collect()

    print("fitting canonical motion normalizer", flush=True)
    normalizer = Normalizer(
        converted["train"].data["pose"], symmetric_indices=(4, 5)
    )
    for dataset in converted.values():
        normalize_in_chunks(dataset.data["pose"], normalizer, args.batch_size)
        dataset.normalizer = normalizer

    for split, dataset in converted.items():
        floor_error = validate_floor(dataset, args.batch_size, device)
        floor_stats[split]["max_abs_floor_after_conversion"] = floor_error
        output = args.output / f"{split}_tensor_dataset.pkl"
        print(f"writing {output}", flush=True)
        with output.open("wb") as handle:
            pickle.dump(dataset, handle, pickle.HIGHEST_PROTOCOL)

    with (args.output / "normalizer.pkl").open("wb") as handle:
        pickle.dump(normalizer, handle, pickle.HIGHEST_PROTOCOL)
    metadata = {
        "height_axis": "z",
        "root_channels": ["velocity_x", "velocity_y", "height_z"],
        "xy_origin": [0.0, 0.0],
        "shared_floor_z": 0.0,
        "velocity_normalization": "symmetric_minmax_zero_centered",
        "foot_joints": list(FOOT_JOINTS),
        "floor_stats": floor_stats,
    }
    (args.output / "representation.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
