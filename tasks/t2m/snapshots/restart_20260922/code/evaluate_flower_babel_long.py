from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from t2m_long.babel_long_benchmark import (
    MultiPredictionEmbeddingCollector,
    TRANSITION_FRAMES,
    centered_transition,
    save_shard,
    stable_seed,
)
from t2m_long.babel_protocol import effective_segments
from t2m_long.data import MotionTextDataset
from t2m_long.evaluator import FloodT2MEvaluator
from t2m_long.meanflow import T2MMeanFlow
from t2m_long.model import OneShotT2MDecoder
from t2m_long.representation import recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_long_v1/flower"))
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    return parser.parse_args()


def load_objective(checkpoint_path: Path, data_root: Path, device: torch.device) -> T2MMeanFlow:
    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy"))
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy"))
    objective = T2MMeanFlow(OneShotT2MDecoder(), mean, std).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    objective.load_state_dict(checkpoint["model"])
    if "ema_decoder" in checkpoint:
        objective.decoder.load_state_dict(checkpoint["ema_decoder"])
    return objective.eval().requires_grad_(False)


@torch.inference_mode()
def sample(
    objective: T2MMeanFlow,
    condition: torch.Tensor,
    steps: int,
    seed: int,
) -> torch.Tensor:
    condition = condition[None].to(next(objective.parameters()).device, non_blocking=True)
    mask = torch.ones(condition.shape[:2], dtype=torch.bool, device=condition.device)
    generator = torch.Generator(device=condition.device).manual_seed(seed)
    normalized = objective.sample(condition, mask, steps, generator)[0]
    return normalized * objective.motion_std[0] + objective.motion_mean[0]


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    data_root = flood_root / "raw_data"
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    dataset = MotionTextDataset(data_root, root / args.embedding_root, "babel", "val", None)
    total_episodes = len(dataset)
    indices = list(range(args.shard_index, total_episodes, args.num_shards))
    objective = load_objective((root / args.checkpoint).resolve(), data_root, device)
    evaluator = FloodT2MEvaluator(flood_root, device, seed=args.seed)
    prediction_names = [
        f"{workload}_{steps}step"
        for workload in ("single", "multi")
        for steps in args.steps
    ]
    collector = MultiPredictionEmbeddingCollector(
        evaluator, prediction_names, args.embedding_batch_size
    )
    transition_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_names = {"1567_3", "2062_9", "11264_1", "7743_8", "9085_3", "12738_4"}

    for dataset_index in tqdm(indices, desc=f"Flower shard {args.shard_index}"):
        item = dataset[dataset_index]
        name = item["name"]
        length = item["length"]
        condition = item["condition"]
        gt_raw = item["motion"].to(device) * objective.motion_std[0] + objective.motion_mean[0]
        segments = effective_segments(item["text_ends"], length)

        full_predictions = {
            steps: sample(
                objective,
                condition,
                steps,
                stable_seed(args.seed, "flower", name, "multi", steps),
            )[:length]
            for steps in args.steps
        }
        gt_joints = recover_joints_from_ric(gt_raw[None])[0]
        prediction_joints = {
            steps: recover_joints_from_ric(value[None])[0]
            for steps, value in full_predictions.items()
        }
        for transition_index, segment in enumerate(segments[1:]):
            gt_curve = centered_transition(gt_joints, segment.start, TRANSITION_FRAMES)
            if gt_curve is None:
                continue
            transition_parts["transition__ground_truth"].append(gt_curve)
            transition_parts["transition_key"].append(
                np.asarray(f"{name}::{transition_index}", dtype=np.str_)
            )
            transition_parts["transition_episode_id"].append(np.asarray(name, dtype=np.str_))
            for steps in args.steps:
                curve = centered_transition(
                    prediction_joints[steps], segment.start, TRANSITION_FRAMES
                )
                if curve is None:
                    raise RuntimeError("prediction transition unexpectedly missing")
                transition_parts[f"transition__multi_{steps}step"].append(curve)

        raw_to_save = {}
        for segment_index, segment in enumerate(segments):
            standalone_condition = condition[segment.start : segment.end]
            standalone_predictions = {
                steps: sample(
                    objective,
                    standalone_condition,
                    steps,
                    stable_seed(args.seed, "flower", name, segment_index, "single", steps),
                )
                for steps in args.steps
            }
            predictions = {}
            for steps in args.steps:
                predictions[f"multi_{steps}step"] = full_predictions[steps][
                    segment.start : segment.end
                ]
                predictions[f"single_{steps}step"] = standalone_predictions[steps]
            collector.add(
                gt_raw[segment.start : segment.end],
                item["text_tokens"][segment.annotation_index],
                predictions,
                {
                    "segment_key": f"{name}::{segment_index}",
                    "episode_id": name,
                    "caption": item["captions"][segment.annotation_index],
                    "segment_index": segment_index,
                    "segment_length": segment.end - segment.start,
                    "episode_length": length,
                },
            )
        if name in selected_names:
            sample_dir = output / "raw" / name
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.save(sample_dir / "gt.npy", gt_raw.float().cpu().numpy())
            for steps, value in full_predictions.items():
                np.save(sample_dir / f"flower_{steps}step.npy", value.float().cpu().numpy())
            (sample_dir / "metadata.json").write_text(json.dumps({
                "name": name,
                "captions": item["captions"],
                "text_ends": item["text_ends"],
                "length": length,
            }, indent=2))

    arrays = collector.finish()
    for key, values in transition_parts.items():
        arrays[key] = np.stack(values) if values else np.empty((0,), dtype=np.float32)
    shard_path = output / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz"
    save_shard(shard_path, arrays, {
        "model": "FlowerMotion tr_t0_full",
        "checkpoint": str((root / args.checkpoint).resolve()),
        "steps": args.steps,
        "feature_fps": 20,
        "maximum_length": None,
        "total_episodes": total_episodes,
        "shard_episodes": len(indices),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "segments": len(arrays["segment_key"]),
        "transitions": len(arrays["transition_key"]),
    })
    print(shard_path, flush=True)


if __name__ == "__main__":
    main()
