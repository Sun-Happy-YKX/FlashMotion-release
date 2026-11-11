from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from t2m_long.babel_long_benchmark import (
    MultiPredictionEmbeddingCollector,
    TRANSITION_FRAMES,
    centered_transition,
    save_shard,
    score_embedding_subset,
    transition_metrics,
)
from t2m_long.babel_protocol import effective_segments
from t2m_long.data import MotionTextDataset
from t2m_long.evaluator import FloodT2MEvaluator
from t2m_long.motion_archive import validate_motion_archive
from t2m_long.representation import recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score one saved BABEL-long-v2 motion archive.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--archive-root", type=Path, default=Path("eval/babel_long_v2/motions"))
    parser.add_argument("--output-root", type=Path, default=Path("eval/babel_long_v2/scores"))
    parser.add_argument("--min-episode-frames", type=int, default=201)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_archives(method_root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted(method_root.glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"]))
        episode_id = str(metadata["episode_id"])
        if episode_id in result:
            raise RuntimeError(f"duplicate archive for {episode_id}")
        result[episode_id] = path
    return result


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    embedding_root = (root / args.embedding_root).resolve()
    method_root = (root / args.archive_root / args.method).resolve()
    output_root = (root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    validation = validate_motion_archive(method_root)
    if validation["failures"]:
        raise RuntimeError(f"archive validation failed: {validation['failures'][:3]}")
    archives = load_archives(method_root)

    dataset = MotionTextDataset(
        flood_root / "raw_data", embedding_root, "babel", "val", None
    )
    eligible = [
        index
        for index, name in enumerate(dataset.ids)
        if np.load(dataset.motion_dir / f"{name}.npy", mmap_mode="r").shape[0]
        >= args.min_episode_frames
    ]
    eligible = [index for index in eligible if len(effective_segments(dataset[index]["text_ends"], dataset[index]["length"])) >= 2]
    expected_ids = {dataset.ids[index] for index in eligible}
    missing = sorted(expected_ids - set(archives))
    extra = sorted(set(archives) - expected_ids)
    if missing or extra:
        raise RuntimeError(f"archive cohort mismatch: missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})")

    device = torch.device("cuda")
    evaluator = FloodT2MEvaluator(flood_root, device, seed=args.seed)
    collector = MultiPredictionEmbeddingCollector(
        evaluator, [args.method], args.embedding_batch_size
    )
    transition_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    mean = torch.from_numpy(dataset.mean).float()
    std = torch.from_numpy(dataset.std).float()

    for progress, dataset_index in enumerate(eligible, start=1):
        item = dataset[dataset_index]
        name = item["name"]
        gt_raw = item["motion"] * std + mean
        with np.load(archives[name], allow_pickle=False) as data:
            prediction = torch.from_numpy(np.asarray(data["motion_263"], dtype=np.float32))
            archived_starts = np.asarray(data["segment_starts"], dtype=np.int64)
            archived_ends = np.asarray(data["segment_ends"], dtype=np.int64)
        if prediction.shape != gt_raw.shape:
            raise RuntimeError(f"{name}: prediction {prediction.shape}, GT {gt_raw.shape}")
        segments = effective_segments(item["text_ends"], item["length"])
        expected_starts = np.asarray([segment.start for segment in segments])
        expected_ends = np.asarray([segment.end for segment in segments])
        if not np.array_equal(archived_starts, expected_starts) or not np.array_equal(archived_ends, expected_ends):
            raise RuntimeError(f"{name}: archived segment boundaries differ from benchmark")

        gt_joints = recover_joints_from_ric(gt_raw[None])[0]
        pred_joints = recover_joints_from_ric(prediction[None])[0]
        for transition_index, segment in enumerate(segments[1:]):
            gt_curve = centered_transition(gt_joints, segment.start, TRANSITION_FRAMES)
            pred_curve = centered_transition(pred_joints, segment.start, TRANSITION_FRAMES)
            if gt_curve is None or pred_curve is None:
                continue
            transition_parts["transition__ground_truth"].append(gt_curve)
            transition_parts[f"transition__{args.method}"].append(pred_curve)
            transition_parts["transition_key"].append(
                np.asarray(f"{name}::{transition_index}", dtype=np.str_)
            )
            transition_parts["transition_episode_id"].append(
                np.asarray(name, dtype=np.str_)
            )
        for segment_index, segment in enumerate(segments):
            caption = item["captions"][segment.annotation_index]
            collector.add(
                gt_raw[segment.start : segment.end],
                item["text_tokens"][segment.annotation_index],
                {args.method: prediction[segment.start : segment.end]},
                {
                    "segment_key": f"{name}::{segment_index}",
                    "episode_id": name,
                    "caption": caption,
                    "segment_index": segment_index,
                    "segment_length": segment.end - segment.start,
                    "episode_length": item["length"],
                },
            )
        if progress % 50 == 0 or progress == len(eligible):
            print(f"[{args.method}] episodes {progress}/{len(eligible)}", flush=True)

    arrays = collector.finish()
    for key, values in transition_parts.items():
        arrays[key] = np.stack(values) if values else np.empty((0,), dtype=np.float32)
    shard_path = output_root / f"{args.method}.npz"
    save_shard(
        shard_path,
        arrays,
        {
            "method": args.method,
            "benchmark": "babel_long_v2_gt10s_no_cap",
            "minimum_episode_frames": args.min_episode_frames,
            "episodes": len(eligible),
            "segments": len(arrays["segment_key"]),
            "transitions": len(arrays["transition_key"]),
            "archive_validation": validation,
        },
    )
    selected = np.ones(len(arrays["segment_key"]), dtype=bool)
    metrics = {
        "method": args.method,
        "benchmark": "babel_long_v2_gt10s_no_cap",
        "semantic": score_embedding_subset(
            arrays,
            f"prediction__{args.method}",
            selected,
            device,
            seed=args.seed,
        ),
        "transition": transition_metrics(arrays, args.method),
        "archive_validation": validation,
    }
    metrics_path = output_root / f"{args.method}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
