from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from evaluate_flower_babel_long import load_objective, sample
from t2m_long.babel_long_benchmark import stable_seed
from t2m_long.babel_protocol import effective_segments
from t2m_long.data import MotionTextDataset
from t2m_long.motion_archive import save_motion_episode
from t2m_long.representation import recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FlowerMotion BABEL-long motions.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("runs/t2m_long_tr_t0_full/checkpoints/latest.pt"),
    )
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument(
        "--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features")
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("eval/babel_long_v2/visualization/selected_20.json"),
    )
    parser.add_argument(
        "--motion-output-root", type=Path, default=Path("eval/babel_long_v2/motions")
    )
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--all-eligible", action="store_true")
    parser.add_argument("--min-episode-frames", type=int, default=201)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    data_root = flood_root / "raw_data"
    output_root = (root / args.motion_output_root).resolve()
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    dataset = MotionTextDataset(data_root, root / args.embedding_root, "babel", "val", None)
    objective = load_objective((root / args.checkpoint).resolve(), data_root, device)
    index_by_name = {name: index for index, name in enumerate(dataset.ids)}
    if args.all_eligible:
        selection_protocol = "babel_long_v2_full_cohort"
        selected_ids = [
            name
            for name in dataset.ids
            if np.load(dataset.motion_dir / f"{name}.npy", mmap_mode="r").shape[0]
            >= args.min_episode_frames
            and len(
                effective_segments(
                    dataset[index_by_name[name]]["text_ends"],
                    dataset[index_by_name[name]]["length"],
                )
            )
            >= 2
        ]
    else:
        selection = json.loads((root / args.selection).resolve().read_text())
        selection_protocol = str(selection["protocol"])
        selected_ids = [str(case["episode_id"]) for case in selection["cases"]]
        if len(selected_ids) != 20 or len(set(selected_ids)) != 20:
            raise ValueError("Selection manifest must contain exactly 20 unique episodes")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    selected_set = set(selected_ids)
    missing = selected_set.difference(index_by_name)
    if missing:
        raise KeyError(f"Selected episodes missing from BABEL validation set: {sorted(missing)}")

    shard_ids = selected_ids[args.shard_index :: args.num_shards]
    for progress, name in enumerate(shard_ids, start=1):
        expected = [
            output_root / f"flowermotion_{steps}step" / f"{name}--"
            for steps in args.steps
        ]
        if not args.overwrite and all(list(path.parent.glob(path.name + "*.npz")) for path in expected):
            print(
                f"[flowermotion shard {args.shard_index}] skip {progress}/{len(shard_ids)} {name}",
                flush=True,
            )
            continue
        item = dataset[index_by_name[name]]
        length = int(item["length"])
        segments = effective_segments(item["text_ends"], length)
        captions = [item["captions"][segment.annotation_index] for segment in segments]
        starts = [segment.start for segment in segments]
        ends = [segment.end for segment in segments]
        for steps in args.steps:
            prediction = sample(
                objective,
                item["condition"],
                steps,
                stable_seed(args.seed, "flower", name, "multi", steps),
            )[:length]
            if prediction.shape != (length, 263):
                raise RuntimeError(f"Unexpected prediction shape for {name}: {prediction.shape}")
            joints = recover_joints_from_ric(prediction[None])[0]
            save_motion_episode(
                output_root,
                method=f"flowermotion_{steps}step",
                episode_id=name,
                motion_263=prediction,
                joints_y_up=joints,
                captions=captions,
                segment_starts=starts,
                segment_ends=ends,
                seed=args.seed,
                metadata={
                    "checkpoint": str((root / args.checkpoint).resolve()),
                    "native_representation": "HumanML3D 263D at 20 fps",
                    "adapter": "native one-shot full-sequence generation; no sliding window",
                    "sampling_steps": steps,
                    "benchmark": "babel_long_v2_gt10s_no_cap",
                    "selection_protocol": selection_protocol,
                },
            )
        print(
            f"[flowermotion shard {args.shard_index}] episodes {progress}/{len(shard_ids)} {name}",
            flush=True,
        )


if __name__ == "__main__":
    main()
