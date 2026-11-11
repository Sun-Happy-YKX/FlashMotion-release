from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluate_flood_babel_long import generate
from evaluate_flooddiffusion_baseline import load_released_models
from t2m_long.babel_protocol import effective_segments
from t2m_long.data import MotionTextDataset
from t2m_long.motion_archive import save_motion_episode
from t2m_long.representation import recover_joints_from_ric
from t2m_long.babel_long_benchmark import stable_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FloodDiffusion BABEL-long motions.")
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--motion-output-root", type=Path, default=Path("eval/babel_long_v2/motions"))
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--min-episode-frames", type=int, default=201)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-episodes", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    data_root = flood_root / "raw_data"
    output_root = (root / args.motion_output_root).resolve()
    device = torch.device("cuda")
    dataset = MotionTextDataset(data_root, root / args.embedding_root, "babel", "val", None)
    eligible = [
        index
        for index, name in enumerate(dataset.ids)
        if np.load(dataset.motion_dir / f"{name}.npy", mmap_mode="r").shape[0]
        >= args.min_episode_frames
    ]
    eligible = [
        index
        for index in eligible
        if len(effective_segments(dataset[index]["text_ends"], dataset[index]["length"])) >= 2
    ]
    if args.max_episodes is not None:
        eligible = eligible[: args.max_episodes]
    indices = eligible[args.shard_index :: args.num_shards]
    vae, model, model_checkpoint, vae_checkpoint = load_released_models(flood_root, device)
    model.cfg_scale = args.cfg_scale
    for progress, dataset_index in enumerate(indices, start=1):
        item = dataset[dataset_index]
        name = item["name"]
        length = item["length"]
        segments = effective_segments(item["text_ends"], length)
        token_ends = [int(end / 4.0 + 0.5) for end in item["text_ends"]]
        prediction = generate(
            model,
            vae,
            item["captions"],
            token_ends,
            length,
            stable_seed(args.seed, "flood", name, "multi", args.denoise_steps),
            args.denoise_steps,
            device,
        )
        pred_joints = recover_joints_from_ric(prediction[None])[0]
        save_motion_episode(
            output_root,
            method="flooddiffusion",
            episode_id=name,
            motion_263=prediction,
            joints_y_up=pred_joints,
            captions=[item["captions"][segment.annotation_index] for segment in segments],
            segment_starts=[segment.start for segment in segments],
            segment_ends=[segment.end for segment in segments],
            seed=args.seed,
            metadata={
                "checkpoint": str(model_checkpoint),
                "vae_checkpoint": str(vae_checkpoint),
                "native_representation": "HumanML3D 263D at 20 fps",
                "adapter": "official one-shot multi-caption generation; no length cap",
                "denoise_steps": args.denoise_steps,
                "cfg_scale": args.cfg_scale,
                "benchmark": "babel_long_v2_gt10s_no_cap",
            },
        )
        if progress % 10 == 0 or progress == len(indices):
            print(
                f"[flooddiffusion shard {args.shard_index}] episodes {progress}/{len(indices)}",
                flush=True,
            )
            model.text_cache.clear()


if __name__ == "__main__":
    main()
