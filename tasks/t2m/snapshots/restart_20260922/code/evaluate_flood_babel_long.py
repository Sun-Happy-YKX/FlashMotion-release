from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from evaluate_flooddiffusion_baseline import load_released_models
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
from t2m_long.representation import recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_long_v1/flood"))
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    return parser.parse_args()


@torch.inference_mode()
def generate(
    model,
    vae,
    captions: list[str],
    token_ends: list[int],
    frame_length: int,
    seed: int,
    denoise_steps: int,
    device: torch.device,
) -> torch.Tensor:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The released causal VAE maps N latent tokens to 4*N-3 frames, rather
    # than exactly 4*N.  Request enough tokens to cover the target duration
    # and crop only the decoded overhang.
    latent_length = (
        frame_length + (vae.downsample_factor - 1) + (vae.downsample_factor - 1)
    ) // vae.downsample_factor
    conditioned_ends = [
        min(latent_length, max(0, int(end))) for end in token_ends
    ]
    if conditioned_ends:
        conditioned_ends[-1] = latent_length
    result = model.generate({
        "feature_length": torch.tensor([latent_length], dtype=torch.long),
        "text": [captions],
        "feature_text_end": [conditioned_ends],
    }, num_denoise_steps=denoise_steps)["generated"][0]
    decoded = vae.decode(result[None].to(device))[0].float()
    if len(decoded) < frame_length:
        raise RuntimeError(f"Flood decoded {len(decoded)} frames for requested {frame_length}")
    return decoded[:frame_length]


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
    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy")).to(device)
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy")).to(device)
    vae, model, model_checkpoint, vae_checkpoint = load_released_models(flood_root, device)
    model.cfg_scale = args.cfg_scale
    if args.denoise_steps % model.chunk_size:
        raise ValueError(
            f"denoise steps must be divisible by Flood chunk_size={model.chunk_size}"
        )
    evaluator = FloodT2MEvaluator(flood_root, device, seed=args.seed)
    prediction_names = ["single_10step", "multi_10step"]
    collector = MultiPredictionEmbeddingCollector(
        evaluator, prediction_names, args.embedding_batch_size
    )
    transition_parts: dict[str, list[np.ndarray]] = defaultdict(list)
    selected_names = {"1567_3", "2062_9", "11264_1", "7743_8", "9085_3", "12738_4"}

    for item_number, dataset_index in enumerate(
        tqdm(indices, desc=f"Flood shard {args.shard_index}")
    ):
        item = dataset[dataset_index]
        name = item["name"]
        length = item["length"]
        gt_raw = item["motion"].to(device) * std + mean
        segments = effective_segments(item["text_ends"], length)
        token_ends = [int(end / 4.0 + 0.5) for end in item["text_ends"]]
        full_prediction = generate(
            model,
            vae,
            item["captions"],
            token_ends,
            length,
            stable_seed(args.seed, "flood", name, "multi", args.denoise_steps),
            args.denoise_steps,
            device,
        )

        gt_joints = recover_joints_from_ric(gt_raw[None])[0]
        pred_joints = recover_joints_from_ric(full_prediction[None])[0]
        for transition_index, segment in enumerate(segments[1:]):
            gt_curve = centered_transition(gt_joints, segment.start, TRANSITION_FRAMES)
            pred_curve = centered_transition(pred_joints, segment.start, TRANSITION_FRAMES)
            if gt_curve is None or pred_curve is None:
                continue
            transition_parts["transition__ground_truth"].append(gt_curve)
            transition_parts["transition__multi_10step"].append(pred_curve)
            transition_parts["transition_key"].append(
                np.asarray(f"{name}::{transition_index}", dtype=np.str_)
            )
            transition_parts["transition_episode_id"].append(np.asarray(name, dtype=np.str_))

        for segment_index, segment in enumerate(segments):
            segment_length = segment.end - segment.start
            caption = item["captions"][segment.annotation_index]
            standalone_prediction = generate(
                model,
                vae,
                [caption],
                [10**9],
                segment_length,
                stable_seed(
                    args.seed, "flood", name, segment_index, "single", args.denoise_steps
                ),
                args.denoise_steps,
                device,
            )
            collector.add(
                gt_raw[segment.start : segment.end],
                item["text_tokens"][segment.annotation_index],
                {
                    "multi_10step": full_prediction[segment.start : segment.end],
                    "single_10step": standalone_prediction,
                },
                {
                    "segment_key": f"{name}::{segment_index}",
                    "episode_id": name,
                    "caption": caption,
                    "segment_index": segment_index,
                    "segment_length": segment_length,
                    "episode_length": length,
                },
            )

        if name in selected_names:
            sample_dir = output / "raw" / name
            sample_dir.mkdir(parents=True, exist_ok=True)
            np.save(sample_dir / "gt.npy", gt_raw.float().cpu().numpy())
            np.save(sample_dir / "flood_10step.npy", full_prediction.float().cpu().numpy())
            (sample_dir / "metadata.json").write_text(json.dumps({
                "name": name,
                "captions": item["captions"],
                "text_ends": item["text_ends"],
                "length": length,
            }, indent=2))
        if (item_number + 1) % 50 == 0:
            model.text_cache.clear()

    arrays = collector.finish()
    for key, values in transition_parts.items():
        arrays[key] = np.stack(values) if values else np.empty((0,), dtype=np.float32)
    shard_path = output / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz"
    save_shard(shard_path, arrays, {
        "model": "FloodDiffusion released BABEL checkpoint",
        "checkpoint": model_checkpoint,
        "vae_checkpoint": vae_checkpoint,
        "denoise_steps": args.denoise_steps,
        "cfg_scale": args.cfg_scale,
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
