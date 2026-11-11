from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from t2m_long.data import MotionTextDataset, collate_motion_text
from t2m_long.evaluator import FloodT2MEvaluator
from t2m_long.meanflow import T2MMeanFlow
from t2m_long.model import OneShotT2MDecoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate whole-sequence 1/4-step generation on multi-action BABEL."
    )
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/t2m_long"))
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda")
    data_root = args.flood_root / "raw_data"
    dataset = MotionTextDataset(
        data_root,
        args.embedding_root,
        dataset="babel",
        split="val",
        max_length=200,
        multi_action_only=True,
    )
    if args.limit:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_motion_text,
    )

    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy"))
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy"))
    objective = T2MMeanFlow(OneShotT2MDecoder(), mean, std).to(device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    objective.load_state_dict(checkpoint["model"])
    if "ema_decoder" in checkpoint:
        objective.decoder.load_state_dict(checkpoint["ema_decoder"])
    objective.eval().requires_grad_(False)

    args.output.mkdir(parents=True, exist_ok=True)
    for sampling_steps in args.steps:
        evaluator = FloodT2MEvaluator(args.flood_root, device, seed=args.seed)
        noise_generator = torch.Generator(device=device).manual_seed(
            args.seed + sampling_steps * 100000
        )
        for batch in tqdm(loader, desc=f"evaluate {sampling_steps}-step"):
            condition = batch["condition"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            ground_truth = batch["motion"].to(device, non_blocking=True)
            prediction = objective.sample(
                condition, mask, steps=sampling_steps, generator=noise_generator
            )
            prediction_raw = prediction * objective.motion_std + objective.motion_mean
            ground_truth_raw = ground_truth * objective.motion_std + objective.motion_mean
            # FloodDiffusion's BABEL evaluator associates the complete sequence
            # with the final action caption; preserve that released convention.
            final_tokens = [per_sample[-1] for per_sample in batch["text_tokens"]]
            evaluator.add_batch(
                prediction_raw, ground_truth_raw, batch["lengths"], final_tokens
            )
        metrics = evaluator.compute()
        record = {
            "checkpoint": str(args.checkpoint.resolve()),
            "sampling_steps": sampling_steps,
            "sampling_mode": "one-shot whole-sequence ODE",
            "split": "BABEL val_processed (public held-out split)",
            "subset": ">=2 action text segments",
            "metrics": metrics,
        }
        output_path = args.output / f"metrics_{sampling_steps}step.json"
        output_path.write_text(json.dumps(record, indent=2))
        print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
