from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t2m_long.evaluator import FloodT2MEvaluator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    input_dir = (root / args.input).resolve()
    shard_paths = sorted(input_dir.glob("embeddings_shard_*_of_*.npz"))
    if not shard_paths:
        raise RuntimeError(f"No shard embeddings in {input_dir}")
    predictions, ground_truths, texts = [], [], []
    for path in shard_paths:
        with np.load(path) as data:
            predictions.append(torch.from_numpy(data["prediction"].copy()))
            ground_truths.append(torch.from_numpy(data["ground_truth"].copy()))
            texts.append(torch.from_numpy(data["text"].copy()))

    # Each worker evaluated ids[shard_index::num_shards]. Interleave rows back
    # into the original dataset order so the fixed evaluator permutation creates
    # exactly the same 32-sample retrieval groups as the FlowerDance sweep.
    def restore_original_order(shards: list[torch.Tensor]) -> torch.Tensor:
        rows = []
        for row_index in range(max(len(shard) for shard in shards)):
            rows.extend(
                shard[row_index : row_index + 1]
                for shard in shards
                if row_index < len(shard)
            )
        return torch.cat(rows)

    evaluator = FloodT2MEvaluator(
        (root / args.flood_root).resolve(), torch.device("cuda"), seed=args.seed
    )
    evaluator.prediction_embeddings = [restore_original_order(predictions)]
    evaluator.ground_truth_embeddings = [restore_original_order(ground_truths)]
    evaluator.text_embeddings = [restore_original_order(texts)]
    metrics = evaluator.compute()
    record = {
        "model": "FloodDiffusion released checkpoint",
        "checkpoint": str(
            (root / args.flood_root / "outputs/20251107_021814_ldf_stream/step_step=240000.ckpt").resolve()
        ),
        "vae_checkpoint": str(
            (root / args.flood_root / "outputs/vae_1d_z4_step=300000.ckpt").resolve()
        ),
        "denoise_steps": args.denoise_steps,
        "cfg_scale": 5.0,
        "split": "BABEL val_processed (public held-out split)",
        "subset": ">=2 action text segments",
        "seed": args.seed,
        "shards": len(shard_paths),
        "metrics": metrics,
    }
    output_path = input_dir / f"metrics_{args.denoise_steps}step.json"
    output_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
