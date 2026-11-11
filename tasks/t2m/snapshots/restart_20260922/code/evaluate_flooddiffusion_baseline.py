from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from t2m_long.data import MotionTextDataset, collate_motion_text
from t2m_long.evaluator import FloodT2MEvaluator


def copy_released_ema(module: torch.nn.Module, ema_state: dict) -> None:
    parameters = list(module.parameters())
    shadow_parameters = ema_state["shadow_params"]
    if len(parameters) != len(shadow_parameters):
        raise RuntimeError(
            f"EMA parameter count mismatch: {len(parameters)} != {len(shadow_parameters)}"
        )
    with torch.no_grad():
        for parameter, shadow in zip(parameters, shadow_parameters):
            parameter.copy_(shadow.to(device=parameter.device, dtype=parameter.dtype))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate FloodDiffusion on the exact FlowerDance multi-action BABEL subset."
    )
    parser.add_argument(
        "--flood-root", type=Path, default=Path("third_party/FloodDiffusion")
    )
    parser.add_argument(
        "--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features")
    )
    parser.add_argument("--output", type=Path, default=Path("eval/flooddiffusion_baseline"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def load_released_models(flood_root: Path, device: torch.device):
    sys.path.insert(0, str(flood_root))
    from models.diffusion_forcing_wan import DiffForcingWanModel
    from models.vae_wan_1d import VAEWanModel

    model_checkpoint_path = (
        flood_root / "outputs/20251107_021814_ldf_stream/step_step=240000.ckpt"
    )
    vae_checkpoint_path = flood_root / "outputs/vae_1d_z4_step=300000.ckpt"
    vae = VAEWanModel(input_dim=263, z_dim=4)
    vae_checkpoint = torch.load(
        vae_checkpoint_path, map_location="cpu", weights_only=False
    )
    vae.load_state_dict(vae_checkpoint["state_dict"], strict=True)
    if "ema_state" in vae_checkpoint:
        copy_released_ema(vae, vae_checkpoint["ema_state"])

    model = DiffForcingWanModel(
        checkpoint_path=str(
            flood_root / "deps/t5_umt5-xxl-enc-bf16/models_t5_umt5-xxl-enc-bf16.pth"
        ),
        tokenizer_path=str(flood_root / "deps/t5_umt5-xxl-enc-bf16/google/umt5-xxl"),
        input_dim=4,
        noise_steps=10,
    )
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    if "ema_state" in checkpoint:
        copy_released_ema(model, checkpoint["ema_state"])

    vae.to(device).eval().requires_grad_(False)
    model.to(device).eval().requires_grad_(False)
    return vae, model, str(model_checkpoint_path.resolve()), str(
        vae_checkpoint_path.resolve()
    )


def exact_token_text_ends(text_path: Path) -> list[int]:
    ends = []
    for line in text_path.read_text().splitlines():
        if not line.strip():
            continue
        _, _, _, end_seconds = line.split("#")
        ends.append(int(float(end_seconds) * 5.0 + 0.5))
    if not ends:
        raise RuntimeError(f"No annotations in {text_path}")
    return ends


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    embedding_root = (root / args.embedding_root).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_float32_matmul_precision("high")

    data_root = flood_root / "raw_data"
    dataset = MotionTextDataset(
        data_root,
        embedding_root,
        dataset="babel",
        split="val",
        max_length=200,
        multi_action_only=True,
    )
    if args.limit:
        dataset.ids = dataset.ids[: min(args.limit, len(dataset))]
    total_samples = len(dataset)
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    dataset.ids = dataset.ids[args.shard_index :: args.num_shards]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_motion_text,
    )

    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy")).to(device)
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy")).to(device)
    vae, model, model_checkpoint, vae_checkpoint = load_released_models(
        flood_root, device
    )
    if args.denoise_steps % model.chunk_size:
        raise ValueError(
            f"FloodDiffusion requires denoise steps divisible by chunk_size={model.chunk_size}"
        )

    evaluator = FloodT2MEvaluator(flood_root, device, seed=args.seed)
    for batch_index, batch in enumerate(tqdm(loader, desc="FloodDiffusion evaluation")):
        frame_lengths = batch["lengths"]
        latent_lengths = torch.div(
            frame_lengths + vae.downsample_factor - 1,
            vae.downsample_factor,
            rounding_mode="floor",
        )
        token_ends = [
            exact_token_text_ends(data_root / "BABEL_streamed/texts" / f"{name}.txt")
            for name in batch["name"]
        ]
        model_input = {
            "feature_length": latent_lengths,
            "text": batch["captions"],
            "feature_text_end": token_ends,
        }
        with torch.inference_mode():
            generated_latents = model.generate(
                model_input, num_denoise_steps=args.denoise_steps
            )["generated"]
            predictions = [vae.decode(latent[None].to(device))[0].float() for latent in generated_latents]

        prediction_lengths = torch.tensor(
            [prediction.shape[0] for prediction in predictions], dtype=torch.long
        )
        prediction_raw = pad_sequence(predictions, batch_first=True)
        ground_truth_normalized = batch["motion"].to(device, non_blocking=True)
        ground_truth_raw = ground_truth_normalized * std + mean
        final_tokens = [per_sample[-1] for per_sample in batch["text_tokens"]]
        evaluator.add_batch(
            prediction_raw,
            ground_truth_raw,
            frame_lengths,
            final_tokens,
            prediction_lengths=prediction_lengths,
            ground_truth_lengths=frame_lengths,
        )

        # The released model caches full token-level UMT5 outputs on CPU. Bound
        # memory use during a full validation-set evaluation without changing outputs.
        if (batch_index + 1) % 100 == 0:
            model.text_cache.clear()

    prediction_embeddings = torch.cat(evaluator.prediction_embeddings).numpy()
    ground_truth_embeddings = torch.cat(evaluator.ground_truth_embeddings).numpy()
    text_embeddings = torch.cat(evaluator.text_embeddings).numpy()
    embedding_path = output / (
        f"embeddings_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz"
    )
    np.savez(
        embedding_path,
        prediction=prediction_embeddings,
        ground_truth=ground_truth_embeddings,
        text=text_embeddings,
    )
    metrics = evaluator.compute() if len(dataset) >= 32 and args.num_shards == 1 else None
    record = {
        "model": "FloodDiffusion released checkpoint",
        "checkpoint": model_checkpoint,
        "vae_checkpoint": vae_checkpoint,
        "denoise_steps": args.denoise_steps,
        "cfg_scale": model.cfg_scale,
        "split": "BABEL val_processed (public held-out split)",
        "subset": ">=2 action text segments",
        "samples": len(dataset),
        "total_samples": total_samples,
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "embedding_path": str(embedding_path),
        "metrics": metrics,
    }
    output_path = output / (
        f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json"
        if args.num_shards > 1
        else f"metrics_{args.denoise_steps}step.json"
    )
    output_path.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
