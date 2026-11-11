from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from evaluate_flooddiffusion_baseline import exact_token_text_ends, load_released_models
from t2m_long.babel_protocol import effective_segments
from t2m_long.data import MotionTextDataset, collate_motion_text


# A qualitative panel spanning locomotion, transitions, acrobatics, dance, and
# fine-grained upper-body actions.  The first six names are the original panel.
SELECTED_NAMES = [
    "1567_3", "2062_9", "11264_1", "7743_8", "9085_3", "12738_4",
    "11597_1", "919_3", "6519_5", "6519_8", "10371_3", "1786_1",
    "252_3", "912_11", "4207_7", "4869_1",
    "4523_3", "6066_5", "3063_5", "6078_3", "10270_3", "10069_10",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-1", type=Path, required=True)
    parser.add_argument("--checkpoint-4", type=Path, required=True)
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_fourway_more_raw"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-flower", action="store_true")
    parser.add_argument("--skip-flood", action="store_true")
    return parser.parse_args()


def save_metadata(sample_dir: Path, batch: dict, index: int, length: int) -> None:
    segments = effective_segments(batch["text_ends"][index], length)
    sample_dir.mkdir(parents=True, exist_ok=True)
    (sample_dir / "metadata.json").write_text(json.dumps({
        "name": batch["name"][index],
        "captions": batch["captions"][index],
        "text_ends": batch["text_ends"][index],
        "effective_segments": [segment.__dict__ for segment in segments],
        "length": length,
    }, indent=2))


def make_loader(dataset: MotionTextDataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        collate_fn=collate_motion_text,
    )


def generate_flower(
    checkpoint: Path,
    steps: int,
    dataset: MotionTextDataset,
    data_root: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> None:
    # Keep the FlowerDance-only Mamba dependency out of FloodDiffusion's
    # official runtime; the two released models intentionally use separate
    # Python environments on the cluster.
    from evaluate_flower_babel_protocols import load_objective

    objective = load_objective(checkpoint, data_root, device)
    generator = torch.Generator(device=device).manual_seed(seed + steps * 100000)
    loader = make_loader(dataset, batch_size, workers)
    label = f"flower_{steps}step"
    for batch in loader:
        condition = batch["condition"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        ground_truth = batch["motion"].to(device, non_blocking=True)
        with torch.inference_mode():
            prediction = objective.sample(condition, mask, steps, generator)
            prediction_raw = prediction * objective.motion_std + objective.motion_mean
            ground_truth_raw = ground_truth * objective.motion_std + objective.motion_mean
        for index, length_value in enumerate(batch["lengths"].tolist()):
            length = int(length_value)
            sample_dir = output / batch["name"][index]
            save_metadata(sample_dir, batch, index, length)
            np.save(sample_dir / f"{label}.npy", prediction_raw[index, :length].float().cpu().numpy())
            np.save(sample_dir / "gt.npy", ground_truth_raw[index, :length].float().cpu().numpy())
            print(sample_dir / f"{label}.npy", flush=True)
    del objective
    torch.cuda.empty_cache()


def generate_flood(
    dataset: MotionTextDataset,
    flood_root: Path,
    data_root: Path,
    output: Path,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy")).to(device)
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy")).to(device)
    vae, model, _, _ = load_released_models(flood_root, device)
    loader = make_loader(dataset, batch_size, workers)
    for batch in loader:
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
        with torch.inference_mode():
            latents = model.generate({
                "feature_length": latent_lengths,
                "text": batch["captions"],
                "feature_text_end": token_ends,
            }, num_denoise_steps=10)["generated"]
            predictions = [vae.decode(latent[None].to(device))[0].float().cpu().numpy() for latent in latents]
            ground_truth = (batch["motion"].to(device) * std + mean).float().cpu().numpy()
        for index, name in enumerate(batch["name"]):
            length = int(frame_lengths[index])
            sample_dir = output / name
            save_metadata(sample_dir, batch, index, length)
            np.save(sample_dir / "flood_10step.npy", predictions[index])
            np.save(sample_dir / "gt.npy", ground_truth[index, :length])
            print(sample_dir / "flood_10step.npy", flush=True)
        model.text_cache.clear()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    data_root = flood_root / "raw_data"
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = MotionTextDataset(data_root, root / args.embedding_root, "babel", "val", 200)
    selected = set(SELECTED_NAMES)
    dataset.ids = [name for name in dataset.ids if name in selected]
    missing = selected.difference(dataset.ids)
    if missing:
        raise RuntimeError(f"Selected samples missing from BABEL validation split: {sorted(missing)}")
    print(f"Selected {len(dataset)} samples: {dataset.ids}", flush=True)
    device = torch.device("cuda")
    if not args.skip_flower:
        generate_flower(
            (root / args.checkpoint_1).resolve(), 1, dataset, data_root, output,
            device, args.batch_size, args.workers, args.seed,
        )
        generate_flower(
            (root / args.checkpoint_4).resolve(), 4, dataset, data_root, output,
            device, args.batch_size, args.workers, args.seed,
        )
    if not args.skip_flood:
        generate_flood(
            dataset, flood_root, data_root, output, device,
            args.batch_size, args.workers, args.seed,
        )


if __name__ == "__main__":
    main()
