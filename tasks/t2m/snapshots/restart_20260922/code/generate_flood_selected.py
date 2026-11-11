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


SELECTED_NAMES = {"1567_3", "2062_9", "11264_1", "7743_8", "9085_3", "12738_4"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_protocol_flood/raw"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    data_root = flood_root / "raw_data"
    output = (root / args.output).resolve()
    dataset = MotionTextDataset(data_root, root / args.embedding_root, "babel", "val", 200)
    dataset.ids = [name for name in dataset.ids if name in SELECTED_NAMES]
    if len(dataset) != len(SELECTED_NAMES):
        raise RuntimeError(f"Selected sample mismatch: {dataset.ids}")
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False, num_workers=4,
                        collate_fn=collate_motion_text)
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy")).to(device)
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy")).to(device)
    vae, model, _, _ = load_released_models(flood_root, device)
    batch = next(iter(loader))
    frame_lengths = batch["lengths"]
    latent_lengths = torch.div(frame_lengths + vae.downsample_factor - 1,
                               vae.downsample_factor, rounding_mode="floor")
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
        segments = effective_segments(batch["text_ends"][index], length)
        sample_dir = output / name
        sample_dir.mkdir(parents=True, exist_ok=True)
        np.save(sample_dir / "flood_10step.npy", predictions[index])
        np.save(sample_dir / "gt.npy", ground_truth[index, :length])
        (sample_dir / "metadata.json").write_text(json.dumps({
            "name": name,
            "captions": batch["captions"][index],
            "text_ends": batch["text_ends"][index],
            "effective_segments": [segment.__dict__ for segment in segments],
            "ground_truth_length": length,
            "prediction_length": len(predictions[index]),
        }, indent=2))
        print(sample_dir / "flood_10step.npy", flush=True)


if __name__ == "__main__":
    main()
