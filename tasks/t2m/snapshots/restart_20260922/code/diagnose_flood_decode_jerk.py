from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from evaluate_flooddiffusion_baseline import exact_token_text_ends, load_released_models
from t2m_long.babel_protocol import effective_segments, transition_jerk_curve
from t2m_long.data import MotionTextDataset
from t2m_long.representation import recover_joints_from_ric


def decode_streaming(vae, latent: torch.Tensor) -> torch.Tensor:
    vae.clear_cache()
    pieces = []
    for index in range(len(latent)):
        piece = vae.stream_decode(
            latent[index : index + 1][None], first_chunk=index == 0
        )[0]
        pieces.append(piece.float())
    vae.clear_cache()
    return torch.cat(pieces, dim=0)


def summarize(curves: list[np.ndarray], baseline: float = 0.033031363) -> None:
    values = np.stack(curves)
    mean_curve = values.mean(axis=0)
    print(
        {
            "count": len(values),
            "peak_of_mean": float(mean_curve.max()),
            "auj_of_mean": float(np.abs(mean_curve - baseline).sum()),
            "mean_of_peak": float(values.max(axis=1).mean()),
            "mean_of_auj": float(np.abs(values - baseline).sum(axis=1).mean()),
            "global_peak": float(values.max()),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg", type=float, default=6.0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    flood_root = root / "third_party/FloodDiffusion"
    data_root = flood_root / "raw_data"
    dataset = MotionTextDataset(
        data_root, root / "data/umt5_xxl_sentence_features", "babel", "val", 200
    )
    dataset.ids = [
        name
        for name in dataset.ids
        if len((data_root / "BABEL_streamed/texts" / f"{name}.txt").read_text().splitlines()) >= 2
    ][: args.limit]
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy")).to(device)
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy")).to(device)
    vae, model, _, _ = load_released_models(flood_root, device)
    model.cfg_scale = args.cfg
    curves = {"full": [], "stream": [], "gt": []}
    lengths = []

    for item in dataset:
        frame_length = int(item["length"])
        latent_length = (frame_length + vae.downsample_factor - 1) // vae.downsample_factor
        text_path = data_root / "BABEL_streamed/texts" / f"{item['name']}.txt"
        token_ends = exact_token_text_ends(text_path)
        with torch.inference_mode():
            latent = model.generate(
                {
                    "feature_length": torch.tensor([latent_length]),
                    "text": [item["captions"]],
                    "feature_text_end": [token_ends],
                },
                num_denoise_steps=10,
            )["generated"][0]
            full = vae.decode(latent[None])[0].float()
            stream = decode_streaming(vae, latent)
            gt = item["motion"].to(device) * std + mean
            joint_sets = {
                "full": recover_joints_from_ric(full[None])[0],
                "stream": recover_joints_from_ric(stream[None])[0],
                "gt": recover_joints_from_ric(gt[None])[0],
            }
        segments = effective_segments(item["text_ends"], frame_length)
        for segment in segments[1:]:
            boundary = segment.start
            if boundary < 15:
                continue
            local: dict[str, np.ndarray] = {}
            for key, motion in joint_sets.items():
                if boundary + 15 <= len(motion):
                    local[key] = transition_jerk_curve(
                        motion[boundary - 15 : boundary + 15].cpu().numpy()
                    )
            if len(local) == 3:
                for key, value in local.items():
                    curves[key].append(value)
        lengths.append((frame_length, len(full), len(stream)))

    print("length triplets", lengths[:10])
    for key, value in curves.items():
        print(key)
        summarize(value)


if __name__ == "__main__":
    main()
