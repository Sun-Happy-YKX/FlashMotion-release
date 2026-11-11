from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluate_flooddiffusion_baseline import exact_token_text_ends, load_released_models
from t2m_long.babel_protocol import effective_segments, transition_jerk_curve
from t2m_long.data import MotionTextDataset, collate_motion_text
from t2m_long.evaluator import FloodT2MEvaluator
from t2m_long.representation import recover_joints_from_ric


SELECTED_NAMES = {"1567_3", "2062_9", "11264_1", "7743_8", "9085_3", "12738_4"}


def feature_jerk_curve(features: np.ndarray) -> np.ndarray:
    """Third difference followed by a max over normalized feature channels."""

    return np.abs(np.diff(features, n=3, axis=0)).max(axis=-1)


def centered_window(features: np.ndarray, boundary: int, length: int) -> np.ndarray:
    left = length // 2
    right = length - left
    start = boundary - left
    end = boundary + right
    window = features[max(start, 0) : min(end, len(features))]
    pad_left = max(0, -start)
    pad_right = max(0, end - len(features))
    if pad_left or pad_right:
        window = np.pad(window, ((pad_left, pad_right), (0, 0)), mode="edge")
    if len(window) != length:
        raise RuntimeError(f"failed to form centered window: {len(window)} != {length}")
    return window


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_protocol_flood"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--cfg-scale", type=float, default=6.0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transition-length", type=int, default=30)
    return parser.parse_args()


class EmbeddingCollector:
    def __init__(self, evaluator: FloodT2MEvaluator, batch_size: int):
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.predictions: list[torch.Tensor] = []
        self.ground_truths: list[torch.Tensor] = []
        self.tokens: list[list[str]] = []
        self.pending_keys: list[int] = []
        self.arrays = {"prediction": [], "ground_truth": [], "text": []}
        self.keys: list[int] = []

    def add(self, prediction, ground_truth, tokens, order_key) -> None:
        if min(len(prediction), len(ground_truth)) < 4:
            return
        self.predictions.append(prediction.detach().cpu())
        self.ground_truths.append(ground_truth.detach().cpu())
        self.tokens.append(tokens)
        self.pending_keys.append(order_key)
        if len(self.predictions) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.predictions:
            return
        prediction_lengths = torch.tensor([len(x) for x in self.predictions])
        ground_truth_lengths = torch.tensor([len(x) for x in self.ground_truths])
        self.arrays["prediction"].append(
            self.evaluator._motion_embeddings(
                pad_sequence(self.predictions, batch_first=True), prediction_lengths
            ).float().cpu().numpy()
        )
        self.arrays["ground_truth"].append(
            self.evaluator._motion_embeddings(
                pad_sequence(self.ground_truths, batch_first=True), ground_truth_lengths
            ).float().cpu().numpy()
        )
        self.arrays["text"].append(
            self.evaluator._text_embeddings(self.tokens).float().cpu().numpy()
        )
        self.keys.extend(self.pending_keys)
        self.predictions.clear()
        self.ground_truths.clear()
        self.tokens.clear()
        self.pending_keys.clear()

    def finish(self) -> dict[str, np.ndarray]:
        self.flush()
        return {
            name: np.concatenate(parts) for name, parts in self.arrays.items()
        } | {"keys": np.asarray(self.keys, dtype=np.int64)}


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    data_root = flood_root / "raw_data"
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = MotionTextDataset(data_root, root / args.embedding_root, "babel", "val", 200)
    name_to_index = {name: index for index, name in enumerate(dataset.ids)}
    total_samples = len(dataset)
    dataset.ids = dataset.ids[args.shard_index :: args.num_shards]
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate_motion_text,
    )
    device = torch.device("cuda")
    torch.manual_seed(args.seed + args.shard_index)
    np.random.seed(args.seed + args.shard_index)
    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy")).to(device)
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy")).to(device)
    vae, model, model_checkpoint, vae_checkpoint = load_released_models(flood_root, device)
    model.cfg_scale = args.cfg_scale
    evaluator = FloodT2MEvaluator(flood_root, device, seed=args.seed)
    collectors = {
        "single": EmbeddingCollector(evaluator, 64),
        "multi": EmbeddingCollector(evaluator, 64),
    }
    sample_counts = {"single": 0, "multi": 0}
    predicted_curves: list[np.ndarray] = []
    ground_truth_curves: list[np.ndarray] = []
    predicted_normalized_feature_curves: list[np.ndarray] = []
    ground_truth_normalized_feature_curves: list[np.ndarray] = []
    predicted_norm_velocity_45_curves: list[np.ndarray] = []
    ground_truth_norm_velocity_45_curves: list[np.ndarray] = []
    diagnostic_transition_keys: list[int] = []
    transition_keys: list[int] = []

    for batch_index, batch in enumerate(tqdm(loader, desc=f"Flood shard {args.shard_index}")):
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
            generated_latents = model.generate({
                "feature_length": latent_lengths,
                "text": batch["captions"],
                "feature_text_end": token_ends,
            }, num_denoise_steps=args.denoise_steps)["generated"]
            predictions = [vae.decode(latent[None].to(device))[0].float() for latent in generated_latents]
            prediction_lengths = [len(prediction) for prediction in predictions]
            prediction_padded = pad_sequence(predictions, batch_first=True)
            ground_truth_raw = batch["motion"].to(device, non_blocking=True) * std + mean
            prediction_joints = recover_joints_from_ric(prediction_padded)
            ground_truth_joints = recover_joints_from_ric(ground_truth_raw)

        for item_index, ground_truth_length in enumerate(frame_lengths.tolist()):
            name = batch["name"][item_index]
            global_index = name_to_index[name]
            prediction_length = prediction_lengths[item_index]
            segments = effective_segments(batch["text_ends"][item_index], ground_truth_length)
            category = "multi" if len(segments) >= 2 else "single"
            sample_counts[category] += 1
            for segment_index, segment in enumerate(segments):
                pred_start = min(segment.start, prediction_length)
                pred_end = min(segment.end, prediction_length)
                collectors[category].add(
                    predictions[item_index][pred_start:pred_end],
                    ground_truth_raw[item_index, segment.start:segment.end],
                    batch["text_tokens"][item_index][segment.annotation_index],
                    global_index * 100 + segment_index,
                )

            if category == "multi":
                prediction_normalized_np = (
                    (predictions[item_index] - mean) / std
                ).cpu().numpy()
                ground_truth_normalized_np = batch["motion"][
                    item_index, :ground_truth_length
                ].cpu().numpy()
                for transition_index, segment in enumerate(segments[1:]):
                    boundary = segment.start
                    predicted_norm_velocity_45_curves.append(feature_jerk_curve(
                        centered_window(prediction_normalized_np, boundary, 45)[:, 193:259]
                    ))
                    ground_truth_norm_velocity_45_curves.append(feature_jerk_curve(
                        centered_window(ground_truth_normalized_np, boundary, 45)[:, 193:259]
                    ))
                    diagnostic_transition_keys.append(global_index * 100 + transition_index)

                margin = args.transition_length // 2
                for transition_index, segment in enumerate(segments[1:]):
                    boundary = segment.start
                    if boundary >= margin and boundary + (args.transition_length - margin) <= min(ground_truth_length, prediction_length):
                        predicted_curves.append(transition_jerk_curve(
                            prediction_joints[item_index, boundary - margin:boundary + (args.transition_length - margin)].cpu().numpy()
                        ))
                        ground_truth_curves.append(transition_jerk_curve(
                            ground_truth_joints[item_index, boundary - margin:boundary + (args.transition_length - margin)].cpu().numpy()
                        ))
                        predicted_normalized_feature_curves.append(feature_jerk_curve(
                            ((predictions[item_index][boundary - margin:boundary + (args.transition_length - margin)] - mean) / std).cpu().numpy()
                        ))
                        ground_truth_normalized_feature_curves.append(feature_jerk_curve(
                            batch["motion"][item_index, boundary - margin:boundary + (args.transition_length - margin)].cpu().numpy()
                        ))
                        transition_keys.append(global_index * 100 + transition_index)

            if name in SELECTED_NAMES:
                raw_dir = output / "raw" / name
                raw_dir.mkdir(parents=True, exist_ok=True)
                np.save(raw_dir / "flood_10step.npy", predictions[item_index].cpu().numpy())
                np.save(raw_dir / "gt.npy", ground_truth_raw[item_index, :ground_truth_length].cpu().numpy())
                (raw_dir / "metadata.json").write_text(json.dumps({
                    "name": name,
                    "captions": batch["captions"][item_index],
                    "text_ends": batch["text_ends"][item_index],
                    "effective_segments": [segment.__dict__ for segment in segments],
                    "ground_truth_length": ground_truth_length,
                    "prediction_length": prediction_length,
                }, indent=2))
        if (batch_index + 1) % 100 == 0:
            model.text_cache.clear()

    arrays = {category: collector.finish() for category, collector in collectors.items()}
    np.savez_compressed(
        output / f"protocol_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz",
        **{f"{category}_{name}": value for category, values in arrays.items() for name, value in values.items()},
        predicted_transition_jerk=np.stack(predicted_curves) if predicted_curves else np.empty((0, args.transition_length - 3)),
        ground_truth_transition_jerk=np.stack(ground_truth_curves) if ground_truth_curves else np.empty((0, args.transition_length - 3)),
        predicted_normalized_feature_jerk=np.stack(predicted_normalized_feature_curves) if predicted_normalized_feature_curves else np.empty((0, args.transition_length - 3)),
        ground_truth_normalized_feature_jerk=np.stack(ground_truth_normalized_feature_curves) if ground_truth_normalized_feature_curves else np.empty((0, args.transition_length - 3)),
        predicted_norm_velocity_45_jerk=np.stack(predicted_norm_velocity_45_curves) if predicted_norm_velocity_45_curves else np.empty((0, 42)),
        ground_truth_norm_velocity_45_jerk=np.stack(ground_truth_norm_velocity_45_curves) if ground_truth_norm_velocity_45_curves else np.empty((0, 42)),
        diagnostic_transition_keys=np.asarray(diagnostic_transition_keys, dtype=np.int64),
        transition_keys=np.asarray(transition_keys, dtype=np.int64),
    )
    record = {
        "model": "FloodDiffusion released checkpoint",
        "checkpoint": model_checkpoint,
        "vae_checkpoint": vae_checkpoint,
        "denoise_steps": args.denoise_steps,
        "cfg_scale": model.cfg_scale,
        "transition_length": args.transition_length,
        "total_samples": total_samples,
        "shard_samples": len(dataset),
        "sample_counts": sample_counts,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
    }
    (output / f"protocol_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json").write_text(
        json.dumps(record, indent=2)
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
