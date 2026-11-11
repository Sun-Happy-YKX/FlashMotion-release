from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from t2m_long.babel_protocol import effective_segments, pj_auj, transition_jerk_curve
from t2m_long.data import MotionTextDataset, collate_motion_text
from t2m_long.evaluator import FloodT2MEvaluator
from t2m_long.meanflow import T2MMeanFlow
from t2m_long.model import OneShotT2MDecoder
from t2m_long.representation import recover_joints_from_ric


SELECTED_NAMES = {"1567_3", "2062_9", "11264_1", "7743_8", "9085_3", "12738_4"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-1", type=Path, required=True)
    parser.add_argument("--checkpoint-4", type=Path, required=True)
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_protocol_flower"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--transition-length", type=int, default=60)
    return parser.parse_args()


class EmbeddingCollector:
    def __init__(self, evaluator: FloodT2MEvaluator, batch_size: int):
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.pending: dict[str, list] = {
            "prediction": [], "ground_truth": [], "tokens": [], "keys": []
        }
        self.arrays: dict[str, list[np.ndarray]] = {
            "prediction": [], "ground_truth": [], "text": []
        }
        self.keys: list[int] = []

    def add(
        self,
        prediction: torch.Tensor,
        ground_truth: torch.Tensor,
        tokens: list[str],
        order_key: int,
    ) -> None:
        if min(len(prediction), len(ground_truth)) < 4:
            return
        self.pending["prediction"].append(prediction.detach().cpu())
        self.pending["ground_truth"].append(ground_truth.detach().cpu())
        self.pending["tokens"].append(tokens)
        self.pending["keys"].append(order_key)
        if len(self.pending["prediction"]) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending["prediction"]:
            return
        prediction_lengths = torch.tensor([len(x) for x in self.pending["prediction"]])
        ground_truth_lengths = torch.tensor([len(x) for x in self.pending["ground_truth"]])
        prediction = pad_sequence(self.pending["prediction"], batch_first=True)
        ground_truth = pad_sequence(self.pending["ground_truth"], batch_first=True)
        self.arrays["prediction"].append(
            self.evaluator._motion_embeddings(prediction, prediction_lengths).float().cpu().numpy()
        )
        self.arrays["ground_truth"].append(
            self.evaluator._motion_embeddings(ground_truth, ground_truth_lengths).float().cpu().numpy()
        )
        self.arrays["text"].append(
            self.evaluator._text_embeddings(self.pending["tokens"]).float().cpu().numpy()
        )
        self.keys.extend(self.pending["keys"])
        for value in self.pending.values():
            value.clear()

    def finish(self) -> dict[str, np.ndarray]:
        self.flush()
        return {
            name: np.concatenate(parts) for name, parts in self.arrays.items()
        } | {"keys": np.asarray(self.keys, dtype=np.int64)}


def compute_arrays(
    evaluator: FloodT2MEvaluator, arrays: dict[str, np.ndarray]
) -> dict[str, float]:
    evaluator.prediction_embeddings = [torch.from_numpy(arrays["prediction"])]
    evaluator.ground_truth_embeddings = [torch.from_numpy(arrays["ground_truth"])]
    evaluator.text_embeddings = [torch.from_numpy(arrays["text"])]
    return evaluator.compute()


def sorted_union(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    merged = {
        name: np.concatenate([part[name] for part in parts])
        for name in ("prediction", "ground_truth", "text", "keys")
    }
    order = np.argsort(merged["keys"], kind="stable")
    return {name: value[order] for name, value in merged.items()}


def load_objective(checkpoint_path: Path, data_root: Path, device: torch.device) -> T2MMeanFlow:
    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy"))
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy"))
    objective = T2MMeanFlow(OneShotT2MDecoder(), mean, std).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    objective.load_state_dict(checkpoint["model"])
    if "ema_decoder" in checkpoint:
        objective.decoder.load_state_dict(checkpoint["ema_decoder"])
    return objective.eval().requires_grad_(False)


def evaluate_model(
    label: str,
    checkpoint_path: Path,
    sampling_steps: int,
    loader: DataLoader,
    dataset: MotionTextDataset,
    evaluator: FloodT2MEvaluator,
    data_root: Path,
    output: Path,
    device: torch.device,
    seed: int,
    transition_length: int,
) -> dict:
    objective = load_objective(checkpoint_path, data_root, device)
    collectors = {
        "single": EmbeddingCollector(evaluator, 64),
        "multi": EmbeddingCollector(evaluator, 64),
    }
    sample_counts = {"single": 0, "multi": 0}
    predicted_curves: list[np.ndarray] = []
    ground_truth_curves: list[np.ndarray] = []
    transition_keys: list[int] = []
    generator = torch.Generator(device=device).manual_seed(seed + sampling_steps * 100000)
    name_to_index = {name: index for index, name in enumerate(dataset.ids)}

    for batch in tqdm(loader, desc=label):
        condition = batch["condition"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        ground_truth = batch["motion"].to(device, non_blocking=True)
        with torch.inference_mode():
            prediction = objective.sample(condition, mask, sampling_steps, generator)
            prediction_raw = prediction * objective.motion_std + objective.motion_mean
            ground_truth_raw = ground_truth * objective.motion_std + objective.motion_mean
            prediction_joints = recover_joints_from_ric(prediction_raw)
            ground_truth_joints = recover_joints_from_ric(ground_truth_raw)

        for batch_index, length in enumerate(batch["lengths"].tolist()):
            name = batch["name"][batch_index]
            global_index = name_to_index[name]
            segments = effective_segments(batch["text_ends"][batch_index], length)
            category = "multi" if len(segments) >= 2 else "single"
            sample_counts[category] += 1
            for segment_index, segment in enumerate(segments):
                collectors[category].add(
                    prediction_raw[batch_index, segment.start : segment.end],
                    ground_truth_raw[batch_index, segment.start : segment.end],
                    batch["text_tokens"][batch_index][segment.annotation_index],
                    global_index * 100 + segment_index,
                )

            if category == "multi":
                margin = transition_length // 2
                for transition_index, segment in enumerate(segments[1:]):
                    boundary = segment.start
                    if boundary >= margin and boundary + (transition_length - margin) <= length:
                        predicted_curves.append(
                            transition_jerk_curve(
                                prediction_joints[batch_index, boundary - margin : boundary + (transition_length - margin)]
                                .float().cpu().numpy()
                            )
                        )
                        ground_truth_curves.append(
                            transition_jerk_curve(
                                ground_truth_joints[batch_index, boundary - margin : boundary + (transition_length - margin)]
                                .float().cpu().numpy()
                            )
                        )
                        transition_keys.append(global_index * 100 + transition_index)

            if name in SELECTED_NAMES:
                raw_dir = output / "raw" / name
                raw_dir.mkdir(parents=True, exist_ok=True)
                np.save(raw_dir / f"{label}.npy", prediction_raw[batch_index, :length].float().cpu().numpy())
                np.save(raw_dir / "gt.npy", ground_truth_raw[batch_index, :length].float().cpu().numpy())
                (raw_dir / "metadata.json").write_text(json.dumps({
                    "name": name,
                    "captions": batch["captions"][batch_index],
                    "text_ends": batch["text_ends"][batch_index],
                    "effective_segments": [segment.__dict__ for segment in segments],
                    "length": length,
                }, indent=2))

    arrays = {category: collector.finish() for category, collector in collectors.items()}
    arrays["all"] = sorted_union(arrays["single"], arrays["multi"])
    metrics = {category: compute_arrays(evaluator, values) for category, values in arrays.items()}
    metrics["multi"]["PJ"] = pj_auj(predicted_curves)["PJ"]
    metrics["multi"]["AUJ"] = pj_auj(predicted_curves)["AUJ"]
    metrics["multi"]["transition_count"] = len(predicted_curves)
    metrics["multi"]["gt_PJ"] = pj_auj(ground_truth_curves)["PJ"]
    metrics["multi"]["gt_AUJ"] = pj_auj(ground_truth_curves)["AUJ"]
    np.savez_compressed(
        output / f"embeddings_{label}.npz",
        **{f"{category}_{name}": value for category, values in arrays.items() for name, value in values.items()},
        predicted_transition_jerk=np.stack(predicted_curves),
        ground_truth_transition_jerk=np.stack(ground_truth_curves),
        transition_keys=np.asarray(transition_keys, dtype=np.int64),
    )
    del objective
    torch.cuda.empty_cache()
    return {
        "label": label,
        "checkpoint": str(checkpoint_path.resolve()),
        "sampling_steps": sampling_steps,
        "sample_counts": sample_counts,
        "metrics": metrics,
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    data_root = flood_root / "raw_data"
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    dataset = MotionTextDataset(data_root, root / args.embedding_root, "babel", "val", 200)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0, collate_fn=collate_motion_text,
    )
    device = torch.device("cuda")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    evaluator = FloodT2MEvaluator(flood_root, device, seed=args.seed)
    records = []
    for label, checkpoint, steps in (
        ("flower_1step", args.checkpoint_1, 1),
        ("flower_4step", args.checkpoint_4, 4),
    ):
        records.append(evaluate_model(
            label, (root / checkpoint).resolve(), steps, loader, dataset, evaluator,
            data_root, output, device, args.seed, args.transition_length,
        ))
    record = {
        "split": "BABEL val_processed, 200-frame horizon",
        "protocols": {
            "single": "one effective text segment; metric item is that segment",
            "multi": "at least two effective text segments; every segment is a metric item",
            "all": "union of single and multi metric items",
            "PJ_AUJ": f"{args.transition_length}-frame windows centered on valid multi-action boundaries",
        },
        "models": records,
    }
    (output / "metrics.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
