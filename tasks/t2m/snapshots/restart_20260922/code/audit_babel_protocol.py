from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t2m_long.evaluator import stable_frechet_distance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--flower-dir", type=Path, default=Path("eval/babel_protocol_flower"))
    parser.add_argument("--flood-dir", type=Path, default=Path("eval/babel_protocol_flood"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_protocol_audit.json"))
    parser.add_argument("--max-length", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_annotations(path: Path) -> list[tuple[str, list[str], float, float]]:
    result = []
    for line in path.read_text().splitlines():
        caption, tokens, start, end = line.split("#")
        result.append((caption, tokens.split(), float(start), float(end)))
    return result


def effective_segment_lengths(annotations, length: int) -> list[int]:
    ends = [min(length, int(item[3] * 20.0 + 0.5)) for item in annotations]
    previous = 0
    segments: list[tuple[int, int, int]] = []
    for annotation_index, source_end in enumerate(ends):
        end = max(previous, min(length, source_end))
        if end > previous:
            segments.append((annotation_index, previous, end))
        previous = end
    if previous < length:
        last_index = len(ends) - 1
        if segments and segments[-1][0] == last_index:
            index, start, _ = segments[-1]
            segments[-1] = (index, start, length)
        else:
            segments.append((last_index, previous, length))
    return [end - start for _, start, end in segments]


def load_arrays(paths: list[Path]) -> dict[str, np.ndarray]:
    fields = ("prediction", "ground_truth", "text", "keys")
    pieces = {field: [] for field in fields}
    for path in paths:
        with np.load(path) as archive:
            for category in ("single", "multi"):
                for field in fields:
                    pieces[field].append(archive[f"{category}_{field}"])
    merged = {field: np.concatenate(values) for field, values in pieces.items()}
    order = np.argsort(merged["keys"], kind="stable")
    return {field: values[order] for field, values in merged.items()}


def select(arrays: dict[str, np.ndarray], keys: set[int]) -> dict[str, np.ndarray]:
    mask = np.fromiter((int(key) in keys for key in arrays["keys"]), dtype=bool)
    return {field: values[mask] for field, values in arrays.items()}


def metric_set(arrays: dict[str, np.ndarray], seed: int) -> dict[str, float]:
    predictions = torch.from_numpy(arrays["prediction"])
    ground_truth = torch.from_numpy(arrays["ground_truth"])
    texts = torch.from_numpy(arrays["text"])
    count = len(predictions)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(count, generator=generator)
    predictions = predictions[order]
    ground_truth = ground_truth[order]
    texts = texts[order]
    usable = count // 32 * 32
    result: dict[str, float] = {"samples": int(count), "retrieval_usable": int(usable)}
    for label, motions in (("", predictions), ("gt_", ground_truth)):
        matching_sum = 0.0
        top_k_sum = torch.zeros(3)
        for start in range(0, usable, 32):
            distances = torch.cdist(texts[start : start + 32], motions[start : start + 32]).nan_to_num()
            matching_sum += float(torch.trace(distances))
            ranking = torch.argsort(distances, dim=1)
            correct = ranking == torch.arange(32)[:, None]
            cumulative = torch.cumsum(correct, dim=1).clamp(max=1)
            top_k_sum += cumulative[:, :3].sum(0)
        result[f"{label}Matching_score"] = matching_sum / usable
        for k in range(3):
            result[f"{label}R_precision_top_{k + 1}"] = float(top_k_sum[k] / usable)

    prediction_np = predictions.numpy()
    ground_truth_np = ground_truth.numpy()
    pred_mean = prediction_np.mean(axis=0)
    gt_mean = ground_truth_np.mean(axis=0)
    pred_cov = np.cov(prediction_np, rowvar=False)
    gt_cov = np.cov(ground_truth_np, rowvar=False)
    result["FID"] = stable_frechet_distance(gt_mean, gt_cov, pred_mean, pred_cov)

    rng = np.random.RandomState(seed)
    times = min(300, count - 1)
    first = rng.choice(count, times, replace=False)
    second = rng.choice(count, times, replace=False)
    result["Diversity"] = float(np.linalg.norm(prediction_np[first] - prediction_np[second], axis=1).mean())
    rng = np.random.RandomState(seed)
    first = rng.choice(count, times, replace=False)
    second = rng.choice(count, times, replace=False)
    result["gt_Diversity"] = float(np.linalg.norm(ground_truth_np[first] - ground_truth_np[second], axis=1).mean())
    return result


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    babel_root = flood_root / "raw_data/BABEL_streamed"
    ids = [line.strip() for line in (babel_root / "val_processed.txt").read_text().splitlines() if line.strip()]

    metadata: dict[int, dict] = {}
    raw_lengths = []
    annotation_counts = []
    visible_segment_counts = []
    transition_full_30 = 0
    transition_full_30_official = 0
    effective_single_official = 0
    effective_single_cropped_long = 0
    last_end_deltas = []
    for global_index, name in enumerate(ids):
        raw_length = int(np.load(babel_root / "motions" / f"{name}.npy", mmap_mode="r").shape[0])
        annotations = read_annotations(babel_root / "texts" / f"{name}.txt")
        cropped_length = min(raw_length, args.max_length)
        lengths = effective_segment_lengths(annotations, cropped_length)
        last_end_frame = int(annotations[-1][3] * 20.0 + 0.5)
        last_end_deltas.append(last_end_frame - raw_length)
        raw_lengths.append(raw_length)
        annotation_counts.append(len(annotations))
        visible_segment_counts.append(len(lengths))
        if len(lengths) == 1:
            if raw_length <= args.max_length:
                effective_single_official += 1
            else:
                effective_single_cropped_long += 1
        previous = 0
        ends = [min(cropped_length, int(item[3] * 20.0 + 0.5)) for item in annotations]
        visible_boundaries = []
        for end in ends:
            end = max(previous, min(cropped_length, end))
            if end > previous:
                if previous > 0:
                    visible_boundaries.append(previous)
                previous = end
        valid_transitions = sum(
            1 for boundary in visible_boundaries
            if boundary >= 15 and boundary + 15 <= cropped_length
        )
        transition_full_30 += valid_transitions
        if raw_length <= args.max_length:
            transition_full_30_official += valid_transitions
        for segment_index, segment_length in enumerate(lengths):
            metadata[global_index * 100 + segment_index] = {
                "name": name,
                "length": int(segment_length),
                "raw_length": raw_length,
                "official_max_length_eligible": raw_length <= args.max_length,
            }

    flower_1 = load_arrays([root / args.flower_dir / "embeddings_flower_1step.npz"])
    flower_4 = load_arrays([root / args.flower_dir / "embeddings_flower_4step.npz"])
    flood_paths = sorted((root / args.flood_dir).glob("protocol_shard_*_of_*.npz"))
    flood = load_arrays(flood_paths)
    common_keys = set(map(int, flower_1["keys"])) & set(map(int, flower_4["keys"])) & set(map(int, flood["keys"]))

    cohorts = {}
    for minimum in (4, 20, 40):
        for official_filter in (False, True):
            label = f"min_{minimum}_" + ("official_le_200" if official_filter else "cropped_all")
            keys = {
                key for key in common_keys
                if key in metadata
                and metadata[key]["length"] >= minimum
                and (not official_filter or metadata[key]["official_max_length_eligible"])
            }
            cohorts[label] = {
                "keys": len(keys),
                "flower_1step": metric_set(select(flower_1, keys), args.seed),
                "flower_4step": metric_set(select(flower_4, keys), args.seed),
                "flood_10step_cfg5": metric_set(select(flood, keys), args.seed),
            }

    raw_lengths_np = np.asarray(raw_lengths)
    last_end_deltas_np = np.asarray(last_end_deltas)
    result = {
        "dataset": {
            "val_ids": len(ids),
            "raw_length_min": int(raw_lengths_np.min()),
            "raw_length_max": int(raw_lengths_np.max()),
            "raw_length_mean": float(raw_lengths_np.mean()),
            "raw_length_gt_200": int((raw_lengths_np > args.max_length).sum()),
            "raw_length_le_200": int((raw_lengths_np <= args.max_length).sum()),
            "annotation_count_histogram": {str(value): annotation_counts.count(value) for value in sorted(set(annotation_counts))},
            "visible_segment_count_histogram_after_crop": {str(value): visible_segment_counts.count(value) for value in sorted(set(visible_segment_counts))},
            "effective_single_official_le_200": effective_single_official,
            "effective_single_from_truncating_gt_200": effective_single_cropped_long,
            "full_centered_30_transitions_after_crop": int(transition_full_30),
            "full_centered_30_transitions_official_le_200": int(transition_full_30_official),
            "last_annotation_end_minus_raw_motion_length_frames": {
                "min": int(last_end_deltas_np.min()),
                "max": int(last_end_deltas_np.max()),
                "mean": float(last_end_deltas_np.mean()),
                "within_4_frames": int((np.abs(last_end_deltas_np) <= 4).sum()),
                "earlier_by_more_than_4_frames": int((last_end_deltas_np < -4).sum()),
                "later_by_more_than_4_frames": int((last_end_deltas_np > 4).sum()),
            },
            "segment_length_histogram": {
                "lt_4": sum(item["length"] < 4 for item in metadata.values()),
                "4_to_19": sum(4 <= item["length"] < 20 for item in metadata.values()),
                "20_to_39": sum(20 <= item["length"] < 40 for item in metadata.values()),
                "ge_40": sum(item["length"] >= 40 for item in metadata.values()),
            },
        },
        "embedding_counts": {
            "flower_1step": len(flower_1["keys"]),
            "flower_4step": len(flower_4["keys"]),
            "flood": len(flood["keys"]),
            "common": len(common_keys),
        },
        "cohorts": cohorts,
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
