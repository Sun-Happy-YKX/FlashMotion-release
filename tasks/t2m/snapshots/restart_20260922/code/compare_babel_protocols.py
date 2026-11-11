from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t2m_long.babel_protocol import pj_auj
from t2m_long.evaluator import FloodT2MEvaluator


FIELDS = ("prediction", "ground_truth", "text", "keys")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flower-root", type=Path, default=Path("eval/babel_protocol_flower"))
    parser.add_argument("--flood-root", type=Path, default=Path("eval/babel_protocol_flood"))
    parser.add_argument("--flood-code", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_protocol_comparison.json"))
    parser.add_argument("--transition-length", type=int, default=30)
    return parser.parse_args()


def flower_category(npz, category: str) -> dict[str, np.ndarray]:
    return {name: npz[f"{category}_{name}"] for name in FIELDS}


def flood_category(shards, category: str) -> dict[str, np.ndarray]:
    categories = ("single", "multi") if category == "all" else (category,)
    values = {
        name: np.concatenate([
            shard[f"{source_category}_{name}"]
            for shard in shards
            for source_category in categories
        ])
        for name in FIELDS
    }
    order = np.argsort(values["keys"], kind="stable")
    return {name: value[order] for name, value in values.items()}


def select(values: dict[str, np.ndarray], common_keys: np.ndarray) -> dict[str, np.ndarray]:
    positions = {int(key): index for index, key in enumerate(values["keys"])}
    indices = np.asarray([positions[int(key)] for key in common_keys])
    return {name: value[indices] for name, value in values.items()}


def compute(evaluator, values, ground_truth: bool = False):
    prediction = values["ground_truth"] if ground_truth else values["prediction"]
    evaluator.prediction_embeddings = [torch.from_numpy(prediction)]
    evaluator.ground_truth_embeddings = [torch.from_numpy(values["ground_truth"])]
    evaluator.text_embeddings = [torch.from_numpy(values["text"])]
    return evaluator.compute()


def jerk_lookup(keys: np.ndarray, curves: np.ndarray) -> dict[int, np.ndarray]:
    return {int(key): curve for key, curve in zip(keys, curves)}


def center_crop_jerk(curves: np.ndarray, transition_length: int) -> np.ndarray:
    """Crop cached jerk curves to a centered transition window.

    A window of N joint frames has N-3 third-difference jerk samples.  This
    keeps the action boundary centered when consuming older 60-frame caches.
    """

    target = transition_length - 3
    if target < 1:
        raise ValueError("transition length must be at least 4 frames")
    if curves.shape[1] < target:
        raise ValueError(
            f"cached jerk curve has {curves.shape[1]} samples, need {target}"
        )
    offset = (curves.shape[1] - target) // 2
    return curves[:, offset : offset + target]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flower_root = (root / args.flower_root).resolve()
    flood_root = (root / args.flood_root).resolve()
    flower = {
        "FlowerDance++ 1-step": np.load(flower_root / "embeddings_flower_1step.npz"),
        "FlowerDance++ 4-step": np.load(flower_root / "embeddings_flower_4step.npz"),
    }
    flood_shards = [np.load(path) for path in sorted(flood_root.glob("protocol_shard_*_of_*.npz"))]
    if not flood_shards:
        raise RuntimeError(f"No Flood shards found in {flood_root}")
    evaluator = FloodT2MEvaluator((root / args.flood_code).resolve(), torch.device("cuda"), 42)
    result = {"Ground truth": {}, "FlowerDance++ 1-step": {},
              "FlowerDance++ 4-step": {}, "FloodDiffusion 10-step": {}}

    selected_values = {}
    for category in ("single", "multi", "all"):
        values = {
            name: flower_category(npz, category) for name, npz in flower.items()
        }
        values["FloodDiffusion 10-step"] = flood_category(flood_shards, category)
        common_keys = values["FloodDiffusion 10-step"]["keys"]
        for other in values.values():
            common_keys = np.intersect1d(common_keys, other["keys"], assume_unique=True)
        common_keys.sort()
        selected_values[category] = {
            name: select(per_model, common_keys) for name, per_model in values.items()
        }
        gt_values = selected_values[category]["FlowerDance++ 1-step"]
        result["Ground truth"][category] = compute(evaluator, gt_values, ground_truth=True)
        for name, per_model in selected_values[category].items():
            result[name][category] = compute(evaluator, per_model)

    flower_jerk = {
        name: jerk_lookup(
            npz["transition_keys"],
            center_crop_jerk(npz["predicted_transition_jerk"], args.transition_length),
        )
        for name, npz in flower.items()
    }
    gt_jerk = jerk_lookup(
        flower["FlowerDance++ 1-step"]["transition_keys"],
        center_crop_jerk(
            flower["FlowerDance++ 1-step"]["ground_truth_transition_jerk"],
            args.transition_length,
        ),
    )
    flood_keys = np.concatenate([shard["transition_keys"] for shard in flood_shards])
    flood_curves = center_crop_jerk(
        np.concatenate([shard["predicted_transition_jerk"] for shard in flood_shards]),
        args.transition_length,
    )
    flood_jerk = jerk_lookup(flood_keys, flood_curves)
    common_transition_keys = sorted(
        set(flood_jerk).intersection(gt_jerk, *[set(values) for values in flower_jerk.values()])
    )
    jerk_sets = {
        "Ground truth": [gt_jerk[key] for key in common_transition_keys],
        **{name: [values[key] for key in common_transition_keys]
           for name, values in flower_jerk.items()},
        "FloodDiffusion 10-step": [flood_jerk[key] for key in common_transition_keys],
    }
    for name, curves in jerk_sets.items():
        result[name]["multi"].update(pj_auj(curves))

    record = {
        "split": "BABEL val_processed",
        "horizon_frames": 200,
        "transition_window_frames": args.transition_length,
        "common_segment_counts": {
            category: int(result["Ground truth"][category]["samples"])
            for category in ("single", "multi", "all")
        },
        "common_transition_count": len(common_transition_keys),
        "metrics": result,
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
