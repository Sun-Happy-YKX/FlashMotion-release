from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flower-root", type=Path, default=Path("eval/babel_protocol_flower"))
    parser.add_argument("--flood-root", type=Path, default=Path("eval/babel_protocol_flood"))
    parser.add_argument("--transition-length", type=int, default=30)
    return parser.parse_args()


def center_crop(curves: np.ndarray, transition_length: int) -> np.ndarray:
    target = transition_length - 3
    offset = (curves.shape[1] - target) // 2
    return curves[:, offset : offset + target]


def lookup(keys: np.ndarray, curves: np.ndarray) -> dict[int, np.ndarray]:
    return {int(key): curve for key, curve in zip(keys, curves)}


def score(curves: list[np.ndarray], constant: float) -> dict[str, float]:
    mean_curve = np.stack(curves).mean(axis=0)
    return {
        "transitions": len(curves),
        "PJ": float(mean_curve.max()),
        "AUJ": float(np.abs(mean_curve - constant).sum()),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flower_root = root / args.flower_root
    flood_root = root / args.flood_root
    flower_archives = {
        "FlowerMotion 1-step": np.load(flower_root / "embeddings_flower_1step.npz"),
        "FlowerMotion 4-step": np.load(flower_root / "embeddings_flower_4step.npz"),
    }
    flower_curves = {
        label: lookup(
            archive["transition_keys"],
            center_crop(archive["predicted_transition_jerk"], args.transition_length),
        )
        for label, archive in flower_archives.items()
    }
    first = flower_archives["FlowerMotion 1-step"]
    ground_truth = lookup(
        first["transition_keys"],
        center_crop(first["ground_truth_transition_jerk"], args.transition_length),
    )
    flood_shards = [np.load(path) for path in sorted(flood_root.glob("protocol_shard_*_of_*.npz"))]
    flood = lookup(
        np.concatenate([archive["transition_keys"] for archive in flood_shards]),
        center_crop(
            np.concatenate([archive["predicted_transition_jerk"] for archive in flood_shards]),
            args.transition_length,
        ),
    )
    common = sorted(set(ground_truth) & set(flood) & set.intersection(*(set(x) for x in flower_curves.values())))
    sets = {
        "Ground truth": [ground_truth[key] for key in common],
        **{label: [values[key] for key in common] for label, values in flower_curves.items()},
        "FloodDiffusion 10-step": [flood[key] for key in common],
    }
    result = {
        "split": "BABEL public val, current cropped cohort",
        "transition_length": args.transition_length,
        "common_transitions": len(common),
        "wrong_humanml_constant_0.033031363": {
            label: score(curves, 0.033031363) for label, curves in sets.items()
        },
        "correct_babel_constant_0.016383045": {
            label: score(curves, 0.016383045) for label, curves in sets.items()
        },
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
