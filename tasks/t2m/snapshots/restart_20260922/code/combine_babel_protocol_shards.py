from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t2m_long.babel_protocol import pj_auj
from t2m_long.evaluator import FloodT2MEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def gather(shards: list, category: str) -> dict[str, np.ndarray]:
    result = {
        name: np.concatenate([shard[f"{category}_{name}"] for shard in shards])
        for name in ("prediction", "ground_truth", "text", "keys")
    }
    order = np.argsort(result["keys"], kind="stable")
    return {name: value[order] for name, value in result.items()}


def union(*parts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {
        name: np.concatenate([part[name] for part in parts])
        for name in ("prediction", "ground_truth", "text", "keys")
    }
    order = np.argsort(result["keys"], kind="stable")
    return {name: value[order] for name, value in result.items()}


def compute(evaluator: FloodT2MEvaluator, values: dict[str, np.ndarray]) -> dict[str, float]:
    evaluator.prediction_embeddings = [torch.from_numpy(values["prediction"])]
    evaluator.ground_truth_embeddings = [torch.from_numpy(values["ground_truth"])]
    evaluator.text_embeddings = [torch.from_numpy(values["text"])]
    return evaluator.compute()


def feature_jerk_summary(curves: np.ndarray) -> dict[str, float]:
    mean_curve = curves.mean(axis=0)
    return {
        "PJ": float(mean_curve.max()),
        "AUJ_minus_flowmdm_javg": float(np.abs(mean_curve - 0.033031363).sum()),
        "AUJ_raw": float(mean_curve.sum()),
    }


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    input_dir = (root / args.input).resolve()
    paths = sorted(input_dir.glob("protocol_shard_*_of_*.npz"))
    if not paths:
        raise RuntimeError(f"No shards in {input_dir}")
    shards = [np.load(path) for path in paths]
    single = gather(shards, "single")
    multi = gather(shards, "multi")
    all_values = union(single, multi)
    evaluator = FloodT2MEvaluator((root / args.flood_root).resolve(), torch.device("cuda"), 42)
    metrics = {
        "single": compute(evaluator, single),
        "multi": compute(evaluator, multi),
        "all": compute(evaluator, all_values),
    }
    predicted_curves = np.concatenate([shard["predicted_transition_jerk"] for shard in shards])
    ground_truth_curves = np.concatenate([shard["ground_truth_transition_jerk"] for shard in shards])
    metrics["multi"].update({
        "PJ": pj_auj(list(predicted_curves))["PJ"],
        "AUJ": pj_auj(list(predicted_curves))["AUJ"],
        "transition_count": len(predicted_curves),
        "gt_PJ": pj_auj(list(ground_truth_curves))["PJ"],
        "gt_AUJ": pj_auj(list(ground_truth_curves))["AUJ"],
    })
    if "predicted_normalized_feature_jerk" in shards[0]:
        predicted_feature_curves = np.concatenate([
            shard["predicted_normalized_feature_jerk"] for shard in shards
        ])
        ground_truth_feature_curves = np.concatenate([
            shard["ground_truth_normalized_feature_jerk"] for shard in shards
        ])
        metrics["multi"]["normalized_263d_feature_diagnostic"] = {
            "prediction": feature_jerk_summary(predicted_feature_curves),
            "ground_truth": feature_jerk_summary(ground_truth_feature_curves),
        }
    if "predicted_norm_velocity_45_jerk" in shards[0]:
        predicted_velocity_curves = np.concatenate([
            shard["predicted_norm_velocity_45_jerk"] for shard in shards
        ])
        ground_truth_velocity_curves = np.concatenate([
            shard["ground_truth_norm_velocity_45_jerk"] for shard in shards
        ])
        metrics["multi"]["normalized_local_velocity_45f_diagnostic"] = {
            "prediction": feature_jerk_summary(predicted_velocity_curves),
            "ground_truth": feature_jerk_summary(ground_truth_velocity_curves),
        }
    record = {
        "model": "FloodDiffusion released checkpoint, 10-step",
        "shards": len(paths),
        "metrics": metrics,
    }
    output = (root / args.output).resolve() if args.output else input_dir / "metrics.json"
    output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
