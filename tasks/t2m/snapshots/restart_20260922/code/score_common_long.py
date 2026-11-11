from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t2m_long.babel_long_benchmark import combine_shards, score_embedding_subset, transition_metrics


METHODS = {
    "FlowerMotion 1-step": ("flower_ckpt260k", "multi_1step"),
    "FlowerMotion 2-step": ("flower_ckpt260k", "multi_2step"),
    "FlowerMotion 4-step": ("flower_ckpt260k", "multi_4step"),
    "FloodDiffusion 10-step": ("flood_cfg6", "multi_10step"),
    "FlowMDM 1000-step": ("flowmdm_gt10s", "multi_flowmdm"),
    "DART 10-step/primitive": ("dart_smplx_yup_gt10s", "multi_dart"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    loaded = {
        directory: combine_shards(sorted((args.root / directory).glob("shard_*_of_*.npz")))
        for directory, _ in set(METHODS.values())
    }
    dart = loaded["dart_smplx_yup_gt10s"]
    dart_finite = np.isfinite(dart["prediction__multi_dart"]).all(axis=1)
    failed_episodes = sorted(set(dart["episode_id"][~dart_finite].tolist()))
    common_episodes = sorted(set(dart["episode_id"].tolist()) - set(failed_episodes))
    common_episode_set = set(common_episodes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result: dict[str, object] = {
        "protocol": {
            "cohort": "BABEL validation episodes longer than 10 seconds",
            "original_episodes": len(set(dart["episode_id"].tolist())),
            "common_valid_episodes": len(common_episodes),
            "failed_episodes": failed_episodes,
            "DART_failure_rate": len(failed_episodes) / len(set(dart["episode_id"].tolist())),
        },
        "methods": {},
    }
    for display_name, (directory, prediction_name) in METHODS.items():
        arrays = loaded[directory]
        selected = (
            (arrays["episode_length"] > 200)
            & (arrays["segment_length"] >= 4)
            & np.asarray([episode in common_episode_set for episode in arrays["episode_id"]])
            & np.isfinite(arrays[f"prediction__{prediction_name}"]).all(axis=1)
        )
        metrics = score_embedding_subset(
            arrays, f"prediction__{prediction_name}", selected, device, seed=42
        )
        transition_selected = np.asarray([
            episode in common_episode_set
            for episode in arrays["transition_episode_id"]
        ]) & np.isfinite(arrays[f"transition__{prediction_name}"]).all(axis=1)
        transition_arrays = {
            key: value[transition_selected]
            if key.startswith("transition") and len(value) == len(transition_selected)
            else value
            for key, value in arrays.items()
        }
        metrics["transition"] = transition_metrics(transition_arrays, prediction_name)
        result["methods"][display_name] = metrics
    output = args.output or args.root / "common_gt10s_metrics.json"
    output.write_text(json.dumps(result, indent=2))
    print(output)


if __name__ == "__main__":
    main()
