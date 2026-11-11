from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from t2m_long.babel_long_benchmark import combine_shards, score_embedding_subset, transition_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    arrays = combine_shards(sorted(args.input.glob("shard_*_of_*.npz")))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    valid = arrays["segment_length"] >= 4
    masks = {
        "multi_gt_10s": valid,
        "multi_10_20s": valid & (arrays["episode_length"] <= 400),
        "multi_gt_20s": valid & (arrays["episode_length"] > 400),
        "multi_min_segment_20f": arrays["segment_length"] >= 20,
        "multi_min_segment_40f": arrays["segment_length"] >= 40,
    }
    result = {
        name: score_embedding_subset(arrays, "prediction__multi_dart", selected, device, seed=42)
        for name, selected in masks.items()
    }
    result["transitions"] = transition_metrics(arrays, "multi_dart")
    output = args.output or args.input / "metrics.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(output)


if __name__ == "__main__":
    main()
