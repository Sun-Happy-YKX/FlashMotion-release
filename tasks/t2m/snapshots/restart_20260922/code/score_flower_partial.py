from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from score_babel_long import score_table
from t2m_long.babel_long_benchmark import combine_shards, transition_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = combine_shards(sorted(args.path.glob("shard_*_of_*.npz")))
    models = {
        "FlowerMotion 1-step": "1step",
        "FlowerMotion 2-step": "2step",
        "FlowerMotion 4-step": "4step",
    }
    selected = arrays["segment_length"] >= 4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = {
        "multi_all": score_table(arrays, models, "multi", selected, device, 42),
        "single_control": score_table(arrays, models, "single", selected, device, 42),
        "transitions": {
            name: transition_metrics(arrays, f"multi_{suffix}")
            for name, suffix in models.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
