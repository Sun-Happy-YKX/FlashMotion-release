from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from t2m_long.babel_long_benchmark import (
    _activation_statistics,
    _episode_weights,
    combine_shards,
)
from t2m_long.evaluator import stable_frechet_distance


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("--prediction", default="prediction__multi_1step")
    args = parser.parse_args()
    arrays = combine_shards(sorted(args.path.glob("shard_*_of_*.npz")))
    prediction = arrays[args.prediction]
    ground_truth = arrays["ground_truth"]
    weights = _episode_weights(arrays["episode_id"])
    print(f"weights_sum={weights.sum()} min={weights.min()} max={weights.max()}")
    pred_mean, pred_cov = _activation_statistics(prediction)
    gt_mean, gt_cov = _activation_statistics(ground_truth)
    print("segment", stable_frechet_distance(gt_mean, gt_cov, pred_mean, pred_cov))
    pred_mean, pred_cov = _activation_statistics(prediction, weights)
    gt_mean, gt_cov = _activation_statistics(ground_truth, weights)
    print("weighted", stable_frechet_distance(gt_mean, gt_cov, pred_mean, pred_cov))


if __name__ == "__main__":
    main()
