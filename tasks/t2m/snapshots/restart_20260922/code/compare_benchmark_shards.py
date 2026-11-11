from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from t2m_long.babel_long_benchmark import combine_shards


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    first = combine_shards(sorted(args.first.glob("shard_*_of_*.npz")))
    second = combine_shards(sorted(args.second.glob("shard_*_of_*.npz")))
    for key in ("ground_truth", "text"):
        delta = np.abs(first[key].astype(np.float64) - second[key].astype(np.float64))
        print(
            f"{key}: shape={delta.shape} max={delta.max():.9g} "
            f"mean={delta.mean():.9g} p99={np.quantile(delta, 0.99):.9g} "
            f"allclose2e-4={np.allclose(first[key], second[key], atol=2e-4, rtol=2e-4)}"
        )


if __name__ == "__main__":
    main()
