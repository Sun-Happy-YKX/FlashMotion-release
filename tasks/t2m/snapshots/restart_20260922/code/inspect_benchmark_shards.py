from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    paths = sorted(args.path.glob("shard_*_of_*.npz"))
    print(f"shards={len(paths)}")
    total_segments = 0
    total_transitions = 0
    schemas: set[tuple[str, ...]] = set()
    episode_ids = []
    episode_lengths = []
    segment_lengths = []
    for path in paths:
        with np.load(path) as shard:
            schemas.add(tuple(sorted(shard.files)))
            total_segments += len(shard["segment_key"])
            total_transitions += len(shard["transition__ground_truth"])
            episode_ids.append(shard["episode_id"])
            episode_lengths.append(shard["episode_length"])
            segment_lengths.append(shard["segment_length"])
    print(f"segments={total_segments}")
    print(f"transitions={total_transitions}")
    print(f"schemas={len(schemas)}")
    if paths:
        print("keys=" + ",".join(schemas.pop()))
        ids = np.concatenate(episode_ids)
        episode_length = np.concatenate(episode_lengths)
        segment_length = np.concatenate(segment_lengths)
        print(f"episodes={len(np.unique(ids))}")
        print(f"segments_in_episodes_over_10s={int((episode_length > 200).sum())}")
        print(f"segments_in_episodes_over_20s={int((episode_length > 400).sum())}")
        print(
            "episode_length_min_median_max="
            f"{episode_length.min()},{np.median(episode_length)},{episode_length.max()}"
        )
        print(
            "segment_length_min_median_max="
            f"{segment_length.min()},{np.median(segment_length)},{segment_length.max()}"
        )


if __name__ == "__main__":
    main()
