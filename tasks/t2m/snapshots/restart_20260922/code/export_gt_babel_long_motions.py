from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from t2m_long.babel_protocol import effective_segments
from t2m_long.data import MotionTextDataset
from t2m_long.motion_archive import save_motion_episode, validate_motion_archive
from t2m_long.representation import recover_joints_from_ric


def main() -> None:
    parser = argparse.ArgumentParser(description="Save the BABEL-long-v2 GT visualization archive.")
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output-root", type=Path, default=Path("eval/babel_long_v2/motions"))
    parser.add_argument("--min-episode-frames", type=int, default=201)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    output_root = (root / args.output_root).resolve()
    dataset = MotionTextDataset(
        flood_root / "raw_data", root / args.embedding_root, "babel", "val", None
    )
    mean = torch.from_numpy(dataset.mean).float()
    std = torch.from_numpy(dataset.std).float()
    count = 0
    for index, name in enumerate(dataset.ids):
        length = int(np.load(dataset.motion_dir / f"{name}.npy", mmap_mode="r").shape[0])
        if length < args.min_episode_frames:
            continue
        item = dataset[index]
        segments = effective_segments(item["text_ends"], item["length"])
        if len(segments) < 2:
            continue
        motion = item["motion"] * std + mean
        joints = recover_joints_from_ric(motion[None])[0]
        save_motion_episode(
            output_root,
            method="gt",
            episode_id=name,
            motion_263=motion,
            joints_y_up=joints,
            captions=[item["captions"][segment.annotation_index] for segment in segments],
            segment_starts=[segment.start for segment in segments],
            segment_ends=[segment.end for segment in segments],
            seed=0,
            metadata={
                "source": "FloodDiffusion-released processed BABEL validation motion",
                "benchmark": "babel_long_v2_gt10s_no_cap",
            },
        )
        count += 1
    print(validate_motion_archive(output_root / "gt"))
    if count != 311:
        raise RuntimeError(f"expected 311 GT episodes, saved {count}")


if __name__ == "__main__":
    main()
