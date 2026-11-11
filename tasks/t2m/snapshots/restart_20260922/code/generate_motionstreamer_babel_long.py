from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from generate_motius_babel_long import existing_episode_ids, read_episodes
from t2m_long.motion_archive import save_motion_episode, validate_motion_archive
from t2m_long.representation import recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate MotionStreamer on BABEL-long-v2.")
    parser.add_argument("--model", default="ZeyuLing/Motius-MotionStreamer-HumanML272")
    parser.add_argument("--data-root", type=Path, default=Path("third_party/FloodDiffusion/raw_data"))
    parser.add_argument("--flow-root", type=Path, default=Path("third_party/FlowMDM"))
    parser.add_argument("--output-root", type=Path, default=Path("eval/babel_long_v2/motions"))
    parser.add_argument("--cache-dir", type=Path, default=Path("third_party/hf_cache"))
    parser.add_argument("--min-episode-frames", type=int, default=201)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance-param", type=float, default=4.0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    return parser.parse_args()


def setup_hml_converter(reference_joints: torch.Tensor, flow_root: Path):
    sys.path.insert(0, str(flow_root))
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.scripts import motion_process
    from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets

    motion_process.l_idx1, motion_process.l_idx2 = 5, 8
    motion_process.fid_r, motion_process.fid_l = [8, 11], [7, 10]
    motion_process.face_joint_indx = [2, 1, 17, 16]
    motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    motion_process.kinematic_chain = t2m_kinematic_chain
    skeleton = Skeleton(motion_process.n_raw_offsets, motion_process.kinematic_chain, "cpu")
    motion_process.tgt_offsets = skeleton.get_offsets_joints(reference_joints.cpu()[0])
    return motion_process


def resample_joints(joints: np.ndarray, target: int) -> np.ndarray:
    value = torch.as_tensor(joints, dtype=torch.float32).reshape(len(joints), -1).T[None]
    value = F.interpolate(value, size=target, mode="linear", align_corners=True)
    return value[0].T.reshape(target, 22, 3).numpy()


def joints_to_hml(joints: np.ndarray, converter) -> np.ndarray:
    padded = np.concatenate([joints, joints[-1:]], axis=0)
    features = converter.process_file(padded, 0.002)[0]
    if features.shape != (len(joints), 263):
        raise RuntimeError(f"HML encoding returned {features.shape}, expected ({len(joints)}, 263)")
    return features.astype(np.float32)


def lengths_at_30fps(episode) -> list[int]:
    boundaries = [0]
    boundaries.extend(int(segment.end * 1.5 + 0.5) for segment in episode.segments)
    total = int(episode.length * 1.5 + 0.5)
    boundaries[-1] = total
    lengths = [boundaries[index + 1] - boundaries[index] for index in range(len(episode.segments))]
    if min(lengths) < 1 or sum(lengths) != total:
        raise RuntimeError(f"invalid 30 fps lengths {lengths}")
    return lengths


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    root = Path(__file__).resolve().parent
    data_root = (root / args.data_root).resolve()
    flow_root = (root / args.flow_root).resolve()
    output_root = (root / args.output_root).resolve()
    cache_dir = (root / args.cache_dir).resolve()
    method_root = output_root / "motionstreamer"
    method_root.mkdir(parents=True, exist_ok=True)
    episodes = read_episodes(data_root, args.min_episode_frames)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    already = set() if args.overwrite else existing_episode_ids(method_root)
    indexed_episodes = list(enumerate(episodes))
    pending = [
        (global_index, episode)
        for global_index, episode in indexed_episodes
        if global_index % args.num_shards == args.shard_index
        and episode.episode_id not in already
    ]

    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy")).float()
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy")).float()
    first_gt = torch.from_numpy(
        np.load(data_root / "BABEL_streamed/motions" / f"{episodes[0].episode_id}.npy")
    ).float()
    converter = setup_hml_converter(recover_joints_from_ric(first_gt[None])[0], flow_root)

    from motius import Pipeline
    from motius.motion import motion272_to_joints

    pipeline = Pipeline.from_pretrained(
        args.model,
        cache_dir=str(cache_dir),
        bundle_kwargs={"device": args.device},
        device=args.device,
    )
    for index, (global_index, episode) in enumerate(pending):
        # Preserve the original unsharded run's seed for bitwise-compatible resume.
        seed = args.seed + global_index
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        captions = [episode.captions[segment.annotation_index] for segment in episode.segments]
        motion272 = pipeline.infer_sequential_text_to_motion(
            [captions],
            [lengths_at_30fps(episode)],
            guidance_param=args.guidance_param,
            temperature=1.0,
            seed=args.seed,
            shard_index=0,
            sample_offset=global_index,
            exact_lengths=True,
        )[0]
        joints30 = motion272_to_joints(motion272)
        joints20 = resample_joints(joints30, episode.length)
        motion263 = joints_to_hml(joints20, converter)
        canonical_joints = recover_joints_from_ric(torch.from_numpy(motion263)[None])[0]
        save_motion_episode(
            output_root,
            method="motionstreamer",
            episode_id=episode.episode_id,
            motion_263=motion263,
            joints_y_up=canonical_joints,
            captions=captions,
            segment_starts=[segment.start for segment in episode.segments],
            segment_ends=[segment.end for segment in episode.segments],
            seed=args.seed,
            metadata={
                "artifact": args.model,
                "checkpoint_origin": "MotionStreamer official HumanML272 checkpoint repackaged by Motius",
                "adapter": "native continuous sequential generation at 30 fps; exact segment lengths; joints linearly resampled to 20 fps and re-encoded as HML263",
                "native_representation": "MotionStreamer 272D at 30 fps",
                "guidance_param": args.guidance_param,
                "benchmark": "babel_long_v2_gt10s_no_cap",
            },
        )
        print(
            f"[motionstreamer shard {args.shard_index}] episodes {index + 1}/{len(pending)} {episode.episode_id}",
            flush=True,
        )

    summary = {
        "model": "motionstreamer",
        "artifact": args.model,
        "seed": args.seed,
        "expected_episodes": len(episodes),
        "previously_complete": len(already & {episode.episode_id for episode in episodes}),
        "newly_generated": len(pending),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "archive_validation": validate_motion_archive(method_root),
    }
    (method_root / f"run_summary_shard_{args.shard_index:02d}_of_{args.num_shards:02d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
