from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from generate_motius_babel_long import existing_episode_ids, read_episodes
from t2m_long.motion_archive import save_motion_episode, validate_motion_archive
from t2m_long.representation import recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run official DoubleTake on BABEL-long-v1.")
    parser.add_argument("--prior-root", type=Path, default=Path("third_party/PriorMDM"))
    parser.add_argument("--flow-root", type=Path, default=Path("third_party/FlowMDM"))
    parser.add_argument("--data-root", type=Path, default=Path("third_party/FloodDiffusion/raw_data"))
    parser.add_argument("--output-root", type=Path, default=Path("eval/babel_long_v2/motions"))
    parser.add_argument("--checkpoint", type=Path, default=Path("save/Babel_TrasnEmb_GeoLoss/model000850000.pt"))
    parser.add_argument("--handshake-size", type=int, default=30)
    parser.add_argument("--blend-len", type=int, default=10)
    parser.add_argument("--skip-steps-double-take", type=int, default=100)
    parser.add_argument("--guidance-param", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-episode-frames", type=int, default=201)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
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


def resample_joints(joints: torch.Tensor | np.ndarray, target: int) -> np.ndarray:
    value = torch.as_tensor(joints, dtype=torch.float32).reshape(len(joints), -1).T[None]
    value = F.interpolate(value, size=int(target), mode="linear", align_corners=True)
    return value[0].T.reshape(int(target), 22, 3).cpu().numpy()


def joints_to_hml(joints: np.ndarray, converter) -> np.ndarray:
    padded = np.concatenate([joints, joints[-1:]], axis=0)
    features = converter.process_file(padded, 0.002)[0]
    if features.shape != (len(joints), 263):
        raise RuntimeError(f"HML encoding returned {features.shape}, expected ({len(joints)}, 263)")
    return features.astype(np.float32)


def split_length(length: int, maximum: int = 180) -> list[int]:
    count = max(1, math.ceil(length / maximum))
    base, remainder = divmod(length, count)
    return [base + (index < remainder) for index in range(count)]


def build_windows(episode, handshake: int, minimum: int) -> tuple[list[str], list[int], list[int]]:
    captions: list[str] = []
    lengths: list[int] = []
    segment_indices: list[int] = []
    for segment_index, segment in enumerate(episode.segments):
        target30 = max(1, int((segment.end - segment.start) * 1.5 + 0.5))
        for contribution in split_length(target30):
            # Every window except the first loses one handshake in unfolding.
            window = contribution + (handshake if lengths else 0)
            captions.append(episode.captions[segment.annotation_index])
            lengths.append(max(minimum, window))
            segment_indices.append(segment_index)
    return captions, lengths, segment_indices


def load_doubletake(prior_root: Path, checkpoint: Path, device: torch.device, args):
    sys.path.insert(0, str(prior_root))
    from model.DoubleTake_MDM import doubleTake_MDM
    from utils.model_util import load_model

    saved = json.loads((checkpoint.parent / "args.json").read_text())
    saved.update(
        {
            "model_path": str(checkpoint),
            "device": 0,
            "batch_size": 2,
            "guidance_param": args.guidance_param,
            "double_take": True,
            "second_take_only": False,
            "handshake_size": args.handshake_size,
            "blend_len": args.blend_len,
            "skip_steps_double_take": args.skip_steps_double_take,
            "debug_double_take": False,
        }
    )
    data_stub = SimpleNamespace(dataset=SimpleNamespace(num_actions=1))
    return (*load_model(Namespace(**saved), data_stub, device, ModelClass=doubleTake_MDM), Namespace(**saved))


@torch.inference_mode()
def generate_episode(model, diffusion, model_args, episode, converter, args, device):
    from data_loaders.amass.transforms import SlimSMPLTransform
    from utils.sampling_utils import double_take_arb_len, unfold_sample_arb_len

    minimum = 2 * args.handshake_size + 2 * args.blend_len + 10
    captions, window_lengths, window_segment_indices = build_windows(
        episode, args.handshake_size, minimum
    )
    lengths = torch.tensor(window_lengths, dtype=torch.long, device=device)
    n_frames = int(lengths.max())
    batch_size = len(captions)
    model_args.batch_size = batch_size
    kwargs = {
        "y": {
            "mask": torch.ones((batch_size, 1, 1, n_frames), device=device),
            "lengths": lengths,
            "text": list(captions),
            "tokens": [""],
            "scale": torch.ones(batch_size, device=device) * args.guidance_param,
        }
    }
    samples, _sample_types = double_take_arb_len(
        model_args, diffusion, model, kwargs, n_frames, eval_mode=True
    )
    if len(samples) != 1:
        raise RuntimeError(f"DoubleTake returned {len(samples)} samples")
    step_sizes = np.zeros(batch_size, dtype=np.int64)
    for index, length in enumerate(window_lengths):
        step_sizes[index] = length if index == 0 else step_sizes[index - 1] + length - args.handshake_size
    unfolded = unfold_sample_arb_len(
        samples[0], args.handshake_size, step_sizes, int(step_sizes[-1]), kwargs
    )
    features135 = unfolded[0, :, 0].T.contiguous()
    transform = SlimSMPLTransform(
        batch_size=1, name="SlimSMPLTransform", ename="smplnh", normalization=True
    )
    joints30 = transform.SlimDatastruct(features=features135).joints.float().cpu()

    contributions: list[torch.Tensor] = []
    starts = [0, *step_sizes[:-1].tolist()]
    for start, end in zip(starts, step_sizes.tolist()):
        contributions.append(joints30[start:end])
    segment_joints20 = []
    for segment_index, segment in enumerate(episode.segments):
        parts = [
            part
            for part, source_index in zip(contributions, window_segment_indices)
            if source_index == segment_index
        ]
        joined = torch.cat(parts, dim=0)
        segment_joints20.append(resample_joints(joined, segment.end - segment.start))
    joints20 = np.concatenate(segment_joints20, axis=0)
    motion263 = joints_to_hml(joints20, converter)
    canonical_joints = recover_joints_from_ric(torch.from_numpy(motion263)[None])[0]
    return motion263, canonical_joints, len(window_lengths)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    root = Path(__file__).resolve().parent
    prior_root = (root / args.prior_root).resolve()
    flow_root = (root / args.flow_root).resolve()
    data_root = (root / args.data_root).resolve()
    output_root = (root / args.output_root).resolve()
    checkpoint = (prior_root / args.checkpoint).resolve()
    method_root = output_root / "doubletake"
    method_root.mkdir(parents=True, exist_ok=True)
    os.chdir(prior_root)

    episodes = read_episodes(data_root, args.min_episode_frames)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    already = set() if args.overwrite else existing_episode_ids(method_root)
    pending = [
        episode
        for index, episode in enumerate(episodes)
        if index % args.num_shards == args.shard_index and episode.episode_id not in already
    ]
    first_gt = torch.from_numpy(
        np.load(data_root / "BABEL_streamed/motions" / f"{episodes[0].episode_id}.npy")
    ).float()
    converter = setup_hml_converter(recover_joints_from_ric(first_gt[None])[0], flow_root)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    model, diffusion, model_args = load_doubletake(prior_root, checkpoint, device, args)

    total_windows = 0
    for index, episode in enumerate(pending):
        seed = args.seed + episodes.index(episode)
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        motion263, joints, windows = generate_episode(
            model, diffusion, model_args, episode, converter, args, device
        )
        total_windows += windows
        save_motion_episode(
            output_root,
            method="doubletake",
            episode_id=episode.episode_id,
            motion_263=motion263,
            joints_y_up=joints,
            captions=[episode.captions[segment.annotation_index] for segment in episode.segments],
            segment_starts=[segment.start for segment in episode.segments],
            segment_ends=[segment.end for segment in episode.segments],
            seed=args.seed,
            metadata={
                "checkpoint": str(checkpoint),
                "checkpoint_iteration": 850000,
                "checkpoint_note": "official public archive contains 850k although README example names 1250k",
                "adapter": "official DoubleTake generation at 30 fps; long segments subwindowed; each annotated segment resampled to exact BABEL-long-v1 boundary at 20 fps",
                "native_representation": "BABEL SMPL 135D at 30 fps",
                "guidance_param": args.guidance_param,
                "handshake_size": args.handshake_size,
                "blend_len": args.blend_len,
                "skip_steps_double_take": args.skip_steps_double_take,
                "benchmark": "babel_long_v2_gt10s_no_cap",
            },
        )
        print(
            f"[doubletake shard {args.shard_index}] episodes {index + 1}/{len(pending)} "
            f"{episode.episode_id} windows={windows}",
            flush=True,
        )

    summary = {
        "model": "doubletake",
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "expected_total_episodes": len(episodes),
        "newly_generated": len(pending),
        "windows": total_windows,
        "archive_validation": validate_motion_archive(method_root),
    }
    (method_root / f"run_summary_shard_{args.shard_index:02d}.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
