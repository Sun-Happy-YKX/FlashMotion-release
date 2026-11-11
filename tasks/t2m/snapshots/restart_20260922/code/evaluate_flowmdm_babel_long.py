from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from t2m_long.babel_long_benchmark import (
    MultiPredictionEmbeddingCollector,
    TRANSITION_FRAMES,
    centered_transition,
    save_shard,
    stable_seed,
)
from t2m_long.babel_protocol import effective_segments
from t2m_long.data import MotionTextDataset
from t2m_long.evaluator import FloodT2MEvaluator
from t2m_long.motion_archive import save_motion_episode
from t2m_long.representation import recover_joints_from_ric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-root", type=Path, default=Path("third_party/FlowMDM"))
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_long_v1/flowmdm"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-episode-frames", type=int, default=201)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--guidance-param", type=float, default=1.5)
    parser.add_argument("--bpe-denoising-step", type=int, default=125)
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--motion-output-root", type=Path)
    return parser.parse_args()


def load_flowmdm(flow_root: Path, device: torch.device, args: argparse.Namespace):
    sys.path.insert(0, str(flow_root))
    from data_loaders.amass.transforms import SlimSMPLTransform
    from diffusion.diffusion_wrappers import DiffusionWrapper_FlowMDM
    from utils import dist_util
    from utils.model_util import load_model

    model_path = flow_root / "results/babel/FlowMDM/model001300000.pt"
    saved = json.loads((model_path.parent / "args.json").read_text())
    saved.update({
        "model_path": str(model_path),
        "device": 0,
        "guidance_param": args.guidance_param,
        "bpe_denoising_step": args.bpe_denoising_step,
        # Sinusoidal absolute positions can be safely extended.  Chunked RPE
        # is the official implementation's inference-only long-sequence path.
        "max_seq_att": max(2048, int(saved.get("max_seq_att", 1024))),
        "use_chunked_att": True,
    })
    flow_args = Namespace(**saved)
    dist_util.setup_dist(flow_args.device)
    model, diffusion = load_model(flow_args, device)
    wrapper = DiffusionWrapper_FlowMDM(flow_args, diffusion, model)
    transform = SlimSMPLTransform(
        batch_size=16,
        name="SlimSMPLTransform",
        ename="smplnh",
        normalization=True,
    )
    return model, wrapper, transform, model_path


def lengths_at_30fps(segments, frame_length: int) -> list[int]:
    boundaries = [0]
    boundaries.extend(int(segment.end * 1.5 + 0.5) for segment in segments)
    boundaries[-1] = int(frame_length * 1.5 + 0.5)
    lengths = [max(1, boundaries[index + 1] - boundaries[index]) for index in range(len(segments))]
    correction = boundaries[-1] - sum(lengths)
    lengths[-1] += correction
    if min(lengths) < 1:
        raise RuntimeError(f"invalid 30-fps segment lengths: {lengths}")
    return lengths


@torch.inference_mode()
def generate_joints(model, wrapper, transform, captions, lengths, seed, device):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    maximum = max(lengths)
    mask = torch.ones((len(lengths), maximum), dtype=torch.float32)
    for index, length in enumerate(lengths):
        mask[index, length:] = 0
    model_kwargs = {"y": {
        "mask": mask,
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "text": list(captions),
        "tokens": [""],
    }}
    sample = wrapper.p_sample_loop(clip_denoised=False, model_kwargs=model_kwargs, progress=False)
    features = sample.squeeze(0).squeeze(1).permute(1, 0)
    datastruct = transform.SlimDatastruct(features=features)
    return datastruct.joints.float().to(device)


def resample_joints(joints: torch.Tensor, frame_length: int) -> torch.Tensor:
    flattened = joints.reshape(len(joints), -1).T[None]
    values = F.interpolate(flattened, size=frame_length, mode="linear", align_corners=True)
    return values[0].T.reshape(frame_length, joints.shape[1], 3)


def setup_hml_converter(reference_joints: torch.Tensor):
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.scripts import motion_process
    from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets

    motion_process.l_idx1, motion_process.l_idx2 = 5, 8
    motion_process.fid_r, motion_process.fid_l = [8, 11], [7, 10]
    motion_process.face_joint_indx = [2, 1, 17, 16]
    motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    motion_process.kinematic_chain = t2m_kinematic_chain
    skeleton = Skeleton(
        motion_process.n_raw_offsets, motion_process.kinematic_chain, "cpu"
    )
    motion_process.tgt_offsets = skeleton.get_offsets_joints(reference_joints.cpu()[0])
    return motion_process


def joints_to_hml_features(joints: torch.Tensor, frame_length: int, converter) -> torch.Tensor:
    joints = resample_joints(joints, frame_length).cpu().numpy()
    # HumanML's feature extraction produces T-1 vectors.  Repeating the last
    # pose supplies the endpoint vector while preserving the requested cohort
    # length; only its terminal velocity/contact channels are affected.
    joints = np.concatenate([joints, joints[-1:]], axis=0)
    features = converter.process_file(joints, 0.002)[0]
    if features.shape != (frame_length, 263):
        raise RuntimeError(f"FlowMDM conversion returned {features.shape}")
    return torch.from_numpy(features.astype(np.float32))


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    root = Path(__file__).resolve().parent
    flow_root = (root / args.flow_root).resolve()
    flood_root = (root / args.flood_root).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    # FlowMDM's released configuration resolves normalization and body-model
    # assets relative to the repository root.
    os.chdir(flow_root)
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")

    dataset = MotionTextDataset(
        flood_root / "raw_data", root / args.embedding_root, "babel", "val", None
    )
    eligible = [
        index for index, name in enumerate(dataset.ids)
        if np.load(dataset.motion_dir / f"{name}.npy", mmap_mode="r").shape[0]
        >= args.min_episode_frames
    ]
    if args.max_episodes is not None:
        eligible = eligible[: args.max_episodes]
    indices = eligible[args.shard_index :: args.num_shards]
    mean = torch.from_numpy(np.load(flood_root / "raw_data/HumanML3D/Mean.npy"))
    std = torch.from_numpy(np.load(flood_root / "raw_data/HumanML3D/Std.npy"))
    first = dataset[eligible[0]]
    first_raw = first["motion"] * std + mean
    model, wrapper, transform, model_path = load_flowmdm(flow_root, device, args)
    converter = setup_hml_converter(recover_joints_from_ric(first_raw[None])[0])
    evaluator = FloodT2MEvaluator(flood_root, device, seed=args.seed)
    collector = MultiPredictionEmbeddingCollector(
        evaluator, ["multi_flowmdm"], args.embedding_batch_size
    )
    transition_parts: dict[str, list[np.ndarray]] = defaultdict(list)

    for dataset_index in tqdm(indices, desc=f"FlowMDM shard {args.shard_index}"):
        item = dataset[dataset_index]
        name = item["name"]
        length = item["length"]
        segments = effective_segments(item["text_ends"], length)
        gt_raw = item["motion"] * std + mean
        generated_joints_30fps = generate_joints(
            model,
            wrapper,
            transform,
            item["captions"],
            lengths_at_30fps(segments, length),
            stable_seed(args.seed, "flowmdm", name),
            device,
        )
        prediction = joints_to_hml_features(
            generated_joints_30fps, length, converter
        )
        gt_joints = recover_joints_from_ric(gt_raw[None])[0]
        pred_joints = recover_joints_from_ric(prediction[None])[0]
        if args.motion_output_root is not None:
            save_motion_episode(
                (root / args.motion_output_root).resolve(),
                method="flowmdm",
                episode_id=name,
                motion_263=prediction,
                joints_y_up=pred_joints,
                captions=[item["captions"][segment.annotation_index] for segment in segments],
                segment_starts=[segment.start for segment in segments],
                segment_ends=[segment.end for segment in segments],
                seed=args.seed,
                metadata={
                    "checkpoint": str(model_path),
                    "native_representation": "BABEL 135D at 30 fps",
                    "adapter": "official continuous FlowMDM generation; joints resampled to 20 fps and re-encoded as HML263",
                    "guidance_param": args.guidance_param,
                    "bpe_denoising_step": args.bpe_denoising_step,
                    "benchmark": "babel_long_v2_gt10s_no_cap",
                },
            )
        for transition_index, segment in enumerate(segments[1:]):
            gt_curve = centered_transition(gt_joints, segment.start, TRANSITION_FRAMES)
            pred_curve = centered_transition(pred_joints, segment.start, TRANSITION_FRAMES)
            if gt_curve is None or pred_curve is None:
                continue
            transition_parts["transition__ground_truth"].append(gt_curve)
            transition_parts["transition__multi_flowmdm"].append(pred_curve)
            transition_parts["transition_key"].append(
                np.asarray(f"{name}::{transition_index}", dtype=np.str_)
            )
            transition_parts["transition_episode_id"].append(
                np.asarray(name, dtype=np.str_)
            )
        for segment_index, segment in enumerate(segments):
            caption = item["captions"][segment.annotation_index]
            collector.add(
                gt_raw[segment.start:segment.end],
                item["text_tokens"][segment.annotation_index],
                {"multi_flowmdm": prediction[segment.start:segment.end]},
                {
                    "segment_key": f"{name}::{segment_index}",
                    "episode_id": name,
                    "caption": caption,
                    "segment_index": segment_index,
                    "segment_length": segment.end - segment.start,
                    "episode_length": length,
                },
            )

    arrays = collector.finish()
    for key, values in transition_parts.items():
        arrays[key] = np.stack(values) if values else np.empty((0,), dtype=np.float32)
    shard_path = output / f"shard_{args.shard_index:02d}_of_{args.num_shards:02d}.npz"
    save_shard(shard_path, arrays, {
        "model": "FlowMDM official BABEL checkpoint",
        "checkpoint": str(model_path),
        "native_representation": "135D at 30 fps",
        "evaluation_representation": "resampled joints -> common 263D at 20 fps",
        "guidance_param": args.guidance_param,
        "bpe_denoising_step": args.bpe_denoising_step,
        "diffusion_steps": 1000,
        "minimum_episode_frames": args.min_episode_frames,
        "eligible_episodes": len(eligible),
        "shard_episodes": len(indices),
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "segments": len(arrays["segment_key"]),
        "transitions": len(arrays["transition_key"]),
    })
    print(shard_path, flush=True)


if __name__ == "__main__":
    main()
