from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from collections import defaultdict
from dataclasses import asdict
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
    parser.add_argument("--dart-root", type=Path, default=Path("third_party/DART"))
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_long_v1/dart_gt10s"))
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-episode-frames", type=int, default=201)
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--guidance-param", type=float, default=5.0)
    parser.add_argument("--use-predicted-joints", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=32)
    parser.add_argument("--motion-output-root", type=Path)
    return parser.parse_args()


def install_dart_compatibility_shims(dart_root: Path, smplh_root: Path) -> str:
    # The released code targets Python 3.10 / NumPy 1.x and imports PyTorch3D
    # solely for its rotation conversion helpers along this inference path.
    for name, value in {
        "float": float,
        "int": int,
        "bool": bool,
        "complex": complex,
        "object": object,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)

    sys.path.insert(0, str(dart_root))
    from utils import rotation_conversions

    pytorch3d = types.ModuleType("pytorch3d")
    pytorch3d.transforms = rotation_conversions
    sys.modules["pytorch3d"] = pytorch3d
    sys.modules["pytorch3d.transforms"] = rotation_conversions

    expected_smplx = (
        dart_root
        / "data/smplx_lockedhead_20230207/models_lockedhead/smplx/SMPLX_MALE.npz"
    )
    if expected_smplx.exists():
        return "smplx"

    # Fallback used only for the diagnostic predicted-joint path when licensed
    # SMPL-X files are unavailable.  Primary results require SMPL-X.
    import smplx

    original_build_layer = smplx.build_layer

    def build_calibration_layer(*_args, **_kwargs):
        return original_build_layer(
            str(smplh_root),
            model_type="smplh",
            gender="male",
            ext="pkl",
            num_pca_comps=12,
        )

    smplx.build_layer = build_calibration_layer
    return "smplh"


def load_dart(
    dart_root: Path,
    device: torch.device,
    guidance_param: float,
    use_predicted_joints: bool,
    body_type: str,
):
    from data_loaders.humanml.data.dataset import SinglePrimitiveDataset
    from mld.rollout_mld import load_mld
    from mld.sample_flowmdm_mld import FlowMDMArgs, rollout
    from mld.train_mld import create_gaussian_diffusion

    checkpoint = dart_root / "mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt"
    denoiser_args, denoiser, vae_args, vae = load_mld(str(checkpoint), device)
    diffusion_args = denoiser_args.diffusion_args
    diffusion_args.respacing = ""
    diffusion = create_gaussian_diffusion(diffusion_args)
    rollout_args = FlowMDMArgs(
        seed=0,
        torch_deterministic=True,
        device=device,
        batch_size=1,
        denoiser_checkpoint=str(checkpoint),
        respacing="",
        guidance_param=guidance_param,
        export_smpl=0,
        zero_noise=0,
        use_predicted_joints=int(use_predicted_joints),
        fix_floor=0,
    )
    dataset = SinglePrimitiveDataset(
        cfg_path=vae_args.data_args.cfg_path,
        dataset_path=vae_args.data_args.data_dir,
        sequence_path="./data/stand.pkl",
        batch_size=1,
        device=device,
        enforce_gender="male",
        enforce_zero_beta=1,
        body_type=body_type,
    )
    return checkpoint, denoiser_args, denoiser, vae_args, vae, diffusion, dataset, rollout_args, rollout


def lengths_at_30fps(segments, frame_length: int) -> list[int]:
    boundaries = [0]
    boundaries.extend(int(segment.end * 1.5 + 0.5) for segment in segments)
    boundaries[-1] = int(frame_length * 1.5 + 0.5)
    lengths = [max(1, boundaries[index + 1] - boundaries[index]) for index in range(len(segments))]
    lengths[-1] += boundaries[-1] - sum(lengths)
    if min(lengths) < 1:
        raise RuntimeError(f"invalid 30-fps segment lengths: {lengths}")
    return lengths


def resample_joints(joints: torch.Tensor, frame_length: int) -> torch.Tensor:
    flattened = joints.reshape(len(joints), -1).T[None]
    values = F.interpolate(flattened, size=frame_length, mode="linear", align_corners=True)
    return values[0].T.reshape(frame_length, joints.shape[1], 3)


def setup_hml_converter(reference_joints: torch.Tensor):
    # Import the common HumanML converter from FloodDiffusion, not DART's
    # same-named data_loaders package already resident in sys.modules.
    flood_root = Path(os.environ["FLOWER_BENCH_FLOOD_ROOT"])
    from importlib.util import module_from_spec, spec_from_file_location

    skeleton_path = flood_root / "data_loaders/humanml/common/skeleton.py"
    process_path = flood_root / "data_loaders/humanml/scripts/motion_process.py"
    # The DART and Flood versions are derived from the same HumanML code.  We
    # configure DART's loaded converter explicitly with Flood's canonical
    # offsets so its 263D output is identical to the rest of this benchmark.
    from data_loaders.humanml.common.skeleton import Skeleton
    from data_loaders.humanml.scripts import motion_process
    from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain, t2m_raw_offsets

    del skeleton_path, process_path, module_from_spec, spec_from_file_location
    motion_process.l_idx1, motion_process.l_idx2 = 5, 8
    motion_process.fid_r, motion_process.fid_l = [8, 11], [7, 10]
    motion_process.face_joint_indx = [2, 1, 17, 16]
    motion_process.n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    motion_process.kinematic_chain = t2m_kinematic_chain
    skeleton = Skeleton(motion_process.n_raw_offsets, motion_process.kinematic_chain, "cpu")
    motion_process.tgt_offsets = skeleton.get_offsets_joints(reference_joints.cpu()[0])
    return motion_process


def joints_to_hml_features(joints: torch.Tensor, frame_length: int, converter) -> torch.Tensor:
    joints = resample_joints(joints, frame_length).cpu().numpy()
    # DART's SMPL-X primitives are canonicalized in a z-up frame.  Match the
    # repository's official HumanML3D conversion before extracting 263D
    # features: joints @ R.from_euler('xyz', [90, 0, 180]).as_matrix().
    # In exact axis notation this is (x, y, z) -> (-x, z, y).
    joints = joints[..., [0, 2, 1]]
    joints[..., 0] *= -1.0
    joints = np.concatenate([joints, joints[-1:]], axis=0)
    features = converter.process_file(joints, 0.002)[0]
    if features.shape != (frame_length, 263):
        raise RuntimeError(f"DART conversion returned {features.shape}")
    return torch.from_numpy(features.astype(np.float32))


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    root = Path(__file__).resolve().parent
    dart_root = (root / args.dart_root).resolve()
    flood_root = (root / args.flood_root).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["FLOWER_BENCH_FLOOD_ROOT"] = str(flood_root)
    os.chdir(dart_root)
    body_type = install_dart_compatibility_shims(
        dart_root, (root / "third_party/FlowMDM/body_models").resolve()
    )
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
    converter = setup_hml_converter(recover_joints_from_ric(first_raw[None])[0])
    (
        checkpoint,
        denoiser_args,
        denoiser,
        vae_args,
        vae,
        diffusion,
        seed_dataset,
        rollout_args,
        rollout,
    ) = load_dart(
        dart_root,
        device,
        args.guidance_param,
        args.use_predicted_joints,
        body_type,
    )
    evaluator = FloodT2MEvaluator(flood_root, device, seed=args.seed)
    collector = MultiPredictionEmbeddingCollector(evaluator, ["multi_dart"], args.embedding_batch_size)
    transition_parts: dict[str, list[np.ndarray]] = defaultdict(list)

    for dataset_index in tqdm(indices, desc=f"DART shard {args.shard_index}"):
        item = dataset[dataset_index]
        name = item["name"]
        length = item["length"]
        segments = effective_segments(item["text_ends"], length)
        gt_raw = item["motion"] * std + mean
        episode_seed = stable_seed(args.seed, "dart", name)
        random.seed(episode_seed)
        np.random.seed(episode_seed % (2**32))
        torch.manual_seed(episode_seed)
        torch.cuda.manual_seed_all(episode_seed)
        config = {
            "text": list(item["captions"]),
            "lengths": lengths_at_30fps(segments, length),
        }
        with torch.inference_mode():
            result = rollout(
                config,
                denoiser_args,
                denoiser,
                vae_args,
                vae,
                diffusion,
                seed_dataset,
                rollout_args,
            )
        joints_30fps = result["joints"][0].float().to(device)
        prediction = joints_to_hml_features(joints_30fps, length, converter)
        gt_joints = recover_joints_from_ric(gt_raw[None])[0]
        pred_joints = recover_joints_from_ric(prediction[None])[0]
        if args.motion_output_root is not None:
            save_motion_episode(
                (root / args.motion_output_root).resolve(),
                method="dart",
                episode_id=name,
                motion_263=prediction,
                joints_y_up=pred_joints,
                captions=[item["captions"][segment.annotation_index] for segment in segments],
                segment_starts=[segment.start for segment in segments],
                segment_ends=[segment.end for segment in segments],
                seed=args.seed,
                metadata={
                    "checkpoint": str(checkpoint),
                    "native_representation": "DART SMPL motion primitives at 30 fps",
                    "adapter": "official autoregressive primitive rollout; z-up converted to y-up; joints resampled to 20 fps and re-encoded as HML263",
                    "guidance_param": args.guidance_param,
                    "body_type": body_type,
                    "benchmark": "babel_long_v2_gt10s_no_cap",
                },
            )
        for transition_index, segment in enumerate(segments[1:]):
            gt_curve = centered_transition(gt_joints, segment.start, TRANSITION_FRAMES)
            pred_curve = centered_transition(pred_joints, segment.start, TRANSITION_FRAMES)
            if gt_curve is None or pred_curve is None:
                continue
            transition_parts["transition__ground_truth"].append(gt_curve)
            transition_parts["transition__multi_dart"].append(pred_curve)
            transition_parts["transition_key"].append(np.asarray(f"{name}::{transition_index}", dtype=np.str_))
            transition_parts["transition_episode_id"].append(np.asarray(name, dtype=np.str_))
        for segment_index, segment in enumerate(segments):
            annotation_index = segment.annotation_index
            collector.add(
                gt_raw[segment.start:segment.end],
                item["text_tokens"][annotation_index],
                {"multi_dart": prediction[segment.start:segment.end]},
                {
                    "segment_key": f"{name}::{segment_index}",
                    "episode_id": name,
                    "caption": item["captions"][annotation_index],
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
        "model": "DART official BABEL checkpoint",
        "checkpoint": str(checkpoint),
        "native_representation": "DART predicted 22 joints at 30 fps",
        "evaluation_representation": "resampled joints -> common 263D at 20 fps",
        "joint_source": (
            "network-predicted joints"
            if args.use_predicted_joints
            else f"{body_type.upper()} regressed joints from DART-predicted rotations"
        ),
        "guidance_param": args.guidance_param,
        "diffusion_steps_per_primitive": int(denoiser_args.diffusion_args.diffusion_steps),
        "primitive_future_frames": int(seed_dataset.future_length),
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
