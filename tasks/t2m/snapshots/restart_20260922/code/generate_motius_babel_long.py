from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from t2m_long.motion_archive import save_motion_episode, validate_motion_archive


MODEL_SPECS = {
    "mdm": {
        "artifact": "ZeyuLing/Motius-MDM-HumanML3D",
        "batch_size": 16,
        "multiple": 4,
        "call_kwargs": {},
        "checkpoint_origin": "MDM official HumanML3D checkpoint repackaged by Motius",
    },
    "momask": {
        "artifact": "ZeyuLing/Motius-MoMask-HumanML3D",
        "batch_size": 32,
        "multiple": 4,
        "call_kwargs": {"time_steps": 18, "cond_scale": 4.0},
        "checkpoint_origin": "MoMask official HumanML3D checkpoint repackaged by Motius",
    },
    "mld": {
        "artifact": "ZeyuLing/Motius-MLD-HumanML3D",
        "batch_size": 32,
        "multiple": 4,
        "call_kwargs": {"num_inference_steps": 50},
        "checkpoint_origin": "MLD official HumanML3D checkpoint repackaged by Motius",
    },
    "t2mgpt": {
        "artifact": "ZeyuLing/Motius-T2M-GPT-HumanML3D",
        "batch_size": 32,
        "multiple": 4,
        "call_kwargs": {"if_categorial": True},
        "checkpoint_origin": "T2M-GPT official HumanML3D checkpoint repackaged by Motius",
    },
    "motionlcm": {
        "artifact": "ZeyuLing/Motius-MotionLCM-HumanML3D",
        "batch_size": 32,
        "multiple": 4,
        "call_kwargs": {"num_inference_steps": 1},
        "checkpoint_origin": "MotionLCM official HumanML3D checkpoint repackaged by Motius",
    },
    "motiongpt": {
        "artifact": "ZeyuLing/Motius-MotionGPT-HumanML3D",
        "batch_size": 8,
        "multiple": 4,
        "call_kwargs": {"do_sample": True},
        "checkpoint_origin": "MotionGPT official HumanML3D checkpoint repackaged by Motius",
    },
}


@dataclass(frozen=True)
class Segment:
    annotation_index: int
    start: int
    end: int


@dataclass(frozen=True)
class Episode:
    episode_id: str
    length: int
    captions: tuple[str, ...]
    segments: tuple[Segment, ...]


@dataclass(frozen=True)
class WorkItem:
    episode_id: str
    segment_index: int
    chunk_index: int
    caption: str
    requested_length: int
    model_length: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HML263 baselines on BABEL-long-v1.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_SPECS))
    parser.add_argument("--data-root", type=Path, default=Path("third_party/FloodDiffusion/raw_data"))
    parser.add_argument("--output-root", type=Path, default=Path("eval/babel_long_v2/motions"))
    parser.add_argument("--cache-dir", type=Path, default=Path("third_party/hf_cache"))
    parser.add_argument("--min-episode-frames", type=int, default=201)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_episodes(data_root: Path, minimum: int) -> list[Episode]:
    root = data_root / "BABEL_streamed"
    motion_root = root / "motions"
    text_root = root / "texts"
    ids = [line.strip() for line in (root / "val_processed.txt").read_text().splitlines() if line.strip()]
    episodes: list[Episode] = []
    for episode_id in ids:
        length = int(np.load(motion_root / f"{episode_id}.npy", mmap_mode="r").shape[0])
        if length < minimum:
            continue
        annotations = []
        for raw in (text_root / f"{episode_id}.txt").read_text().splitlines():
            caption, _tokens, _start, end = raw.split("#")
            annotations.append((caption, min(length, int(float(end) * 20.0 + 0.5))))
        if not annotations:
            raise RuntimeError(f"{episode_id} has no BABEL text annotations")
        segments: list[Segment] = []
        previous = 0
        for annotation_index, (_caption, source_end) in enumerate(annotations):
            end = max(previous, min(length, source_end))
            if end > previous:
                segments.append(Segment(annotation_index, previous, end))
            previous = end
        if previous < length:
            last_index = len(annotations) - 1
            if segments and segments[-1].annotation_index == last_index:
                last = segments[-1]
                segments[-1] = Segment(last.annotation_index, last.start, length)
            else:
                segments.append(Segment(last_index, previous, length))
        if len(segments) < 2:
            continue
        episodes.append(
            Episode(
                episode_id=episode_id,
                length=length,
                captions=tuple(item[0] for item in annotations),
                segments=tuple(segments),
            )
        )
    return episodes


def existing_episode_ids(method_root: Path) -> set[str]:
    ids: set[str] = set()
    for path in method_root.glob("*.npz"):
        try:
            with np.load(path, allow_pickle=False) as data:
                ids.add(str(json.loads(str(data["metadata_json"]))["episode_id"]))
        except Exception:
            pass
    return ids


def split_length(length: int, maximum: int = 196) -> list[int]:
    count = max(1, math.ceil(length / maximum))
    base, remainder = divmod(length, count)
    return [base + (index < remainder) for index in range(count)]


def model_length(requested: int, multiple: int) -> int:
    length = max(40, min(196, requested))
    if multiple > 1:
        length = int(math.ceil(length / multiple) * multiple)
        length = min(196, length)
    return length


def fit_exact_length(motion: np.ndarray, requested: int) -> tuple[np.ndarray, int]:
    motion = np.asarray(motion, dtype=np.float32)
    if motion.ndim != 2 or motion.shape[1] != 263 or len(motion) == 0:
        raise RuntimeError(f"invalid model output {motion.shape}")
    padded = max(0, requested - len(motion))
    if padded:
        motion = np.concatenate([motion, np.repeat(motion[-1:], padded, axis=0)], axis=0)
    return motion[:requested], padded


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_root = (root / args.data_root).resolve()
    output_root = (root / args.output_root).resolve()
    cache_dir = (root / args.cache_dir).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    method_root = output_root / args.model
    method_root.mkdir(parents=True, exist_ok=True)

    spec = MODEL_SPECS[args.model]
    episodes = read_episodes(data_root, args.min_episode_frames)
    if args.max_episodes is not None:
        episodes = episodes[: args.max_episodes]
    already = set() if args.overwrite else existing_episode_ids(method_root)
    pending = [episode for episode in episodes if episode.episode_id not in already]
    episode_map = {episode.episode_id: episode for episode in pending}

    work: list[WorkItem] = []
    for episode in pending:
        for segment_index, segment in enumerate(episode.segments):
            caption = episode.captions[segment.annotation_index]
            for chunk_index, requested in enumerate(split_length(segment.end - segment.start)):
                work.append(
                    WorkItem(
                        episode.episode_id,
                        segment_index,
                        chunk_index,
                        caption,
                        requested,
                        model_length(requested, int(spec["multiple"])),
                    )
                )

    from motius import Pipeline

    pipeline = Pipeline.from_pretrained(
        str(spec["artifact"]),
        cache_dir=str(cache_dir),
        device=args.device,
    )
    batch_size = int(args.batch_size or spec["batch_size"])
    generated: dict[tuple[str, int], list[np.ndarray]] = {}
    padded_frames = 0
    for start in range(0, len(work), batch_size):
        batch = work[start : start + batch_size]
        batch_seed = args.seed + start
        random.seed(batch_seed)
        np.random.seed(batch_seed % (2**32 - 1))
        torch.manual_seed(batch_seed)
        torch.cuda.manual_seed_all(batch_seed)
        call_kwargs = dict(spec["call_kwargs"])
        if args.model == "motiongpt":
            call_kwargs["seed"] = batch_seed
        motions = pipeline.infer_text_to_motion(
            [item.caption for item in batch],
            [item.model_length for item in batch],
            **call_kwargs,
        )
        if len(motions) != len(batch):
            raise RuntimeError(f"model returned {len(motions)} outputs for batch of {len(batch)}")
        for item, motion in zip(batch, motions):
            exact, padded = fit_exact_length(motion, item.requested_length)
            padded_frames += padded
            generated.setdefault((item.episode_id, item.segment_index), []).append(exact)
        print(f"[{args.model}] chunks {min(start + len(batch), len(work))}/{len(work)}", flush=True)

    for index, episode in enumerate(pending):
        segment_motions = []
        for segment_index, segment in enumerate(episode.segments):
            pieces = generated[(episode.episode_id, segment_index)]
            segment_motion = np.concatenate(pieces, axis=0)
            expected = segment.end - segment.start
            if segment_motion.shape != (expected, 263):
                raise RuntimeError(
                    f"{episode.episode_id} segment {segment_index}: {segment_motion.shape} != ({expected}, 263)"
                )
            segment_motions.append(segment_motion)
        motion = np.concatenate(segment_motions, axis=0)
        if motion.shape != (episode.length, 263):
            raise RuntimeError(f"{episode.episode_id}: {motion.shape} != ({episode.length}, 263)")
        save_motion_episode(
            output_root,
            method=args.model,
            episode_id=episode.episode_id,
            motion_263=motion,
            captions=[episode.captions[segment.annotation_index] for segment in episode.segments],
            segment_starts=[segment.start for segment in episode.segments],
            segment_ends=[segment.end for segment in episode.segments],
            seed=args.seed,
            metadata={
                "artifact": spec["artifact"],
                "checkpoint_origin": spec["checkpoint_origin"],
                "adapter": "segment-conditioned hard concatenation; >196-frame segments evenly subchunked; no seam smoothing",
                "native_representation": "HumanML3D 263D at 20 fps",
                "benchmark": "babel_long_v2_gt10s_no_cap",
            },
        )
        if (index + 1) % 25 == 0 or index + 1 == len(pending):
            print(f"[{args.model}] episodes {index + 1}/{len(pending)}", flush=True)

    validation = validate_motion_archive(method_root)
    summary = {
        "model": args.model,
        "artifact": spec["artifact"],
        "seed": args.seed,
        "minimum_episode_frames": args.min_episode_frames,
        "expected_episodes": len(episodes),
        "previously_complete": len(already & {item.episode_id for item in episodes}),
        "newly_generated": len(pending),
        "work_chunks": len(work),
        "exact_length_padding_frames": padded_frames,
        "call_kwargs": spec["call_kwargs"],
        "archive_validation": validation,
    }
    (method_root / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
