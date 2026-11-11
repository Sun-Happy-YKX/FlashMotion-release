from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .representation import recover_joints_from_ric


ARCHIVE_VERSION = 1


def _safe_episode_name(episode_id: str) -> str:
    readable = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in episode_id)
    digest = hashlib.sha1(episode_id.encode("utf-8")).hexdigest()[:10]
    return f"{readable[:80]}--{digest}.npz"


def save_motion_episode(
    output_root: Path,
    *,
    method: str,
    episode_id: str,
    motion_263: torch.Tensor | np.ndarray,
    captions: Iterable[str],
    segment_starts: Iterable[int],
    segment_ends: Iterable[int],
    seed: int,
    metadata: dict[str, Any] | None = None,
    joints_y_up: torch.Tensor | np.ndarray | None = None,
) -> Path:
    """Persist one generated episode in the benchmark's visualization format."""

    motion = torch.as_tensor(motion_263, dtype=torch.float32).detach().cpu()
    if motion.ndim != 2 or motion.shape[1] != 263:
        raise ValueError(f"expected (T, 263) motion, received {tuple(motion.shape)}")
    if joints_y_up is None:
        joints = recover_joints_from_ric(motion[None])[0].detach().cpu().numpy()
    else:
        joints = torch.as_tensor(joints_y_up, dtype=torch.float32).detach().cpu().numpy()
    if joints.shape != (len(motion), 22, 3):
        raise ValueError(f"expected (T, 22, 3) joints, received {joints.shape}")

    starts = np.asarray(list(segment_starts), dtype=np.int64)
    ends = np.asarray(list(segment_ends), dtype=np.int64)
    texts = np.asarray(list(captions), dtype=np.str_)
    if not (len(starts) == len(ends) == len(texts)):
        raise ValueError("caption and segment-boundary counts differ")
    if len(starts) == 0 or starts[0] != 0 or ends[-1] != len(motion):
        raise ValueError("segments must cover the complete episode")
    if np.any(ends <= starts) or np.any(starts[1:] != ends[:-1]):
        raise ValueError("segments must be positive and contiguous")

    method_root = Path(output_root) / method
    method_root.mkdir(parents=True, exist_ok=True)
    path = method_root / _safe_episode_name(episode_id)
    payload_metadata = {
        "archive_version": ARCHIVE_VERSION,
        "method": method,
        "episode_id": episode_id,
        "fps": 20,
        "length": len(motion),
        "seed": int(seed),
        **(metadata or {}),
    }
    np.savez_compressed(
        path,
        motion_263=motion.numpy(),
        joints_y_up=joints.astype(np.float32, copy=False),
        captions=texts,
        segment_starts=starts,
        segment_ends=ends,
        metadata_json=np.asarray(json.dumps(payload_metadata, sort_keys=True), dtype=np.str_),
    )
    return path


def validate_motion_archive(method_root: Path) -> dict[str, Any]:
    files = sorted(Path(method_root).glob("*.npz"))
    episodes: set[str] = set()
    total_frames = 0
    total_segments = 0
    failures: list[str] = []
    for path in files:
        try:
            with np.load(path) as data:
                motion = data["motion_263"]
                joints = data["joints_y_up"]
                starts = data["segment_starts"]
                ends = data["segment_ends"]
                texts = data["captions"]
                meta = json.loads(str(data["metadata_json"]))
            if motion.shape != (len(motion), 263):
                raise ValueError(f"bad motion shape {motion.shape}")
            if joints.shape != (len(motion), 22, 3):
                raise ValueError(f"bad joint shape {joints.shape}")
            if not np.isfinite(motion).all() or not np.isfinite(joints).all():
                raise ValueError("non-finite values")
            if not (len(starts) == len(ends) == len(texts)):
                raise ValueError("segment counts differ")
            if len(starts) == 0 or starts[0] != 0 or ends[-1] != len(motion):
                raise ValueError("segments do not cover episode")
            episode_id = str(meta["episode_id"])
            if episode_id in episodes:
                raise ValueError(f"duplicate episode id {episode_id}")
            episodes.add(episode_id)
            total_frames += len(motion)
            total_segments += len(starts)
        except Exception as error:  # validation must report all damaged files
            failures.append(f"{path.name}: {error}")
    return {
        "method_root": str(method_root),
        "episodes": len(episodes),
        "frames": total_frames,
        "segments": total_segments,
        "failures": failures,
    }
