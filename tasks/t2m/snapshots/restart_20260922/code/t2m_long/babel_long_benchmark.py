from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from .evaluator import FloodT2MEvaluator, stable_frechet_distance


FEATURE_FPS = 20
TRANSITION_FRAMES = 20  # one second in FloodDiffusion's released 263D BABEL data


def stable_seed(base_seed: int, *parts: object) -> int:
    payload = "::".join(str(part) for part in (base_seed, *parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def normalize_caption(caption: str) -> str:
    return " ".join(caption.casefold().strip().split())


class MultiPredictionEmbeddingCollector:
    """Embed paired GT/text segments and any number of prediction variants."""

    def __init__(
        self,
        evaluator: FloodT2MEvaluator,
        prediction_names: Iterable[str],
        batch_size: int = 32,
    ):
        self.evaluator = evaluator
        self.prediction_names = tuple(prediction_names)
        self.batch_size = int(batch_size)
        self.pending_gt: list[torch.Tensor] = []
        self.pending_tokens: list[list[str]] = []
        self.pending_predictions = {name: [] for name in self.prediction_names}
        self.pending_meta: list[dict] = []
        self.parts: dict[str, list[np.ndarray]] = {
            "ground_truth": [],
            "text": [],
            **{f"prediction__{name}": [] for name in self.prediction_names},
        }
        self.meta: dict[str, list] = defaultdict(list)

    def add(
        self,
        ground_truth: torch.Tensor,
        tokens: list[str],
        predictions: dict[str, torch.Tensor],
        metadata: dict,
    ) -> None:
        if len(ground_truth) < 4:
            return
        if set(predictions) != set(self.prediction_names):
            raise ValueError("prediction variants changed during collection")
        expected = len(ground_truth)
        if any(len(value) != expected for value in predictions.values()):
            raise ValueError("prediction/GT segment length mismatch")
        self.pending_gt.append(ground_truth.detach().cpu())
        self.pending_tokens.append(tokens)
        for name, value in predictions.items():
            self.pending_predictions[name].append(value.detach().cpu())
        self.pending_meta.append(metadata)
        if len(self.pending_gt) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.pending_gt:
            return
        lengths = torch.tensor([len(value) for value in self.pending_gt], dtype=torch.long)
        gt = pad_sequence(self.pending_gt, batch_first=True)
        self.parts["ground_truth"].append(
            self.evaluator._motion_embeddings(gt, lengths).float().cpu().numpy()
        )
        self.parts["text"].append(
            self.evaluator._text_embeddings(self.pending_tokens).float().cpu().numpy()
        )
        for name in self.prediction_names:
            batch = pad_sequence(self.pending_predictions[name], batch_first=True)
            self.parts[f"prediction__{name}"].append(
                self.evaluator._motion_embeddings(batch, lengths).float().cpu().numpy()
            )
        for item in self.pending_meta:
            for key, value in item.items():
                self.meta[key].append(value)
        self.pending_gt.clear()
        self.pending_tokens.clear()
        self.pending_meta.clear()
        for values in self.pending_predictions.values():
            values.clear()

    def finish(self) -> dict[str, np.ndarray]:
        self.flush()
        result = {key: np.concatenate(values) for key, values in self.parts.items()}
        for key, values in self.meta.items():
            if key in {"segment_index", "segment_length", "episode_length"}:
                result[key] = np.asarray(values, dtype=np.int64)
            else:
                result[key] = np.asarray(values, dtype=np.str_)
        order = np.argsort(result["segment_key"], kind="stable")
        return {key: value[order] for key, value in result.items()}


def transition_jerk_curve(joints: np.ndarray) -> np.ndarray:
    velocity = joints[1:] - joints[:-1]
    acceleration = velocity[1:] - velocity[:-1]
    jerk = acceleration[1:] - acceleration[:-1]
    return np.abs(jerk).sum(axis=-1).max(axis=-1)


def centered_transition(joints: torch.Tensor, boundary: int, frames: int = TRANSITION_FRAMES):
    left = frames // 2
    right = frames - left
    if boundary < left or boundary + right > len(joints):
        return None
    return transition_jerk_curve(
        joints[boundary - left : boundary + right].float().cpu().numpy()
    )


def save_shard(path: Path, arrays: dict[str, np.ndarray], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2))


def combine_shards(paths: Iterable[Path]) -> dict[str, np.ndarray]:
    loaded = [np.load(path) for path in sorted(paths)]
    if not loaded:
        raise RuntimeError("no benchmark shards found")
    keys = set(loaded[0].files)
    if any(set(shard.files) != keys for shard in loaded[1:]):
        raise RuntimeError("shards have different array schemas")
    result = {key: np.concatenate([shard[key] for shard in loaded]) for key in keys}
    if "segment_key" in result:
        order = np.argsort(result["segment_key"], kind="stable")
        segment_fields = {
            "ground_truth", "text", "segment_key", "episode_id", "caption",
            "segment_index", "segment_length", "episode_length",
        } | {key for key in result if key.startswith("prediction__")}
        for key in segment_fields:
            if key in result:
                result[key] = result[key][order]
    return result


def _activation_statistics(features: np.ndarray, weights: np.ndarray | None = None):
    """Return deterministic float64 statistics for FID.

    The NumPy/OpenBLAS build on the training image can return a visibly
    asymmetric result for the mathematically symmetric weighted product
    ``(X * w).T @ X`` (up to roughly 3e-2 on the BABEL embeddings).  Averaging
    those two corrupted triangles is not sufficient and can make the FID
    spuriously negative.  PyTorch's CPU float64 GEMM is deterministic on this
    image and agrees with SciPy's reference FID once the covariance is formed.
    """

    values = torch.as_tensor(np.asarray(features), dtype=torch.float64)
    if len(values) < 2:
        raise ValueError("FID covariance requires at least two feature vectors")
    if weights is None:
        normalized_weights = torch.full(
            (len(values),), 1.0 / len(values), dtype=torch.float64
        )
    else:
        normalized_weights = torch.as_tensor(
            np.asarray(weights), dtype=torch.float64
        )
        normalized_weights = normalized_weights / normalized_weights.sum()
    mean = torch.sum(values * normalized_weights[:, None], dim=0)
    centered = values - mean
    correction = 1.0 - torch.square(normalized_weights).sum()
    covariance = (
        (centered * normalized_weights[:, None]).T @ centered
        / correction.clamp_min(1e-12)
    )
    covariance = (covariance + covariance.T) * 0.5
    return mean.numpy(), covariance.numpy()


def _episode_weights(episode_ids: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(episode_ids, return_inverse=True, return_counts=True)
    return 1.0 / (len(unique) * counts[inverse])


def _diversity(features: np.ndarray, seed: int, times: int = 300) -> float:
    if len(features) < 2:
        return float("nan")
    rng = np.random.default_rng(seed)
    times = min(times, len(features) - 1)
    first = rng.choice(len(features), times, replace=False)
    second = rng.choice(len(features), times, replace=False)
    collision = first == second
    while collision.any():
        second[collision] = rng.integers(0, len(features), int(collision.sum()))
        collision = first == second
    return float(np.linalg.norm(features[first] - features[second], axis=1).mean())


@torch.inference_mode()
def retrieval_metrics(
    text: np.ndarray,
    motion: np.ndarray,
    captions: np.ndarray,
    episode_ids: np.ndarray,
    device: torch.device,
    seed: int = 42,
    repeats: int = 20,
    candidate_size: int = 32,
) -> dict[str, float]:
    """Both legacy diagonal and duplicate-safe multi-positive R-precision."""

    text_t = torch.from_numpy(text).float().to(device)
    motion_t = torch.from_numpy(motion).float().to(device)
    label_values = [normalize_caption(value) for value in captions.tolist()]
    label_map = {value: index for index, value in enumerate(sorted(set(label_values)))}
    labels = torch.tensor([label_map[value] for value in label_values], device=device)
    count = len(text_t)
    hit_sum = torch.zeros((count, 3), dtype=torch.float64)
    diagonal_sum = torch.zeros(3, dtype=torch.float64)
    matching_sum = 0.0
    usable_total = 0

    for repeat in range(repeats):
        order = torch.randperm(
            count, generator=torch.Generator().manual_seed(seed + repeat)
        )
        usable = count // candidate_size * candidate_size
        for start in range(0, usable, candidate_size):
            indices_cpu = order[start : start + candidate_size]
            indices = indices_cpu.to(device)
            distances = torch.cdist(text_t[indices], motion_t[indices]).nan_to_num()
            ranking = torch.argsort(distances, dim=1)
            positive = labels[indices, None] == labels[indices][None, :]
            ranked_positive = positive.gather(1, ranking[:, :3])
            hits = ranked_positive.cumsum(1).clamp_max(1).cpu().to(torch.float64)
            hit_sum[indices_cpu] += hits
            diagonal = ranking[:, :3] == torch.arange(candidate_size, device=device)[:, None]
            diagonal_sum += diagonal.cumsum(1).clamp_max(1).sum(0).cpu()
            matching_sum += float(distances.masked_fill(~positive, float("inf")).min(1).values.sum())
            usable_total += candidate_size

    appearances = hit_sum[:, 2].clone()
    # R@3 can be zero even when the item appeared, so count appearances separately.
    per_item_counts = torch.zeros(count, dtype=torch.float64)
    for repeat in range(repeats):
        order = torch.randperm(count, generator=torch.Generator().manual_seed(seed + repeat))
        usable = count // candidate_size * candidate_size
        per_item_counts[order[:usable]] += 1
    per_item = hit_sum / per_item_counts.clamp_min(1)[:, None]
    episode_scores = []
    for episode in np.unique(episode_ids):
        mask = torch.from_numpy(episode_ids == episode)
        episode_scores.append(per_item[mask].mean(0))
    macro = torch.stack(episode_scores).mean(0)
    micro = hit_sum.sum(0) / per_item_counts.sum().clamp_min(1)

    return {
        **{f"R@{k}_episode_macro": float(macro[k - 1]) for k in (1, 2, 3)},
        **{f"R@{k}_segment_micro": float(micro[k - 1]) for k in (1, 2, 3)},
        **{
            f"R@{k}_legacy_diagonal": float(diagonal_sum[k - 1] / max(usable_total, 1))
            for k in (1, 2, 3)
        },
        "MM_Dist_multi_positive": matching_sum / max(usable_total, 1),
        "candidate_size": candidate_size,
        "repeats": repeats,
    }


def score_embedding_subset(
    arrays: dict[str, np.ndarray],
    prediction_key: str,
    selected: np.ndarray,
    device: torch.device,
    seed: int = 42,
) -> dict[str, float]:
    prediction = arrays[prediction_key][selected]
    ground_truth = arrays["ground_truth"][selected]
    text = arrays["text"][selected]
    captions = arrays["caption"][selected]
    episodes = arrays["episode_id"][selected]
    result = retrieval_metrics(text, prediction, captions, episodes, device, seed=seed)
    result.update({f"gt_{key}": value for key, value in retrieval_metrics(
        text, ground_truth, captions, episodes, device, seed=seed
    ).items() if key.startswith("R@") or key.startswith("MM_")})

    pred_mean, pred_cov = _activation_statistics(prediction)
    gt_mean, gt_cov = _activation_statistics(ground_truth)
    weights = _episode_weights(episodes)
    pred_weighted_mean, pred_weighted_cov = _activation_statistics(prediction, weights)
    gt_weighted_mean, gt_weighted_cov = _activation_statistics(ground_truth, weights)
    result.update({
        "segments": int(selected.sum()),
        "episodes": int(len(np.unique(episodes))),
        "FID_segment": stable_frechet_distance(gt_mean, gt_cov, pred_mean, pred_cov),
        "FID_episode_balanced": stable_frechet_distance(
            gt_weighted_mean, gt_weighted_cov, pred_weighted_mean, pred_weighted_cov
        ),
        "Diversity": _diversity(prediction, seed),
        "gt_Diversity": _diversity(ground_truth, seed),
    })
    return result


def transition_metrics(arrays: dict[str, np.ndarray], prediction_name: str) -> dict[str, float]:
    pred_key = f"transition__{prediction_name}"
    if pred_key not in arrays or not len(arrays[pred_key]):
        return {"transitions": 0}
    prediction = arrays[pred_key]
    ground_truth = arrays["transition__ground_truth"]
    prediction_mean = prediction.mean(0)
    ground_truth_mean = ground_truth.mean(0)
    return {
        "transitions": int(len(prediction)),
        "PJ": float(prediction_mean.max()),
        "gt_PJ": float(ground_truth_mean.max()),
        "PJ_gap": float(abs(prediction_mean.max() - ground_truth_mean.max())),
        "AUJ_gap_to_cohort_GT": float(np.abs(prediction_mean - ground_truth_mean).sum()),
        "jerk_area": float(prediction_mean.sum()),
        "gt_jerk_area": float(ground_truth_mean.sum()),
    }
