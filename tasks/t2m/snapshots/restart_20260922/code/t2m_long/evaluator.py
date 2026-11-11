from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def stable_frechet_distance(
    mean_a: np.ndarray,
    covariance_a: np.ndarray,
    mean_b: np.ndarray,
    covariance_b: np.ndarray,
) -> float:
    """Compute the standard T2M FID with a stable symmetric PSD square root.

    This is algebraically the same Fréchet formula used by FloodDiffusion,
    but avoids scipy/numpy ``sqrtm`` failures on this runtime's LAPACK build.
    """

    mean_a_t = torch.as_tensor(mean_a, dtype=torch.float64)
    mean_b_t = torch.as_tensor(mean_b, dtype=torch.float64)
    covariance_a_t = torch.as_tensor(covariance_a, dtype=torch.float64)
    covariance_b_t = torch.as_tensor(covariance_b, dtype=torch.float64)
    covariance_a_t = (covariance_a_t + covariance_a_t.T) * 0.5
    covariance_b_t = (covariance_b_t + covariance_b_t.T) * 0.5

    eigenvalues_a, eigenvectors_a = torch.linalg.eigh(covariance_a_t)
    root_a = (
        eigenvectors_a
        * torch.sqrt(eigenvalues_a.clamp_min(0.0)).unsqueeze(0)
    ) @ eigenvectors_a.T
    middle = root_a @ covariance_b_t @ root_a
    middle = (middle + middle.T) * 0.5
    middle_eigenvalues = torch.linalg.eigvalsh(middle).clamp_min(0.0)

    value = (
        (mean_a_t - mean_b_t).square().sum()
        + torch.trace(covariance_a_t)
        + torch.trace(covariance_b_t)
        - 2.0 * torch.sqrt(middle_eigenvalues).sum()
    )
    # A tiny negative is possible from cancellation when both sets are equal.
    scale = (
        torch.trace(covariance_a_t).abs()
        + torch.trace(covariance_b_t).abs()
        + 1.0
    )
    if value < -1e-6 * scale:
        raise RuntimeError(f"Numerically invalid negative FID: {value.item()}")
    return float(value.clamp_min(0.0))


class FloodT2MEvaluator:
    """FloodDiffusion's released HumanML3D evaluator without Lightning wrappers."""

    def __init__(self, flood_root: str | Path, device: torch.device, seed: int = 42):
        flood_root = Path(flood_root).resolve()
        sys.path.insert(0, str(flood_root))
        from metrics.tools.t2m_evaluator import (
            MotionEncoderBiGRUCo,
            MovementConvEncoder,
            TextEncoderBiGRUCo,
        )
        from metrics.tools.word_vectorizer import WordVectorizer

        deps = flood_root / "deps"
        self.device = device
        self.seed = int(seed)
        self.metric_mean = torch.from_numpy(np.load(deps / "t2m/meta/mean.npy")).float().to(device)
        self.metric_std = torch.from_numpy(np.load(deps / "t2m/meta/std.npy")).float().to(device)
        self.word_vectorizer = WordVectorizer(str(deps / "glove"), "our_vab")

        self.text_encoder = TextEncoderBiGRUCo(300, 15, 512, 512).to(device)
        self.move_encoder = MovementConvEncoder(259, 512, 512).to(device)
        self.motion_encoder = MotionEncoderBiGRUCo(512, 1024, 512).to(device)
        self.text_encoder.load_state_dict(
            torch.load(deps / "t2m/humanml3d/text_encoder.pt", map_location="cpu")
        )
        self.move_encoder.load_state_dict(
            torch.load(deps / "t2m/humanml3d/movement_encoder.pt", map_location="cpu")
        )
        self.motion_encoder.load_state_dict(
            torch.load(deps / "t2m/humanml3d/motion_encoder.pt", map_location="cpu")
        )
        for module in (self.text_encoder, self.move_encoder, self.motion_encoder):
            module.eval().requires_grad_(False)
        self.prediction_embeddings: list[torch.Tensor] = []
        self.ground_truth_embeddings: list[torch.Tensor] = []
        self.text_embeddings: list[torch.Tensor] = []

    @torch.inference_mode()
    def _motion_embeddings(self, motions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(lengths, descending=True)
        inverse = torch.argsort(order)
        motions = motions[order].to(self.device).float()
        lengths = lengths[order].to(self.device)
        normalized = (motions - self.metric_mean) / self.metric_std
        movement = self.move_encoder(normalized[..., :-4])
        embeddings = self.motion_encoder(movement, torch.div(lengths, 4, rounding_mode="floor"))
        return embeddings[inverse]

    @torch.inference_mode()
    def _text_embeddings(self, token_lists: list[list[str]]) -> torch.Tensor:
        all_words = []
        all_pos = []
        all_lengths = []
        max_text_len = 20
        for source_tokens in token_lists:
            tokens = list(source_tokens)
            if len(tokens) < max_text_len:
                tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
                sentence_length = len(tokens)
                tokens += ["unk/OTHER"] * (max_text_len + 2 - sentence_length)
            else:
                tokens = ["sos/OTHER"] + tokens[:max_text_len] + ["eos/OTHER"]
                sentence_length = len(tokens)
            words, positions = zip(*(self.word_vectorizer[token] for token in tokens))
            all_words.append(np.stack(words).astype(np.float32))
            all_pos.append(np.stack(positions).astype(np.float32))
            all_lengths.append(sentence_length)
        word_tensor = torch.from_numpy(np.stack(all_words)).to(self.device)
        pos_tensor = torch.from_numpy(np.stack(all_pos)).to(self.device)
        length_tensor = torch.tensor(all_lengths, device=self.device)
        return self.text_encoder(word_tensor, pos_tensor, length_tensor)

    @torch.inference_mode()
    def add_batch(
        self,
        prediction_raw: torch.Tensor,
        ground_truth_raw: torch.Tensor,
        lengths: torch.Tensor,
        token_lists: list[list[str]],
        prediction_lengths: torch.Tensor | None = None,
        ground_truth_lengths: torch.Tensor | None = None,
    ) -> None:
        if prediction_lengths is None:
            prediction_lengths = lengths
        if ground_truth_lengths is None:
            ground_truth_lengths = lengths
        self.prediction_embeddings.append(
            self._motion_embeddings(prediction_raw, prediction_lengths).float().cpu()
        )
        self.ground_truth_embeddings.append(
            self._motion_embeddings(ground_truth_raw, ground_truth_lengths).float().cpu()
        )
        self.text_embeddings.append(self._text_embeddings(token_lists).float().cpu())

    def compute(self, r_size: int = 32, diversity_times: int = 300) -> dict[str, float]:
        from metrics.tools.utils import (
            calculate_activation_statistics_np,
            calculate_diversity_np,
            calculate_top_k,
            euclidean_distance_matrix,
        )

        predictions = torch.cat(self.prediction_embeddings)
        ground_truth = torch.cat(self.ground_truth_embeddings)
        texts = torch.cat(self.text_embeddings)
        count = len(predictions)
        generator = torch.Generator().manual_seed(self.seed)
        order = torch.randperm(count, generator=generator)
        predictions = predictions[order]
        ground_truth = ground_truth[order]
        texts = texts[order]

        usable = count // r_size * r_size
        result: dict[str, float] = {"samples": count}
        for label, motions in (("", predictions), ("gt_", ground_truth)):
            matching_sum = 0.0
            top_k_sum = torch.zeros(3)
            for start in range(0, usable, r_size):
                distances = euclidean_distance_matrix(
                    texts[start : start + r_size], motions[start : start + r_size]
                ).nan_to_num()
                matching_sum += float(torch.trace(distances))
                ranking = torch.argsort(distances, dim=1)
                top_k_sum += calculate_top_k(ranking, top_k=3).sum(0)
            result[f"{label}Matching_score"] = matching_sum / usable
            for k in range(3):
                result[f"{label}R_precision_top_{k + 1}"] = float(top_k_sum[k] / usable)

        prediction_np = predictions.numpy()
        ground_truth_np = ground_truth.numpy()
        pred_mean, pred_cov = calculate_activation_statistics_np(prediction_np)
        gt_mean, gt_cov = calculate_activation_statistics_np(ground_truth_np)
        result["FID"] = stable_frechet_distance(
            gt_mean, gt_cov, pred_mean, pred_cov
        )
        np.random.seed(self.seed)
        diversity_times = min(diversity_times, count - 1)
        result["Diversity"] = float(calculate_diversity_np(prediction_np, diversity_times))
        np.random.seed(self.seed)
        result["gt_Diversity"] = float(
            calculate_diversity_np(ground_truth_np, diversity_times)
        )
        return result
