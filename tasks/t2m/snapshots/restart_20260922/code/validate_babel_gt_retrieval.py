from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from t2m_long.babel_protocol import effective_segments
from t2m_long.data import MotionTextDataset, collate_motion_text
from t2m_long.evaluator import FloodT2MEvaluator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flood-root", type=Path, default=Path("third_party/FloodDiffusion"))
    parser.add_argument("--embedding-root", type=Path, default=Path("data/umt5_xxl_sentence_features"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_gt_protocol_check.json"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


class SegmentBuffer:
    def __init__(self, evaluator: FloodT2MEvaluator, batch_size: int):
        self.evaluator = evaluator
        self.batch_size = batch_size
        self.motions: list[torch.Tensor] = []
        self.tokens: list[list[str]] = []

    def add(self, motion: torch.Tensor, tokens: list[str]) -> None:
        if len(motion) < 4:
            return
        self.motions.append(motion.cpu())
        self.tokens.append(tokens)
        if len(self.motions) >= self.batch_size:
            self.flush()

    def flush(self) -> None:
        if not self.motions:
            return
        lengths = torch.tensor([len(motion) for motion in self.motions])
        padded = pad_sequence(self.motions, batch_first=True)
        self.evaluator.add_batch(padded, padded, lengths, self.tokens)
        self.motions.clear()
        self.tokens.clear()


def combine(single: FloodT2MEvaluator, multi: FloodT2MEvaluator) -> FloodT2MEvaluator:
    combined = single
    combined.prediction_embeddings += multi.prediction_embeddings
    combined.ground_truth_embeddings += multi.ground_truth_embeddings
    combined.text_embeddings += multi.text_embeddings
    return combined


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flood_root = (root / args.flood_root).resolve()
    data_root = flood_root / "raw_data"
    dataset = MotionTextDataset(
        data_root, root / args.embedding_root, "babel", "val", max_length=200
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_motion_text,
    )
    device = torch.device("cuda")
    mean = torch.from_numpy(np.load(data_root / "HumanML3D/Mean.npy"))
    std = torch.from_numpy(np.load(data_root / "HumanML3D/Std.npy"))
    single_evaluator = FloodT2MEvaluator(flood_root, device, seed=42)
    multi_evaluator = FloodT2MEvaluator(flood_root, device, seed=42)
    single_buffer = SegmentBuffer(single_evaluator, args.batch_size)
    multi_buffer = SegmentBuffer(multi_evaluator, args.batch_size)
    sample_counts = {"single": 0, "multi": 0}
    segment_counts = {"single": 0, "multi": 0}

    for batch in tqdm(loader, desc="GT retrieval protocol check"):
        raw = batch["motion"] * std + mean
        for index, length_value in enumerate(batch["lengths"].tolist()):
            segments = effective_segments(batch["text_ends"][index], length_value)
            category = "multi" if len(segments) >= 2 else "single"
            sample_counts[category] += 1
            buffer = multi_buffer if category == "multi" else single_buffer
            for segment in segments:
                buffer.add(
                    raw[index, segment.start : segment.end],
                    batch["text_tokens"][index][segment.annotation_index],
                )
                if segment.end - segment.start >= 4:
                    segment_counts[category] += 1
    single_buffer.flush()
    multi_buffer.flush()

    single_metrics = single_evaluator.compute()
    multi_metrics = multi_evaluator.compute()
    all_metrics = combine(single_evaluator, multi_evaluator).compute()
    record = {
        "definition": {
            "single": "samples with exactly one effective text segment inside the 200-frame crop",
            "multi": "each effective segment from samples with at least two visible text segments",
            "all": "union of single and multi segments",
        },
        "sample_counts": sample_counts,
        "segment_counts": segment_counts,
        "single": single_metrics,
        "multi": multi_metrics,
        "all": all_metrics,
    }
    output = (root / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2))
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
