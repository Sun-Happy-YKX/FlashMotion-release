from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import ConcatDataset, Dataset


def _read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def _parse_text(path: Path) -> list[tuple[str, list[str], float, float]]:
    result = []
    for raw in path.read_text().splitlines():
        caption, tokens, start, end = raw.split("#")
        result.append((caption, tokens.split(), float(start), float(end)))
    if not result:
        raise RuntimeError(f"No text annotations in {path}")
    return result


class TextEmbeddingStore:
    def __init__(self, root: str | Path):
        root = Path(root)
        self.embeddings = np.load(root / "embeddings.npy", mmap_mode="r")
        self.index = json.loads((root / "text_index.json").read_text())
        if self.embeddings.ndim != 2:
            raise ValueError(f"Expected a 2D embedding bank, got {self.embeddings.shape}")

    @property
    def dim(self) -> int:
        return int(self.embeddings.shape[1])

    def get(self, caption: str) -> np.ndarray:
        try:
            index = self.index[caption]
        except KeyError as exc:
            raise KeyError(f"Caption missing from precomputed embedding bank: {caption!r}") from exc
        return self.embeddings[index]


class MotionTextDataset(Dataset):
    """HumanML3D or BABEL in the common raw 263D representation."""

    def __init__(
        self,
        data_root: str | Path,
        embedding_root: str | Path,
        dataset: str,
        split: str,
        max_length: int | None = 200,
        multi_action_only: bool = False,
    ):
        self.data_root = Path(data_root)
        self.dataset = dataset
        self.split = split
        self.max_length = None if max_length is None or int(max_length) <= 0 else int(max_length)
        self.multi_action_only = bool(multi_action_only)
        self.embedding_store = TextEmbeddingStore(embedding_root)

        hml_root = self.data_root / "HumanML3D"
        self.mean = np.load(hml_root / "Mean.npy").astype(np.float32)
        self.std = np.load(hml_root / "Std.npy").astype(np.float32)
        if self.mean.shape != (263,) or self.std.shape != (263,) or np.any(self.std <= 0):
            raise ValueError("Invalid HumanML3D Mean/Std")

        if dataset == "humanml3d":
            self.root = hml_root
            self.motion_dir = self.root / "new_joint_vecs"
            split_file = {"train": "train.txt", "val": "val.txt", "test": "test.txt"}[split]
        elif dataset == "babel":
            self.root = self.data_root / "BABEL_streamed"
            self.motion_dir = self.root / "motions"
            split_file = {"train": "train_processed.txt", "val": "val_processed.txt"}[split]
        else:
            raise ValueError(f"Unknown dataset: {dataset}")

        self.text_dir = self.root / "texts"
        ids = _read_ids(self.root / split_file)
        if self.multi_action_only:
            if dataset != "babel":
                raise ValueError("multi_action_only is defined only for BABEL")
            ids = [name for name in ids if len(_parse_text(self.text_dir / f"{name}.txt")) >= 2]
        self.ids = ids
        if not self.ids:
            raise RuntimeError(f"No samples for {dataset}/{split}")

    def __len__(self) -> int:
        return len(self.ids)

    def _frame_condition(
        self, annotations: Iterable[tuple[str, list[str], float, float]], length: int
    ) -> tuple[np.ndarray, list[str], list[list[str]], list[int]]:
        annotations = list(annotations)
        captions = [item[0] for item in annotations]
        tokens = [item[1] for item in annotations]
        if self.dataset == "humanml3d":
            choice = random.randrange(len(annotations))
            captions = [captions[choice]]
            tokens = [tokens[choice]]
            ends = [length]
        else:
            # Match FloodDiffusion: ignore overlapping f_tag values and use the
            # ordered to_tag boundaries, clamped to the actual sequence length.
            ends = [min(length, int(item[3] * 20.0 + 0.5)) for item in annotations]

        condition = np.zeros((length, self.embedding_store.dim), dtype=np.float16)
        previous = 0
        for caption, end in zip(captions, ends):
            end = max(previous, min(length, end))
            if end > previous:
                condition[previous:end] = self.embedding_store.get(caption)
            previous = end
        if previous < length:
            condition[previous:] = self.embedding_store.get(captions[-1])
        return condition, captions, tokens, ends

    def __getitem__(self, index: int) -> dict:
        name = self.ids[index]
        motion = np.load(self.motion_dir / f"{name}.npy").astype(np.float32)
        if motion.ndim != 2 or motion.shape[1] != 263 or not np.isfinite(motion).all():
            raise RuntimeError(f"Invalid motion {self.dataset}/{name}: {motion.shape}")
        if self.max_length is not None:
            motion = motion[: self.max_length]
        length = int(motion.shape[0])
        annotations = _parse_text(self.text_dir / f"{name}.txt")
        condition, captions, tokens, ends = self._frame_condition(annotations, length)
        motion = (motion - self.mean) / self.std
        return {
            "motion": torch.from_numpy(motion),
            "condition": torch.from_numpy(condition),
            "length": length,
            # ``None`` means the evaluation protocol preserves the complete
            # episode. Keep the actual length so collation pads concretely.
            "max_length": length if self.max_length is None else self.max_length,
            "name": name,
            "dataset": self.dataset,
            "captions": captions,
            "text_tokens": tokens,
            "text_ends": ends,
        }


def collate_motion_text(batch: list[dict]) -> dict:
    motions = pad_sequence([item["motion"] for item in batch], batch_first=True, padding_value=0.0)
    conditions = pad_sequence(
        [item["condition"] for item in batch], batch_first=True, padding_value=0.0
    )
    # The backbone is bidirectional. A fixed padded horizon keeps the result of
    # one sample independent of whichever other sequence happens to share its
    # batch, while the model-side mask prevents padding leakage.
    padded_length = max(item["max_length"] for item in batch)
    motions = torch.nn.functional.pad(motions, (0, 0, 0, padded_length - motions.shape[1]))
    conditions = torch.nn.functional.pad(
        conditions, (0, 0, 0, padded_length - conditions.shape[1])
    )
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)
    steps = torch.arange(motions.shape[1])[None, :]
    mask = steps < lengths[:, None]
    return {
        "motion": motions,
        "condition": conditions,
        "mask": mask,
        "lengths": lengths,
        "name": [item["name"] for item in batch],
        "dataset": [item["dataset"] for item in batch],
        "captions": [item["captions"] for item in batch],
        "text_tokens": [item["text_tokens"] for item in batch],
        "text_ends": [item["text_ends"] for item in batch],
    }


def build_joint_train_dataset(
    data_root: str | Path, embedding_root: str | Path, max_length: int = 200
) -> ConcatDataset:
    return ConcatDataset(
        [
            MotionTextDataset(data_root, embedding_root, "babel", "train", max_length),
            MotionTextDataset(data_root, embedding_root, "humanml3d", "train", max_length),
        ]
    )
