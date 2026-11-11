"""Explicit BEAT2 splits and invertible, Y-up local SMPL-X representation.

No axis search, clipping, validation statistics, or ground-truth hand filling.
The independent cache contains complete clips, including short clips.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation

PERSON_IDS = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18,
              20, 21, 22, 23, 24, 25, 27, 28, 30)
EXPECTED_COUNTS = {"train": 1093, "val": 106, "test": 265}
# The official 25-person dataset does not contain validation clips for 6/22.
# Do not invent validation rows or redraw its split to force 25 per split.
EXPECTED_SPLIT_PERSON_IDS = {"train": set(PERSON_IDS),
                            "val": set(PERSON_IDS) - {6, 22},
                            "test": set(PERSON_IDS)}


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(raw_root, output):
    raw_root = Path(raw_root).resolve()
    csv_path = raw_root / "train_test_split.csv"
    splits = {key: [] for key in EXPECTED_COUNTS}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            ident = row["id"]
            if int(ident.split("_", 1)[0]) in PERSON_IDS and row["type"] in splits:
                splits[row["type"]].append(ident)
    all_ids = []
    for split, ids in splits.items():
        assert len(ids) == len(set(ids)) == EXPECTED_COUNTS[split], (split, len(ids))
        assert {int(i.split("_", 1)[0]) for i in ids} == EXPECTED_SPLIT_PERSON_IDS[split], split
        for ident in ids:
            assert (raw_root / "smplxflame_30" / (ident + ".npz")).is_file(), ident
            assert (raw_root / "wave16k" / (ident + ".wav")).is_file(), ident
            assert (raw_root / "textgrid" / (ident + ".TextGrid")).is_file(), ident
        all_ids.extend(ids)
    assert len(all_ids) == len(set(all_ids)), "Cross-split leakage"
    value = {"schema": 1, "protocol": "semtalk25-val-selected",
             "raw_data_root": str(raw_root), "split_csv_sha256": sha256(csv_path),
             "person_ids": PERSON_IDS, "splits": splits, "additional_data": False,
             "speaker_coverage": {split: sorted(EXPECTED_SPLIT_PERSON_IDS[split]) for split in splits},
             "official_validation_absent_speakers": [6, 22],
             "fps": 30, "rotation_space": "SMPL-X local", "height_axis": "y",
             "root_representation": "per-frame dx,dz; absolute y; not velocity/second",
             "train_frames": 240, "train_stride": 120,
             "train_window_policy": "All complete clips; pad short clips with valid-frame masks",
             "test_selection": False, "statistics_split": "train"}
    atomic_json(output, value)
    return value


def encode_motion(poses, translation, mode="full55"):
    poses, translation = np.asarray(poses), np.asarray(translation)
    assert poses.ndim == 2 and poses.shape[1] == 165
    assert translation.shape == (len(poses), 3)
    assert np.isfinite(poses).all() and np.isfinite(translation).all()
    joints = 55 if mode == "full55" else 22
    assert mode in ("full55", "body22")
    matrices = Rotation.from_rotvec(poses.reshape(-1, 3)).as_matrix().reshape(-1, 55, 3, 3)
    # PyTorch3D/SemTalk rotation6D uses the first TWO ROWS, not columns.
    rotations = matrices[:, :joints, :2, :].reshape(len(poses), -1)
    delta = np.diff(translation[:, (0, 2)], axis=0, prepend=translation[:1, (0, 2)])
    root = np.column_stack((delta[:, 0], delta[:, 1], translation[:, 1]))
    return np.concatenate((rotations, root), axis=-1).astype(np.float32)


def decode_motion(value, mode="full55", initial_xz=(0., 0.)):
    value = np.asarray(value, dtype=np.float64)
    joints = 55 if mode == "full55" else 22
    assert mode in ("full55", "body22") and value.shape == (len(value), joints * 6 + 3)
    assert np.isfinite(value).all()
    d6 = value[:, :-3].reshape(-1, joints, 6)
    a, b = d6[..., :3], d6[..., 3:]
    na = np.linalg.norm(a, axis=-1, keepdims=True)
    if (na < 1e-10).any():
        raise ValueError("Degenerate 6D first vector; no silent identity replacement")
    a = a / na
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    nb = np.linalg.norm(b, axis=-1, keepdims=True)
    if (nb < 1e-10).any():
        raise ValueError("Degenerate 6D parallel vectors; no silent replacement")
    b = b / nb
    matrices = np.stack((a, b, np.cross(a, b)), axis=-2)
    poses = np.zeros((len(value), 55, 3), dtype=np.float64)
    poses[:, :joints] = Rotation.from_matrix(matrices.reshape(-1, 3, 3)).as_rotvec().reshape(-1, joints, 3)
    delta_xz = value[:, -3:-1].copy()
    delta_xz[0] = 0.  # The first generated horizontal delta never shifts origin.
    trans = np.column_stack((np.cumsum(delta_xz[:, 0]) + initial_xz[0],
                             value[:, -1], np.cumsum(delta_xz[:, 1]) + initial_xz[1]))
    return poses.reshape(-1, 165).astype(np.float32), trans.astype(np.float32)


class Standardizer:
    def __init__(self, mean, std):
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        assert self.mean.shape == self.std.shape and (self.std > 0).all()
        assert np.isfinite(self.mean).all() and np.isfinite(self.std).all()

    def normalize(self, value):
        return (value - self.mean) / self.std

    def denormalize(self, value):
        # Deliberately NO normalized [-1,1] clamp or quantile clipping.
        return value * self.std + self.mean

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls(z["mean"], z["std"])


def train_statistics(manifest, cache_root, mode, output):
    count, total, square = 0, None, None
    for ident in manifest["splits"]["train"]:
        with np.load(Path(cache_root) / (ident + ".npz"), allow_pickle=False) as z:
            assert str(z["source_id"].item()) == ident
            x = encode_motion(z["poses"], z["trans"], mode).astype(np.float64)
        total = x.sum(0) if total is None else total + x.sum(0)
        square = (x*x).sum(0) if square is None else square + (x*x).sum(0)
        count += len(x)
    mean = total / count
    std = np.maximum(np.sqrt(np.maximum(square / count - mean * mean, 0)), 1e-4)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.with_suffix(".tmp").open("wb") as handle:
        np.savez(handle, mean=mean.astype(np.float32), std=std.astype(np.float32),
                 frame_count=count, split=np.asarray("train"), mode=np.asarray(mode),
                 split_csv_sha256=np.asarray(manifest["split_csv_sha256"]))
    output.with_suffix(".tmp").replace(output)
    return Standardizer(mean, std)


def windows(frames, length=240, stride=120):
    if frames <= length:
        return [0]
    starts = list(range(0, frames-length+1, stride))
    if starts[-1] != frames-length:
        starts.append(frames-length)
    return starts


class ClipWindowDataset:
    """Duck-typed torch Dataset; consumes audited complete-clip feature caches."""
    def __init__(self, manifest, cache_root, normalizer, mode, length=240, stride=120):
        self.cache_root, self.normalizer, self.mode = Path(cache_root), normalizer, mode
        self.length, self.index = length, []
        for ident in manifest["splits"]["train"]:
            with np.load(self.cache_root / (ident + ".npz"), allow_pickle=False) as z:
                frames = len(z["poses"])
                assert z["audio"].shape == (frames, 1024), ident
                assert z["text"].shape == (frames, 300), ident
                assert z["rest_joints"].shape == (55, 3), ident
                assert z["pose_mean"].shape == (165,), ident
                assert str(z["geometry_schema"].item()).startswith("unposed-shaped-rest-v2"), ident
                assert str(z["source_id"].item()) == ident
                assert frames > 0, ident
            self.index.extend((ident, start, frames) for start in windows(frames, length, stride))

    def __len__(self):
        return len(self.index)

    @lru_cache(maxsize=2)
    def _read(self, ident):
        with np.load(self.cache_root / (ident + ".npz"), allow_pickle=False) as z:
            return {key: z[key].copy() for key in ("poses", "trans", "audio", "text", "rest_joints")}

    def __getitem__(self, item):
        ident, start, frames = self.index[item]
        z = self._read(ident)
        usable = min(self.length, frames-start)
        motion = encode_motion(z["poses"][start:start+usable], z["trans"][start:start+usable], self.mode)
        arrays = {"motion": self.normalizer.normalize(motion),
                  "audio": z["audio"][start:start+usable].astype(np.float32),
                  "text": z["text"][start:start+usable].astype(np.float32)}
        for key, array in arrays.items():
            arrays[key] = np.pad(array, ((0, self.length-usable), (0, 0)), mode="edge").astype(np.float32)
        arrays.update(person_id=np.int64(ident.split("_", 1)[0]),
                      padding_mask=np.arange(self.length) < usable,
                      rest_joints=z["rest_joints"].astype(np.float32))
        return arrays


def roundtrip_audit(poses, trans, standardizer, mode="full55"):
    physical = encode_motion(poses, trans, mode)
    restored = standardizer.denormalize(standardizer.normalize(physical))
    poses_back, trans_back = decode_motion(restored, mode, trans[0, (0, 2)])
    n = 55 if mode == "full55" else 22
    original = Rotation.from_rotvec(np.asarray(poses)[:, :n*3].reshape(-1, 3)).as_matrix()
    reconstructed = Rotation.from_rotvec(poses_back[:, :n*3].reshape(-1, 3)).as_matrix()
    result = {"standardizer_max_error": float(np.max(np.abs(restored-physical))),
              "rotation_matrix_max_error": float(np.max(np.abs(original-reconstructed))),
              "root_xyz_max_error": float(np.max(np.abs(trans_back-trans))),
              "height_axis": "y", "rot6d_convention": "first_two_rows",
              "clamping": False, "mode": mode}
    assert result["standardizer_max_error"] < 2e-5, result
    assert result["rotation_matrix_max_error"] < 3e-5, result
    assert result["root_xyz_max_error"] < 2e-4, result
    return result
