from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def read_ends(path: Path, fps: float = 20.0) -> list[int]:
    ends = []
    for line in path.read_text().splitlines():
        if line.strip():
            ends.append(int(float(line.split("#")[3]) * fps + 0.5))
    return ends


def fixed_window(values: np.ndarray, boundary: int, length: int) -> np.ndarray:
    left = length // 2
    right = length - left
    start = boundary - left
    end = boundary + right
    pad_left = max(0, -start)
    pad_right = max(0, end - len(values))
    window = values[max(start, 0) : min(end, len(values))]
    if pad_left or pad_right:
        window = np.pad(window, ((pad_left, pad_right), (0, 0)), mode="edge")
    assert len(window) == length
    return window


def feature_curve(values: np.ndarray, channels: slice) -> np.ndarray:
    return np.abs(np.diff(values[:, channels], n=3, axis=0)).max(axis=-1)


def summarize(curves: list[np.ndarray], baseline: float = 0.033031363) -> dict:
    array = np.stack(curves)
    mean_curve = array.mean(axis=0)
    return {
        "count": len(array),
        "peak_of_mean": float(mean_curve.max()),
        "AUJ_minus_javg": float(np.abs(mean_curve - baseline).sum()),
        "AUJ_raw": float(mean_curve.sum()),
        "mean_of_peak": float(array.max(axis=1).mean()),
        "mean_of_AUJ_raw": float(array.sum(axis=1).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path("third_party/FloodDiffusion/raw_data")
    )
    args = parser.parse_args()
    babel = args.root / "BABEL_streamed"
    hml = args.root / "HumanML3D"
    mean = np.load(hml / "Mean.npy")
    std = np.load(hml / "Std.npy")
    ids = [x.strip() for x in (babel / "val_processed.txt").read_text().splitlines() if x.strip()]

    channels = {
        "all_263": slice(0, 263),
        "pose_4_259": slice(4, 259),
        "local_velocity_193_259": slice(193, 259),
    }
    sequences = []
    for name in ids:
        text_path = babel / "texts" / f"{name}.txt"
        ends = read_ends(text_path)
        if len(ends) < 2:
            continue
        raw = np.load(babel / "motions" / f"{name}.npy")[:200]
        normalized = (raw - mean) / std
        boundary = min(max(0, ends[0]), len(normalized))
        sequences.append((normalized, boundary))

    for length in (20, 24, 27, 30, 32, 36, 40, 45, 50, 60):
        curves = {name: [] for name in channels}
        for normalized, boundary in sequences:
            window = fixed_window(normalized, boundary, length)
            for key, channel_slice in channels.items():
                curves[key].append(feature_curve(window, channel_slice))
        print(f"WINDOW={length}")
        for key, values in curves.items():
            print(key, summarize(values))


if __name__ == "__main__":
    main()
