from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT / "third_party/FloodDiffusion"))
from utils.motion_process import recover_joint_positions_263


CHAINS = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
COLORS = ["#ef8a17", "#00a7d8", "#154c89", "#e8a500", "#00ae78"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flower-root", type=Path, default=Path("eval/babel_protocol_flower/raw"))
    parser.add_argument("--flood-root", type=Path, default=Path("eval/babel_protocol_flood/raw"))
    parser.add_argument("--output", type=Path, default=Path("eval/babel_fourway_videos"))
    parser.add_argument("--fps", type=int, default=20)
    return parser.parse_args()


def recover(path: Path) -> np.ndarray:
    joints = recover_joint_positions_263(np.load(path).astype(np.float32), 22)
    joints[..., 1] -= joints[..., 1].min()
    return joints


def active_caption(metadata: dict, frame: int) -> str:
    for segment in metadata["effective_segments"]:
        if segment["start"] <= frame < segment["end"]:
            return metadata["captions"][segment["annotation_index"]]
    return metadata["captions"][-1]


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    flower_root = (root / args.flower_root).resolve()
    flood_root = (root / args.flood_root).resolve()
    output = (root / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    names = sorted(
        path.name for path in flower_root.iterdir()
        if path.is_dir() and (flood_root / path.name / "flood_10step.npy").exists()
    )
    if not names:
        raise RuntimeError("No selected samples are present in both generation outputs")

    for name in names:
        sample_dir = flower_root / name
        metadata = json.loads((sample_dir / "metadata.json").read_text())
        series = [
            ("Ground truth", recover(sample_dir / "gt.npy")),
            ("FlowerDance++ 1-step", recover(sample_dir / "flower_1step.npy")),
            ("FlowerDance++ 4-step", recover(sample_dir / "flower_4step.npy")),
            ("FloodDiffusion 10-step", recover(flood_root / name / "flood_10step.npy")),
        ]
        length = min(len(joints) for _, joints in series)
        height = max(float(np.ptp(joints[:length, :, 1])) for _, joints in series)
        radius = max(1.0, height * 0.75)
        fig = plt.figure(figsize=(16, 5), dpi=110, facecolor="white")
        axes = [fig.add_subplot(1, 4, index + 1, projection="3d") for index in range(4)]
        lines = []
        for axis, (title, _) in zip(axes, series):
            axis.set_title(title, fontsize=12, fontweight="bold")
            axis.view_init(elev=16, azim=-125)
            axis.set_box_aspect((1, 1, 1.3))
            axis.set_axis_off()
            per_axis = []
            for color in COLORS:
                line, = axis.plot([], [], [], color=color, linewidth=3.0, solid_capstyle="round")
                per_axis.append(line)
            lines.append(per_axis)
        caption_text = fig.text(0.5, 0.035, "", ha="center", va="center", fontsize=11)
        fig.suptitle(f"BABEL {name}: synchronized comparison", fontsize=15, fontweight="bold")
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.12, wspace=0.01)

        def draw(frame: int) -> None:
            caption = active_caption(metadata, frame)
            caption_text.set_text(
                f"t={frame / args.fps:4.1f}s  |  text: {textwrap.shorten(caption, width=145, placeholder='…')}"
            )
            for axis, (_, joints), per_axis in zip(axes, series, lines):
                pose = joints[frame]
                root_x, root_z = pose[0, 0], pose[0, 2]
                axis.set_xlim(root_x - radius, root_x + radius)
                axis.set_ylim(root_z - radius, root_z + radius)
                axis.set_zlim(0.0, radius * 2.0)
                for line, chain in zip(per_axis, CHAINS):
                    points = pose[chain]
                    line.set_data(points[:, 0], points[:, 2])
                    line.set_3d_properties(points[:, 1])

        writer = FFMpegWriter(fps=args.fps, codec="libx264", bitrate=3500,
                              extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
        out_path = output / f"{name}_gt_vs_flower1_vs_flower4_vs_flood.mp4"
        with writer.saving(fig, str(out_path), dpi=110):
            for frame in range(length):
                draw(frame)
                writer.grab_frame()
        transitions = [item["start"] for item in metadata["effective_segments"][1:]]
        preview_frame = min(length - 1, (transitions[0] + 5) if transitions else length // 2)
        draw(preview_frame)
        fig.savefig(output / f"{name}_preview.png", dpi=130, facecolor="white")
        plt.close(fig)
        print(out_path, flush=True)


if __name__ == "__main__":
    main()
