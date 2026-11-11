from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np


METHODS = [
    ("gt", "Ground truth", (50, 170, 60)),
    ("flowermotion_1step", "FlowerMotion 1-step", (34, 88, 230)),
    ("flowermotion_2step", "FlowerMotion 2-step", (36, 130, 235)),
    ("flowermotion_4step", "FlowerMotion 4-step", (24, 165, 245)),
    ("flooddiffusion", "FloodDiffusion", (190, 70, 190)),
    ("flowmdm", "FlowMDM", (180, 75, 110)),
    ("dart", "DART", (190, 115, 50)),
    ("motionstreamer", "MotionStreamer", (185, 145, 35)),
    ("doubletake", "DoubleTake", (150, 155, 25)),
    ("mdm", "MDM [concat]", (80, 145, 25)),
    ("momask", "MoMask [concat]", (30, 150, 110)),
    ("mld", "MLD [concat]", (25, 145, 165)),
    ("t2mgpt", "T2M-GPT [concat]", (30, 120, 205)),
    ("motiongpt", "MotionGPT [concat]", (70, 95, 215)),
    ("motionlcm", "MotionLCM-1 [concat]", (125, 80, 205)),
]

CHAINS = [
    [0, 2, 5, 8, 11],
    [0, 1, 4, 7, 10],
    [0, 3, 6, 9, 12, 15],
    [9, 14, 17, 19, 21],
    [9, 13, 16, 18, 20],
]
CHAIN_COLORS = [(30, 130, 255), (235, 175, 45), (75, 75, 220), (75, 185, 90), (180, 90, 195)]
PANEL_W = 480
PANEL_H = 235
CANVAS_W = 1920
CANVAS_H = 1080
GRID_H = PANEL_H * 4
FPS = 20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render 15-way BABEL-long comparisons.")
    parser.add_argument(
        "--motion-root", type=Path, default=Path("eval/babel_long_v2/motions")
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("eval/babel_long_v2/visualization/selected_20.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("eval/babel_long_v2/visualization/selected_20"),
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def archive_path(root: Path, method: str, episode_id: str) -> Path:
    paths = list((root / method).glob(f"{episode_id}--*.npz"))
    if len(paths) != 1:
        raise RuntimeError(f"{method}/{episode_id}: expected one archive, found {len(paths)}")
    return paths[0]


def load_case(root: Path, episode_id: str) -> tuple[dict[str, np.ndarray], dict]:
    motions = {}
    reference = None
    for method, _, _ in METHODS:
        path = archive_path(root, method, episode_id)
        with np.load(path, allow_pickle=False) as data:
            joints = np.asarray(data["joints_y_up"], dtype=np.float32)
            captions = [str(value) for value in data["captions"].tolist()]
            starts = np.asarray(data["segment_starts"], dtype=np.int64)
            ends = np.asarray(data["segment_ends"], dtype=np.int64)
            metadata = json.loads(str(data["metadata_json"].item()))
        signature = (len(joints), tuple(captions), tuple(starts), tuple(ends))
        if reference is None:
            reference = signature
        elif signature != reference:
            raise RuntimeError(f"{method}/{episode_id}: length, captions, or boundaries differ")
        if not np.isfinite(joints).all():
            raise RuntimeError(f"{method}/{episode_id}: non-finite joints")
        motions[method] = joints
    assert reference is not None
    length, captions, starts, ends = reference
    return motions, {
        "episode_id": episode_id,
        "length": int(length),
        "captions": list(captions),
        "starts": np.asarray(starts),
        "ends": np.asarray(ends),
        "fps": int(metadata.get("fps", FPS)),
    }


def project(points: np.ndarray, root: np.ndarray, left: int, top: int) -> np.ndarray:
    relative = points - root[None]
    horizontal = 0.78 * relative[:, 0] - 0.48 * relative[:, 2]
    vertical = relative[:, 1] + 0.16 * (relative[:, 0] + relative[:, 2])
    scale = 78.0
    x = left + PANEL_W // 2 + horizontal * scale
    y = top + PANEL_H - 32 - vertical * scale
    return np.stack([x, y], axis=1).round().astype(np.int32)


def draw_floor(image: np.ndarray, root: np.ndarray, left: int, top: int) -> None:
    for offset in (-2, -1, 0, 1, 2):
        a = root + np.asarray([-3.0, 0.0, float(offset)])
        b = root + np.asarray([3.0, 0.0, float(offset)])
        pts = project(np.stack([a, b]), root, left, top)
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), (220, 224, 229), 1, cv2.LINE_AA)
        a = root + np.asarray([float(offset), 0.0, -3.0])
        b = root + np.asarray([float(offset), 0.0, 3.0])
        pts = project(np.stack([a, b]), root, left, top)
        cv2.line(image, tuple(pts[0]), tuple(pts[1]), (220, 224, 229), 1, cv2.LINE_AA)


def draw_trajectory(
    image: np.ndarray,
    roots: np.ndarray,
    frame: int,
    common_radius: float,
    left: int,
    top: int,
    color: tuple[int, int, int],
) -> None:
    x0, y0, w, h = left + PANEL_W - 112, top + PANEL_H - 78, 98, 62
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (244, 246, 248), -1)
    cv2.rectangle(image, (x0, y0), (x0 + w, y0 + h), (190, 196, 203), 1)
    relative = roots[: frame + 1, (0, 2)] - roots[0, (0, 2)]
    relative = np.clip(relative / common_radius, -1.0, 1.0)
    px = x0 + w / 2 + relative[:, 0] * (w / 2 - 5)
    py = y0 + h / 2 - relative[:, 1] * (h / 2 - 5)
    points = np.stack([px, py], axis=1).round().astype(np.int32)
    if len(points) > 1:
        cv2.polylines(image, [points], False, color, 2, cv2.LINE_AA)
    cv2.circle(image, tuple(points[-1]), 3, (20, 20, 20), -1, cv2.LINE_AA)


def draw_panel(
    image: np.ndarray,
    joints: np.ndarray,
    roots: np.ndarray,
    frame: int,
    row: int,
    col: int,
    title: str,
    accent: tuple[int, int, int],
    common_radius: float,
) -> None:
    left, top = col * PANEL_W, row * PANEL_H
    image[top : top + PANEL_H, left : left + PANEL_W] = (250, 250, 249)
    cv2.rectangle(image, (left, top), (left + PANEL_W - 1, top + PANEL_H - 1), (200, 204, 210), 1)
    cv2.rectangle(image, (left, top), (left + 7, top + PANEL_H), accent, -1)
    cv2.putText(image, title, (left + 17, top + 24), cv2.FONT_HERSHEY_DUPLEX, 0.55, (30, 32, 36), 1, cv2.LINE_AA)
    current_root = roots[frame].copy()
    current_root[1] = 0.0
    draw_floor(image, current_root, left, top)
    projected = project(joints[frame], current_root, left, top)
    for chain, chain_color in zip(CHAINS, CHAIN_COLORS):
        cv2.polylines(image, [projected[np.asarray(chain)]], False, chain_color, 3, cv2.LINE_AA)
    for point in projected:
        cv2.circle(image, tuple(point), 2, (30, 33, 38), -1, cv2.LINE_AA)
    draw_trajectory(image, roots, frame, common_radius, left, top, accent)


def draw_legend_panel(image: np.ndarray) -> None:
    left, top = 3 * PANEL_W, 3 * PANEL_H
    image[top : top + PANEL_H, left : left + PANEL_W] = (243, 246, 250)
    cv2.rectangle(image, (left, top), (left + PANEL_W - 1, top + PANEL_H - 1), (190, 198, 208), 1)
    lines = [
        "LONG-SEQUENCE PROTOCOL",
        "FlowerMotion: one-shot whole episode",
        "Native long: Flood / FlowMDM / DART",
        "MotionStreamer / DoubleTake",
        "[concat]: independent text segments",
        "All panels: 20 fps, same frame & floor",
        "Inset: root trajectory, shared scale",
    ]
    for index, line in enumerate(lines):
        scale = 0.55 if index == 0 else 0.47
        color = (35, 55, 90) if index == 0 else (65, 70, 78)
        cv2.putText(image, line, (left + 24, top + 36 + index * 27), cv2.FONT_HERSHEY_DUPLEX, scale, color, 1, cv2.LINE_AA)


def draw_footer(image: np.ndarray, metadata: dict, frame: int) -> None:
    top = GRID_H
    image[top:] = (24, 27, 34)
    starts, ends = metadata["starts"], metadata["ends"]
    segment = int(np.searchsorted(ends, frame, side="right"))
    segment = min(segment, len(ends) - 1)
    caption = metadata["captions"][segment]
    cv2.putText(
        image,
        f"{metadata['episode_id']}   {frame / FPS:05.2f}s / {metadata['length'] / FPS:05.2f}s   text {segment + 1}/{len(ends)}",
        (28, top + 34),
        cv2.FONT_HERSHEY_DUPLEX,
        0.62,
        (205, 211, 220),
        1,
        cv2.LINE_AA,
    )
    display_caption = caption if len(caption) <= 115 else caption[:112] + "..."
    cv2.putText(image, display_caption, (28, top + 69), cv2.FONT_HERSHEY_DUPLEX, 0.72, (255, 255, 255), 1, cv2.LINE_AA)
    x0, x1, y0, y1 = 28, CANVAS_W - 28, top + 96, top + 116
    palette = [(57, 168, 255), (83, 196, 123), (235, 160, 69), (172, 111, 225), (220, 105, 105)]
    for index, (start, end) in enumerate(zip(starts, ends)):
        a = x0 + int((x1 - x0) * start / metadata["length"])
        b = x0 + int((x1 - x0) * end / metadata["length"])
        cv2.rectangle(image, (a, y0), (max(a + 1, b), y1), palette[index % len(palette)], -1)
    marker = x0 + int((x1 - x0) * frame / max(metadata["length"] - 1, 1))
    cv2.line(image, (marker, y0 - 5), (marker, y1 + 5), (255, 255, 255), 3, cv2.LINE_AA)


def render_case(output: Path, motions: dict[str, np.ndarray], metadata: dict) -> None:
    roots_by_method = {method: joints[:, 0] for method, joints in motions.items()}
    max_radius = 0.0
    for roots in roots_by_method.values():
        displacement = roots[:, (0, 2)] - roots[0, (0, 2)]
        max_radius = max(max_radius, float(np.abs(displacement).max()))
    common_radius = max(2.0, min(25.0, max_radius * 1.05))
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s:v", f"{CANVAS_W}x{CANVAS_H}",
        "-r", str(FPS), "-i", "-", "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    preview = None
    try:
        for frame in range(metadata["length"]):
            image = np.empty((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
            for index, (method, title, accent) in enumerate(METHODS):
                draw_panel(
                    image,
                    motions[method],
                    roots_by_method[method],
                    frame,
                    index // 4,
                    index % 4,
                    title,
                    accent,
                    common_radius,
                )
            draw_legend_panel(image)
            draw_footer(image, metadata, frame)
            if frame == metadata["length"] // 3:
                preview = image.copy()
            assert process.stdin is not None
            process.stdin.write(image.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError(f"ffmpeg failed for {output}")
    if preview is not None:
        cv2.imwrite(str(output.with_suffix(".jpg")), preview, [cv2.IMWRITE_JPEG_QUALITY, 92])


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard")
    root = Path(__file__).resolve().parent
    motion_root = (root / args.motion_root).resolve()
    output_root = (root / args.output_root).resolve()
    manifest = json.loads((root / args.selection).resolve().read_text())
    cases = manifest["cases"][args.shard_index :: args.num_shards]
    for index, case in enumerate(cases, start=1):
        episode_id = str(case["episode_id"])
        output = output_root / f"{episode_id}_15way.mp4"
        if output.exists() and not args.overwrite:
            print(f"[render shard {args.shard_index}] skip {index}/{len(cases)} {episode_id}", flush=True)
            continue
        motions, metadata = load_case(motion_root, episode_id)
        render_case(output, motions, metadata)
        print(f"[render shard {args.shard_index}] {index}/{len(cases)} {output}", flush=True)


if __name__ == "__main__":
    main()
