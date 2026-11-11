"""Paired real SMPL-X mesh renders for fixed validation clips (Y-up).

Pyrender uses a Y-up camera directly. Rendering never rewrites evaluation
poses or optimizes a rotation to improve a score. GT expressions and predicted
expressions are both neutral because face quality is outside this experiment.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path

from data import atomic_json, sha256


COLOR_QUALITY_THRESHOLDS = {
    "person_seg_pixels_gt": 1000,
    "nonblack_rgb_mean_gt": 16.,
    "whole_nonblack_fraction_gt": .5,
    "person_nonblack_fraction_gt": .9,
    "normal_depth_positive_person_fraction_gt": .99,
}


def retained_color_quality(color, depth, segmentation, context=""):
    """Fail closed on the SAME retained NORMAL and person-only SEG draw.

    Pure CPU pixel checks, not semantic motion/face evaluation or human review.
    SEG/depth alone can be normal while the initial EGL color readback is zero.
    """
    import numpy as np

    def require(condition, message):
        if not condition:
            raise RuntimeError(f"SMPL-X retained color/depth/SEG quality failed [{context}]: {message}")

    color, depth, segmentation = map(np.asarray, (color, depth, segmentation))
    require(color.ndim == 3 and color.shape[-1] == 3 and min(color.shape[:2]) > 0
            and color.dtype == np.uint8, "NORMAL must be nonempty RGB uint8")
    require(segmentation.shape == color.shape and segmentation.dtype == np.uint8, "SEG must match NORMAL RGB shape")
    require(depth.shape == color.shape[:2] and np.issubdtype(depth.dtype, np.floating), "NORMAL depth must match viewport")
    require(np.isfinite(color).all() and np.isfinite(segmentation).all()
            and np.isfinite(depth).all(), "nonfinite NORMAL/SEG/depth")
    require(np.all(depth >= 0), "negative NORMAL depth")
    person = ((segmentation[..., 0] == 255) & (segmentation[..., 1] == 0)
              & (segmentation[..., 2] == 0))
    count = int(np.count_nonzero(person))
    require(count > COLOR_QUALITY_THRESHOLDS["person_seg_pixels_gt"], f"person SEG pixels={count}")
    nonblack = color.astype(np.float32).mean(axis=-1) > COLOR_QUALITY_THRESHOLDS["nonblack_rgb_mean_gt"]
    whole_fraction = float(nonblack.mean())
    person_fraction = float(nonblack[person].mean())
    depth_fraction = float((depth[person] > 0).mean())
    require(whole_fraction > COLOR_QUALITY_THRESHOLDS["whole_nonblack_fraction_gt"], f"whole NORMAL nonblack fraction={whole_fraction}")
    require(person_fraction > COLOR_QUALITY_THRESHOLDS["person_nonblack_fraction_gt"], f"person NORMAL nonblack fraction={person_fraction}")
    require(depth_fraction > COLOR_QUALITY_THRESHOLDS["normal_depth_positive_person_fraction_gt"], f"person NORMAL depth fraction={depth_fraction}")
    return {"passed": True, "viewport_hw": list(color.shape[:2]), "finite_normal_color_depth_seg": True,
            "person_seg_pixels": count, "whole_nonblack_fraction": whole_fraction,
            "person_nonblack_fraction": person_fraction, "normal_depth_positive_person_fraction": depth_fraction,
            "person_normal_mean_rgb": color[person].mean(axis=0).tolist(),
            "normal_depth_nonzero_pixels": int(np.count_nonzero(depth > 0))}


def retained_frame(renderer, scene, node, frame, seg_flags, context=""):
    """Deterministic per-clip/per-side frame-zero warmup; no frame skipped."""
    import numpy as np

    kwargs = {"flags": seg_flags,
              "seg_node_map": {node: np.array([255, 0, 0], dtype=np.uint8)}}
    if frame == 0:
        renderer.render(scene)  # Discard cold NORMAL at this exact pose/time.
        renderer.render(scene, **kwargs)  # Discard SEG; initialize both draw paths.
    color, depth = renderer.render(scene)
    # Preserve the actual NORMAL bytes even if a future SEG backend reuses buffers.
    color, depth = np.array(color, copy=True), np.array(depth, copy=True)
    segmentation, _ = renderer.render(scene, **kwargs)
    quality = retained_color_quality(color, depth, segmentation, context)
    quality["same_pose_frame_zero_normal_seg_warmup"] = frame == 0
    return color, quality


@contextmanager
def managed_video_writer(writer):
    """Release encoder even when open/retained-frame quality/write fails."""
    try:
        if not writer.isOpened():
            raise RuntimeError("SMPL-X paired VideoWriter did not open")
        yield writer
    finally:
        writer.release()


def fixed_validation_ids(manifest):
    candidates = manifest["splits"]["val"]
    chosen = [next(ident for ident in candidates if int(ident.split("_", 1)[0]) == pid)
              for pid in (1, 2, 30)]
    assert len(set(chosen)) == 3
    return chosen


def camera_pose(position, target):
    import numpy as np
    position, target = np.asarray(position, float), np.asarray(target, float)
    backward = position-target
    backward /= np.linalg.norm(backward)
    right = np.cross(np.array([0., 1., 0.]), backward)
    right /= np.linalg.norm(right)
    upward = np.cross(backward, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack((right, upward, backward))
    pose[:3, 3] = position
    return pose


def render_bundle(manifest_path, predictions, raw_root, smplx_root, output,
                  clips=None, pose_policy="full55", device="cuda:0", seconds=10., sampling_step=4,
                  prediction_label="Prediction", split="val"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import cv2
    import numpy as np
    import pyrender
    import smplx
    import torch
    import trimesh
    manifest = json.loads(Path(manifest_path).read_text())
    raw_root, predictions, output = Path(raw_root), Path(predictions), Path(output)
    clips = clips or fixed_validation_ids(manifest)
    assert split in ("val", "test")
    assert len(clips) == len(set(clips)) and all(i in manifest["splits"][split] for i in clips)
    assert pose_policy in ("full55", "body22")
    output.mkdir(parents=True, exist_ok=True)
    model = smplx.create(str(smplx_root), model_type="smplx", gender="NEUTRAL_2020",
        num_betas=300, num_expression_coeffs=100, use_pca=False, use_face_contour=False,
        ext="npz").eval().requires_grad_(False).to(device)
    renderer = pyrender.OffscreenRenderer(640, 720)
    scenes = []
    material = [pyrender.MetallicRoughnessMaterial(baseColorFactor=(.35, .65, .95, 1.), metallicFactor=0., roughnessFactor=.7),
                pyrender.MetallicRoughnessMaterial(baseColorFactor=(.9, .55, .35, 1.), metallicFactor=0., roughnessFactor=.7)]
    for _ in range(2):
        scene = pyrender.Scene(bg_color=(245, 245, 245, 255), ambient_light=(.55, .55, .55))
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi/4), pose=camera_pose((0, 1.05, 3.5), (0, 1.05, 0)))
        for position in ((2, 3, 2), (-2, 2, 2)):
            scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2), pose=camera_pose(position, (0, 1, 0)))
        floor = trimesh.creation.box(extents=(3.6, .01, 3.6))
        floor.apply_translation((0, -.015, 0))
        scene.add(pyrender.Mesh.from_trimesh(floor, smooth=False))
        scenes.append(scene)
    reports = []
    mesh_nodes = [None, None]
    try:
        for ident in clips:
            gt_path = raw_root / "smplxflame_30" / (ident + ".npz")
            pred_path = predictions / ("res_" + ident + ".npz")
            with np.load(gt_path, allow_pickle=False) as z:
                gt_pose, gt_trans = z["poses"][:, :165].copy(), z["trans"].copy()
                betas = np.asarray(z["betas"]).reshape(-1)
            with np.load(pred_path, allow_pickle=False) as z:
                pred_pose, pred_trans = z["poses"][:, :165].copy(), z["trans"].copy()
            assert np.isfinite(gt_pose).all() and np.isfinite(pred_pose).all()
            assert np.isfinite(pred_trans).all() and np.isfinite(gt_trans).all()
            # Canonical origins for display only; no rotation, scale matching,
            # or GT root trajectory injected into the prediction.
            gt_trans[:, (0, 2)] -= gt_trans[0, (0, 2)]
            pred_trans[:, (0, 2)] -= pred_trans[0, (0, 2)]
            if pose_policy == "body22":
                gt_pose[:, 66:] = 0.
                pred_pose[:, 66:] = 0.
            length = min(len(gt_pose), len(pred_pose), len(gt_trans), len(pred_trans))
            frames = min(length, int(seconds*30))
            frames -= frames % 2  # Exact 30 -> 15 fps duration and frame count.
            assert frames >= 32, (ident, frames)
            betas = np.pad(betas, (0, max(0, 300-len(betas))))[:300]
            vertices = []
            with torch.inference_mode():
                for poses, trans in ((gt_pose, gt_trans), (pred_pose, pred_trans)):
                    pieces = []
                    for begin in range(0, frames, 30):
                        n = min(30, frames-begin)
                        p = torch.as_tensor(poses[begin:begin+n], dtype=torch.float32, device=device)
                        result = model(betas=torch.as_tensor(betas[None], dtype=torch.float32, device=device).expand(n, -1),
                            global_orient=p[:, :3], body_pose=p[:, 3:66], jaw_pose=p[:, 66:69],
                            leye_pose=p[:, 69:72], reye_pose=p[:, 72:75], left_hand_pose=p[:, 75:120],
                            right_hand_pose=p[:, 120:165], expression=torch.zeros(n, 100, device=device),
                            transl=torch.as_tensor(trans[begin:begin+n], dtype=torch.float32, device=device))
                        pieces.append(result.vertices.cpu().numpy())
                    vertices.append(np.concatenate(pieces))
            assert all(np.isfinite(v).all() for v in vertices)
            final = output / (ident + "_gt_pred_smplx.mp4")
            if final.exists():
                raise FileExistsError(f"Refusing to overwrite render: {final}")
            with tempfile.TemporaryDirectory(prefix="m2g_render_", dir=output) as temporary:
                temporary = Path(temporary)
                silent = temporary / "silent.mp4"
                visible_count = [0, 0]
                frame_quality = []
                with managed_video_writer(cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 720))) as writer:
                    for frame in range(0, frames, 2):
                        pictures, sides = [], []
                        for side, (scene, v) in enumerate(zip(scenes, vertices)):
                            if mesh_nodes[side] is not None:
                                scene.remove_node(mesh_nodes[side])
                            mesh = trimesh.Trimesh(v[frame], model.faces, process=False)
                            mesh_nodes[side] = scene.add(pyrender.Mesh.from_trimesh(mesh, material=material[side], smooth=True))
                            color, quality = retained_frame(renderer, scene, mesh_nodes[side], frame,
                                pyrender.RenderFlags.SEG, context=f"{ident}/{pose_policy}/source_frame={frame}/side={side}")
                            sides.append({"side": "GT" if side == 0 else "prediction", **quality})
                            visible_count[side] += 1  # Only after NORMAL/depth/SEG all pass.
                            image = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)
                            label = "GT" if side == 0 else prediction_label
                            cv2.putText(image, f"{label} | {pose_policy} | {ident}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .55, (30, 30, 30), 1, cv2.LINE_AA)
                            cv2.putText(image, f"t={frame/30:.2f}s | {sampling_step}-step", (12, 54), cv2.FONT_HERSHEY_SIMPLEX, .55, (30, 30, 30), 1, cv2.LINE_AA)
                            pictures.append(image)
                        writer.write(np.concatenate(pictures, axis=1))
                        frame_quality.append({"source_frame": frame, "video_frame": frame//2,
                                              "time_seconds": frame/30., "sides": sides})
                expected_frames = len(range(0, frames, 2))
                assert visible_count == [expected_frames, expected_frames]
                assert len(frame_quality) == expected_frames and frame_quality[0]["source_frame"] == 0
                muxed = temporary / "paired.mp4"
                audio = raw_root / "wave16k" / (ident + ".wav")
                subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(silent), "-i", str(audio),
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22",
                    "-c:a", "aac", "-t", str(frames/30), str(muxed)], check=True)
                probe = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-show_streams", "-of", "json", str(muxed)], text=True))
                video = next(s for s in probe["streams"] if s["codec_type"] == "video")
                assert int(video["nb_read_frames"]) == expected_frames
                assert any(s["codec_type"] == "audio" for s in probe["streams"])
                muxed.replace(final)
            report = {"id": ident, "split": split, "pose_policy": pose_policy, "source_fps": 30, "video_fps": 15,
                      "source_frames": frames, "rendered_frames": expected_frames, "native_y_up": True,
                      "gt_sha256": sha256(gt_path), "prediction_sha256": sha256(pred_path), "video_sha256": sha256(final),
                      "sampling_step": sampling_step, "smplx_mesh_vertices": int(vertices[0].shape[1]),
                      "prediction_label": prediction_label,
                      "finite": True, "nonempty_all_frames": True, "audio_muxed": True, "visual_human_review": "pending",
                      "retained_normal_color_depth_seg_all_frames": True,
                      "frame_zero_warmup": "per clip AND side: NORMAL discard, SEG discard, NORMAL retained, SEG checked; same pose/time; no frame skipped",
                      "retained_frame_quality": {"kind": "same_frame_normal_color_depth_person_SEG",
                          "thresholds": dict(COLOR_QUALITY_THRESHOLDS), "frames": frame_quality,
                          "both_side_minima": [{"side": side,
                              "checked_frames": expected_frames,
                              "person_seg_pixels": min(row["sides"][index]["person_seg_pixels"] for row in frame_quality),
                              "whole_nonblack_fraction": min(row["sides"][index]["whole_nonblack_fraction"] for row in frame_quality),
                              "person_nonblack_fraction": min(row["sides"][index]["person_nonblack_fraction"] for row in frame_quality),
                              "normal_depth_positive_person_fraction": min(row["sides"][index]["normal_depth_positive_person_fraction"] for row in frame_quality)}
                              for index, side in enumerate(("GT", "prediction"))]},
                      "expressions": "neutral on BOTH GT/pred; no face-quality claim", "video": str(final)}
            atomic_json(output / (ident + "_render_audit.json"), report)
            reports.append(report)
    finally:
        renderer.delete()
    atomic_json(output / "render_complete.json", {"clips": reports, "completed": True,
                                                  "requires_visual_review_before_quality_claims": True})
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--raw-root", required=True)
    p.add_argument("--smplx", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--clips", nargs="+")
    p.add_argument("--pose-policy", choices=("full55", "body22"), default="full55")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seconds", type=float, default=10.)
    p.add_argument("--sampling-step", type=int, default=4)
    p.add_argument("--prediction-label", default="Prediction")
    p.add_argument("--split", choices=("val", "test"), default="val")
    a = p.parse_args()
    render_bundle(a.manifest, a.predictions, a.raw_root, a.smplx, a.output, a.clips,
                  a.pose_policy, a.device, a.seconds, a.sampling_step, a.prediction_label, a.split)


if __name__ == "__main__":
    main()
