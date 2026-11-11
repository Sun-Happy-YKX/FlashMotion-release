"""Read-only, fail-closed visual-review gate for actual precheck renders.

This module NEVER authors clearance. The root operator must first view all
18 PNGs (first/middle/last of full55/body22 x the three fixed validation clips)
and only then author the review JSON and its gate reference. Technical render
completion, image decoding, and this validator do not constitute pixel review.
Synthetic unit-test fixtures cannot clear the separate real-GPU/training gates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


REVIEW_KIND = "actual_render_frame_review"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class VisualReviewError(RuntimeError):
    pass


def _require(condition, message):
    if not condition:
        raise VisualReviewError(message)


def _sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file(path, label):
    _require(isinstance(path, str) and path, f"Missing {label} path")
    value = Path(path)
    _require(value.is_absolute(), f"{label} path must be absolute")
    _require(value.is_file() and value.stat().st_size > 0, f"Missing or empty {label}: {value}")
    return value.resolve()


def _hashed_file(path, claimed_hash, label):
    _require(isinstance(claimed_hash, str) and len(claimed_hash) == 64
             and all(char in "0123456789abcdef" for char in claimed_hash), f"Invalid {label} SHA256")
    value = _file(path, label)
    _require(_sha256(value) == claimed_hash, f"Changed {label}: SHA256 mismatch")
    return value


def _json(path, label):
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise VisualReviewError(f"Cannot read {label} JSON: {error}") from error
    _require(isinstance(result, dict), f"{label} JSON must be an object")
    return result


def _png(path):
    with path.open("rb") as handle:
        _require(handle.read(8) == PNG_SIGNATURE, f"Observed frame is not PNG: {path}")
    try:
        import numpy as np
        from PIL import Image
        with Image.open(path) as image:
            _require(image.format == "PNG" and min(image.size) > 0 and image.width >= 2,
                     f"Invalid paired PNG dimensions (requires BOTH halves): {path}")
            image.verify()
        with Image.open(path) as image:
            color = np.asarray(image.convert("RGB"), dtype=np.uint8)
            split = image.width//2
            for label, half in (("left/GT", color[:, :split]), ("right/prediction", color[:, split:])):
                nonblack_fraction = float((half.astype(np.float32).mean(axis=-1) > 16.).mean())
                _require(nonblack_fraction > .5,
                         f"Observed paired PNG {label} has insufficient actual NORMAL color "
                         f"(RGB mean>16 fraction={nonblack_fraction}, requires>.5): {path}")
    except VisualReviewError:
        raise
    except Exception as error:
        raise VisualReviewError(f"Cannot validate observed PNG {path}: {error}") from error


def _fixed_ids(manifest):
    candidates = manifest.get("splits", {}).get("val")
    _require(isinstance(candidates, list), "Manifest has no validation inventory")
    try:
        result = [next(ident for ident in candidates if int(ident.split("_", 1)[0]) == person)
                  for person in (1, 2, 30)]
    except (StopIteration, AttributeError, ValueError) as error:
        raise VisualReviewError("Manifest lacks the fixed person 1/2/30 validation clips") from error
    _require(len(set(result)) == 3, "Fixed validation IDs must be distinct")
    return result


def require_visual_review(config, gate):
    """Validate current videos/PNGs against operator-authored review evidence.

    ``gate.visual_review`` is {passed, kind, evidence: absolute JSON, sha256}.
    That JSON has the same passed/kind/task_name/hostname, reviewer="root",
    pixels_viewed=true, actual_three_clip_render={evidence,sha256}, and videos
    with {pose_policy,id,video,video_sha256,rendered_frames,observed_frames}.
    Each observed_frames list contains {position,frame_index,path,sha256,
    pixels_viewed:true} for first=0, middle=N//2, last=N-1 (video-frame indices,
    not 30fps source indices). Returns the checked proof; writes nothing.
    """
    task, host = config.get("task_name"), config.get("expected_hostname")
    _require(isinstance(task, str) and task and isinstance(host, str) and host,
             "Visual review requires configured task and expected hostname")
    _require(gate.get("passed") is True and gate.get("task_name") == task
             and gate.get("hostname") == host, "Visual review belongs to an incompatible technical gate")
    review = gate.get("visual_review", {})
    _require(isinstance(review, dict) and review.get("passed") is True
             and review.get("kind") == REVIEW_KIND,
             "Actual rendered pixels require root visual review before any BS probe/training")
    review_path = _hashed_file(review.get("evidence"), review.get("sha256"), "visual review evidence")
    proof = _json(review_path, "visual review evidence")
    _require(proof.get("passed") is True and proof.get("kind") == REVIEW_KIND
             and proof.get("task_name") == task and proof.get("hostname") == host,
             "Visual review proof has wrong pass/kind/task/host")
    _require(proof.get("reviewer") == "root" and proof.get("pixels_viewed") is True,
             "Proof must attest that root actually viewed the rendered pixels")

    render_check = gate.get("checks", {}).get("actual_three_clip_render", {})
    _require(isinstance(render_check, dict) and render_check.get("passed") is True
             and render_check.get("kind") == "smplx_mesh", "Missing actual three-clip render check")
    render_path = _hashed_file(render_check.get("evidence"), render_check.get("sha256"), "actual render evidence")
    binding = proof.get("actual_three_clip_render", {})
    _require(isinstance(binding, dict), "Visual review lacks actual render evidence binding")
    bound_path = _hashed_file(binding.get("evidence"), binding.get("sha256"), "review-bound render evidence")
    _require(bound_path == render_path and binding.get("sha256") == render_check.get("sha256"),
             "Visual review does not bind the current actual render evidence")
    render = _json(render_path, "actual render evidence")
    _require(render.get("passed") is True and render.get("kind") == "smplx_mesh",
             "Actual render evidence is not a completed SMPL-X mesh render")
    manifest = _json(_file(config.get("manifest"), "manifest"), "manifest")
    ids = _fixed_ids(manifest)
    renders = render.get("renders")
    _require(isinstance(renders, dict) and set(renders) == {"full55", "body22"},
             "Actual render evidence must cover both full55 and body22")
    expected = {}
    video_paths = set()
    for mode in ("full55", "body22"):
        records = renders[mode]
        _require(isinstance(records, list) and len(records) == 3, "Each pose mode needs all three fixed clips")
        for record in records:
            _require(isinstance(record, dict), "Render record must be an object")
            ident = record.get("id")
            _require(ident in ids and record.get("pose_policy") == mode and record.get("split") == "val",
                     "Render record has wrong fixed clip, pose policy, or split")
            key = (mode, ident)
            _require(key not in expected, "Duplicate actual render coverage")
            count, source_frames = record.get("rendered_frames"), record.get("source_frames")
            _require(type(count) is int and count >= 16 and type(source_frames) is int
                     and source_frames == 2 * count and record.get("source_fps") == 30
                     and record.get("video_fps") == 15, "Render count must match actual 30-to-15fps video")
            _require(record.get("finite") is True and record.get("nonempty_all_frames") is True,
                     "Actual paired meshes must be finite and visible in all rendered frames")
            video = _hashed_file(record.get("video"), record.get("video_sha256"), "actual rendered video")
            _require(video not in video_paths, "Each pose/clip must have a distinct actual video")
            video_paths.add(video)
            expected[key] = (video, record["video_sha256"], count)
    _require(set(expected) == {(mode, ident) for mode in ("full55", "body22") for ident in ids},
             "Actual render evidence is missing required coverage")

    reviewed = proof.get("videos")
    _require(isinstance(reviewed, list) and len(reviewed) == 6, "Review must bind all six actual videos")
    seen, image_paths = set(), set()
    for record in reviewed:
        _require(isinstance(record, dict), "Reviewed video must be an object")
        key = (record.get("pose_policy"), record.get("id"))
        _require(key in expected and key not in seen, "Missing, duplicate, or extra reviewed pose/clip")
        seen.add(key)
        video, digest, count = expected[key]
        _require(_hashed_file(record.get("video"), record.get("video_sha256"), "reviewed video") == video
                 and record.get("video_sha256") == digest and record.get("rendered_frames") == count,
                 "Reviewed video path/hash/count does not match the actual precheck video")
        frames = record.get("observed_frames")
        _require(isinstance(frames, list) and len(frames) == 3, "Each video needs first/middle/last observed PNGs")
        positions = {"first": 0, "middle": count // 2, "last": count - 1}
        seen_positions = set()
        for frame in frames:
            _require(isinstance(frame, dict), "Observed frame must be an object")
            position, index = frame.get("position"), frame.get("frame_index")
            _require(position in positions and position not in seen_positions
                     and type(index) is int and index == positions[position],
                     "Observed frame index must be actual video's first/middle/last 15fps index")
            _require(frame.get("pixels_viewed") is True, "Root must actually view each observed PNG")
            seen_positions.add(position)
            png = _hashed_file(frame.get("path"), frame.get("sha256"), "observed frame PNG")
            _require(png not in image_paths, "Observed frames must have distinct extracted PNG paths")
            image_paths.add(png)
            _png(png)
    _require(seen == set(expected), "Visual review is incomplete")
    return proof


__all__ = ["require_visual_review", "VisualReviewError", "REVIEW_KIND"]
