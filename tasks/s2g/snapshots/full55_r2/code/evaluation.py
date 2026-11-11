"""Audited SemTalk metrics for explicit BEAT2 manifests; no downloads or GT completion.

Primary BC reproduces SemTalk's historical evaluator, including its DIV input
mutation. ``bc_corrected`` fixes that mutation and the audio/crop index quirks.
FGD uses the released AESK weights but SemTalk's (not EMAGE's) distance formula.
All public functions return raw units, never the display multipliers in a paper.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Sequence

import numpy as np

AESK_SHA256 = "5cd9566b24264f34d44003b3de62cdfd50aa85b7cdde2d369214599023c40f55"
PERSON_IDS = (1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 30)
UPPER_BODY = (3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
FPS, AUDIO_SR, ALIGN_MASK = 30, 16000, 60


class ContractError(ValueError):
    """Refuse a missing clip, wrong representation, or incompatible protocol."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rollout_frames(frames: int, pose_length: int = 64, seed_frames: int = 4) -> int:
    """SemTalk full-clip 8-frame truncation then 64/60 autoregressive budget."""
    available = int(frames) // 8 * 8
    step = int(pose_length) - int(seed_frames)
    if step <= 0 or available < pose_length:
        raise ContractError("clip cannot hold one complete rollout window")
    return int(seed_frames) + step * ((available - int(seed_frames)) // step)


def manifest_records(manifest: Any, split: str | None = None,
                     expected_count: int | None = None) -> list[dict[str, Any]]:
    """Accept IDs, clip records, or the shared {splits:{train,val,test}} manifest."""
    if isinstance(manifest, (str, Path)):
        manifest = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if isinstance(manifest, dict):
        if "splits" in manifest:
            if split not in manifest["splits"]:
                raise ContractError(f"manifest has no requested split {split!r}")
            manifest = manifest["splits"][split]
        elif "clips" in manifest:
            manifest = manifest["clips"]
        elif "ids" in manifest:
            manifest = manifest["ids"]
        else:
            raise ContractError("manifest must have splits, clips, or ids")
    if not isinstance(manifest, (list, tuple)) or not manifest:
        raise ContractError("explicit nonempty clip IDs are required")
    records = [dict(value) if isinstance(value, dict) else {"id": str(value)} for value in manifest]
    ids = [value.get("id") for value in records]
    if any(not isinstance(value, str) or not value or Path(value).name != value for value in ids):
        raise ContractError("IDs must be nonempty file stems without paths")
    if len(ids) != len(set(ids)):
        raise ContractError("manifest contains duplicate IDs")
    if expected_count is not None and len(ids) != int(expected_count):
        raise ContractError(f"expected {expected_count} clips; manifest has {len(ids)}")
    for value in ids:
        try:
            person = int(value.split("_", 1)[0])
        except ValueError as exc:
            raise ContractError(f"invalid BEAT2 ID {value}") from exc
        if person not in PERSON_IDS:
            raise ContractError(f"non-English/unreleased BEAT2 person ID {person}")
    return records


def pose_mask(poses: np.ndarray, policy: str) -> np.ndarray:
    value = np.asarray(poses, dtype=np.float32).copy()
    if value.ndim != 2 or value.shape[1] != 165 or not np.isfinite(value).all():
        raise ContractError(f"expected finite local SMPL-X AA [T,165], got {value.shape}")
    if policy in ("body22", "body22only", "zero_nonbody", "zero_non_body"):
        value[:, 66:165] = 0.0
    elif policy not in ("full55", "full", "preserve"):
        raise ContractError(f"unknown pose policy {policy!r}")
    return value


def aa_to_rot6d(poses: Any, torch_module: Any) -> Any:
    """PantoMatrix quaternion conversion, with row-based rotation6D convention."""
    vectors = poses.reshape(*poses.shape[:-1], 55, 3)
    angles = torch_module.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    half = angles * 0.5
    small = angles.abs() < 1e-6
    ratio = torch_module.empty_like(angles)
    ratio[~small] = torch_module.sin(half[~small]) / angles[~small]
    ratio[small] = 0.5 - angles[small].square() / 48
    q = torch_module.cat((torch_module.cos(half), vectors * ratio), -1)
    w, x, y, z = torch_module.unbind(q, -1)
    s = 2.0 / q.square().sum(-1)
    matrix = torch_module.stack((
        1-s*(y*y+z*z), s*(x*y-z*w), s*(x*z+y*w),
        s*(x*y+z*w), 1-s*(x*x+z*z), s*(y*z-x*w),
        s*(x*z-y*w), s*(y*z+x*w), 1-s*(x*x+y*y),
    ), -1).reshape(*q.shape[:-1], 3, 3)
    return matrix[..., :2, :].clone().reshape(*poses.shape[:-1], 330)


def audit_rotation_roundtrip(original_poses: np.ndarray, reconstructed_poses: np.ndarray,
                             torch_module: Any, device: Any, *, atol: float = 2e-5) -> dict[str, Any]:
    """Audit the production-device float32 converter against two independent checks.

    This does not change the AESK input converter or select a scoring backend.
    CPU float32 is diagnostic only when AESK runs elsewhere. Old unexplained
    CPU anomalies remain evidence, not a reason to relax the rotation tolerance.
    """
    from scipy.spatial.transform import Rotation

    original = np.ascontiguousarray(original_poses, dtype=np.float32)
    reconstructed = np.ascontiguousarray(reconstructed_poses, dtype=np.float32)
    if original.shape != reconstructed.shape or original.ndim != 2 or original.shape[1] != 165 or not len(original):
        raise ContractError("rotation audit requires matching nonempty local AA [T,165]")
    if not np.isfinite(original).all() or not np.isfinite(reconstructed).all():
        raise ContractError("rotation audit requires finite AA inputs")
    if not math.isfinite(atol) or atol <= 0:
        raise ContractError("rotation audit tolerance must be positive and finite")
    arrays = {"original": original, "reconstructed": reconstructed}
    hashes = {side: hashlib.sha256(value.tobytes(order="C")).hexdigest() for side, value in arrays.items()}

    def converted(dtype: Any, selected_device: Any) -> dict[str, np.ndarray]:
        result = {}
        with torch_module.inference_mode():
            for side, value in arrays.items():
                # Exactly the production AESK batch/AA feature shape and dtype.
                tensor = torch_module.as_tensor(value[None], dtype=dtype, device=selected_device)
                result[side] = aa_to_rot6d(tensor, torch_module)[0].detach().cpu().numpy()
        return result

    primary = converted(torch_module.float32, device)
    double = converted(torch_module.float64, "cpu")
    scipy_result = {side: Rotation.from_rotvec(value.astype(np.float64).reshape(-1, 3)).as_matrix()
                    .reshape(-1, 55, 3, 3)[..., :2, :].reshape(-1, 330)
                    for side, value in arrays.items()}
    cpu32 = converted(torch_module.float32, "cpu")
    backends = {"primary_production_float32": primary, "cpu_float64_same_formula": double,
                "scipy_float64_matrix": scipy_result, "cpu_float32_diagnostic": cpu32}
    expected = (len(original), 330)
    if any(value.shape != expected or not np.isfinite(value).all()
           for pair in backends.values() for value in pair.values()):
        raise ContractError("rotation audit backend returned wrong-shaped or nonfinite rotation6D")

    def error(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        difference = np.abs(left.astype(np.float64) - right.astype(np.float64))
        frame, feature = (int(index) for index in np.unravel_index(int(np.argmax(difference)), difference.shape))
        joint = feature // 6
        return {"max_abs_error": float(difference[frame, feature]), "within_tolerance": bool(difference[frame, feature] <= atol),
                "worst": {"frame": frame, "feature": feature, "joint": joint,
                          "matrix_row": feature % 6 // 3, "matrix_column": feature % 3,
                          "left_value": float(left[frame, feature]), "right_value": float(right[frame, feature]),
                          "original_aa": original.reshape(-1, 55, 3)[frame, joint].tolist(),
                          "reconstructed_aa": reconstructed.reshape(-1, 55, 3)[frame, joint].tolist(),
                          "original_aa_norm_float64": float(np.linalg.norm(original.reshape(-1, 55, 3)[frame, joint].astype(np.float64))),
                          "reconstructed_aa_norm_float64": float(np.linalg.norm(reconstructed.reshape(-1, 55, 3)[frame, joint].astype(np.float64)))}}

    roundtrips = {name: error(pair["original"], pair["reconstructed"]) for name, pair in backends.items()}
    crosschecks = {}
    for label, left, right in (("primary_vs_cpu64", primary, double), ("primary_vs_scipy", primary, scipy_result),
                               ("cpu64_vs_scipy", double, scipy_result)):
        crosschecks[label] = {side: error(left[side], right[side]) for side in arrays}
    cpu_diagnostic = {"roundtrip": roundtrips["cpu_float32_diagnostic"],
                      "vs_cpu64": {side: error(cpu32[side], double[side]) for side in arrays},
                      "vs_scipy": {side: error(cpu32[side], scipy_result[side]) for side in arrays},
                      "used_as_fallback": False, "used_for_metric_encoding": False}
    cpu_diagnostic["discrepancy_detected"] = not all(
        value["within_tolerance"] for value in (cpu_diagnostic["roundtrip"],
        *cpu_diagnostic["vs_cpu64"].values(), *cpu_diagnostic["vs_scipy"].values()))
    unchanged = hashes == {side: hashlib.sha256(value.tobytes(order="C")).hexdigest() for side, value in arrays.items()}
    required = [roundtrips[name]["within_tolerance"] for name in
                ("primary_production_float32", "cpu_float64_same_formula", "scipy_float64_matrix")]
    required.extend(value["within_tolerance"] for sides in crosschecks.values() for value in sides.values())
    if torch_module.device(device).type == "cpu":
        # Here CPU32 is the production backend as well, so a inconsistent
        # repeated conversion cannot be treated as an optional diagnostic.
        required.append(not cpu_diagnostic["discrepancy_detected"])
    return {"passed": bool(unchanged and all(required)), "kind": "production_device_rotation6d_with_independent_references",
            "primary_device": str(device), "primary_dtype": "float32", "primary_input_shape": [1, len(original), 165],
            "full_clip_frames_compared": len(original), "tolerance": atol, "thresholds_relaxed": False,
            "production_converter_unchanged": True, "metric_backend_selected_by_score": False,
            "input_array_sha256": hashes, "input_arrays_unchanged": unchanged,
            "primary_max_abs_error": roundtrips["primary_production_float32"]["max_abs_error"],
            "backend_roundtrips": roundtrips, "required_crosschecks": crosschecks,
            "cpu_float32_diagnostic": cpu_diagnostic,
            "cpu_float32_anomaly_root_cause": "not_determined_by_this_audit"}


def feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(features)
    if value.ndim != 2 or len(value) < 2 or not np.isfinite(value).all():
        raise ContractError("FGD needs finite [rows,dim] features with >=2 rows")
    # Preserve SemTalk's input mean dtype (AESK emits float32); np.cov is float64.
    return np.mean(value, axis=0), np.atleast_2d(np.cov(value, rowvar=False))


def _psd_sqrt(matrix: np.ndarray) -> np.ndarray:
    from scipy import linalg
    symmetric = (matrix + matrix.T) * 0.5
    # Actual AESK covariances exposed a broken NumPy1.23.5 LAPACK path:
    # V.T@V error exceeded165 while SciPy ev/evd/evr and Torch reconstructed
    # the identical matrix at1e-16. Keep the same PSD mathematics, use the
    # independently verified driver, and audit its invariants every call.
    eigenvalues, eigenvectors = linalg.eigh(symmetric, driver="evd", check_finite=True)
    scale = max(1.0, float(np.max(np.abs(symmetric))))
    orthogonality_error = float(np.max(np.abs(eigenvectors.T @ eigenvectors - np.eye(len(eigenvalues)))))
    reconstruction_error = float(np.max(np.abs((eigenvectors * eigenvalues) @ eigenvectors.T - symmetric)))
    if not np.isfinite(eigenvalues).all() or orthogonality_error > 1e-9 or reconstruction_error > 1e-9 * scale:
        raise ContractError(f"PSD eigendecomposition audit failed: orthogonality={orthogonality_error}, reconstruction={reconstruction_error}")
    tolerance = max(1.0, float(np.max(np.abs(eigenvalues)))) * 1e-9
    if float(np.min(eigenvalues)) < -tolerance:
        raise ContractError("covariance is not positive semidefinite")
    root = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0.0))) @ eigenvectors.T
    if float(np.max(np.abs(root @ root - symmetric))) > 2e-9 * scale:
        raise ContractError("PSD square-root reconstruction audit failed")
    return root


def frechet_from_statistics(pred_stats: tuple[np.ndarray, np.ndarray],
                           gt_stats: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    """SemTalk sqrtm result checked against a symmetric PSD formulation.

    SemTalk adds epsilon only if sqrtm is nonfinite. EMAGE's unconditional
    epsilon is intentionally not used: that gives a different GT-self floor.
    """
    from scipy import linalg

    mu_p, cov_p = pred_stats
    mu_g, cov_g = gt_stats
    if mu_p.shape != mu_g.shape or cov_p.shape != cov_g.shape:
        raise ContractError("FGD feature dimensions do not match")
    root = linalg.sqrtm(cov_p.dot(cov_g))
    fallback_epsilon = 0.0
    if not np.isfinite(root).all():
        fallback_epsilon = 1e-6
        offset = np.eye(len(mu_p)) * fallback_epsilon
        root = linalg.sqrtm((cov_p + offset).dot(cov_g + offset))
    imaginary_max = float(np.max(np.abs(root.imag))) if np.iscomplexobj(root) else 0.0
    if np.iscomplexobj(root):
        if not np.allclose(np.diagonal(root).imag, 0.0, atol=1e-3):
            raise ContractError(f"complex FGD matrix root: max imag {imaginary_max}")
        root = root.real
    difference = mu_p - mu_g
    trace_sum = float(np.trace(cov_p) + np.trace(cov_g))
    raw = float(difference.dot(difference) + trace_sum - 2.0 * np.trace(root))
    offset = np.eye(len(mu_p)) * fallback_epsilon
    left = _psd_sqrt(cov_p + offset)
    symmetric_product = left @ (cov_g + offset) @ left
    stable = float(difference.dot(difference) + trace_sum - 2.0 * np.trace(_psd_sqrt(symmetric_product)))
    tolerance = max(1e-6, 2e-5 * max(1.0, abs(trace_sum)))
    disagreement = abs(raw - stable)
    if not math.isfinite(raw) or not math.isfinite(stable) or disagreement > tolerance:
        raise ContractError(f"FGD numerical audit failed: scipy={raw}, PSD={stable}, tolerance={tolerance}")
    # The returned main number is the author formula, never silently clamped.
    return {"value": raw, "stable_psd_value": stable, "sqrtm_psd_abs_error": disagreement,
            "psd_eigh_backend": "scipy.linalg.eigh(driver=evd); orthogonality/reconstruction audited",
            "sqrtm_imaginary_max": imaginary_max, "fallback_epsilon": fallback_epsilon,
            "numerical_tolerance": tolerance, "latent_dimension": int(len(mu_p))}


def frechet_distance(pred_features: np.ndarray, gt_features: np.ndarray) -> dict[str, Any]:
    return frechet_from_statistics(feature_statistics(pred_features), feature_statistics(gt_features))


def l1_diversity(positions: np.ndarray) -> float:
    flat = np.asarray(positions).reshape(len(positions), -1)
    return float(np.sum(np.abs(flat - np.mean(flat, axis=0))) / len(flat))


def motion_beats(positions: np.ndarray, mean_velocity: np.ndarray,
                 legacy: bool, threshold: float = 0.3, order: int = 7) -> list[np.ndarray]:
    from scipy.signal import argrelextrema

    flat = np.asarray(positions).reshape(len(positions), -1)
    joints = flat.T
    dt = 1.0 / FPS
    velocity = np.concatenate(((joints[:, 1:2]-joints[:, :1])/dt,
                               (joints[:, 2:]-joints[:, :-2])/(2*dt),
                               (joints[:, -1:]-joints[:, -2:-1])/dt), axis=1)
    magnitude = np.linalg.norm(velocity.T.reshape(len(flat), 55, 3), axis=2) / mean_velocity
    result = []
    for joint in range(55):
        indices = argrelextrema(magnitude[ALIGN_MASK:-ALIGN_MASK, joint], np.less, order=order)[0]
        # SemTalk compares crop-relative extrema with a whole-sequence mask.
        lookup = indices if legacy else indices + ALIGN_MASK
        result.append(indices[magnitude[lookup, joint] > threshold])
    return result


def beat_score(audio_beats: np.ndarray, beats: Sequence[np.ndarray], sigma: float = 0.3) -> float:
    if not len(audio_beats):
        raise ContractError("BC is undefined without detected audio onsets; no silent zero fallback")
    scores = []
    for joint in UPPER_BODY:
        if not len(beats[joint]):
            scores.append(0.0)
            continue
        times = beats[joint] / FPS
        nearest = np.min(np.abs(times[:, None] - audio_beats[None, :]), axis=0)
        scores.append(float(np.mean(np.exp(-np.square(nearest) / (2*sigma*sigma)))))
    return float(np.mean(scores))


def normalization_roundtrip_report(features: np.ndarray, mean: np.ndarray,
                                   std: np.ndarray, atol: float = 2e-6) -> dict[str, Any]:
    """CPU algebra audit in the actual model representation; no metric claims."""
    value = np.asarray(features, dtype=np.float64)
    mean, std = np.asarray(mean, dtype=np.float64), np.asarray(std, dtype=np.float64)
    if mean.shape != std.shape or value.shape[-1:] != mean.shape or not np.isfinite(value).all():
        raise ContractError("normalizer shape or finite-value contract failed")
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ContractError("normalizer must have finite means and strictly positive std")
    recovered = ((value - mean) / std) * std + mean
    error = float(np.max(np.abs(recovered - value)))
    if error > atol:
        raise ContractError(f"normalization roundtrip max error {error} exceeds {atol}")
    return {"max_abs_error": error, "tolerance": atol, "dimension": int(value.shape[-1]),
            "passed": True, "audit_kind": "normalization_algebra_not_aesk_fgd"}


def coordinate_axis_evidence(positions: np.ndarray) -> dict[str, Any]:
    """Anatomical axis evidence from sampled GT, never a metric-minimizing search.

    Head-to-feet vectors use SMPL-X bone indices head=15 and feet=10,11.
    Root translation cannot affect this relative anatomical measurement.
    An upright sample supports positive-Y height only when its head-to-feet
    median has positive dominant Y. Unusual inverted/lying motions must be
    interpreted explicitly, not silently rotated to improve a model score.
    """
    joints = np.asarray(positions)
    if joints.ndim != 3 or joints.shape[1:] != (55, 3) or not len(joints) or not np.isfinite(joints).all():
        raise ContractError("axis audit expects finite nonempty [T,55,3] GT joints")
    vertical = joints[:, 15] - (joints[:, 10] + joints[:, 11]) * 0.5
    median = np.median(vertical, axis=0)
    dominant = int(np.argmax(np.abs(median)))
    return {"kind": "anatomical_GT_head_to_feet_not_score_search", "sampled_frames": len(joints),
            "head_minus_mean_feet_median_xyz": median.tolist(),
            "head_minus_mean_feet_mean_xyz": np.mean(vertical, axis=0).tolist(),
            "dominant_anatomical_axis": "xyz"[dominant],
            "dominant_axis_sign": "positive" if median[dominant] > 0 else "negative",
            "upright_sample_supports_positive_y_height": bool(dominant == 1 and median[1] > 0),
            "coordinate_transform_applied": "none"}


def _read_archive(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {name: np.asarray(archive[name]) for name in ("poses", "betas") if name in archive}
        if "mocap_frame_rate" in archive and float(archive["mocap_frame_rate"]) != FPS:
            raise ContractError(f"wrong motion FPS in {path}")
    if "poses" not in values:
        raise ContractError(f"missing poses in {path}")
    return values


def _gt_path(root: Path, file_id: str) -> Path:
    paths = [path for path in (root/f"{file_id}.npz", root/f"gt_{file_id}.npz") if path.is_file()]
    if len(paths) != 1:
        raise ContractError(f"expected exactly one raw/gt archive for {file_id}, found {len(paths)}")
    return paths[0]


def _encoder_class(encoder_dir: Path) -> tuple[Any, bool]:
    """Load vendored official modules from an explicit directory, without network."""
    name = "_m2g_aesk_" + hashlib.sha256(str(encoder_dir.resolve()).encode()).hexdigest()[:12]
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(encoder_dir.resolve())]
        sys.modules[name] = package
    if (encoder_dir / "motion_encoder.py").is_file():
        module = importlib.import_module(name + ".motion_encoder")
        return module.VAESKConv, True
    if (encoder_dir / "test_vae.py").is_file():
        module = importlib.import_module(name + ".test_vae")
        return module.VAESKConv, False
    raise ContractError(f"no official motion_encoder.py/test_vae.py in {encoder_dir}")


class SemTalkEvaluator:
    """Persistent AESK and SMPL-X, caching GT only in this in-memory evaluator.

    ``metric_assets`` contains AESKConv_240_100.bin, mean_vel_smplxflame_30.npy,
    and smplx_models/smplx/SMPLX_NEUTRAL_2020.npz. ``encoder_dir`` is the vendored
    emage_evaltools directory or StreamTalk Scripts/FM/EMAGE_VAE directory.
    No model initialization or file downloads occur merely by importing this module.
    """
    def __init__(self, metric_assets: str | Path, encoder_dir: str | Path,
                 device: str = "cuda:0", smplx_batch_size: int = 128):
        import torch
        import smplx
        from importlib.metadata import version

        self.torch = torch
        self.device = torch.device(device)
        self.assets = Path(metric_assets).resolve()
        checkpoint = self.assets / "AESKConv_240_100.bin"
        if sha256_file(checkpoint) != AESK_SHA256:
            raise ContractError("unverified AESK checkpoint; expected official SHA256")
        cls, needs_smplx_path = _encoder_class(Path(encoder_dir))
        args = types.SimpleNamespace(vae_test_dim=330, vae_length=240, vae_layer=4,
                                     vae_grow=[1, 1, 2, 1], variational=False)
        self.aesk = (cls(args, model_save_path=str(self.assets)+"/") if needs_smplx_path else cls(args))
        states = torch.load(checkpoint, map_location="cpu", weights_only=True)
        weights = states.get("model_state", states.get("state_dict", states))
        weights = {key[7:] if key.startswith("module.") else key: val for key, val in weights.items()}
        self.aesk.load_state_dict(weights, strict=True)
        self.aesk = self.aesk.requires_grad_(False).eval().to(self.device)
        self.smplx = smplx.create(str(self.assets / "smplx_models"), model_type="smplx",
                                  gender="NEUTRAL_2020", use_face_contour=False,
                                  num_betas=300, num_expression_coeffs=100, ext="npz", use_pca=False)
        self.smplx = self.smplx.requires_grad_(False).eval().to(self.device)
        self.mean_velocity = np.asarray(np.load(self.assets / "mean_vel_smplxflame_30.npy"))
        if self.mean_velocity.shape != (55,) or not np.isfinite(self.mean_velocity).all() or np.any(self.mean_velocity <= 0):
            raise ContractError("mean-velocity asset must be finite positive [55]")
        if smplx_batch_size < 1:
            raise ContractError("SMPL-X metric batch size must be positive")
        self.smplx_batch_size = int(smplx_batch_size)
        self._gt_cache: dict[Any, Any] = {}
        encoder_files = sorted(Path(encoder_dir).glob("*.py"))
        self.instrument = {"aesk_sha256": sha256_file(checkpoint),
                           "encoder_source": str(Path(encoder_dir).resolve()),
                           "encoder_source_sha256": {path.name: sha256_file(path) for path in encoder_files},
                           "mean_velocity_sha256": sha256_file(self.assets / "mean_vel_smplxflame_30.npy"),
                           "smplx_sha256": sha256_file(self.assets / "smplx_models/smplx/SMPLX_NEUTRAL_2020.npz"),
                           "device": str(self.device), "aesk_input": "local AA165 to row-rot6D330",
                           "coordinate_transform": "none; BEAT2 SMPL-X Y-up", "normalization": "none before AESK",
                           "evaluation_source_sha256": sha256_file(__file__),
                           "dependency_versions": {name: version(name) for name in
                                                   ("numpy", "scipy", "torch", "smplx", "librosa", "soundfile")}}

    def encode(self, poses: np.ndarray) -> np.ndarray:
        length = len(poses) // 32 * 32
        if length < 32:
            raise ContractError("FGD clip must have >=32 frames")
        with self.torch.inference_mode():
            tensor = self.torch.as_tensor(poses[:length][None], dtype=self.torch.float32, device=self.device)
            latent = self.aesk.map2latent(aa_to_rot6d(tensor, self.torch)).detach().cpu().numpy()
        result = latent.reshape(-1, latent.shape[-1])
        if result.shape[1] != 240 or not np.isfinite(result).all():
            raise ContractError("AESK returned wrong/nonfinite latent")
        return result

    def positions(self, poses: np.ndarray, betas: np.ndarray) -> np.ndarray:
        betas = np.asarray(betas, dtype=np.float32).reshape(-1)
        if betas.shape != (300,) or not np.isfinite(betas).all():
            raise ContractError("GT betas must be finite static SMPL-X [300]")
        output = []
        torch = self.torch
        with torch.inference_mode():
            for start in range(0, len(poses), self.smplx_batch_size):
                pose = torch.as_tensor(poses[start:start+self.smplx_batch_size], device=self.device)
                count = len(pose)
                zeros = torch.zeros(count, 3, device=self.device)
                mesh = self.smplx(betas=torch.as_tensor(betas, device=self.device)[None].expand(count, -1),
                                  transl=zeros, expression=torch.zeros(count, 100, device=self.device),
                                  global_orient=pose[:, :3], body_pose=pose[:, 3:66], jaw_pose=pose[:, 66:69],
                                  leye_pose=pose[:, 69:72], reye_pose=pose[:, 72:75],
                                  left_hand_pose=pose[:, 75:120], right_hand_pose=pose[:, 120:165],
                                  return_joints=True)
                output.append(mesh.joints[:, :55].detach().cpu().numpy())
        return np.concatenate(output, axis=0)

    def audit_roundtrip(self, original_poses: np.ndarray, reconstructed_poses: np.ndarray,
                        betas: np.ndarray, *, pose_policy: str = "full55",
                        rot6d_atol: float = 2e-5, div_atol: float = 2e-5) -> dict[str, Any]:
        """Gate actual encode/normalize/inverse/decode GT with the metric assets.

        Caller reconstructs GT using its real model representation and train-only
        normalizer, then passes the resulting AA165 here. Both sides get the same
        pose mask. Rotation6D comparison avoids false failures at +/-pi AA wraps.
        No audio, inference, training, arbitrary coordinate transform or file I/O.
        """
        original = pose_mask(original_poses, pose_policy)
        reconstructed = pose_mask(reconstructed_poses, pose_policy)
        if original.shape != reconstructed.shape:
            raise ContractError("roundtrip must preserve the complete motion shape")
        torch = self.torch
        rotation_audit = audit_rotation_roundtrip(original, reconstructed, torch, self.device, atol=rot6d_atol)
        rotation_error = rotation_audit["primary_max_abs_error"]
        latent_original, latent_reconstructed = self.encode(original), self.encode(reconstructed)
        distance = frechet_distance(latent_reconstructed, latent_original)
        self_distance = frechet_distance(latent_original, latent_original)
        joints_original = self.positions(original, betas)
        joints_reconstructed = self.positions(reconstructed, betas)
        diversity_original, diversity_reconstructed = l1_diversity(joints_original), l1_diversity(joints_reconstructed)
        diversity_error = abs(diversity_original-diversity_reconstructed)
        passed = (rotation_audit["passed"] and rotation_error <= rot6d_atol and diversity_error <= div_atol
                  and abs(distance["value"]) <= min(1e-5, distance["numerical_tolerance"])
                  and abs(self_distance["value"]) <= min(1e-5, self_distance["numerical_tolerance"]))
        report = {"passed": bool(passed), "kind": "actual_GT_model_roundtrip_aesk_and_smplx",
                  "pose_policy": pose_policy, "frames": len(original),
                  "max_abs_rot6d_error": rotation_error, "rot6d_tolerance": rot6d_atol,
                  "rotation_conversion_audit": rotation_audit,
                  "max_abs_aa_error_wrap_sensitive": float(np.max(np.abs(original-reconstructed))),
                  "fgd_roundtrip": distance, "fgd_gt_self": self_distance,
                  "self_roundtrip_absolute_fgd_cap": 1e-5,
                  "div_gt": diversity_original, "div_roundtrip": diversity_reconstructed,
                  "div_abs_error": diversity_error, "div_tolerance": div_atol,
                  "axis_evidence": coordinate_axis_evidence(joints_original),
                  "instrument": self.instrument}
        if not passed:
            raise ContractError("real GT normalization roundtrip failed: " + json.dumps(report, allow_nan=False))
        return report

    def evaluate(self, prediction_root: str | Path, ground_truth_root: str | Path,
                 audio_root: str | Path, manifest: Any, *, split: str | None = None,
                 pose_policy: str = "full55", expected_count: int | None = None,
                 model_pose_policy: str | None = None) -> dict[str, Any]:
        import librosa
        import soundfile as sf

        manifest_payload = json.loads(Path(manifest).read_text(encoding="utf-8")) if isinstance(manifest, (str, Path)) else manifest
        records = manifest_records(manifest_payload, split, expected_count)
        prediction_root, ground_truth_root, audio_root = map(Path, (prediction_root, ground_truth_root, audio_root))
        features_p, features_g, per_clip = [], [], []
        accum = {name: 0.0 for name in ("div", "div_gt", "bc", "bc_gt", "bc_corrected", "bc_corrected_gt")}
        total_frames, bc_frames = 0, 0
        for record in records:
            file_id = record["id"]
            pred_path = prediction_root / f"res_{file_id}.npz"
            gt_path = _gt_path(ground_truth_root, file_id)
            audio_path = audio_root / f"{file_id}.wav"
            pred, gt = _read_archive(pred_path), _read_archive(gt_path)
            audio_info = sf.info(audio_path)
            available = min(len(gt["poses"]) // FPS, math.floor(audio_info.frames/audio_info.samplerate)) * FPS
            frame_override = record.get("frames", record.get("evaluation_frames"))
            frames = rollout_frames(available) if frame_override is None else int(frame_override)
            if frames > available or frames <= 2*ALIGN_MASK or frames % 4:
                raise ContractError(f"invalid explicit full-clip frame budget {frames} for {file_id}")
            if len(pred["poses"]) < frames:
                raise ContractError(f"prediction {file_id} has {len(pred['poses'])} frames, required {frames}; no silent min-crop")
            poses_p, poses_g = pose_mask(pred["poses"][:frames], pose_policy), pose_mask(gt["poses"][:frames], pose_policy)
            if "betas" not in gt:
                raise ContractError(f"GT missing static SMPL-X betas: {file_id}")
            gt_key = (str(gt_path.resolve()), gt_path.stat().st_mtime_ns, frames, pose_policy)
            if gt_key not in self._gt_cache:
                self._gt_cache[gt_key] = (self.encode(poses_g), self.positions(poses_g, gt["betas"]))
            latent_g, positions_g = self._gt_cache[gt_key]
            latent_p, positions_p = self.encode(poses_p), self.positions(poses_p, gt["betas"])
            if latent_p.shape != latent_g.shape:
                raise ContractError("FGD latent frame budget differs between pred and GT")
            features_p.append(latent_p)
            features_g.append(latent_g)
            wave_legacy, legacy_rate = librosa.load(audio_path)
            wave_legacy = librosa.resample(wave_legacy, orig_sr=legacy_rate, target_sr=AUDIO_SR)
            offset = int(ALIGN_MASK*AUDIO_SR/FPS)
            # Retain the official call's original-WAV end argument on the already
            # generated-length-trimmed wave, even if it exceeds the trimmed wave.
            legacy_segment = wave_legacy[:int(frames*AUDIO_SR/FPS)][offset:len(wave_legacy)-offset]
            wave_corrected = librosa.load(audio_path, sr=AUDIO_SR)[0]
            corrected_segment = wave_corrected[offset:int((frames-ALIGN_MASK)*AUDIO_SR/FPS)]
            audio_l = librosa.onset.onset_detect(y=legacy_segment, sr=AUDIO_SR, hop_length=512, units="time")
            audio_c = librosa.onset.onset_detect(y=corrected_segment, sr=AUDIO_SR, hop_length=512, units="time")
            clip = {"id": file_id, "frames": frames, "fgd_frames": frames//32*32,
                    "prediction_frames": len(pred["poses"]), "gt_frames": len(gt["poses"]),
                    "audio_frames_30fps": math.floor(audio_info.frames/audio_info.samplerate*FPS),
                    "audio_onsets_legacy": len(audio_l), "audio_onsets_corrected": len(audio_c)}
            for label, positions in (("", positions_p), ("_gt", positions_g)):
                flat = positions.reshape(frames, 165)
                div = l1_diversity(flat)
                mutated = np.abs(flat - np.mean(flat, axis=0))
                legacy_bc = beat_score(audio_l, motion_beats(mutated, self.mean_velocity, legacy=True))
                corrected_bc = beat_score(audio_c, motion_beats(flat, self.mean_velocity, legacy=False))
                clip.update({"div"+label: div, "bc"+label: legacy_bc, "bc_corrected"+label: corrected_bc})
                accum["div"+label] += div * frames
                accum["bc"+label] += legacy_bc * (frames-120)
                accum["bc_corrected"+label] += corrected_bc * (frames-120)
            per_clip.append(clip)
            total_frames += frames
            bc_frames += frames-120
        latent_p, latent_g = np.concatenate(features_p), np.concatenate(features_g)
        fgd = frechet_distance(latent_p, latent_g)
        self_fgd = frechet_distance(latent_g, latent_g)
        if abs(self_fgd["value"]) > min(1e-5, self_fgd["numerical_tolerance"]):
            raise ContractError(f"GT-self FGD is not approximately zero: {self_fgd}")
        ids_hash = hashlib.sha256(json.dumps([v["id"] for v in records], separators=(",", ":")).encode()).hexdigest()
        noncontrolled = model_pose_policy in ("body22", "body22only") and pose_policy == "full55"
        return {"schema": "music2gesture-semtalk-metrics-v1", "split": split,
                "sample_count": len(records), "ids_sha256": ids_hash, "pose_policy": pose_policy,
                "manifest_sha256": hashlib.sha256(json.dumps(manifest_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                "split_csv_sha256": manifest_payload.get("split_csv_sha256") if isinstance(manifest_payload, dict) else None,
                "model_pose_policy": model_pose_policy, "controlled_cross_model_comparison": not noncontrolled,
                "comparison_warning": "body-only full55 score includes missing hands; do not controlled-compare with full55 model" if noncontrolled else None,
                "fgd": fgd["value"], "fgd_numerical_audit": fgd, "fgd_gt_self_audit": self_fgd,
                "bc": accum["bc"]/bc_frames, "bc_gt": accum["bc_gt"]/bc_frames,
                "bc_corrected": accum["bc_corrected"]/bc_frames,
                "bc_corrected_gt": accum["bc_corrected_gt"]/bc_frames,
                "div": accum["div"]/total_frames, "div_gt": accum["div_gt"]/total_frames,
                "frame_count": total_frames, "fgd_frame_count": sum(v["fgd_frames"] for v in per_clip),
                "latent_rows": len(latent_g), "instrument": self.instrument, "raw_units": True,
                "fgd_protocol": "SemTalk sqrtm(Cpred*Cgt); epsilon only nonfinite fallback; independent symmetric PSD audit",
                "bc_protocol": "SemTalk legacy: DIV mutation, default librosa 22050->16000, original audio-end argument, crop-relative velocity mask; sigma=.3 order=7 threshold=.3 align_mask=60",
                "bc_corrected_protocol": "unmutated joints, direct 16000 audio, synchronized start/end crop, crop-offset-correct velocity mask; otherwise identical hyperparameters",
                "div_protocol": "per-clip sum abs(joints - temporal_mean)/frames; frame-weighted across clips",
                "pose_mask_application": "same local-AA mask on pred and GT; no GT oracle completion, no coordinate rotation",
                "face_metrics": "not reported: this experiment does not predict facial expression coefficients",
                "per_clip": per_clip}


def cpu_self_test() -> dict[str, Any]:
    """Synthetic CPU guards, not an assertion that a trained model is correct."""
    rng = np.random.default_rng(20261003)
    x = rng.normal(size=(160, 24)).astype(np.float64)
    self_fgd = frechet_distance(x, x)
    if abs(self_fgd["value"]) > 1e-7:
        raise AssertionError(f"FGD self-test failed: {self_fgd}")
    shifted = frechet_distance(x + 0.25, x)
    if abs(shifted["value"] - 24*0.25**2) > 1e-6:
        raise AssertionError("FGD mean-shift self-test failed")
    mean, std = rng.normal(size=24), rng.uniform(0.01, 2.0, size=24)
    roundtrip = normalization_roundtrip_report(x, mean, std)
    recovered = ((x-mean)/std)*std+mean
    rt_fgd = frechet_distance(recovered, x)
    if abs(rt_fgd["value"]) > 1e-7 or abs(l1_diversity(recovered)-l1_diversity(x)) > 1e-12:
        raise AssertionError("normalization roundtrip FGD/DIV self-test failed")
    raw = rng.normal(size=(124, 165)).astype(np.float32)
    masked = pose_mask(raw, "body22")
    if np.any(masked[:, 66:]) or not np.array_equal(masked[:, :66], raw[:, :66]):
        raise AssertionError("body mask self-test failed")
    if rollout_frames(1906) != 1864 or rollout_frames(1200) != 1144:
        raise AssertionError("rollout boundary self-test failed")
    identical_beats = [np.array([30, 60]) for _ in range(55)]
    if abs(beat_score(np.array([1., 2.]), identical_beats)-1.) > 1e-12:
        raise AssertionError("directed BC self-test failed")
    return {"passed": True, "kind": "synthetic_cpu_contract_tests", "fgd_self": self_fgd,
            "normalization_roundtrip": roundtrip, "normalization_roundtrip_fgd": rt_fgd,
            "normalization_roundtrip_div_error": abs(l1_diversity(recovered)-l1_diversity(x)),
            "gpu_or_real_aesk_run": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--predictions")
    parser.add_argument("--ground-truth")
    parser.add_argument("--audio")
    parser.add_argument("--manifest")
    parser.add_argument("--split", choices=("val", "test"))
    parser.add_argument("--expected-count", type=int)
    parser.add_argument("--pose-policy", choices=("full55", "body22"), default="full55")
    parser.add_argument("--model-pose-policy", choices=("full55", "body22"))
    parser.add_argument("--metric-assets")
    parser.add_argument("--encoder-dir")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.self_test:
        result = cpu_self_test()
    else:
        for key in ("predictions", "ground_truth", "audio", "manifest", "metric_assets", "encoder_dir"):
            if not getattr(args, key):
                parser.error(f"--{key.replace('_', '-')} is required for evaluation")
        expected = args.expected_count if args.expected_count is not None else {"val": 106, "test": 265}.get(args.split)
        evaluator = SemTalkEvaluator(args.metric_assets, args.encoder_dir, args.device)
        result = evaluator.evaluate(args.predictions, args.ground_truth, args.audio, args.manifest,
                                    split=args.split, pose_policy=args.pose_policy, expected_count=expected,
                                    model_pose_policy=args.model_pose_policy)
    payload = json.dumps(result, indent=2, allow_nan=False)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload+"\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
