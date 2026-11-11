"""Opt-in, single-GPU test265 replay. Default is a stdlib-only verified plan.

Four hash-verified function definitions are loaded from train.py.
The training module and controller are not executed by this inference entry.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import random
import re
import socket
import sys
import time

SOURCE_PREFIX = "tasks/s2g/snapshots/full55_r2/code/"
SOURCE_FILES = (
    "model.py", "loss.py", "data.py", "evaluation.py", "train.py", "prepare.py",
    "render.py", "render_review.py", "verify.py", "native_jvp_check.py",
    "native_higherorder_jvp.py", "controller.py", "config.target.json",
    "baselines.template.json", "paper_table.json",
)
FUNCTIONS = ("config_hash", "require_numerical_environment", "project_rotations", "export_predictions")
COUNTS = {"train": 1093, "val": 106, "test": 265}
PERSON_IDS = {1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 27, 28, 30}
NUMERICAL_ENV = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
ASSET_HASHES = {
    "AESKConv_240_100.bin": "5cd9566b24264f34d44003b3de62cdfd50aa85b7cdde2d369214599023c40f55",
    "mean_vel_smplxflame_30.npy": "53b5e48f2a7bf78c41a6de6395d6bb4f29018465ca5d0ee2820a2be3eebb7137",
    "smplx_models/smplx/SMPLX_NEUTRAL_2020.npz": "bdf06146e27d92022fe5dadad3b9203373f6879eca8e4d8235359ee3ec6a5a74",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def clip_seed(ident):
    return 42 + int(hashlib.sha256(ident.encode()).hexdigest()[:8], 16)


def digest_arg(value):
    if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
        raise argparse.ArgumentTypeError("expected a 64-hex SHA256")
    return value.lower()


def fraction_arg(value):
    result = float(value)
    if not math.isfinite(result) or not 0 < result <= 0.08:
        raise argparse.ArgumentTypeError("GPU memory fraction must be >0 and <=0.08")
    return result


def absolute(value):
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("paths must be explicit absolute paths without '..'")
    return path.resolve()


def verified_file(path, expected):
    path = absolute(path)
    if not path.is_file() or sha256(path) != expected:
        raise ValueError("missing or SHA256-mismatched input: " + str(path))
    return path


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def native_ast(source):
    tree = ast.parse(Path(source).read_text(encoding="utf-8"))
    definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    if any(name not in definitions for name in FUNCTIONS):
        raise ValueError("frozen inference function missing")
    # Preserve actual native AST, including its stable seed, projection and exports.
    return ast.Module(body=[definitions[name] for name in FUNCTIONS], type_ignores=[])


def verify_sources(source_dir, proof_path, proof_sha):
    source_dir = absolute(source_dir)
    proof = read_json(verified_file(proof_path, proof_sha))
    records = {}
    for record in proof.get("files", []):
        name = record.get("path", "")
        if name.startswith(SOURCE_PREFIX):
            leaf = name[len(SOURCE_PREFIX):]
            if leaf not in SOURCE_FILES or leaf in records:
                raise ValueError("unexpected/duplicate frozen source proof entry")
            records[leaf] = record
    if set(records) != set(SOURCE_FILES):
        raise ValueError("source proof must bind all 15 full55-r2 files")
    for leaf, record in records.items():
        path = verified_file(source_dir / leaf, record["sha256"])
        if path.stat().st_size != record["bytes"]:
            raise ValueError("source byte count differs: " + leaf)
    native_ast(source_dir / "train.py")
    return {leaf: record["sha256"] for leaf, record in records.items()}


def fresh_output(output, protected):
    output = absolute(output)
    if output.exists() or not output.parent.is_dir() or output == output.parent:
        raise ValueError("output must not exist and its parent must already exist")
    for value in protected:
        path = absolute(value)
        if output.is_relative_to(path) or path.is_relative_to(output):
            raise ValueError("output overlaps a protected input/historical tree")
    return output


def make_plan(args):
    if args.expected_epoch < 1 or args.metric_batch_size < 1:
        raise ValueError("epoch and metric batch size must be positive")
    sources = verify_sources(args.source_dir, args.source_proof, args.source_proof_sha256)
    config_path = verified_file(args.config, args.config_sha256)
    if args.config_sha256 != sources["config.target.json"]:
        raise ValueError("config must match the frozen source proof (no silent relocation)")
    config = read_json(config_path)
    for key in ("cache_root", "raw_root", "metric_assets", "evaluator_assets"):
        if absolute(getattr(args, key)) != absolute(config[key]):
            raise ValueError(key + " must match the original bound config for this replay")
    if config.get("seed") != 42 or config.get("numerical_environment") != NUMERICAL_ENV:
        raise ValueError("config seed/numerical protocol differs")
    checkpoint = verified_file(args.checkpoint, args.checkpoint_sha256)
    normalizer = verified_file(args.normalizer, args.normalizer_sha256)
    manifest_path = verified_file(args.manifest, args.manifest_sha256)
    cache_root = absolute(args.cache_root)
    if normalizer != cache_root / "stats_full55.npz" or manifest_path != cache_root / "manifest.json":
        raise ValueError("normalizer/manifest must be the bound full55 cache files")
    manifest = read_json(manifest_path)
    splits = manifest.get("splits", {})
    if {key: len(splits.get(key, [])) for key in COUNTS} != COUNTS:
        raise ValueError("manifest must preserve train1093/val106/test265")
    all_ids = []
    for split in COUNTS:
        values = splits[split]
        if any(not isinstance(ident, str) or not re.fullmatch(r"[0-9]+_[A-Za-z0-9_-]+", ident) for ident in values):
            raise ValueError("unsafe/non-native manifest ID")
        coverage = {int(ident.split("_", 1)[0]) for ident in values}
        if coverage != (PERSON_IDS - {6, 22} if split == "val" else PERSON_IDS):
            raise ValueError("official speaker coverage differs")
        all_ids.extend(values)
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("manifest duplicates/cross-split leakage")
    integrity_path = verified_file(args.cache_integrity, args.cache_integrity_sha256)
    integrity = read_json(integrity_path)
    clips = integrity.get("clips", [])
    if (integrity.get("kind") != "complete_feature_cache_integrity_inventory" or
            integrity.get("passed") is not True or integrity.get("clip_count") != 1464 or
            integrity.get("manifest_sha256") != args.manifest_sha256 or
            [item.get("id") for item in clips] != all_ids):
        raise ValueError("cache integrity proof does not cover the original exact manifest")
    if integrity.get("stats", {}).get("full55", {}).get("sha256") != args.normalizer_sha256:
        raise ValueError("normalizer is not bound by cache proof")
    clip_map = {item["id"]: item for item in clips}
    tested = []
    for ident in splits["test"]:
        record = clip_map[ident]
        path = cache_root / (ident + ".npz")
        if absolute(record["path"]) != path:
            raise ValueError("cache proof path differs: " + ident)
        verified_file(path, record["sha256"])
        for folder, suffix in (("smplxflame_30", ".npz"), ("wave16k", ".wav")):
            if not (absolute(args.raw_root) / folder / (ident + suffix)).is_file():
                raise ValueError("missing raw test input: " + ident)
        tested.append({"id": ident, "cache_sha256": record["sha256"], "sampling_seed": clip_seed(ident)})
    assets = {name: str(verified_file(absolute(args.metric_assets) / name, digest)) for name, digest in ASSET_HASHES.items()}
    vendor = {}
    for name in ("motion_encoder.py", "decoders.py", "skeleton.py", "skeleton_DME.py"):
        path = absolute(args.evaluator_assets) / name
        if not path.is_file():
            raise ValueError("missing original evaluator source: " + name)
        vendor[name] = sha256(path)
    numerical_path = verified_file(config["numerical_environment_evidence"], config["numerical_environment_evidence_sha256"])
    output = fresh_output(args.output, [args.source_dir, args.cache_root, args.raw_root, args.metric_assets,
        args.evaluator_assets, checkpoint.parent, config["output_root"], numerical_path.parent])
    return {"schema": "s2g-independent-test265-replay-v1", "status": "verified_plan_not_executed",
        "training": False, "gpu_used": False, "checkpoint_tensors_loaded": False,
        "checkpoint": str(checkpoint), "checkpoint_sha256": args.checkpoint_sha256,
        "expected_epoch": args.expected_epoch, "config_sha256": canonical_hash(config),
        "config_file_sha256": args.config_sha256, "normalizer_sha256": args.normalizer_sha256,
        "manifest_sha256": args.manifest_sha256, "cache_integrity_sha256": args.cache_integrity_sha256,
        "source_proof_sha256": args.source_proof_sha256, "source_sha256": sources,
        "evaluator_source_sha256_observed": vendor, "asset_paths": assets,
        "environment": NUMERICAL_ENV, "seed": 42, "sampling_steps": args.steps,
        "split": "test", "count": 265, "mode": "full55", "test_clips": tested,
        "gpu_memory_fraction": args.gpu_memory_fraction,
        "memory_cap_scope": "PyTorch caching allocator; root must separately monitor total driver/native/context memory",
        "metric_batch_size": args.metric_batch_size, "output": str(output),
        "prediction_directory": str(output / "predictions" / ("nfe" + str(args.steps))),
        "metric_output": str(output / ("metrics.full55.test265.nfe" + str(args.steps) + ".json")),
        "original_training_gates_modified": False}


def write_new_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def execute(args, plan):
    # No native import or output creation occurs before explicit --execute.
    if any(os.environ.get(key) != value for key, value in NUMERICAL_ENV.items()):
        raise ValueError("set pinned numerical environment before starting Python")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
    if len(visible) != 1 or not visible[0].strip() or visible[0].strip() == "-1":
        raise ValueError("explicitly expose exactly one checked GPU with CUDA_VISIBLE_DEVICES")
    for name in ("model", "data", "loss", "evaluation", "native_higherorder_jvp"):
        if name in sys.modules:
            raise ValueError("native module namespace already occupied: " + name)
    source_dir = absolute(args.source_dir)
    sys.path.insert(0, str(source_dir))
    import numpy as np
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("exactly one available CUDA GPU required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, device)
    torch.cuda.reset_peak_memory_stats(device)
    total = torch.cuda.get_device_properties(device).total_memory
    free, _ = torch.cuda.mem_get_info(device)
    if free < total * args.gpu_memory_fraction:
        raise ValueError("checked GPU lacks the requested bounded free memory")
    data = importlib.import_module("data")
    loss = importlib.import_module("loss")
    model_module = importlib.import_module("model")
    evaluation = importlib.import_module("evaluation")
    namespace = {"torch": torch, "np": np, "Path": Path, "os": os, "sys": sys,
        "socket": socket, "hashlib": hashlib, "json": json, "SEED": 42,
        "EXPECTED_COUNTS": COUNTS, "sha256": sha256, "decode_motion": data.decode_motion,
        "matrix_to_rotation_6d": loss.matrix_to_rotation_6d,
        "rotation_6d_to_matrix": loss.rotation_6d_to_matrix}
    exec(compile(native_ast(source_dir / "train.py"), str(source_dir / "train.py"), "exec"), namespace)
    config = read_json(args.config)
    namespace["require_numerical_environment"](config)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Formal epoch checkpoint only, no unsafe pickle fallback or latest_full resume.
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    metadata = state.get("metadata", {})
    runtime_sources = {name: plan["source_sha256"][name] for name in SOURCE_FILES if name.endswith(".py")}
    if (state.get("epoch") != args.expected_epoch or state.get("config_sha256") != plan["config_sha256"] or
            metadata.get("config_sha256") != plan["config_sha256"] or metadata.get("mode") != "full55" or
            metadata.get("seed") != 42 or metadata.get("normalizer_sha256") != args.normalizer_sha256 or
            metadata.get("manifest_sha256") != args.manifest_sha256 or metadata.get("source_hashes") != runtime_sources):
        raise ValueError("checkpoint epoch/config/source/normalizer/manifest binding differs")
    model = model_module.GestureM2D(mode="full55", **config["architecture"])
    architecture = model.architecture_config()
    if state.get("architecture") != architecture or metadata.get("architecture") != architecture or architecture.get("test_backend"):
        raise ValueError("checkpoint architecture/native backend differs")
    model.load_state_dict(state["model"], strict=True)
    del state
    model = model.eval().requires_grad_(False).to(device)
    normalizer = data.Standardizer.load(args.normalizer)
    manifest = read_json(args.manifest)
    output = Path(plan["output"])
    output.mkdir(exist_ok=False)
    plan.update(status="started", gpu_used=True, checkpoint_tensors_loaded=True,
        started_at_unix=time.time(), torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
        gpu_total_bytes=total)
    write_new_json(output / "started.json", plan)
    try:
        sampled = namespace["export_predictions"](model, normalizer, config, manifest, "test", args.steps,
            plan["prediction_directory"], device, args.checkpoint_sha256)
        if [record["id"] for record in sampled] != manifest["splits"]["test"]:
            raise ValueError("inference did not preserve exact test265 IDs/order")
        if any(record["sampling_seed"] != clip_seed(record["id"]) for record in sampled):
            raise ValueError("native per-ID noise seed differs")
        artifacts = []
        for record in sampled:
            path = Path(plan["prediction_directory"]) / ("res_" + record["id"] + ".npz")
            artifacts.append({**record, "path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size})
        write_new_json(output / "prediction_artifacts.json", {"count": 265, "checkpoint_sha256": args.checkpoint_sha256,
            "manifest_sha256": args.manifest_sha256, "sampling_steps": args.steps, "seed": 42, "files": artifacts})
        del model
        gc.collect()
        torch.cuda.empty_cache()
        evaluator = evaluation.SemTalkEvaluator(args.metric_assets, args.evaluator_assets,
            device="cuda:0", smplx_batch_size=args.metric_batch_size)
        metrics = evaluator.evaluate(plan["prediction_directory"], Path(args.raw_root) / "smplxflame_30",
            Path(args.raw_root) / "wave16k", manifest, split="test", pose_policy="full55",
            expected_count=265, model_pose_policy="full55")
        if metrics.get("sample_count") != 265:
            raise ValueError("evaluator did not score every test ID")
        write_new_json(plan["metric_output"], metrics)
        torch.cuda.synchronize(device)
        memory = {"peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device), "total_bytes": total,
            "allocator_cap_fraction": args.gpu_memory_fraction, "total_driver_memory_not_measured": True}
        if max(memory["peak_allocated_bytes"], memory["peak_reserved_bytes"]) > total * args.gpu_memory_fraction:
            raise ValueError("observed PyTorch peak exceeds explicit allocator cap")
        result = {**plan, "status": "complete", "completed_at_unix": time.time(), "memory": memory,
            "metrics": {key: metrics[key] for key in ("fgd", "bc", "div", "bc_corrected", "frame_count")},
            "metric_file_sha256": sha256(plan["metric_output"]), "rendered_media": False,
            "independent_original_metric_acceptance_claimed": False}
        write_new_json(output / "complete.json", result)
        return result
    except Exception as error:
        write_new_json(output / "failed.json", {"status": "failed", "type": type(error).__name__,
            "message": str(error)[:1000], "completed_at_unix": time.time(), "resume_supported": False})
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--dry-run", action="store_true")
    for name in ("checkpoint", "config", "normalizer", "manifest", "source-proof", "cache-integrity"):
        result.add_argument("--" + name, type=Path, required=True)
        result.add_argument("--" + name + "-sha256", type=digest_arg, required=True)
    for name in ("source-dir", "cache-root", "raw-root", "metric-assets", "evaluator-assets", "output"):
        result.add_argument("--" + name, type=Path, required=True)
    result.add_argument("--seed", type=int, choices=(42,), required=True)
    result.add_argument("--steps", type=int, choices=(1, 4), required=True)
    result.add_argument("--expected-epoch", type=int, required=True)
    result.add_argument("--gpu-memory-fraction", type=fraction_arg, default=0.08)
    result.add_argument("--metric-batch-size", type=int, default=128)
    return result


def main(argv=None):
    arguments = parser()
    args = arguments.parse_args(argv)
    try:
        plan = make_plan(args)
        result = execute(args, plan) if args.execute else plan
    except (ValueError, OSError, KeyError, TypeError) as error:
        arguments.error(str(error))
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
