#!/usr/bin/env python3
"""Opt-in fresh reproduction of the two selected Full main-table views.

Default is a stdlib-only dry-run. Native kernels are imported only by --execute.
This is neither the historical launcher nor the original-periodic evaluator CLI.
No cached generation, old-output reuse, training, resume, retry, or GPU polling.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import runpy
import sys
import time


RELEASE_ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
LIVE_ROOT = "/efs/yangkaixing/FlowerDance_T2M_Long"
HISTORICAL_SOURCE_SHA = "bff7bf408d22bb398ba75811a04b2d256ab7eb91912803b218be9cc9b171957c"
CANONICAL_SHA = "9ed7762556bb307bc147358215f849f2f43a20676f703445408269beafd4af9d"
CANONICAL_FIELDS = {"ground_truth", "text", "segment_key", "caption", "episode_id", "segment_length"}
SELECTED = {
    1: {"step": 100100, "epoch": 7150, "tag": "mixed_1step_e7150",
        "checkpoint_sha256": "d1f5848ca560c7812566abef413d1f532e3652f44030af4388ef3447a5932846"},
    4: {"step": 81900, "epoch": 5850, "tag": "mixed_4step_e5850",
        "checkpoint_sha256": "43cecbb482982352754364706ee14932812a22da70606fc1ec412c4663e19c9d"},
}


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fresh_output(path: Path, root: Path) -> None:
    path, root = path.resolve(), root.resolve()
    if path == root or root not in path.parents:
        raise ValueError("Output must be a new child of configured output_root")
    normalized = path.as_posix().lower()
    if LIVE_ROOT.lower() in normalized:
        raise ValueError("Output inside the original live T2M tree is forbidden")
    if (HERE / "snapshots").resolve() in path.parents:
        raise ValueError("Output inside frozen snapshots is forbidden")
    if path.exists() or path.is_symlink():
        raise ValueError("Output exists: no reuse, skip, resume, overwrite, or automatic retry")


def build_plan(args: argparse.Namespace) -> dict:
    native = runpy.run_path(str(HERE / "run.py"), run_name="t2m_reproduction_planner")
    config_path = Path(args.paths).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    paths = {key: native["resolve_path"](raw[key], config_path.parent)
             for key in ("data_root", "embedding_root", "flood_root", "cohort_root", "output_root")}
    if args.seed != 42:
        raise ValueError("Historical main-table base seed is fixed to 42")
    if not math.isfinite(args.cap) or not 0 < args.cap <= .08:
        raise ValueError("Evaluation allocator cap must be finite and in (0, .08]")
    if args.checkpoint_sha256 != SELECTED[args.nfe]["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA does not match the actual selected main-table checkpoint")
    if args.canonical_sha256 != CANONICAL_SHA:
        raise ValueError("Canonical SHA differs from the independently recorded canonical fixture")
    gpu = args.gpu if args.gpu is not None else raw.get("evaluation_gpus", "0")
    if not isinstance(gpu, str) or not gpu.isdigit():
        raise ValueError("Choose exactly one nonnegative GPU id; no forced historical GPU0")
    output = native["resolve_path"](args.output, config_path.parent)
    fresh_output(output, paths["output_root"])
    checkpoint = native["resolve_path"](args.checkpoint, config_path.parent)
    canonical = native["resolve_path"](args.canonical, config_path.parent)
    historical = native["resolve_path"](args.historical_source, config_path.parent)
    packaged_code, _, source = native["frozen_snapshot"]("full")
    code = native["resolve_path"](args.code_root, config_path.parent) if args.code_root else packaged_code
    if code != packaged_code:
        collection = json.loads((RELEASE_ROOT / "manifests/frozen_source_collection.json").read_text(encoding="utf-8"))
        prefix = "tasks/t2m/snapshots/restart_20260922/code/"
        for record in collection["files"]:
            if record["path"].startswith(prefix):
                path = code / record["path"][len(prefix):]
                if sha256(path) != record["sha256"]:
                    raise ValueError("External frozen code differs: " + record["path"])
    missing = native["missing_inputs"](paths, "eval", checkpoint)
    for path in (canonical, historical):
        if not path.is_file():
            missing.append("Missing input file: " + str(path))
    if historical.is_file() and sha256(historical) != HISTORICAL_SOURCE_SHA:
        raise ValueError("Historical reference-script bytes changed; its launcher is never executed")
    return {"schema_version": 1, "mode": "main_uniform_reencoding_fresh_generation",
            "dry_run": not args.execute, "executed": False, "selected": SELECTED[args.nfe],
            "nfe": args.nfe, "seed": 42, "checkpoint": str(checkpoint),
            "canonical": str(canonical), "canonical_sha256": args.canonical_sha256,
            "canonical_sha_provenance": "Input SHA256 recorded on 2026-10-02; original launch input bytes were not verified.",
            "historical_source": str(historical), "historical_source_sha256": HISTORICAL_SOURCE_SHA,
            "code": str(code), "source": source, "paths": {key: str(value) for key, value in paths.items()},
            "output": str(output), "gpu": gpu, "allocator_cap": args.cap,
            "precision": {"generation_matmul": "high", "embedding_matmul": "highest", "embedding_batch": 32},
            "protocol": "BABEL val, full episodes in dataset id order intersect canonical cohort; Mixed single+multi",
            "reuse_raw": False, "resume": False, "retry": False,
            "missing_inputs": missing, "runtime_verification": "NOT_RUN"}


def finite_json(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_json(item) for item in value)
    return True


def execute(plan: dict) -> int:
    if plan["missing_inputs"]:
        raise ValueError("Execution refused until all explicit runtime assets are supplied")
    checkpoint, canonical = Path(plan["checkpoint"]), Path(plan["canonical"])
    # Hash trusted checkpoint BEFORE unpickling; use actual selected weights, not matching names.
    if sha256(checkpoint) != plan["selected"]["checkpoint_sha256"]:
        raise ValueError("Actual checkpoint bytes differ from selected SHA")
    if sha256(canonical) != plan["canonical_sha256"]:
        raise ValueError("Actual canonical fixture bytes differ from pinned SHA")
    output = Path(plan["output"])
    fresh_output(output, Path(plan["paths"]["output_root"]))
    # Native imports begin only beyond the explicit --execute boundary.
    os.environ["CUDA_VISIBLE_DEVICES"] = plan["gpu"]
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    code = Path(plan["code"])
    sys.path.insert(0, str(code))
    import numpy as np
    import torch

    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(plan["allocator_cap"])
    torch.set_float32_matmul_precision("high")
    spec = importlib.util.spec_from_file_location("flashmotion_main_native_kernels", code / "evaluate_t2m_ablation.py")
    kernels = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kernels)
    from t2m_long.evaluator import FloodT2MEvaluator
    from t2m_long.babel_long_benchmark import MultiPredictionEmbeddingCollector, centered_transition, TRANSITION_FRAMES, score_embedding_subset, transition_metrics

    with np.load(canonical, allow_pickle=False) as stored:
        if set(stored.files) != CANONICAL_FIELDS:
            raise ValueError("Canonical fixture must contain exactly the authenticated six fields")
        reference = {key: stored[key].copy() for key in stored.files}
    allowed = set(reference["episode_id"].tolist())
    if len(allowed) != 311 or len(reference["segment_key"]) != 621:
        raise ValueError("Canonical coverage must be 311 episodes / 621 segments")
    if not all(np.isfinite(reference[key]).all() for key in ("ground_truth", "text", "segment_length")):
        raise ValueError("Canonical numeric fields are nonfinite")
    paths = {key: Path(value) for key, value in plan["paths"].items()}
    cohort = {path.stem.split("--", 1)[0] for path in paths["cohort_root"].glob("*.npz") if "--" in path.stem}
    if cohort != allowed:
        raise ValueError("Explicit cohort differs from canonical episode identities")
    dataset = kernels.MotionTextDataset(paths["data_root"], paths["embedding_root"], "babel", "val", max_length=None)
    if not allowed.issubset(dataset.ids):
        raise ValueError("Canonical episode absent from BABEL validation")
    device = torch.device("cuda")
    model, record, variant = kernels.load_objective(checkpoint, paths["data_root"], device)
    if int(record["step"]) != plan["selected"]["step"] or float(record["epoch"]) != plan["selected"]["epoch"]:
        raise ValueError("Checkpoint metadata step/epoch differs from selected main view")
    if "ema_decoder" not in record:
        raise ValueError("Selected main reproduction requires actual EMA; no live-weight fallback")
    if not all(torch.isfinite(value).all() for value in record["ema_decoder"].values()):
        raise ValueError("Nonfinite selected EMA")
    ema_hash = hashlib.sha256()
    for name, value in sorted(record["ema_decoder"].items()):
        ema_hash.update(name.encode())
        ema_hash.update(str(tuple(value.shape)).encode())
        ema_hash.update(value.detach().cpu().numpy().tobytes())
    ema_evidence = {"sha256": ema_hash.hexdigest(), "algorithm": "sorted(name UTF8 + shape str + original dtype tensor bytes)",
                    "tensor_count": len(record["ema_decoder"]), "all_finite": True}
    del record
    sampling_spec = importlib.util.spec_from_file_location("flashmotion_t2m_direct_sampling", HERE / "sampling.py")
    sampling = importlib.util.module_from_spec(sampling_spec)
    sampling_spec.loader.exec_module(sampling)
    sampler_evidence = sampling.install_direct_sampler(model)
    evaluator = FloodT2MEvaluator(paths["flood_root"], device, seed=42)
    collector = MultiPredictionEmbeddingCollector(evaluator, ["single", "multi"], 32)
    transitions = {key: [] for key in ("transition__ground_truth", "transition__multi", "transition_key", "transition_episode_id")}
    fresh_output(output, paths["output_root"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fresh_output(output, paths["output_root"])
    output.mkdir(exist_ok=False)
    raw_root = output / "raw"
    raw_root.mkdir(exist_ok=False)
    started = time.time()
    (output / "reproduction_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    count = 0
    try:
        for index, name in enumerate(dataset.ids):
            if name not in allowed:
                continue
            item = dataset[index]
            segments = kernels.effective_segments(item["text_ends"], item["length"])
            ground_truth = item["motion"] * torch.from_numpy(dataset.std) + torch.from_numpy(dataset.mean)
            torch.set_float32_matmul_precision("high")
            with torch.inference_mode():
                arrays = {"multi": kernels.sample_episode(model, item["condition"], plan["nfe"],
                          kernels.stable_seed(42, variant, name, plan["nfe"]))[:item["length"]].cpu().numpy()}
                for segment_index, segment in enumerate(segments):
                    arrays[f"single_{segment_index}"] = kernels.sample_episode(model,
                        item["condition"][segment.start:segment.end], plan["nfe"],
                        kernels.stable_seed(42, variant, name, segment_index, "single", plan["nfe"])).cpu().numpy()
            if not all(np.isfinite(value).all() for value in arrays.values()):
                raise FloatingPointError("Nonfinite freshly generated raw motion")
            np.savez_compressed(raw_root / f"{name}.npz", **arrays)
            multi = torch.from_numpy(arrays["multi"])
            torch.set_float32_matmul_precision("highest")
            for segment_index, segment in enumerate(segments):
                collector.add(ground_truth[segment.start:segment.end], item["text_tokens"][segment.annotation_index],
                              {"multi": multi[segment.start:segment.end], "single": torch.from_numpy(arrays[f"single_{segment_index}"])},
                              {"segment_key": f"{name}::{segment_index}", "episode_id": name,
                               "caption": item["captions"][segment.annotation_index], "segment_length": segment.end - segment.start})
            gt_joints = kernels.recover_joints_from_ric(ground_truth[None])[0]
            pred_joints = kernels.recover_joints_from_ric(multi[None])[0]
            for transition_index, segment in enumerate(segments[1:]):
                gt = centered_transition(gt_joints, segment.start, TRANSITION_FRAMES)
                pred = centered_transition(pred_joints, segment.start, TRANSITION_FRAMES)
                if gt is None or pred is None:
                    continue
                transitions["transition__ground_truth"].append(gt)
                transitions["transition__multi"].append(pred)
                transitions["transition_key"].append(f"{name}::{transition_index}")
                transitions["transition_episode_id"].append(name)
            count += 1
            print(json.dumps({"stage": "fresh_generated", "episodes": count, "total": 311}), flush=True)
        torch.set_float32_matmul_precision("highest")
        embeddings = collector.finish()
        embeddings.update({key: np.asarray(value) for key, value in transitions.items()})
        if (count, len(embeddings["segment_key"]), len(embeddings["transition_key"])) != (311, 621, 297):
            raise ValueError("Fresh generation count differs from 311/621/297")
        for key in CANONICAL_FIELDS:
            if not np.array_equal(embeddings[key], reference[key]):
                raise ValueError("Canonical exact equality failed: " + key)
        if not all(np.isfinite(value).all() for value in embeddings.values() if value.dtype.kind in "fiu"):
            raise FloatingPointError("Nonfinite fresh embedding/transition arrays")
        np.savez_compressed(output / "embeddings.npz", **embeddings)
        mixed = {key: np.concatenate([embeddings[key], embeddings[key]])
                 for key in ("ground_truth", "text", "caption", "episode_id", "segment_length")}
        mixed["prediction"] = np.concatenate([embeddings["prediction__single"], embeddings["prediction__multi"]])
        metrics = score_embedding_subset(mixed, "prediction", mixed["segment_length"] >= 4, torch.device("cpu"), seed=42)
        result = {"tag": plan["selected"]["tag"], "epoch": plan["selected"]["epoch"], "step": plan["selected"]["step"],
                  "mode": "mixed", "sampling_steps": plan["nfe"], "metrics": metrics,
                  "transition": transition_metrics(embeddings, "multi"), "canonical_gt_exact": True,
                  "selection": "Evaluate the supplied paper checkpoint with fresh generation and uniform re-encoding; checkpoint ranking is not performed"}
        if not finite_json(result):
            raise FloatingPointError("Nonfinite metric output")
        (output / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        runtime = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
                   "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0),
                   "cudnn": torch.backends.cudnn.version(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                   "cudnn_deterministic": torch.backends.cudnn.deterministic, "cudnn_benchmark": torch.backends.cudnn.benchmark,
                   "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                   "seed_policy": "Historical explicit stable_seed generators and metric seed42; no added global eval reseeding"}
        raw = [{"path": path.name, "sha256": sha256(path), "bytes": path.stat().st_size} for path in sorted(raw_root.glob("*.npz"))]
        receipt = {"status": "completed", "executed": True, "started": started, "ended": time.time(),
                   "checkpoint_sha256": plan["selected"]["checkpoint_sha256"], "canonical_sha256": plan["canonical_sha256"],
                   "ema": ema_evidence, "episodes": count, "segments": 621, "transitions": 297, "mixed_segments": 1242,
                   "sampling_optimization": sampler_evidence,
                   "canonical_gt_exact": True, "metrics_sha256": sha256(output / "metrics.json"),
                   "embeddings_sha256": sha256(output / "embeddings.npz"), "fresh_raw": raw,
                   "runtime": runtime, "no_cached_raw_reuse": True, "automatic_retry": False,
                   "historical_metrics_match": "NOT_CHECKED_BY_THIS_RUNNER; requires independent full-precision comparison"}
        (output / "reproduction_receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        return 0
    except Exception as exc:
        (output / "reproduction_failure.json").write_text(json.dumps({"status": "failed", "executed": True,
            "exception_type": type(exc).__name__, "completed_episodes": count, "automatic_retry": False}, indent=2) + "\n", encoding="utf-8")
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paths", default=str(RELEASE_ROOT / "configs/t2m.paths.example.json"))
    parser.add_argument("--nfe", type=int, choices=tuple(SELECTED), required=True)
    parser.add_argument("--checkpoint", required=True, help="Explicit trusted selected checkpoint pickle")
    parser.add_argument("--checkpoint-sha256", required=True, help="Actual fresh selected checkpoint SHA; validated before load")
    parser.add_argument("--canonical", required=True, help="Authenticated six-field canonical_gt.npz")
    parser.add_argument("--canonical-sha256", default=CANONICAL_SHA)
    parser.add_argument("--historical-source", default=str(HERE / "reference/ours_reencode_20260925.py"))
    parser.add_argument("--code-root", help="Optional existing frozen restart code, verified against every collection pin")
    parser.add_argument("--output", required=True, help="Nonexistent new child of paths output_root, outside original live tree")
    parser.add_argument("--gpu", help="One GPU id, otherwise paths evaluation_gpus; caller owns shared flock")
    parser.add_argument("--cap", type=float, default=.08)
    parser.add_argument("--seed", type=int, default=42)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true")
    action.add_argument("--execute", action="store_true", help="Explicitly authorize fresh native inference in this process")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        print(json.dumps(plan, indent=2), flush=True)
        return execute(plan) if args.execute else 0
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        print("Main reproduction refused: " + str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
