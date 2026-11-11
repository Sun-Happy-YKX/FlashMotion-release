"""Full-epoch, fail-closed eight-GPU Music2Gesture training.

No remote launch occurs here. PRECHECK_PASSED.json must contain real-data/GPU
evidence; calibration never manufactures that gate. Each mode/BS run starts
from seed-42 random initialization, independently of all prior experiments.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler, Subset

from data import ClipWindowDataset, EXPECTED_COUNTS, Standardizer, atomic_json, decode_motion, sha256, windows
from loss import MeanFlowLoss, matrix_to_rotation_6d, rotation_6d_to_matrix
from model import GestureM2D
from render_review import require_visual_review

SEED, EPOCHS, SAVE_EVERY, WORLD_SIZE = 42, 10000, 50, 8
BS_PROBE_PROTOCOL = "real_long_and_mixed_short_v2"
BS_UPWARD = (16, 32, 64, 128, 256, 512)
# B=1 cannot contain both an actual long AND actual short training window.
BS_FALLBACK = (8, 4, 2)
BS_SCENARIOS = ("long240", "mixed_real_short")
BS_NO_SHORT_KIND = "actual_training_clip_header_inventory_NOT_GPU_clearance"
BS_WINDOW_POLICY = "Long tails are end-aligned full240; only a full training clip<240 yields a real masked short window"
GATE_CHECKS = ("gpu8_idle", "gt_self_fgd", "normalization", "axis", "smplx_fk",
               "actual_three_clip_render", "native_cuda_jvp")
SOURCE_FILES = ("model.py", "loss.py", "data.py", "evaluation.py", "train.py",
                "prepare.py", "render.py", "render_review.py", "verify.py", "native_jvp_check.py",
                "native_higherorder_jvp.py", "controller.py")


class Adan(Optimizer):
    """Reference FlowerDance Adan, including its first-step behavior.

    Copied from streamtalk_metric_calibration/flowerdance_adan.py, not AdamW.
    Its beta values are update fractions (0.02, 0.08, 0.01), not Adam betas.
    """
    def __init__(self, params, lr=2e-4, betas=(0.02, 0.08, 0.01), eps=1e-8,
                 weight_decay=0.02, restart_cond=None):
        if len(betas) != 3:
            raise ValueError("Adan requires three beta values")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                         weight_decay=weight_decay, restart_cond=restart_cond))

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2, beta3 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                data, gradient = parameter.data, parameter.grad
                if gradient.is_sparse:
                    raise RuntimeError("Adan does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state.update(step=0, prev_grad=torch.zeros_like(gradient),
                                 m=torch.zeros_like(gradient), v=torch.zeros_like(gradient),
                                 n=torch.zeros_like(gradient))
                step = state["step"]
                first, difference_average, square_average = state["m"], state["v"], state["n"]
                previous_gradient = state["prev_grad"]
                if step > 0:
                    first.mul_(1-beta1).add_(gradient, alpha=beta1)
                    difference = gradient-previous_gradient
                    difference_average.mul_(1-beta2).add_(difference, alpha=beta2)
                    square_average.mul_(1-beta3).add_((gradient+(1-beta2)*difference).square(), alpha=beta3)
                step += 1
                correction1 = 1/(1-(1-beta1)**step)
                correction2 = 1/(1-(1-beta2)**step)
                correction3 = 1/(1-(1-beta3)**step)

                def update():
                    step_size = lr/(square_average*correction3).sqrt().add_(group["eps"])
                    data.addcmul_(step_size, first*correction1+(1-beta2)*difference_average*correction2,
                                  value=-1).div_(1+group["weight_decay"]*lr)

                update()
                if group["restart_cond"] is not None and group["restart_cond"](state):
                    first.copy_(gradient)
                    difference_average.zero_()
                    square_average.copy_(gradient.square())
                    update()
                previous_gradient.copy_(gradient)
                state["step"] = step
        return loss


def load_config(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if "runtime" in value:
        value = {**value, **value["runtime"]}
    if value.get("template_only_not_launchable") is True:
        raise ValueError("Template configuration is not launchable; verified deployment must explicitly clear this marker")
    required = ("task_name", "expected_hostname", "output_root", "manifest", "cache_root",
                "raw_root", "metric_assets", "evaluator_assets", "smplx", "precheck_path")
    missing = [key for key in required if not value.get(key)]
    if missing:
        raise ValueError(f"Runtime configuration missing {missing}")
    if value.get("epochs", EPOCHS) != EPOCHS or value.get("save_every", SAVE_EVERY) != SAVE_EVERY:
        raise ValueError("This authorization is exactly 10000 epochs, saving/evaluating every 50")
    if value.get("world_size", WORLD_SIZE) != WORLD_SIZE or value.get("seed", SEED) != SEED:
        raise ValueError("All six runs require the same eight-GPU world and seed 42")
    require_numerical_environment(value)
    return value


def require_numerical_environment(config):
    """Bind verified CPU arithmetic runtime; this NEVER grants training clearance.

    NumPy may already have been imported by dependencies. Check environment
    BEFORE trusting any loaded provider, then inspect those actual providers;
    setting an environment variable after BLAS loads cannot fake this guard.
    """
    names = ("numerical_environment", "numerical_environment_evidence",
             "numerical_environment_evidence_sha256")
    if not any(name in config for name in names):
        return None
    expected = {"OPENBLAS_CORETYPE": "Haswell", "OPENBLAS_NUM_THREADS": "1",
                "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    if config.get("numerical_environment") != expected:
        raise RuntimeError("Numerical environment must be the verified Haswell/single-thread configuration")
    # This ordering is intentional: do not query/trust a loaded BLAS first.
    if any(os.environ.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Required numerical environment was not set before loading providers")
    evidence = Path(config.get("numerical_environment_evidence", ""))
    digest = config.get("numerical_environment_evidence_sha256")
    if (not evidence.is_absolute() or not evidence.is_file() or not isinstance(digest, str) or
            len(digest) != 64 or sha256(evidence) != digest):
        raise RuntimeError("Numerical arithmetic evidence is missing or changed")
    proof = json.loads(evidence.read_text(encoding="utf-8"))
    prefix = config.get("runtime_prefix")
    if (proof.get("kind") != "cpu_arithmetic_verified_numerical_environment_NOT_training_clearance" or
            proof.get("passed") is not True or proof.get("expected_environment") != expected or
            not isinstance(prefix, str) or not Path(prefix).is_absolute() or
            proof.get("runtime_prefix") != prefix or Path(sys.prefix).resolve() != Path(prefix).resolve() or
            proof.get("hostname") != config.get("expected_hostname") or
            socket.gethostname() != config.get("expected_hostname") or
            not config.get("task_id") or proof.get("task_id") != config["task_id"]):
        raise RuntimeError("Numerical arithmetic evidence does not bind this exact task/host/runtime")
    from threadpoolctl import threadpool_info
    fields = ("filepath", "sha256", "internal_api", "user_api", "prefix",
              "architecture", "version", "num_threads")
    declared = proof.get("libraries")
    if not isinstance(declared, list) or not declared:
        raise RuntimeError("Numerical evidence has no actual loaded BLAS library inventory")
    recorded, current = {}, {}
    for entry in declared:
        if not isinstance(entry, dict):
            raise RuntimeError("Malformed numerical library evidence")
        if entry.get("user_api") != "blas":
            continue
        if (not all(key in entry for key in fields) or not isinstance(entry.get("filepath"), str) or
                entry["filepath"] in recorded):
            raise RuntimeError("Incomplete/duplicate BLAS library evidence")
        recorded[entry["filepath"]] = {key: entry[key] for key in fields}
    for entry in threadpool_info():
        if entry.get("user_api") != "blas":
            continue
        library = Path(entry.get("filepath", ""))
        if not library.is_absolute() or not library.is_file() or str(library) in current:
            raise RuntimeError("Actual loaded BLAS path is missing/ambiguous")
        current[str(library)] = {key: entry.get(key) for key in fields}
        current[str(library)]["filepath"] = str(library)
        current[str(library)]["sha256"] = sha256(library)
    if not current or current != recorded:
        raise RuntimeError("Actual loaded BLAS library bytes/architecture/threads differ from numerical evidence")
    if any(type(entry["num_threads"]) is not int or entry["num_threads"] != 1 for entry in current.values()):
        raise RuntimeError("All verified live BLAS providers must retain one actual thread")
    numpy_package = Path(np.__file__).resolve().parent
    numpy_library_dirs = (numpy_package, numpy_package.parent/"numpy.libs")
    numpy_providers = [entry for entry in current.values() if any(
        Path(entry["filepath"]).resolve().is_relative_to(directory) for directory in numpy_library_dirs)]
    if (not numpy_providers or any(entry["internal_api"] != "openblas" or
            entry["architecture"] != "Haswell" or type(entry["num_threads"]) is not int or
            entry["num_threads"] != 1 for entry in numpy_providers)):
        raise RuntimeError("Actual NumPy BLAS must be the verified Haswell/single-thread provider")
    return proof


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def require_evaluation_budget_inventory(config, gate):
    """All val/test clips must support official rollout and BC before launch."""
    kind = "complete_clip_frame_budget_inventory_not_metrics"
    manifest_digest = sha256(config["manifest"])
    ready = json.loads((Path(config["cache_root"])/"ready.json").read_text())
    if (ready.get("passed") is not True or ready.get("full_clips") != sum(EXPECTED_COUNTS.values()) or
            ready.get("manifest_sha256") != manifest_digest):
        raise RuntimeError("Full-clip preparation is incomplete or belongs to another manifest")
    summary = ready.get("evaluation_budget_audit", {})
    if (summary.get("passed") is not True or summary.get("kind") != kind or
            summary.get("manifest_sha256") != manifest_digest or
            gate.get("evaluation_budget_audit") != summary):
        raise RuntimeError("Missing gate-bound complete val/test frame-budget coverage")
    evidence = Path(summary.get("evidence", ""))
    if not evidence.is_absolute() or not evidence.is_file() or sha256(evidence) != summary.get("sha256"):
        raise RuntimeError("Full val/test frame-budget evidence missing/changed")
    budgets = json.loads(evidence.read_text())
    if (budgets.get("passed") is not True or budgets.get("kind") != kind or
            budgets.get("manifest_sha256") != manifest_digest):
        raise RuntimeError("Incompatible val/test frame-budget inventory")
    expected_counts = {split: EXPECTED_COUNTS[split] for split in ("val", "test")}
    if summary.get("split_counts") != expected_counts or budgets.get("split_counts") != expected_counts:
        raise RuntimeError("Frame-budget inventory does not cover val106/test265")
    from evaluation import manifest_records, rollout_frames
    manifest = json.loads(Path(config["manifest"]).read_text())
    minimum_frames = {}
    for split in ("val", "test"):
        expected_ids = [item["id"] for item in manifest_records(manifest, split, EXPECTED_COUNTS[split])]
        records = budgets.get("splits", {}).get(split, [])
        if [item.get("id") for item in records] != expected_ids:
            raise RuntimeError("Frame-budget coverage omits/reorders an official evaluation clip")
        for record in records:
            source_frames, frames = record.get("source_frames", 0), record.get("evaluation_frames", 0)
            if (not isinstance(source_frames, int) or source_frames % 30 or
                    not isinstance(frames, int) or frames <= 120 or frames % 4 or
                    frames != rollout_frames(source_frames) or record.get("fgd_frames") != frames//32*32):
                raise RuntimeError("An official evaluation clip has an unsupported rollout/BC frame budget")
        minimum_frames[split] = min(record["evaluation_frames"] for record in records)
    if summary.get("minimum_frames_by_split") != minimum_frames or budgets.get("minimum_frames_by_split") != minimum_frames:
        raise RuntimeError("Frame-budget inventory minimums do not match actual complete coverage")
    return budgets


def require_cache_integrity(config, gate):
    """Bind current train statistics and every complete cache to the gate."""
    kind = "complete_feature_cache_integrity_inventory"
    summary = gate.get("cache_integrity", {})
    manifest_digest = sha256(config["manifest"])
    if (summary.get("passed") is not True or summary.get("kind") != kind or
            summary.get("manifest_sha256") != manifest_digest or
            summary.get("clip_count") != sum(EXPECTED_COUNTS.values())):
        raise RuntimeError("Missing gate-bound complete feature-cache integrity inventory")
    evidence = Path(summary.get("evidence", ""))
    if not evidence.is_absolute() or not evidence.is_file() or sha256(evidence) != summary.get("sha256"):
        raise RuntimeError("Feature-cache integrity evidence missing/changed")
    inventory = json.loads(evidence.read_text())
    if (inventory.get("passed") is not True or inventory.get("kind") != kind or
            inventory.get("manifest_sha256") != manifest_digest or
            inventory.get("clip_count") != sum(EXPECTED_COUNTS.values())):
        raise RuntimeError("Incompatible complete feature-cache integrity inventory")
    cache_root = Path(config["cache_root"]).resolve()

    def check_file(record, expected_path, label):
        path = Path(record.get("path", ""))
        if not path.is_absolute() or path.resolve() != expected_path.resolve() or not path.is_file():
            raise RuntimeError(f"Feature-cache inventory has a missing/incompatible path: {label}")
        before = path.stat()
        digest = sha256(path)
        after = path.stat()
        if (digest != record.get("sha256") or
                (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise RuntimeError(f"Feature cache/statistics changed after technical precheck: {label}")
        return digest

    stats = inventory.get("stats", {})
    if set(stats) != {"full55", "body22"} or set(summary.get("stats_sha256", {})) != {"full55", "body22"}:
        raise RuntimeError("Cache inventory must bind both pose-mode train-only statistics")
    for mode in ("full55", "body22"):
        digest = check_file(stats[mode], cache_root/f"stats_{mode}.npz", f"stats_{mode}")
        if summary["stats_sha256"][mode] != digest:
            raise RuntimeError("Current normalization statistics differ from gate summary")
    from evaluation import manifest_records
    manifest = json.loads(Path(config["manifest"]).read_text())
    expected_ids = [item["id"] for split in ("train", "val", "test")
                    for item in manifest_records(manifest, split, EXPECTED_COUNTS[split])]
    records = inventory.get("clips", [])
    if len(expected_ids) != len(set(expected_ids)) or [item.get("id") for item in records] != expected_ids:
        raise RuntimeError("Cache integrity evidence omits/reorders official complete clips")
    for record in records:
        check_file(record, cache_root/(record["id"]+".npz"), record["id"])
    return inventory


def require_bc_audio_onset_inventory(config, gate, budgets=None):
    """Both official BC audio branches must be defined for all val/test IDs.

    This consumes a CPU-only onset-availability inventory, not completed BC
    scores. No onsets are recomputed and no clips are omitted at launch.
    """
    kind = "complete_val_test_bc_audio_onset_inventory_not_metrics"
    ready = json.loads((Path(config["cache_root"])/"ready.json").read_text())
    summary = ready.get("bc_audio_onset_audit", {})
    counts = {split: EXPECTED_COUNTS[split] for split in ("val", "test")}
    manifest_digest = sha256(config["manifest"])
    from evaluation import AUDIO_SR, ALIGN_MASK, manifest_records
    import evaluation
    current_evaluation_digest = sha256(evaluation.__file__)
    budget_digest = gate.get("evaluation_budget_audit", {}).get("sha256")
    if (summary != gate.get("bc_audio_onset_audit") or summary.get("passed") is not True or
            summary.get("kind") != kind or summary.get("split_counts") != counts or
            summary.get("total_clips") != sum(counts.values()) or
            summary.get("manifest_sha256") != manifest_digest or
            summary.get("evaluation_source_sha256") != current_evaluation_digest or
            summary.get("evaluation_budget_sha256") != budget_digest):
        raise RuntimeError("Missing compatible gate-bound complete BC audio-onset inventory")
    evidence = Path(summary.get("evidence", ""))
    if not evidence.is_absolute() or not evidence.is_file() or sha256(evidence) != summary.get("sha256"):
        raise RuntimeError("Complete BC audio-onset evidence missing/changed")
    inventory = json.loads(evidence.read_text())
    for key in ("passed", "kind", "split_counts", "total_clips", "manifest_sha256",
                "evaluation_source_sha256", "evaluation_budget_sha256", "minimum_onsets_by_protocol"):
        if inventory.get(key) != summary.get(key):
            raise RuntimeError(f"BC audio-onset inventory summary disagrees: {key}")
    if (inventory.get("cuda_used") is not False or inventory.get("motion_predictions_used") is not False or
            inventory.get("metric_values_computed") is not False or
            inventory.get("audio_sample_rate") != AUDIO_SR or inventory.get("align_mask") != ALIGN_MASK or
            inventory.get("hop_length") != 512 or inventory.get("onset_units") != "time" or
            inventory.get("prepare_source_sha256") != sha256(Path(__file__).parent/"prepare.py")):
        raise RuntimeError("BC inventory is not the current CPU-only official audio-onset availability audit")
    raw_root = Path(config["raw_root"]).resolve()
    if Path(inventory.get("raw_data_root", "")).resolve() != raw_root:
        raise RuntimeError("BC audio-onset inventory belongs to another raw dataset")
    if budgets is None:
        budgets = require_evaluation_budget_inventory(config, gate)
    manifest = json.loads(Path(config["manifest"]).read_text())
    all_records = []
    for split in ("val", "test"):
        expected_ids = [record["id"] for record in manifest_records(manifest, split, EXPECTED_COUNTS[split])]
        records = inventory.get("splits", {}).get(split, [])
        if [record.get("id") for record in records] != expected_ids:
            raise RuntimeError("BC audio-onset coverage omits/reorders an official clip")
        for record, budget in zip(records, budgets["splits"][split]):
            if (record.get("source_frames") != budget["source_frames"] or
                    record.get("evaluation_frames") != budget["evaluation_frames"] or
                    type(record.get("original_audio_samples")) is not int or record["original_audio_samples"] <= 0 or
                    type(record.get("original_audio_sample_rate")) is not int or record["original_audio_sample_rate"] <= 0):
                raise RuntimeError("BC audio-onset source/frame budget differs from complete inventory")
            audio_path = Path(record.get("audio_path", ""))
            expected_audio_path = (raw_root/"wave16k"/(record["id"]+".wav")).resolve()
            if (not audio_path.is_absolute() or audio_path.resolve() != expected_audio_path or
                    not audio_path.is_file() or sha256(audio_path) != record.get("audio_sha256")):
                raise RuntimeError("BC audio source missing/changed after its availability audit")
            for protocol in ("legacy", "corrected"):
                branch = record.get(protocol, {})
                first, last = branch.get("first_onset_seconds"), branch.get("last_onset_seconds")
                if (type(branch.get("sample_count")) is not int or branch["sample_count"] <= 0 or
                        type(branch.get("onset_count")) is not int or branch["onset_count"] <= 0 or
                        not isinstance(first, (int, float)) or not isinstance(last, (int, float)) or
                        not math.isfinite(first) or not math.isfinite(last) or not 0 <= first <= last or
                        last*AUDIO_SR > branch["sample_count"]+512):
                    raise RuntimeError(f"An official {protocol} BC audio segment has no finite valid onsets")
            all_records.append(record)
    minimums = {protocol: min(record[protocol]["onset_count"] for record in all_records)
                for protocol in ("legacy", "corrected")}
    if summary.get("minimum_onsets_by_protocol") != minimums:
        raise RuntimeError("BC onset inventory minimums disagree with actual complete coverage")
    return inventory


def require_bs_preflight(config, gate, path):
    report = json.loads(Path(path).read_text())
    if (report.get("passed") is not True or report.get("kind") != "real_cuda_forward_backward" or
            report.get("task_name") != config["task_name"] or report.get("hostname") != config["expected_hostname"] or
            report.get("world_size") != WORLD_SIZE or report.get("config_sha256") != config_hash(config) or
            report.get("precheck_sha256") != sha256(config["precheck_path"]) or
            report.get("cache_integrity_evidence_sha256") != gate["cache_integrity"]["sha256"] or
            report.get("code_hashes") != {name: sha256(Path(__file__).parent/name) for name in SOURCE_FILES} or
            report.get("probe_includes_real_ddp") is not True or report.get("rank0_resident_evaluator") is not True or
            report.get("probe_protocol") != BS_PROBE_PROTOCOL):
        raise RuntimeError("Missing compatible real DDP + resident-evaluator BS preflight")
    if set(report.get("modes", {})) != {"full55", "body22"}:
        raise RuntimeError("BS preflight is missing a required model mode")
    manifest = json.loads(Path(config["manifest"]).read_text())
    required_scenarios, policy = bs_scenario_policy(config, manifest)
    actual_inventory = bs_current_window_inventory(config, manifest)
    if report.get("required_scenarios") != list(required_scenarios) or report.get("scenario_policy") != policy:
        raise RuntimeError("BS scenario applicability/authorization differs from current audited policy")
    if any(report.get(key) != value for key, value in validate_bs_candidates_config(config).items()):
        raise RuntimeError("BS configured candidate order/rejections do not match evidence")
    for mode in ("full55", "body22"):
        group = report["modes"][mode]
        validate_bs_search_inventory(group)
        summary = {key: actual_inventory[key] for key in
            ("long_window_count", "short_window_count", "training_window_inventory_sha256")}
        if group.get("window_availability") != summary:
            raise RuntimeError("BS recorded windows do not match the current original train window inventory")
        selected = group.get("selected_batch_per_gpu", [])
        if (len(selected) != 3 or len(set(selected)) != 3 or
                any(type(size) is not int or size < 2 for size in selected) or group.get("model_state_bytes", 0) <= 0):
            raise RuntimeError("BS preflight must retain exactly three actual safe batches per mode")
        for size in selected:
            attempts = [attempt for attempt in group.get("attempts", []) if attempt.get("batch_per_gpu") == size]
            if len(attempts) != 1 or attempts[0].get("actual_ddp_attempted") is not True:
                raise RuntimeError("Selected batch lacks its unique real DDP probe")
            ranks = attempts[0].get("ranks", [])
            rank_plans = attempts[0].get("rank_scenario_inputs", [])
            if len(rank_plans) != WORLD_SIZE or attempts[0].get("scenario_inputs") != rank_plans[0]:
                raise RuntimeError("Selected batch lacks exact separate real input provenance for all ranks")
            for index, plan in enumerate(rank_plans):
                expected = bs_scenario_inputs(actual_inventory, size, rank=index, world_size=WORLD_SIZE,
                    required_scenarios=required_scenarios)
                if plan != expected:
                    raise RuntimeError("BS rank input is not its actual distinct unmodified training-window subset")
            if len(ranks) != WORLD_SIZE or {item.get("rank") for item in ranks} != set(range(WORLD_SIZE)):
                raise RuntimeError("Selected batch did not pass on all eight ranks")
            for item in ranks:
                if (item.get("passed") is not True or item.get("real_ddp") is not True or
                        item.get("rank0_evaluator_resident") != (item["rank"] == 0) or
                        (item["rank"] == 0 and item.get("evaluator_allocated_bytes", 0) <= 0) or
                        item.get("peak_allocated_bytes", 0) <= 0 or item.get("gpu_total_bytes", 0) <= 0 or
                        max(item["peak_allocated_bytes"], item.get("peak_reserved_bytes", 0)) >= .8*item["gpu_total_bytes"]):
                    raise RuntimeError("Selected batch lacks passing real DDP/evaluator memory evidence")
                validate_bs_rank_scenarios(item, rank_plans[item["rank"]], size, mode,
                                           required_scenarios=required_scenarios)
            screens = attempts[0].get("screen_ranks", [])
            if len(screens) != WORLD_SIZE or {item.get("rank") for item in screens} != set(range(WORLD_SIZE)):
                raise RuntimeError("Selected batch lacks complete all-rank bare safety screens")
            for item in screens:
                validate_bs_rank_scenarios(item, rank_plans[item["rank"]], size, mode,
                    required_scenarios=required_scenarios, real_ddp=False)
    return report


def bs_window_inventory(dataset):
    """Metadata only; no manufactured crops, extra padding or held-out clips."""
    if dataset.length != 240:
        raise RuntimeError("BS probes must retain the original T240 window protocol")
    long_windows, short_windows = [], []
    identities = set()
    for index, (ident, start, frames) in enumerate(dataset.index):
        if not isinstance(ident, str) or type(start) is not int or type(frames) is not int or not 0 <= start < frames:
            raise RuntimeError("Invalid actual training-window metadata")
        if (ident, start) in identities:
            raise RuntimeError("Duplicate training window; no probe occupancy duplicates")
        identities.add((ident, start))
        usable = min(240, frames-start)
        record = dict(dataset_index=index, id=ident, window_start=start,
                      source_frames=frames, valid_frames=usable)
        if usable == 240:
            long_windows.append(record)
        elif frames < 240 and start == 0:
            short_windows.append(record)
        else:
            raise RuntimeError("A short probe window must be an actual short whole train clip")
    digest = hashlib.sha256(json.dumps(dataset.index, separators=(",", ":")).encode()).hexdigest()
    return dict(long=long_windows, short=short_windows,
                long_window_count=len(long_windows), short_window_count=len(short_windows),
                training_window_inventory_sha256=digest)


def current_train_cache_frames(config, train_ids):
    """Read actual NPZ poses.npy HEADERS only; never write/rebuild the cache."""
    records = []
    for ident in train_ids:
        with zipfile.ZipFile(Path(config["cache_root"])/(ident+".npz")) as archive:
            with archive.open("poses.npy") as handle:
                version = np.lib.format.read_magic(handle)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
                else:
                    raise RuntimeError("Unsupported actual poses header format in BS audit")
        if len(shape) != 2 or shape[1] != 165 or shape[0] <= 0 or dtype.kind != "f" or fortran:
            raise RuntimeError("Actual cached poses header has invalid dimensions/layout")
        records.append(dict(id=ident, frames=int(shape[0])))
    return records


def bs_current_window_inventory(config, manifest):
    records = current_train_cache_frames(config, manifest["splits"]["train"])
    index = [(item["id"], start, item["frames"]) for item in records
             for start in windows(item["frames"], length=240, stride=120)]
    return bs_window_inventory(SimpleNamespace(length=240, index=index))


def bs_scenario_policy(config, manifest, inventory=None):
    """Default dual coverage; explicitly authorized zero-short exception only.

    A header inventory is NOT numerical/GPU clearance. This read-only check
    binds the exception to all current official train IDs and actual caches.
    """
    if config.get("allow_bs_no_short_full240_only") is not True:
        return BS_SCENARIOS, {"mixed_real_short": "required", "no_fake_short_windows": True}
    authorization = config.get("bs_no_short_authorization")
    evidence = Path(config.get("bs_train_window_inventory", ""))
    digest = config.get("bs_train_window_inventory_sha256", "")
    if (not isinstance(authorization, str) or not authorization.strip() or
            not evidence.is_absolute() or not evidence.is_file() or
            len(digest) != 64 or sha256(evidence) != digest):
        raise RuntimeError("No-short exception needs explicit authorization and unchanged complete header evidence")
    proof = json.loads(evidence.read_text())
    train_ids = manifest.get("splits", {}).get("train", [])
    records = proof.get("all_train_clips", [])
    if (proof.get("schema") != 1 or proof.get("kind") != BS_NO_SHORT_KIND or
            proof.get("hostname") != config["expected_hostname"] or proof.get("task_name") != config["task_name"] or
            proof.get("task_id") != config.get("task_id") or not config.get("task_id") or
            proof.get("manifest_sha256") != sha256(config["manifest"]) or proof.get("split") != "train" or
            len(train_ids) != EXPECTED_COUNTS["train"] or len(set(train_ids)) != len(train_ids) or
            len(records) != len(train_ids) or [item.get("id") for item in records] != train_ids or
            proof.get("clips") != len(train_ids) or proof.get("window_frames") != 240 or
            proof.get("short_clips") != [] or proof.get("short_clip_count") != 0 or
            proof.get("full_length_clip_count") != len(train_ids) or
            proof.get("data_modified") is not False or proof.get("training_started") is not False or
            proof.get("window_policy") != BS_WINDOW_POLICY or
            any(type(item.get("frames")) is not int or item["frames"] < 240 for item in records)):
        raise RuntimeError("No-short header evidence is incomplete, wrong-host or contains actual short clips")
    lengths = [item["frames"] for item in records]
    if proof.get("minimum_frames") != min(lengths) or proof.get("maximum_frames") != max(lengths):
        raise RuntimeError("No-short header inventory minimum/maximum disagrees with full coverage")
    if current_train_cache_frames(config, train_ids) != records:
        raise RuntimeError("No-short evidence differs from actual current cache poses headers")
    if inventory is not None:
        actual_frames = {}
        for window in inventory["long"] + inventory["short"]:
            ident, frames = window["id"], window["source_frames"]
            if ident in actual_frames and actual_frames[ident] != frames:
                raise RuntimeError("Actual dataset index disagrees within one training clip")
            actual_frames[ident] = frames
        if inventory["short"] or actual_frames != {item["id"]: item["frames"] for item in records}:
            raise RuntimeError("Actual unmodified dataset windows disagree with complete no-short audit")
    return ("long240",), dict(mixed_real_short="not_applicable_no_actual_short_train_clips",
        no_fake_short_windows=True, authorization=authorization,
        train_window_inventory=str(evidence), train_window_inventory_sha256=digest,
        training_clip_count=len(records), minimum_actual_frames=min(lengths), maximum_actual_frames=max(lengths))


def bs_scenario_inputs(inventory, batch_size, *, required_scenarios=BS_SCENARIOS, rank=0, world_size=1):
    if type(batch_size) is not int or batch_size < 2:
        raise RuntimeError("Dual-scenario BS needs at least two actual windows")
    if type(world_size) is not int or type(rank) is not int or not 0 <= rank < world_size:
        raise RuntimeError("Invalid real rank/window partition")
    if len(inventory["long"]) < batch_size*world_size:
        raise RuntimeError("Not enough distinct real T240 training windows for requested batch")
    if required_scenarios not in (BS_SCENARIOS, ("long240",)):
        raise RuntimeError("Unsupported BS scenario policy")
    if "mixed_real_short" in required_scenarios and len(inventory["short"]) < world_size:
        raise RuntimeError("No actual short train clips; mixed-short coverage cannot be fabricated")
    if required_scenarios == ("long240",) and inventory["short"]:
        raise RuntimeError("Existing genuine short clips cannot be waived by no-short policy")
    long_inputs = inventory["long"][rank*batch_size:(rank+1)*batch_size]
    selected_windows = {"long240": long_inputs}
    if "mixed_real_short" in required_scenarios:
        short_count = min(len(inventory["short"])//world_size, batch_size//2)
        selected_windows["mixed_real_short"] = (inventory["short"][rank*short_count:(rank+1)*short_count]
                                               + long_inputs[:batch_size-short_count])
    result = {}
    for name, records in selected_windows.items():
        digest = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        result[name] = dict(windows=records, input_window_sha256=digest,
                            valid_lengths=[item["valid_frames"] for item in records])
    return result


def search_bs_candidates(probe, *, attempts=None):
    """Bounded upward genuine probes; choose largest three, never first three."""
    attempts = [] if attempts is None else attempts
    if attempts:
        raise RuntimeError("A BS search must start with a fresh, empty attempt inventory")
    passed, stop_reason = [], "bounded_search_cap_512"
    for size in BS_UPWARD:
        attempt = probe(size)
        if attempt.get("batch_per_gpu") != size or type(attempt.get("passed")) is not bool:
            raise RuntimeError("Malformed actual BS probe result")
        attempts.append(attempt)
        if not attempt["passed"]:
            stop_reason = attempt.get("boundary_reason", "actual_80pct_or_oom_boundary")
            break
        passed.append(size)
    if len(passed) < 3:
        for size in BS_FALLBACK:
            attempt = probe(size)
            if attempt.get("batch_per_gpu") != size or type(attempt.get("passed")) is not bool:
                raise RuntimeError("Malformed actual BS fallback result")
            attempts.append(attempt)
            if attempt["passed"]:
                passed.append(size)
            if len(passed) >= 3:
                break
    if len(passed) < 3:
        raise RuntimeError("Cannot find three distinct dual-scenario safe genuine batch sizes")
    return dict(selected_batch_per_gpu=sorted(passed)[-3:], attempts=attempts,
                search_stop_reason=stop_reason, searched_upper_bound=512)


def validate_bs_search_inventory(group):
    """Replay ordering/selection from immutable records, with NO GPU work."""
    recorded = group.get("attempts", [])
    position = 0
    def consume(size):
        nonlocal position
        if position >= len(recorded) or recorded[position].get("batch_per_gpu") != size:
            raise RuntimeError("BS evidence skipped/reordered the bounded genuine search")
        value = recorded[position]
        position += 1
        return value
    reconstructed = search_bs_candidates(consume)
    if position != len(recorded) or any(group.get(key) != reconstructed[key] for key in
            ("selected_batch_per_gpu", "search_stop_reason", "searched_upper_bound")):
        raise RuntimeError("BS evidence did not select the largest three actual safe search results")


def validate_bs_candidates_config(config):
    candidates = config.get("bs_probe_candidates", list(BS_UPWARD + BS_FALLBACK))
    if candidates not in (list(BS_UPWARD + BS_FALLBACK), list(BS_UPWARD + BS_FALLBACK) + [1]):
        raise RuntimeError("BS candidates must match bounded upward16..512 then descending8/4/2")
    return {"configured_candidates": candidates,
            "rejected_candidates": [{"batch_per_gpu": 1,
                "reason": "B1 excluded: dual genuine long/short coverage cannot fit one batch"}] if 1 in candidates else []}


def validate_bs_rank_scenarios(rank_record, scenario_inputs, batch_size, mode, *, required_scenarios=BS_SCENARIOS,
                              real_ddp=True):
    """Read-only structural checks; synthetic numbers are not GPU clearance."""
    scenarios = rank_record.get("scenarios", {})
    if rank_record.get("passed") is not True or rank_record.get("real_ddp") is not real_ddp:
        raise RuntimeError("BS rank aggregate does not match actual probe kind/pass")
    if (required_scenarios not in (BS_SCENARIOS, ("long240",)) or
            set(scenarios) != set(required_scenarios) or set(scenario_inputs) != set(required_scenarios)):
        raise RuntimeError("BS evidence is missing required genuine window scenarios")
    expected_dim = 333 if mode == "full55" else 135
    for name in required_scenarios:
        plan, leaf = scenario_inputs[name], scenarios[name]
        windows, lengths = plan.get("windows", []), plan.get("valid_lengths", [])
        digest = hashlib.sha256(json.dumps(windows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if (len(windows) != batch_size or len(lengths) != batch_size or
                len({(item.get("id"), item.get("window_start")) for item in windows}) != batch_size or
                digest != plan.get("input_window_sha256") or lengths != [item.get("valid_frames") for item in windows]):
            raise RuntimeError("BS actual training-window provenance is incomplete/inconsistent")
        for item in windows:
            frames, start, usable = item.get("source_frames"), item.get("window_start"), item.get("valid_frames")
            if (type(frames) is not int or type(start) is not int or type(usable) is not int or
                    not 0 <= start < frames or usable != min(240, frames-start) or
                    (usable < 240 and not (frames < 240 and start == 0))):
                raise RuntimeError("BS masks do not belong to actual unmodified training windows")
        if name == "long240" and any(value != 240 for value in lengths):
            raise RuntimeError("Long-window probe must contain only complete genuine T240 windows")
        if name == "mixed_real_short" and not (batch_size >= 2 and any(value < 240 for value in lengths)
                                               and any(value == 240 for value in lengths)):
            raise RuntimeError("Mixed-window probe must contain actual long AND actual short windows")
        duration = leaf.get("end_to_end_seconds", 0.)
        waits = [leaf.get(key, -1.) for key in ("loading_wait_seconds", "transfer_seconds", "compute_seconds")]
        if (leaf.get("passed") is not True or leaf.get("real_ddp") is not real_ddp or
                leaf.get("rank") != rank_record.get("rank") or leaf.get("batch_per_gpu") != batch_size or
                leaf.get("frames") != 240 or leaf.get("motion_dim") != expected_dim or
                leaf.get("optimizer_updates") != 2 or leaf.get("actual_input_mask_checked") is not True or
                leaf.get("input_valid_lengths") != [lengths, lengths] or
                leaf.get("input_window_sha256") != digest or leaf.get("effective_frames_total") != 2*sum(lengths) or
                leaf.get("samples_total") != 2*batch_size or
                not math.isfinite(duration) or duration <= 0 or
                any(not math.isfinite(value) or value < 0 for value in waits) or
                duration + 1e-6 < sum(waits) or
                any(not math.isfinite(leaf.get(key, float("nan"))) or leaf.get(key, 0) <= 0
                    for key in ("effective_frames_per_second", "samples_per_second")) or
                not math.isclose(leaf["effective_frames_per_second"], 2*sum(lengths)/duration, rel_tol=1e-8) or
                not math.isclose(leaf["samples_per_second"], 2*batch_size/duration, rel_tol=1e-8) or
                leaf.get("peak_allocated_bytes", 0) <= 0 or leaf.get("peak_reserved_bytes", 0) <= 0 or
                leaf.get("gpu_total_bytes", 0) != rank_record.get("gpu_total_bytes") or
                max(leaf["peak_allocated_bytes"], leaf["peak_reserved_bytes"]) >= .8*leaf["gpu_total_bytes"] or
                leaf.get("rank0_evaluator_resident") != (leaf["rank"] == 0) or
                (leaf["rank"] == 0 and leaf.get("evaluator_allocated_bytes", 0) <= 0)):
            raise RuntimeError("BS leaf lacks genuine two-update DDP/memory/throughput evidence")
    for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
        if rank_record.get(key) != max(leaf[key] for leaf in scenarios.values()):
            raise RuntimeError("BS aggregate hides the worst actual scenario memory peak")


def require_native_binary_bindings(proof, scan_module, conv_module):
    """Bind the actual imported compiled kernels, not merely wheel versions."""
    from native_jvp_check import kernel_binary_bindings
    current = kernel_binary_bindings(scan_module, conv_module)
    if proof.get("kernel_binary_sources") != current:
        raise RuntimeError("Native CUDA kernel binary changed after JVP verification")
    return current


def require_native_derivative_repair(proof):
    """The deployed exact helper and all mandatory numerical checks must bind."""
    from native_higherorder_jvp import install
    from native_jvp_check import require_extended_native_checks
    current = install()
    if current.get("installed") is not True or proof.get("native_derivative_repair") != current:
        raise RuntimeError("Native derivative repair changed or was not actually installed during GPU verification")
    if proof.get("source_hashes", {}).get("native_higherorder_jvp.py") != sha256(Path(__file__).parent/"native_higherorder_jvp.py"):
        raise RuntimeError("Exact native derivative helper source changed after GPU verification")
    require_extended_native_checks(proof)
    return current


def require_precheck(config):
    gate = json.loads(Path(config["precheck_path"]).read_text(encoding="utf-8"))
    if gate.get("passed") is not True or gate.get("task_name") != config["task_name"]:
        raise RuntimeError("Missing or incompatible real PRECHECK_PASSED gate")
    if gate.get("hostname") != config["expected_hostname"]:
        raise RuntimeError("PRECHECK evidence belongs to another hostname")
    if gate.get("manifest_sha256") != sha256(config["manifest"]):
        raise RuntimeError("Data manifest changed after real precheck")
    budgets = require_evaluation_budget_inventory(config, gate)
    require_cache_integrity(config, gate)
    require_bc_audio_onset_inventory(config, gate, budgets)
    require_visual_review(config, gate)
    for name in GATE_CHECKS:
        check = gate.get("checks", {}).get(name, {})
        if check.get("passed") is not True or check.get("kind") not in ("real_gpu", "real_data", "smplx_mesh"):
            raise RuntimeError(f"Real precheck evidence has not passed: {name}")
        evidence = Path(check.get("evidence", ""))
        if not evidence.is_absolute() or not evidence.is_file():
            raise RuntimeError(f"Missing precheck evidence artifact: {name}")
        proof = json.loads(evidence.read_text(encoding="utf-8"))
        if proof.get("passed") is not True or proof.get("kind") != check["kind"]:
            raise RuntimeError(f"Evidence content is not an actual passing audit: {name}")
        if check.get("sha256") and sha256(evidence) != check["sha256"]:
            raise RuntimeError(f"Changed precheck evidence artifact: {name}")
        if name == "native_cuda_jvp":
            if proof.get("task_name") != config["task_name"] or proof.get("hostname") != config["expected_hostname"]:
                raise RuntimeError("Native CUDA/JVP evidence belongs to another task or host")
            if proof.get("cuda_available") is not True or proof.get("torch_version") != str(torch.__version__):
                raise RuntimeError("Native CUDA/JVP evidence does not match installed PyTorch")
            if proof.get("torch_cuda_version") != torch.version.cuda:
                raise RuntimeError("Native CUDA/JVP CUDA runtime changed")
            for source in ("model.py", "loss.py", "native_jvp_check.py", "native_higherorder_jvp.py"):
                if proof.get("source_hashes", {}).get(source) != sha256(Path(__file__).parent/source):
                    raise RuntimeError(f"Native CUDA/JVP diagnostic source changed: {source}")
            import mamba_ssm.ops.selective_scan_interface as scan_module
            import causal_conv1d.causal_conv1d_interface as conv_module
            for name_on_disk, module in (("selective_scan_interface.py", scan_module),
                                         ("causal_conv1d_interface.py", conv_module)):
                if proof.get("kernel_python_source_hashes", {}).get(name_on_disk) != sha256(module.__file__):
                    raise RuntimeError(f"Native kernel interface changed after JVP verification: {name_on_disk}")
            require_native_binary_bindings(proof, scan_module, conv_module)
            require_native_derivative_repair(proof)
            for package, recorded in (("mamba-ssm", "mamba_ssm_version"), ("causal-conv1d", "causal_conv1d_version")):
                if proof.get(recorded) != importlib.metadata.version(package):
                    raise RuntimeError(f"Native CUDA/JVP package version changed: {package}")
    for name in SOURCE_FILES:
        if gate.get("code_hashes", {}).get(name) != sha256(Path(__file__).parent/name):
            raise RuntimeError(f"Source changed after precheck: {name}")
    return gate


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def capture_rng(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device).cpu())


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    torch.cuda.set_rng_state(state["cuda"].cpu(), device)


@contextlib.contextmanager
def evaluation_rng(device):
    """Evaluation/render scheduling never advances training RNG streams."""
    state = capture_rng(device)
    try:
        with torch.random.fork_rng(devices=[device.index]):
            seed_all(SEED)
            yield
    finally:
        restore_rng(state, device)


def build_evaluator(config, device):
    # AESK initializes parameters before loading weights. Keep initialization
    # independent of both fresh and restored training RNG streams.
    from evaluation import SemTalkEvaluator
    with evaluation_rng(device):
        return SemTalkEvaluator(config["metric_assets"], config["evaluator_assets"],
            device=str(device), smplx_batch_size=int(config.get("metric_batch_size", 128)))


def torch_save_atomic(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def append_json(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, allow_nan=False)+"\n")
        handle.flush()


def batch_on_device(batch, device):
    batch = {key: torch.as_tensor(value).to(device, non_blocking=True) for key, value in batch.items()}
    condition = {key: batch[key] for key in ("audio", "text", "person_id", "padding_mask")}
    return batch["motion"], batch["rest_joints"], condition


def load_pose_mean(config, manifest):
    """SMPL-X has a nonzero default hand pose; static mean is not motion GT."""
    reference = None
    for ident in sum((manifest["splits"][split] for split in ("train", "val", "test")), []):
        if isinstance(ident, dict):
            ident = ident["id"]
        with np.load(Path(config["cache_root"])/(ident+".npz"), allow_pickle=False) as archive:
            pose_mean = np.asarray(archive["pose_mean"], dtype=np.float32).reshape(-1)
            if pose_mean.shape != (165,) or not np.isfinite(pose_mean).all():
                raise RuntimeError(f"Invalid SMPL-X static pose_mean in {ident}")
            if reference is None:
                reference = pose_mean.copy()
            elif not np.array_equal(reference, pose_mean):
                raise RuntimeError(f"SMPL-X pose_mean differs across clip caches: {ident}")
    return reference


def build_model(config, mode, normalizer, device, pose_mean):
    model = GestureM2D(mode=mode, **config.get("architecture", {})).to(device)
    if model.test_backend:
        raise RuntimeError("Fake/test Mamba is forbidden in real training")
    loss = MeanFlowLoss(model, torch.as_tensor(normalizer.mean), torch.as_tensor(normalizer.std),
                        mode=mode, pose_mean=torch.as_tensor(pose_mean), **config.get("loss", {})).to(device)
    optimizer = Adan(model.parameters(), lr=float(config.get("learning_rate", 2e-4)),
                     weight_decay=float(config.get("weight_decay", 0.02)))
    return model, loss, optimizer


def probe_updates(model, objective, optimizer, clean, rest, condition):
    """Two genuine forward/JVP/backward/Adan updates, with no occupancy filler."""
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        total, terms = objective(clean, rest, condition)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite real MeanFlow loss")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
    return float(total.detach()), float(grad_norm)


def aggregate_bs_rank(rank, batch_size, scenarios, *, real_ddp, evaluator_delta, required_scenarios=BS_SCENARIOS):
    """Keep legacy controller fields, but derive peaks from every actual leaf."""
    complete = set(scenarios) == set(required_scenarios)
    values = list(scenarios.values())
    totals = {item["gpu_total_bytes"] for item in values}
    if len(totals) != 1:
        raise RuntimeError("Scenario device-memory capacities disagree")
    return dict(rank=rank, batch_per_gpu=batch_size,
        passed=complete and all(item["passed"] for item in values),
        real_ddp=real_ddp, rank0_evaluator_resident=rank == 0,
        evaluator_allocated_bytes=evaluator_delta,
        gpu_total_bytes=next(iter(totals)),
        peak_allocated_bytes=max(item.get("peak_allocated_bytes", 0) for item in values),
        peak_reserved_bytes=max(item.get("peak_reserved_bytes", 0) for item in values),
        model_state_bytes=max(item.get("model_state_bytes", 0) for item in values),
        scenarios=scenarios)


def probe_scenario_updates(model, objective, optimizer, dataset, plan, device):
    """Two real updates INCLUDING actual loading/transfer/compute timing.

    Uses exactly the original dataset windows, normalization and prefix masks.
    The same distinct real input windows are reloaded for the two optimization
    updates; this is explicitly a num_workers=0 probe, not a claim about future
    asynchronous DataLoader throughput. No synthetic tensors fill device memory.
    """
    if device.type != "cuda":
        raise RuntimeError("Actual BS update probes require real CUDA; CPU is not clearance")
    records, lengths = plan["windows"], plan["valid_lengths"]
    size = len(records)
    loader = DataLoader(Subset(dataset, [item["dataset_index"] for item in records]),
        batch_size=size, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    loading, transfer, compute = 0., 0., 0.
    observed = []
    begin_all = time.monotonic()
    for _ in range(2):
        begin = time.monotonic()
        batch = next(iter(loader))
        loading += time.monotonic()-begin
        expected_mask = torch.arange(240)[None] < torch.tensor(lengths)[:, None]
        if (batch["motion"].shape != (size, 240, model.motion_dim) or
                not torch.equal(batch["padding_mask"], expected_mask)):
            raise RuntimeError("Actual probe batch does not match its real window/mask provenance")
        observed.append(batch["padding_mask"].sum(1).tolist())
        begin = time.monotonic()
        clean, rest, condition = batch_on_device(batch, device)
        torch.cuda.synchronize(device)
        transfer += time.monotonic()-begin
        begin = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        total, terms = objective(clean, rest, condition)
        if not torch.isfinite(total):
            raise FloatingPointError("Nonfinite real MeanFlow BS-probe loss")
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
        optimizer.step()
        torch.cuda.synchronize(device)
        compute += time.monotonic()-begin
    duration = time.monotonic()-begin_all
    frames = 2*sum(lengths)
    return dict(loss=float(total.detach()), grad_norm=float(grad_norm),
        optimizer_updates=2, frames=240, motion_dim=model.motion_dim,
        actual_input_mask_checked=True, input_valid_lengths=observed,
        input_window_sha256=plan["input_window_sha256"],
        effective_frames_total=frames, samples_total=2*size,
        end_to_end_seconds=duration, loading_wait_seconds=loading,
        transfer_seconds=transfer, compute_seconds=compute,
        effective_frames_per_second=frames/duration, samples_per_second=2*size/duration,
        loader_num_workers=0, same_real_windows_reloaded_for_two_updates=True)


def calibration(config, manifest, rank, device, gate):
    """Bounded genuine BS search; no-short exception is explicit/audit-bound."""
    if device.type != "cuda" or not dist.is_initialized() or dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError("Calibration requires the actual eight-rank CUDA DDP job")
    candidate_config = validate_bs_candidates_config(config)
    required_scenarios, policy = bs_scenario_policy(config, manifest)
    report = dict(passed=False, task_name=config["task_name"], hostname=socket.gethostname(),
        world_size=WORLD_SIZE, kind="real_cuda_forward_backward", modes={},
        probe_protocol=BS_PROBE_PROTOCOL, required_scenarios=list(required_scenarios), scenario_policy=policy,
        config_sha256=config_hash(config), memory_limit_fraction=.8,
        precheck_sha256=sha256(config["precheck_path"]),
        cache_integrity_evidence_sha256=gate["cache_integrity"]["sha256"],
        code_hashes={name: sha256(Path(__file__).parent/name) for name in SOURCE_FILES},
        probe_includes_real_ddp=True, rank0_resident_evaluator=True,
        upward_candidates=list(BS_UPWARD), descending_fallback=list(BS_FALLBACK),
        oom_screen="bare real objective plus 2x model-state bytes +64MiB safety reserve; not training occupancy",
        actual_ddp_failure_policy="abort all ranks; no reuse or silent fallback")
    report.update(candidate_config)
    evaluator, evaluator_delta = None, 0

    def initialize_probe_evaluator():
        nonlocal evaluator, evaluator_delta
        before = torch.cuda.memory_allocated(device)
        evaluator = build_evaluator(config, device)
        evaluator_delta = torch.cuda.memory_allocated(device)-before
        report["evaluator_instrument"] = evaluator.instrument
        report["rank0_evaluator_allocated_bytes"] = evaluator_delta
    rank_zero_action(rank, initialize_probe_evaluator)
    pose_mean = load_pose_mean(config, manifest)

    def failure_evidence(reason):
        if rank == 0:
            path = Path(config["output_root"])/"BS_PREFLIGHT_FAILED.json"
            if not path.exists():
                atomic_json(path, {**report, "passed": False, "reason": reason,
                    "training_started": False, "no_fake_short_windows": True})

    for mode in ("full55", "body22"):
        normalizer = Standardizer.load(Path(config["cache_root"])/f"stats_{mode}.npz")
        dataset = ClipWindowDataset(manifest, config["cache_root"], normalizer, mode)
        inventory = bs_window_inventory(dataset)
        summary = {key: inventory[key] for key in
                   ("long_window_count", "short_window_count", "training_window_inventory_sha256")}
        report["modes"][mode] = {"window_availability": summary, "attempts": []}
        if required_scenarios == ("long240",):
            bs_scenario_policy(config, manifest, inventory)
        elif not inventory["short"]:
            reason = f"{mode}: no actual short train clips; mixed scenario unavailable, not fabricated"
            failure_evidence(reason)
            raise RuntimeError(reason)

        def run_leaf(size, name, plan, real_ddp):
            seed_all(SEED)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            model = objective = optimizer = reserve = None
            leaf = dict(rank=rank, batch_per_gpu=size, scenario=name, passed=False,
                real_ddp=real_ddp, rank0_evaluator_resident=rank == 0,
                evaluator_allocated_bytes=evaluator_delta,
                gpu_total_bytes=torch.cuda.get_device_properties(device).total_memory)
            try:
                model, objective, optimizer = build_model(config, mode, normalizer, device, pose_mean)
                state_bytes = sum(value.numel()*value.element_size() for value in model.state_dict().values())
                leaf["model_state_bytes"] = state_bytes
                if real_ddp:
                    objective = DDP(objective, device_ids=[device.index],
                        broadcast_buffers=True, find_unused_parameters=False)
                else:
                    reserve = torch.empty(2*state_bytes+64*1024**2, dtype=torch.uint8, device=device)
                leaf.update(probe_scenario_updates(model, objective, optimizer, dataset, plan, device))
                torch.cuda.synchronize(device)
                allocated = torch.cuda.max_memory_allocated(device)
                reserved = torch.cuda.max_memory_reserved(device)
                leaf.update(peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                    passed=max(allocated, reserved) < .8*leaf["gpu_total_bytes"])
            except Exception as error:
                if real_ddp:
                    # Even genuine DDP OOM must abort the worker group. It is
                    # NOT a candidate fallback on a possibly broken collective.
                    raise
                leaf.update(error_type=type(error).__name__, error=str(error)[:1000],
                            fatal=not isinstance(error, torch.cuda.OutOfMemoryError))
            finally:
                model = objective = optimizer = reserve = None
                gc.collect()
                torch.cuda.empty_cache()
            return leaf

        def actual_probe(size):
            if len(inventory["long"]) < size*WORLD_SIZE:
                return dict(batch_per_gpu=size, passed=False, ranks=[], screen_ranks=[],
                    scenario_inputs={}, actual_ddp_attempted=False,
                    boundary_reason="not_enough_distinct_real_long_windows")
            rank_plans = [bs_scenario_inputs(inventory, size, required_scenarios=required_scenarios,
                          rank=index, world_size=WORLD_SIZE) for index in range(WORLD_SIZE)]
            plans = rank_plans[rank]
            screens_by_rank = [dict() for _ in range(WORLD_SIZE)]
            for name in required_scenarios:
                leaf = run_leaf(size, name, plans[name], False)
                gathered = [None]*WORLD_SIZE
                dist.all_gather_object(gathered, leaf)
                for item in gathered:
                    screens_by_rank[item["rank"]][name] = item
                if any(item.get("fatal") for item in gathered):
                    raise RuntimeError(f"Native CUDA MeanFlow screen failed; no fallback: {gathered}")
                if not all(item["passed"] for item in gathered):
                    screens = [aggregate_bs_rank(index, size, cases,
                        real_ddp=False, evaluator_delta=cases[name]["evaluator_allocated_bytes"],
                        required_scenarios=required_scenarios)
                        for index, cases in enumerate(screens_by_rank)]
                    return dict(batch_per_gpu=size, passed=False, scenario_inputs=rank_plans[0], rank_scenario_inputs=rank_plans,
                        screen_ranks=screens, ranks=[], actual_ddp_attempted=False,
                        boundary_reason="actual_bare_oom_or_80pct_boundary")
            screens = [aggregate_bs_rank(index, size, cases, real_ddp=False,
                evaluator_delta=cases[required_scenarios[0]]["evaluator_allocated_bytes"],
                required_scenarios=required_scenarios)
                for index, cases in enumerate(screens_by_rank)]
            ddp_by_rank = [dict() for _ in range(WORLD_SIZE)]
            # Every applicable bare scenario passed on ALL ranks before DDP.
            for name in required_scenarios:
                leaf = run_leaf(size, name, plans[name], True)
                gathered = [None]*WORLD_SIZE
                dist.all_gather_object(gathered, leaf)
                for item in gathered:
                    ddp_by_rank[item["rank"]][name] = item
                if not all(item["passed"] for item in gathered):
                    break
            results = [aggregate_bs_rank(index, size, cases, real_ddp=True,
                evaluator_delta=cases[required_scenarios[0]]["evaluator_allocated_bytes"],
                required_scenarios=required_scenarios)
                for index, cases in enumerate(ddp_by_rank)]
            passed = all(item["passed"] for item in results)
            if passed:
                for item in results:
                    validate_bs_rank_scenarios(item, rank_plans[item["rank"]], size, mode, required_scenarios=required_scenarios)
            return dict(batch_per_gpu=size, passed=passed, scenario_inputs=rank_plans[0], rank_scenario_inputs=rank_plans,
                screen_ranks=screens, ranks=results, actual_ddp_attempted=True,
                boundary_reason=None if passed else "actual_ddp_80pct_boundary",
                worst_all_rank_allocated_bytes=max(item["peak_allocated_bytes"] for item in results),
                worst_all_rank_reserved_bytes=max(item["peak_reserved_bytes"] for item in results))

        try:
            searched = search_bs_candidates(actual_probe, attempts=report["modes"][mode]["attempts"])
        except Exception as error:
            failure_evidence(f"{mode}: {type(error).__name__}: {error}")
            raise
        model_bytes = max(item["model_state_bytes"] for attempt in searched["attempts"]
                          for item in attempt["ranks"] if item["passed"])
        report["modes"][mode] = {**searched, "window_availability": summary,
            "model_state_bytes": model_bytes,
            "global_batches": [WORLD_SIZE*value for value in searched["selected_batch_per_gpu"]]}
    report["passed"] = True
    if rank == 0:
        atomic_json(Path(config["output_root"])/"BS_PREFLIGHT_PASSED.json", report)
    evaluator = None
    return report


def project_rotations(value, normalizer, mode):
    mean = torch.as_tensor(normalizer.mean, device=value.device)
    std = torch.as_tensor(normalizer.std, device=value.device)
    physical = value*std+mean
    joints = 55 if mode == "full55" else 22
    rotations = physical[..., :joints*6].reshape(*physical.shape[:-1], joints, 6)
    projected = matrix_to_rotation_6d(rotation_6d_to_matrix(rotations)).flatten(-2)
    return (torch.cat((projected, physical[..., -3:]), -1)-mean)/std


def export_predictions(model, normalizer, config, manifest, split, step, output, device, checkpoint_digest):
    """All explicit split IDs, complete official budgets, without GT seeds."""
    from evaluation import manifest_records, rollout_frames
    records = manifest_records(manifest, split, EXPECTED_COUNTS[split])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    sampled = []
    normalizer_digest = sha256(Path(config["cache_root"])/f"stats_{model.mode}.npz")
    with torch.inference_mode():
        for record in records:
            ident = record["id"]
            with np.load(Path(config["cache_root"])/(ident+".npz"), allow_pickle=False) as archive:
                # Cache is audited to whole common audio/GT seconds. No posed
                # motion values, root origins or expressions enter sampling.
                frames = int(record.get("frames", record.get("evaluation_frames", rollout_frames(len(archive["audio"])))))
                if frames > len(archive["audio"]):
                    raise RuntimeError(f"Features shorter than fixed evaluation budget: {ident}")
                audio = torch.as_tensor(archive["audio"][:frames].astype(np.float32)[None], device=device)
                text = torch.as_tensor(archive["text"][:frames].astype(np.float32)[None], device=device)
                betas = archive["betas"].copy()
            # Each step-count starts from exactly the same per-ID noise.
            clip_seed = SEED+int(hashlib.sha256(ident.encode()).hexdigest()[:8], 16)
            destination = output/("res_"+ident+".npz")
            if destination.exists():
                with np.load(destination, allow_pickle=False) as previous:
                    compatible = (str(previous["source_id"].item()) == ident,
                        str(previous["model_pose_policy"].item()) == model.mode,
                        str(previous["checkpoint_sha256"].item()) == checkpoint_digest,
                        str(previous["normalizer_sha256"].item()) == normalizer_digest,
                        str(previous["config_sha256"].item()) == config_hash(config),
                        int(previous["sampling_step"].item()) == step,
                        int(previous["sampling_seed"].item()) == clip_seed,
                        previous["poses"].shape == (frames, 165),
                        previous["trans"].shape == (frames, 3),
                        np.isfinite(previous["poses"]).all(), np.isfinite(previous["trans"]).all())
                    if not all(compatible):
                        raise RuntimeError("Partial prediction belongs to another checkpoint/configuration")
                sampled.append({"id": ident, "frames": frames, "sampling_seed": clip_seed})
                continue
            generator = torch.Generator(device=device).manual_seed(clip_seed)
            noise = torch.randn(1, frames, model.motion_dim, generator=generator, device=device)
            prediction = model.sample(noise, steps=step,
                project_func=lambda value: project_rotations(value, normalizer, model.mode),
                audio=audio, text=text, person_id=torch.tensor([int(ident.split("_", 1)[0])], device=device),
                padding_mask=torch.ones(1, frames, dtype=torch.bool, device=device))
            physical = normalizer.denormalize(prediction[0].float().cpu().numpy())
            # The model's first horizontal delta is ignored by its canonical
            # root decoder. No GT origin is added to generated translation.
            physical[0, -3:-1] = 0
            poses, trans = decode_motion(physical, model.mode)
            temporary = destination.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, poses=poses, trans=trans, betas=betas,
                         expressions=np.zeros((frames, 100), dtype=np.float32),
                         mocap_frame_rate=np.asarray(30), source_id=np.asarray(ident),
                         model_pose_policy=np.asarray(model.mode), seed_pose_used=np.asarray(False),
                         checkpoint_sha256=np.asarray(checkpoint_digest), normalizer_sha256=np.asarray(normalizer_digest),
                         config_sha256=np.asarray(config_hash(config)), sampling_step=np.asarray(step),
                         sampling_seed=np.asarray(clip_seed))
            os.replace(temporary, destination)
            sampled.append({"id": ident, "frames": frames, "sampling_seed": clip_seed})
    return sampled


def fixed_render_clips(manifest, split):
    ids = [value["id"] if isinstance(value, dict) else value for value in manifest["splits"][split]]
    result = []
    for person in (1, 2, 30):
        candidates = [value for value in ids if int(value.split("_", 1)[0]) == person]
        if not candidates:
            raise RuntimeError(f"No fixed render clip for person {person} in {split}")
        result.append(candidates[0])
    return result


def render_checkpoint(config, manifest, split, predictions, output, mode):
    clips = fixed_render_clips(manifest, split)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports, missing = [], []
    for ident in clips:
        audit_path = output/(ident+"_render_audit.json")
        video_path = output/(ident+"_gt_pred_smplx.mp4")
        if not audit_path.exists():
            if video_path.exists():
                raise RuntimeError("Partial render video has no audit; needs operator decision")
            missing.append(ident)
            continue
        report = json.loads(audit_path.read_text())
        gt_path = Path(config["raw_root"])/"smplxflame_30"/(ident+".npz")
        pred_path = Path(predictions)/("res_"+ident+".npz")
        checks = (report.get("id") == ident, report.get("pose_policy") == mode,
                  report.get("split") == split,
                  report.get("sampling_step") == 4, report.get("finite") is True,
                  report.get("native_y_up") is True, report.get("nonempty_all_frames") is True,
                  report.get("audio_muxed") is True, video_path.is_file(),
                  report.get("prediction_sha256") == sha256(pred_path),
                  report.get("gt_sha256") == sha256(gt_path))
        if not all(checks) or report.get("video_sha256") != sha256(video_path):
            raise RuntimeError("Existing render audit does not match this checkpoint's actual sources")
        reports.append(report)
    command = [sys.executable, str(Path(__file__).parent/"render.py"),
        "--manifest", config["manifest"], "--predictions", str(predictions),
        "--raw-root", config["raw_root"], "--smplx", config["smplx"],
        "--output", str(output), "--clips", *missing, "--pose-policy", mode,
        "--sampling-step", "4", "--split", split, "--device", "cuda:0", "--seconds", "10"]
    if missing:
        with (output/"render.log").open("a", encoding="utf-8") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True,
                           timeout=int(config.get("render_timeout_seconds", 21600)))
        # Re-enter once: all newly rendered files must now pass source/hash,
        # finite, visibility and real audio/video audit checks above.
        return render_checkpoint(config, manifest, split, predictions, output, mode)
    videos = [output/(ident+"_gt_pred_smplx.mp4") for ident in clips]
    qa = [output/(ident+"_render_audit.json") for ident in clips]
    if len(reports) != 3 or any(path.stat().st_size <= 1024 for path in videos):
        raise RuntimeError("Renderer returned without three actual paired SMPL-X videos and audits")
    atomic_json(output/"training_render_bundle_complete.json", {"completed": True, "clips": reports,
        "pose_policy": mode, "split": split, "sampling_step": 4, "visual_human_review": "pending"})
    qa.append(output/"training_render_bundle_complete.json")
    return {"fixed_clip_ids": clips, "sampling_step": 4, "pose_policy": mode,
            "videos": [str(path) for path in videos], "qa": [str(path) for path in qa]}


def completed_bundle(path, config, checkpoint):
    if not Path(path).is_file():
        return None
    result = json.loads(Path(path).read_text())
    if result.get("status") != "complete" or result.get("config_sha256") != config_hash(config):
        raise RuntimeError("Existing bundle is incompatible; refusing to overwrite")
    if result.get("checkpoint_sha256") != sha256(checkpoint):
        raise RuntimeError("Checkpoint changed under an existing completed bundle")
    for item in result.get("artifact_hashes", []):
        if not Path(item["path"]).is_file() or sha256(item["path"]) != item["sha256"]:
            raise RuntimeError("Completed bundle artifact missing/changed")
    if not result.get("artifact_hashes"):
        raise RuntimeError("Completed bundle has no verified artifacts")
    prediction_manifest_path = Path(result.get("prediction_artifact_manifest", ""))
    if not prediction_manifest_path.is_file():
        raise RuntimeError("Completed bundle has no complete prediction artifact manifest")
    inventory = json.loads(prediction_manifest_path.read_text())
    if (inventory.get("schema") != "m2g-prediction-artifacts-v1" or
            inventory.get("config_sha256") != config_hash(config) or
            inventory.get("checkpoint_sha256") != result["checkpoint_sha256"] or
            inventory.get("split") != result.get("split") or
            inventory.get("model_pose_policy") != result.get("mode")):
        raise RuntimeError("Completed prediction artifact manifest is incompatible")
    from evaluation import manifest_records
    manifest = json.loads(Path(config["manifest"]).read_text())
    split = result["split"]
    expected_ids = [record["id"] for record in manifest_records(manifest, split, EXPECTED_COUNTS[split])]
    expected_ids_hash = hashlib.sha256(json.dumps(expected_ids, separators=(",", ":")).encode()).hexdigest()
    artifact_hashes = {str(Path(item["path"])): item["sha256"] for item in result["artifact_hashes"]}
    mode = result["mode"]
    if mode not in ("full55", "body22"):
        raise RuntimeError("Completed bundle has an invalid model pose policy")
    policies = ("full55", "body22") if mode == "full55" else ("body22",)
    if (set(result.get("metrics", {})) != {"1", "2", "4"} or
            set(result.get("renders", {})) != set(policies) or result.get("primary_pose_policy") != mode):
        raise RuntimeError("Completed bundle is missing required step/mask metric or render coverage")
    bundle_root = Path(path).parent
    for step in ("1", "2", "4"):
        if set(result["metrics"][step]) != set(policies):
            raise RuntimeError("Completed bundle has incomplete pose-mask metrics")
        for policy in policies:
            metric_path = bundle_root/f"metrics_step{step}_{policy}.json"
            if (artifact_hashes.get(str(metric_path)) != sha256(metric_path) or
                    json.loads(metric_path.read_text()) != result["metrics"][step][policy] or
                    not math.isfinite(result["metrics"][step][policy].get("fgd", float("nan")))):
                raise RuntimeError("Completed metrics are not bound to the actual metric artifacts")
    if result.get("selection_fgd_4step") != result["metrics"]["4"][mode]["fgd"]:
        raise RuntimeError("Completed bundle selection score differs from its saved primary metric")
    clips = fixed_render_clips(manifest, split)
    for policy in policies:
        render = result["renders"][policy]
        render_root = bundle_root/f"render_step4_{policy}"
        videos = [str(render_root/(ident+"_gt_pred_smplx.mp4")) for ident in clips]
        qa = [str(render_root/(ident+"_render_audit.json")) for ident in clips]
        qa.append(str(render_root/"training_render_bundle_complete.json"))
        if (render.get("pose_policy") != policy or render.get("sampling_step") != 4 or
                render.get("fixed_clip_ids") != clips or render.get("videos") != videos or render.get("qa") != qa or
                any(item not in artifact_hashes for item in videos+qa)):
            raise RuntimeError("Completed bundle is missing required fixed paired-render artifacts")
    if artifact_hashes.get(str(prediction_manifest_path)) != sha256(prediction_manifest_path):
        raise RuntimeError("Prediction artifact manifest is not bound to this completed bundle")
    if set(inventory.get("steps", {})) != {"1", "2", "4"}:
        raise RuntimeError("Completed bundle must retain predictions for all three sampling steps")
    for step in ("1", "2", "4"):
        group = inventory["steps"][step]
        files = group.get("files", [])
        if (group.get("count") != EXPECTED_COUNTS[split] or
                group.get("ids_sha256") != expected_ids_hash or
                [item.get("id") for item in files] != expected_ids):
            raise RuntimeError("Completed bundle prediction coverage is incomplete")
        for item in files:
            prediction_path = Path(item["path"])
            if (item.get("checkpoint_sha256") != result["checkpoint_sha256"] or
                    item.get("sampling_step") != int(step) or
                    prediction_path.name != "res_"+item["id"]+".npz" or
                    str(prediction_path.parent) != result["predictions"][step] or
                    artifact_hashes.get(str(prediction_path)) != item.get("sha256")):
                raise RuntimeError("Completed prediction provenance is incompatible")
            # The generic artifact loop already streamed and checked this
            # exact path/hash. No second full read of every motion archive.
    return result


def prediction_artifact_inventory(config, split, mode, checkpoint_digest, predictions, sampled_records):
    """Stream hashes of every prediction; never embed motion arrays in JSON."""
    inventory = {"schema": "m2g-prediction-artifacts-v1", "config_sha256": config_hash(config),
                 "checkpoint_sha256": checkpoint_digest, "split": split,
                 "model_pose_policy": mode, "steps": {}}
    for step in ("1", "2", "4"):
        records = sampled_records[step]
        if len(records) != EXPECTED_COUNTS[split] or len({record["id"] for record in records}) != len(records):
            raise RuntimeError("Export did not retain every official prediction ID")
        ids = [record["id"] for record in records]
        files = []
        for record in records:
            path = Path(predictions[step])/("res_"+record["id"]+".npz")
            if not path.is_file():
                raise RuntimeError("Exported prediction artifact is missing")
            files.append({"id": record["id"], "path": str(path), "sha256": sha256(path),
                          "frames": record["frames"], "sampling_step": int(step),
                          "sampling_seed": record["sampling_seed"], "checkpoint_sha256": checkpoint_digest})
        inventory["steps"][step] = {"count": len(files), "files": files,
            "ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()}
    return inventory


def evaluate_bundle(model, normalizer, config, manifest, evaluator, checkpoint, epoch, split, device, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    previous = completed_bundle(output/"complete.json", config, checkpoint)
    if previous:
        return previous
    pending = dict(status="pending", epoch=epoch, split=split, mode=model.mode,
                   checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
                   config_sha256=config_hash(config), started_at_unix=time.time())
    atomic_json(output/"pending.json", pending)
    model.eval()
    try:
        with evaluation_rng(device):
            metrics, artifacts, predictions, sampled_records = {}, [], {}, {}
            policies = ("full55", "body22") if model.mode == "full55" else ("body22",)
            for step in (1, 2, 4):
                prediction_root = output/f"predictions_step{step}"
                records = export_predictions(model, normalizer, config, manifest, split, step, prediction_root, device,
                                             pending["checkpoint_sha256"])
                predictions[str(step)] = str(prediction_root)
                sampled_records[str(step)] = records
                atomic_json(output/f"sampling_step{step}.json", records)
                for policy in policies:
                    result = evaluator.evaluate(prediction_root=prediction_root,
                        ground_truth_root=Path(config["raw_root"])/"smplxflame_30",
                        audio_root=Path(config["raw_root"])/"wave16k", manifest=manifest,
                        split=split, pose_policy=policy, expected_count=EXPECTED_COUNTS[split],
                        model_pose_policy=model.mode)
                    if not math.isfinite(result["fgd"]):
                        raise FloatingPointError("Nonfinite validation/test FGD")
                    metrics.setdefault(str(step), {})[policy] = result
                    metric_path = output/f"metrics_step{step}_{policy}.json"
                    atomic_json(metric_path, result)
                    artifacts.append(metric_path)
            renders = {}
            for policy in policies:
                render = render_checkpoint(config, manifest, split, predictions["4"],
                                           output/f"render_step4_{policy}", policy)
                renders[policy] = render
                artifacts.extend(Path(path) for path in render["videos"]+render["qa"])
            inventory = prediction_artifact_inventory(config, split, model.mode,
                pending["checkpoint_sha256"], predictions, sampled_records)
            inventory_path = output/"prediction_artifacts.json"
            atomic_json(inventory_path, inventory)
            artifacts.append(inventory_path)
            artifact_hashes = [{"path": str(path), "sha256": sha256(path)} for path in artifacts]
            artifact_hashes.extend({"path": item["path"], "sha256": item["sha256"]}
                                   for group in inventory["steps"].values() for item in group["files"])
            result = {**pending, "status": "complete", "completed_at_unix": time.time(),
                      "metrics": metrics, "predictions": predictions, "renders": renders,
                      "prediction_artifact_manifest": str(inventory_path),
                      "primary_pose_policy": model.mode,
                      "selection_fgd_4step": metrics["4"][model.mode]["fgd"],
                      "artifact_hashes": artifact_hashes}
            atomic_json(output/"complete.json", result)
            atomic_json(output/"pending.json", {**pending, "status": "complete",
                        "completed_bundle": str(output/"complete.json")})
            return result
    except Exception:
        atomic_json(output/"failed.json", {**pending, "status": "failed", "traceback": traceback.format_exc()})
        raise
    finally:
        model.train()


def rank_zero_action(rank, action):
    """Other ranks wait; rank-zero metric/render failures abort every rank."""
    result = [None]
    if rank == 0:
        try:
            result[0] = {"passed": True, "result": action()}
        except Exception:
            result[0] = {"passed": False, "traceback": traceback.format_exc()}
    dist.broadcast_object_list(result, src=0)
    if not result[0]["passed"]:
        raise RuntimeError("Rank-zero checkpoint bundle failed; no further epochs:\n"+result[0]["traceback"])
    return result[0]["result"]


def train_run(config, manifest, mode, batch_size, resume, rank, device):
    normalizer = Standardizer.load(Path(config["cache_root"])/f"stats_{mode}.npz")
    dataset = ClipWindowDataset(manifest, config["cache_root"], normalizer, mode, length=240, stride=120)
    sampler = DistributedSampler(dataset, num_replicas=WORLD_SIZE, rank=rank, shuffle=True, seed=SEED, drop_last=False)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
        num_workers=int(config.get("num_workers", 4)), pin_memory=True, drop_last=False,
        persistent_workers=int(config.get("num_workers", 4)) > 0)
    pose_mean = load_pose_mean(config, manifest)
    model, objective, optimizer = build_model(config, mode, normalizer, device, pose_mean)
    # Wrap the complete objective, not each of its prediction/JVP forward
    # calls. Reducer setup occurs once after the complete loss forward.
    objective = DDP(objective, device_ids=[device.index], broadcast_buffers=True, find_unused_parameters=False)
    output = Path(config["output_root"])/f"{mode}_bs{batch_size}"
    output.mkdir(parents=True, exist_ok=True)
    state_path = output/"latest_full.pt"
    cfg_hash = config_hash(config)
    metadata = dict(config_sha256=cfg_hash, mode=mode, seed=SEED, epochs=EPOCHS,
        batch_per_gpu=batch_size, global_batch=WORLD_SIZE*batch_size, world_size=WORLD_SIZE,
        unique_training_windows=len(dataset), sampler_windows_per_epoch=sampler.total_size,
        repeated_tail_indices_per_epoch=sampler.total_size-len(dataset), no_omitted_windows=True,
        steps_per_epoch=len(loader), full_epoch_definition="Every train window, distributed sampler padding only; no step cutoff",
        optimizer="Reference FlowerDance Adan", learning_rate=config.get("learning_rate", 2e-4),
        weight_decay=config.get("weight_decay", 0.02), schedule="constant learning rate",
        architecture=model.architecture_config(), loss=objective.module.protocol_config(),
        smplx_pose_mean_sha256=hashlib.sha256(pose_mean.tobytes()).hexdigest(),
        normalizer_sha256=sha256(Path(config["cache_root"])/f"stats_{mode}.npz"),
        manifest_sha256=sha256(config["manifest"]), save_every_epochs=SAVE_EVERY,
        selection="minimum independent val106 4-step FGD; ties earlier epoch",
        source_hashes={name: sha256(Path(__file__).parent/name) for name in SOURCE_FILES})
    def initialize_metadata():
        if (output/"run.json").exists():
            old = json.loads((output/"run.json").read_text())
            if not resume or old != metadata:
                raise RuntimeError("Existing run directory; explicit compatible resume required")
        else:
            if resume:
                raise RuntimeError("No existing run to resume")
            atomic_json(output/"run.json", metadata)
        return None
    rank_zero_action(rank, initialize_metadata)
    start_epoch, updates = 0, 0
    seed_all(SEED+rank)
    if resume:
        state = torch.load(state_path, map_location=device, weights_only=False)
        if state["config_sha256"] != cfg_hash or state["metadata"] != metadata:
            raise RuntimeError("Resume configuration is incompatible")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, updates = state["epoch"], state["updates"]
        restore_rng(state["rng_by_rank"][rank], device)
    evaluator = None
    def initialize_evaluator():
        nonlocal evaluator
        evaluator = build_evaluator(config, device)
        return None
    rank_zero_action(rank, initialize_evaluator)

    def finish_bundle(epoch):
        checkpoint = output/"checkpoints"/f"epoch_{epoch:05d}.pt"
        return evaluate_bundle(model, normalizer, config, manifest, evaluator, checkpoint, epoch, "val",
                               device, output/"validation"/f"epoch_{epoch:05d}")
    if resume and start_epoch and start_epoch % SAVE_EVERY == 0:
        rank_zero_action(rank, lambda: finish_bundle(start_epoch))
    for epoch in range(start_epoch+1, EPOCHS+1):
        sampler.set_epoch(epoch-1)
        objective.train()
        begin, local_samples, local_frames = time.monotonic(), 0, 0
        loss_sum = torch.zeros(1, dtype=torch.float64, device=device)
        for batch_index, batch in enumerate(loader):
            clean, rest, condition = batch_on_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            total, terms = objective(clean, rest, condition)
            if not torch.isfinite(total):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {batch_index}")
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            updates += 1
            local_samples += len(clean)
            local_frames += int(condition["padding_mask"].sum())
            loss_sum += total.detach().double()*len(clean)
            if rank == 0 and updates % int(config.get("log_every_updates", 10)) == 0:
                append_json(output/"train_updates.jsonl", dict(epoch=epoch, batch_index=batch_index, updates=updates,
                    loss=float(total), grad_norm=float(grad_norm), **{key: float(value.detach()) for key, value in terms.items()}))
        counts = torch.tensor([local_samples, local_frames], dtype=torch.long, device=device)
        dist.all_reduce(counts)
        dist.all_reduce(loss_sum)
        if int(counts[0]) != sampler.total_size:
            raise RuntimeError("Epoch did not process every distributed training sample")
        if rank == 0:
            append_json(output/"epochs.jsonl", dict(epoch=epoch, updates=updates, real_samples=int(counts[0]),
                unique_windows=len(dataset), repeated_tail_indices=sampler.total_size-len(dataset),
                valid_frames_including_repeats=int(counts[1]), mean_loss=float(loss_sum/counts[0]),
                seconds=time.monotonic()-begin))
        if epoch % SAVE_EVERY == 0:
            rngs = [None]*WORLD_SIZE
            dist.all_gather_object(rngs, capture_rng(device))
            def save_checkpoint():
                target = output/"checkpoints"/f"epoch_{epoch:05d}.pt"
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists():
                    raise RuntimeError("Historical checkpoint already exists; refusing overwrite")
                cpu_model = {key: value.detach().cpu() for key, value in model.state_dict().items()}
                torch_save_atomic(dict(model=cpu_model, epoch=epoch, updates=updates,
                    architecture=model.architecture_config(), metadata=metadata, config_sha256=cfg_hash), target)
                torch_save_atomic(dict(model=cpu_model, optimizer=optimizer.state_dict(), epoch=epoch,
                    updates=updates, rng_by_rank=rngs, metadata=metadata, config_sha256=cfg_hash), state_path)
                atomic_json(output/"progress.json", dict(epoch=epoch, updates=updates, checkpoint=str(target),
                    status="checkpoint_saved_evaluation_pending"))
                return str(target)
            rank_zero_action(rank, save_checkpoint)
            result = rank_zero_action(rank, lambda: finish_bundle(epoch))
            if rank == 0:
                atomic_json(output/"progress.json", dict(epoch=epoch, updates=updates, status="bundle_complete",
                    validation_fgd_4step=result["selection_fgd_4step"]))
    def select_and_test():
        candidates = []
        for epoch in range(SAVE_EVERY, EPOCHS+1, SAVE_EVERY):
            checkpoint = output/"checkpoints"/f"epoch_{epoch:05d}.pt"
            result = completed_bundle(output/"validation"/f"epoch_{epoch:05d}"/"complete.json", config, checkpoint)
            if not result:
                raise RuntimeError(f"Missing required checkpoint validation/render bundle {epoch}")
            candidates.append((result["selection_fgd_4step"], epoch, checkpoint))
        best_fgd, best_epoch, checkpoint = min(candidates, key=lambda value: (value[0], value[1]))
        chosen = torch.load(checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(chosen["model"], strict=True)
        result = evaluate_bundle(model, normalizer, config, manifest, evaluator, checkpoint, best_epoch,
                                 "test", device, output/"test_selected")
        selection = dict(status="complete", mode=mode, batch_per_gpu=batch_size,
            selected_epoch=best_epoch, independent_val_fgd_4step=best_fgd,
            checkpoint=str(checkpoint), config_sha256=cfg_hash, test=result,
            selection_uses_test=False, epochs_completed=EPOCHS, historical_checkpoint_count=200)
        atomic_json(output/"TRAINING_COMPLETE.json", selection)
        return selection
    rank_zero_action(rank, select_and_test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("full55", "body22"))
    parser.add_argument("--batch-per-gpu", type=int)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    gate = require_precheck(config)
    if socket.gethostname() != config["expected_hostname"]:
        raise RuntimeError("Wrong task hostname; refusing GPU allocation")
    if not torch.cuda.is_available() or torch.cuda.device_count() != WORLD_SIZE:
        raise RuntimeError("Exactly eight visible CUDA GPUs are required")
    rank, local_rank = int(os.environ.get("RANK", "-1")), int(os.environ.get("LOCAL_RANK", "-1"))
    if int(os.environ.get("WORLD_SIZE", "0")) != WORLD_SIZE or not 0 <= local_rank < WORLD_SIZE:
        raise RuntimeError("Launch through torchrun --nproc_per_node=8")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_per_process_memory_fraction(0.8, local_rank)
    dist.init_process_group("nccl", timeout=timedelta(hours=max(6, int(config.get("ddp_timeout_hours", 8)))))
    try:
        manifest = json.loads(Path(config["manifest"]).read_text())
        if {key: len(manifest["splits"][key]) for key in EXPECTED_COUNTS} != EXPECTED_COUNTS:
            raise RuntimeError("Wrong full25 official train/val/test split counts")
        if not (Path(config["cache_root"])/"ready.json").exists():
            raise RuntimeError("Whole-clip feature preparation/statistics not complete")
        seed_all(SEED)
        if args.preflight:
            calibration(config, manifest, rank, device, gate)
        else:
            if not args.mode or not args.batch_per_gpu or args.batch_per_gpu < 1:
                parser.error("training requires --mode and positive --batch-per-gpu")
            preflight = require_bs_preflight(config, gate, Path(config["output_root"])/"BS_PREFLIGHT_PASSED.json")
            if args.batch_per_gpu not in preflight["modes"][args.mode]["selected_batch_per_gpu"]:
                raise RuntimeError("Requested batch was not one of the three actual safe probes")
            train_run(config, manifest, args.mode, args.batch_per_gpu, args.resume, rank, device)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
