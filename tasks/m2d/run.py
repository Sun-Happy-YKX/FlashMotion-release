#!/usr/bin/env python3
"""Plan M2D commands, prepare source copies and launch checkpoint evaluation.

The source uses fixed paths and writes at module import time. Use --prepare
to create a separate configured copy. Eval --execute validates the current
inputs and runtime before launching GPU inference; planning is CPU-only.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
import shlex
import stat
import socket
import subprocess
import sys
from typing import Any


sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOT_RELATIVE = "tasks/m2d/snapshots/genre_20260929"
HISTORICAL_PROJECT = PurePosixPath("/efs/yangkaixing/FlowerDance_x0_ablation")
HISTORICAL_RUN = HISTORICAL_PROJECT / "runs/music_genre_ablation_20260929"
TRAIN_SHA256 = "03f1a2f6dc276c3aad600600fa29db3b5ca2c5d02b415773cb70c355a4ceb1b9"
CORRECTED_EVAL_SHA256 = "bd28059623a1b00fc278e131f9664ae2fa78b9f4addec237d01c868c715dd3fb"
PATH_KEYS = (
    "release_root", "python", "data_root", "checkpoint_root", "output_root",
    "generated_code_root", "evaluation_output_root", "asset_fetch_receipt",
    "runtime_evidence", "precheck_path",
)
CONTRACT = {
    "seed": 42, "world_size": 4, "batch_per_rank": 64, "global_batch": 256,
    "epochs": 4000, "archived_updates_per_epoch": 77,
    "archived_total_updates": 308000, "train_frames": 240, "test_frames": 1020,
    "motion_fps": 30, "native_motion_dim": 151, "pose_dim_after_four_channels": 147,
    "music_feature_dim": 35, "mixed_precision": "no", "save_every_epochs": 50,
    "eval_batch": 2, "eval_seed": 42, "eval_nfe": [1, 4],
    "pretrained": None, "resume": None,
}
SHARED_OWNER_IDENTITIES = [
    {"pid": 1079227, "startticks": 790086285,
     "cwd": "/efs/yangkaixing/FlowerDance_T2M_Long/runs/no_fk_retrain200_20261008/code"},
    {"pid": 1101724, "startticks": 796655821,
     "cwd": "/efs/yangkaixing/FlowerDance_T2M_Long/runs/uncond_full_official_r3_20261009/code"},
]
SHARED_CAP_SOURCES = {
    "/efs/yangkaixing/FlowerDance_T2M_Long/runs/no_fk_retrain200_20261008/code/train_t2m_long.py":
        "db485e2b183552fba81b399f1ad2a858ec56e8b23cc37c2a940abd19c9a36ac4",
    "/efs/yangkaixing/FlowerDance_T2M_Long/runs/uncond_full_official_r3_20261009/code/train_t2m_long.py":
        "9a02aa16525d1366dbe131b73a9bdeb07525e088744aa6c237d3a0c18b9038fa",
}


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def linux_path(value: Any, key: str, allow_historical_input: bool = False) -> PurePosixPath:
    if not isinstance(value, str) or not value.startswith("/"):
        raise ValueError(f"{key} must be an explicit absolute Linux path")
    if value.startswith("//") or "\\" in value or ":" in value:
        raise ValueError(f"{key} contains an ambiguous/non-Linux path")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{key} contains control characters")
    if any(part in ("", ".", "..") for part in value.split("/")[1:]):
        raise ValueError(f"{key} must not contain empty, '.' or '..' segments")
    result = PurePosixPath(value)
    if result == PurePosixPath("/"):
        raise ValueError(f"{key} must not be the filesystem root")
    if not allow_historical_input and (result.is_relative_to(HISTORICAL_PROJECT) or HISTORICAL_PROJECT.is_relative_to(result)):
        raise ValueError(f"{key} overlaps the historical EFS project")
    return result


def overlaps(first: PurePosixPath, second: PurePosixPath) -> bool:
    return first.is_relative_to(second) or second.is_relative_to(first)


def validate_paths(paths: dict[str, Any]) -> None:
    if type(paths.get("schema_version")) is not int or paths["schema_version"] != 1:
        raise ValueError("paths JSON requires schema_version=1")
    normalized = {key: linux_path(paths.get(key), key, key in ("python", "data_root", "checkpoint_root")) for key in PATH_KEYS}
    for key in ("task_id", "expected_hostname"):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise ValueError(f"Supply an independent {key}")
    if normalized["generated_code_root"] != normalized["output_root"] / "code":
        raise ValueError("generated_code_root must be output_root/code to preserve native R derivation")
    for key in ("output_root", "evaluation_output_root", "data_root", "checkpoint_root"):
        if overlaps(normalized[key], normalized["release_root"]):
            raise ValueError(f"{key} must be outside the source checkout")
    for output in ("output_root", "evaluation_output_root"):
        for data in ("data_root", "checkpoint_root", "python"):
            if overlaps(normalized[output], normalized[data]):
                raise ValueError(f"{output} overlaps input/runtime path {data}")
    if overlaps(normalized["output_root"], normalized["evaluation_output_root"]):
        raise ValueError("Training and independent corrected-evaluation outputs must be separate")
    for key in ("asset_fetch_receipt", "runtime_evidence", "precheck_path"):
        if any(overlaps(normalized[key], normalized[output])
               for output in ("output_root", "evaluation_output_root", "release_root")):
            raise ValueError(f"{key} must not be inside/above source or generated outputs")
    devices = paths.get("visible_devices")
    if (not isinstance(devices, list) or len(devices) != 4
            or any(not isinstance(device, str) or not re.fullmatch(r"[0-9]+", device) for device in devices)
            or len(set(devices)) != 4):
        raise ValueError("visible_devices must name four distinct numeric device IDs; availability is unverified")
    ports = paths.get("main_process_ports")
    if not isinstance(ports, dict) or set(ports) != {"genre", "no_genre"}:
        raise ValueError("main_process_ports must contain genre and no_genre")
    if any(type(port) is not int or not 1024 <= port <= 65535 for port in ports.values()):
        raise ValueError("main_process_ports must be integers in 1024..65535")
    if len(set(ports.values())) != 2:
        raise ValueError("genre and no_genre ports must differ")


def safe_source_file(repo_root: Path, relative: Any) -> Path:
    if (not isinstance(relative, str) or "\\" in relative or ":" in relative
            or any(ord(character) < 32 for character in relative)):
        raise ValueError("Unsafe frozen source path")
    parts = relative.split("/")
    if any(part in ("", ".", "..") for part in parts) or PurePosixPath(relative).is_absolute():
        raise ValueError("Unsafe frozen source path")
    root = repo_root.resolve()
    current = root
    for part in parts:
        current = current / part
        info = current.lstat()
        if current.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
            raise ValueError(f"Symlink/reparse point in frozen source: {relative}")
    if not current.is_file() or not current.resolve().is_relative_to(root):
        raise ValueError(f"Unsafe/missing frozen source: {relative}")
    return current


def verify_frozen_source(repo_root: Path) -> dict[str, Any]:
    receipt = read_json(repo_root / "manifests/frozen_source_collection.json")
    entries = [item for item in receipt["files"]
               if isinstance(item, dict) and isinstance(item.get("path"), str)
               and (item["path"].startswith(SNAPSHOT_RELATIVE + "/code/")
                    or item["path"] in {SNAPSHOT_RELATIVE + "/provenance/" + name
                                         for name in ("formal_code_sha256.json", "full_control.py",
                                                      "music_corrected_eval_20260930.py")})]
    if not entries:
        raise ValueError("No M2D source entries in frozen collection receipt")
    by_path: dict[str, dict[str, Any]] = {}
    for item in entries:
        relative = item["path"]
        if relative in by_path:
            raise ValueError(f"Duplicate frozen source entry: {relative}")
        payload = safe_source_file(repo_root, relative).read_bytes()
        if (type(item.get("bytes")) is not int or len(payload) != item["bytes"]
                or hashlib.sha256(payload).hexdigest() != item.get("sha256")):
            raise ValueError(f"Frozen source receipt mismatch: {relative}")
        if relative.endswith(".py"):
            ast.parse(payload, filename=relative)
        by_path[relative] = item
    formal_path = SNAPSHOT_RELATIVE + "/provenance/formal_code_sha256.json"
    train_path = SNAPSHOT_RELATIVE + "/code/train_pair.py"
    formal = read_json(safe_source_file(repo_root, formal_path))
    if not formal or train_path not in by_path:
        raise ValueError("Formal M2D source/training entry missing from receipt")
    for relative, expected in formal.items():
        full_relative = SNAPSHOT_RELATIVE + "/code/" + relative
        if full_relative not in by_path or by_path[full_relative]["sha256"] != expected:
            raise ValueError(f"Formal source SHA mismatch or missing entry: {relative}")
    if formal.get("train_pair.py") != TRAIN_SHA256:
        raise ValueError("Unexpected train_pair.py identity")
    native = safe_source_file(repo_root, train_path).read_text(encoding="utf-8")
    required = (
        "R=Path(__file__).resolve().parent.parent",
        "S=Path('/efs/yangkaixing/FlowerDance_x0_ablation')",
        "version,phase=sys.argv[1:3]",
        "assert version in ('genre','no_genre') and phase in ('verify','train')",
        "use_genre=version=='genre';out=R/version/phase",
        "out.mkdir(exist_ok=True,parents=True)",
    )
    if any(fragment not in native for fragment in required):
        raise ValueError("Native path/CLI/write contract changed; re-review before planning")
    return {"receipt_files_verified": len(entries), "formal_code_files_verified": len(formal),
            "train_sha256": TRAIN_SHA256, "frozen_bytes_modified": True}


def build_plan(phase: str, version: str, paths: dict[str, Any], repo_root: Path = REPO_ROOT,
               selection: dict[str, Any] | None = None) -> dict[str, Any]:
    if phase not in ("verify", "train", "eval") or version not in ("genre", "no_genre"):
        raise ValueError("Training native arguments are genre|no_genre and verify|train only")
    validate_paths(paths)
    frozen = verify_frozen_source(repo_root)
    source = PurePosixPath(paths["release_root"]) / SNAPSHOT_RELATIVE / "code"
    proposed = PurePosixPath(paths["generated_code_root"])
    data = PurePosixPath(paths["data_root"])
    output = PurePosixPath(paths["output_root"])
    if phase == "eval":
        if (not selection or selection.get("version") != version
                or type(selection.get("epoch")) is not int or selection["epoch"] < 1
                or type(selection.get("n")) is not int or selection["n"] not in (1, 4)
                or not isinstance(selection.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", selection["sha256"])):
            raise ValueError("Eval requires one explicit checkpoint, SHA256, epoch and NFE 1|4")
        checkpoint = linux_path(selection.get("checkpoint"), "checkpoint", allow_historical_input=True)
        if not checkpoint.is_relative_to(PurePosixPath(paths["checkpoint_root"])):
            raise ValueError("Eval checkpoint must be inside the independent checkpoint_root")
        output = PurePosixPath(paths["evaluation_output_root"])
        proposed = output / "code"
    argv = [paths["python"], "-m", "accelerate.commands.launch", "--multi_gpu",
            "--num_processes", "4", "--num_machines", "1", "--main_process_port",
            str(paths["main_process_ports"][version]), "--mixed_precision", "no",
            "--dynamo_backend", "no", "train_pair.py", version, phase]
    environment = {"CUDA_VISIBLE_DEVICES": ",".join(paths["visible_devices"]),
                   "OMP_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1", "NCCL_DEBUG": "WARN",
                   "NCCL_IB_DISABLE": "1", "WANDB_MODE": "offline", "PYTHONDONTWRITEBYTECODE": "1"}
    controller_argv = [paths["python"], str(output / "full_control.py"), "launch", version, phase]
    if phase == "eval":
        argv = [paths["python"], str(output / "music_corrected_eval_20260930.py")]
        controller_argv = None
        environment.update(CUDA_VISIBLE_DEVICES=paths["visible_devices"][0], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    return {
        "status": "BLOCKED_PLAN_ONLY", "phase": phase, "version": version,
        "plan_only": True, "launch_performed": False, "ready_to_execute": False,
        "model_imports": False, "gpu_queried": False, "network_used": False,
        "files_written": False, "paths_checked_on_target_host": False,
        "frozen_source": frozen,
        "contract": (dict(CONTRACT, genre_input=version == "genre") if phase != "eval" else {
            "seed": 42, "world_size": 1, "batch_size": 2, "coverage": 84, "frames": 1020,
            "motion_fps": 30, "native_motion_dim": 151, "music_feature_dim": 35,
            "nfe": selection["n"], "genre_input": version == "genre", "mixed_precision": "no"}),
        "argv": argv, "environment": environment,
        "controller_argv_preserving_guards": controller_argv,
        "prepared_root": str(output), "eval_selection": selection,
        "proposed_cwd_after_reviewed_adapter": str(proposed),
        "proposed_command_after_reviewed_adapter": "cd " + shlex.quote(str(proposed)) + " && "
            + shlex.join(["env", *(f"{key}={value}" for key, value in environment.items()), *argv]),
        "command_is_not_safe_to_execute_yet": True,
        "native_source_on_target_host": str(source / "train_pair.py") if phase != "eval" else
            str(PurePosixPath(paths["release_root"]) / SNAPSHOT_RELATIVE / "provenance/music_corrected_eval_20260930.py"),
        "path_adaptation_plan": [
            {"symbol": "S", "native": str(HISTORICAL_PROJECT), "proposed": str(data),
             "required_change": "--prepare writes an exclusive generated copy only; no native --data-root or environment override exists."},
            {"symbol": "R", "native": "Path(__file__).resolve().parent.parent",
             "proposed": str(output), "required_change": "Generate code only at new root/code; preserve derived R and never run frozen copy."},
            {"symbol": "out" if phase != "eval" else "OUT", "native": "R/version/phase" if phase != "eval" else "R/'results'",
             "proposed": str(output / version / phase) if phase != "eval" else str(output / "results"),
             "required_change": "New exclusive output namespace, launch/worker locks and duplicate-run refusal required."},
        ],
        "required_data_paths": [str(data / relative) for relative in (
            (["data/dataset_backups_stable_v2/train_tensor_dataset.pkl"] if phase != "eval" else []) +
            ["data/dataset_backups_stable_v2/test_tensor_dataset.pkl", "data/test/motions_sliced", "data/test/baseline_feats"])],
        "private_artifact_targets": [
            {"repo_id": "yangkaixing/FlashMotion-data", "repo_type": "dataset", "private": True, "prefix": "m2d/"},
            {"repo_id": "yangkaixing/FlashMotion-checkpoints", "repo_type": "model", "private": True, "prefix": "m2d/"},
        ],
        "quality_eval": {"status": "REPAIRED_SCORER_COLLECTED_EXECUTION_UNVERIFIED",
                         "release_source": SNAPSHOT_RELATIVE + "/provenance/music_corrected_eval_20260930.py",
                         "historical_source": str(HISTORICAL_RUN / "music_corrected_eval_20260930.py"),
                         "expected_sha256": CORRECTED_EVAL_SHA256,
                         "native_training_eval_is_paper_protocol": False,
                         "native_test_py_checkpoint_route_is_valid": False},
        "blockers": [
            "Training --execute refuses; bounded eval --execute requires independently supplied current runtime/resource/input proof and prepared-copy rehash.",
            "The frozen entry writes at module top; importing/running it would write to its derived R and use historical S.",
            "Verify authentic private assets with immutable HF revision, file SHA/size, and ordered dataset/normalizer provenance; never unpickle in preflight.",
            "Capture the actual native Mamba/causal-conv1d/PyTorch3D, Python/CUDA and numerical environment; reference requirements are not a tested lock.",
            "For training only: independently reproduce historical controller guards, identity/locks/no duplicate, four free slots (>60000 MiB each), successful verify exit and two metric JSONs. Independent single-checkpoint eval does not require these training proofs.",
            "The verify phase performs two real training updates, checkpoint save and old evaluation; it is not CPU verification.",
            "Preserve fresh initialization, no resume/pretrained/smoke reuse, 4x64 batch, 240 frames and no AMP; independently confirm 77 updates/epoch on exact data.",
            "Paper checkpoint selection is recorded separately using a no-PFC criterion. Evaluate the supplied checkpoint with the paper-protocol scorer; this entry does not rerank checkpoints. The native training scorer uses a different protocol.",
            "Applicable inherited skeleton/body/render asset licenses and written FineDance private-distribution scope must be retained; any actually used external body model needs its own verified license/assets.",
        ],
        "evidence_paths_not_validated": {key: paths[key] for key in (
            "asset_fetch_receipt", "runtime_evidence", "precheck_path")},
    }


def validate_eval_authorization(plan: dict[str, Any], paths: dict[str, Any],
                                actual_hostname: str | None = None) -> dict[str, Any]:
    """Validate independently supplied evidence without ML imports/GPU queries.

    Runtime/shape/license/device claims must be produced by the authorized
    operator, not by this planner. All concrete input bytes are rehashed here.
    """
    try:
        from . import adapter
    except ImportError:
        import adapter
    if plan["phase"] != "eval":
        raise ValueError("Only bounded independent eval can execute; training remains blocked")
    hostname = actual_hostname or socket.gethostname()
    if hostname != paths["expected_hostname"] or "REPLACE" in paths["task_id"].upper():
        raise ValueError("Independent execution hostname/task identity is not supplied/matched")
    proof_path = Path(paths["runtime_evidence"])
    adapter.no_links(proof_path)
    proof = read_json(proof_path)
    if type(proof.get("schema_version")) is not int:
        raise ValueError("Runtime proof schema_version must be integer 1")
    selection = plan["eval_selection"]
    required = {"schema_version": 1, "stage": "m2d_eval",
                "task_id": paths["task_id"], "expected_hostname": hostname,
                "python": paths["python"], "data_root": paths["data_root"],
                "checkpoint": selection["checkpoint"], "checkpoint_sha256": selection["sha256"],
                "visible_device": paths["visible_devices"][0]}
    if any(proof.get(key) != value for key, value in required.items()):
        raise ValueError("Independent eval runtime/resource proof does not match this request")
    if proof.get("status") not in ("PASS", "CPU_INPUT_AUDIT_PASS_RESOURCE_PENDING"):
        raise ValueError("CPU/input audit has not passed; resource checks cannot authorize it")
    timestamp = proof.get("validated_at_utc")
    if not isinstance(timestamp, str):
        raise ValueError("Runtime proof needs an explicit UTC validation timestamp")
    checked_at = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    if checked_at.tzinfo is None:
        raise ValueError("Runtime proof timestamp must include UTC timezone")
    age = (datetime.now(timezone.utc) - checked_at).total_seconds()
    if not -300 <= age <= 3600:
        raise ValueError("Runtime/resource proof is stale or from the future")
    checks = proof.get("checks", {})
    if not isinstance(checks, dict):
        raise ValueError("Runtime proof checks must be an object")
    for key in ("native_runtime_inventory", "finedance_written_private_permission", "inherited_skeleton_license",
                "ordered_test84", "feature35_frame1020", "checkpoint_normalizer"):
        if checks.get(key) is not True:
            raise ValueError(f"Independent eval proof missing check: {key}")
    versions = proof.get("versions", {})
    if not isinstance(versions, dict):
        raise ValueError("Runtime proof versions must be an object")
    for key in ("python", "torch", "numpy", "scipy", "mamba_ssm", "causal_conv1d", "pytorch3d"):
        if not isinstance(versions.get(key), str) or not versions[key].strip():
            raise ValueError(f"Independent native runtime version missing: {key}")
    data = Path(paths["data_root"])
    motions = data / "data/test/motions_sliced"
    music = data / "data/test/baseline_feats"
    adapter.no_links(motions)
    adapter.no_links(music)
    motion_files = sorted(motions.glob("*.pkl"))
    music_files = sorted(music.glob("*.npy"))
    if len(motion_files) != 84 or len(music_files) != 84:
        raise ValueError("Independent eval requires exactly 84 motions and 84 feature files")
    names = [path.stem for path in motion_files]
    if names != [path.stem for path in music_files] or len(set(names)) != 84:
        raise ValueError("Test motion/music file names do not match exactly")
    required_files = {path.as_posix() for path in motion_files + music_files} | {
        (data / "data/dataset_backups_stable_v2/test_tensor_dataset.pkl").as_posix(),
        selection["checkpoint"],
    }
    records = proof.get("files")
    if not isinstance(records, list):
        raise ValueError("Runtime proof requires all concrete input file SHA/size records")
    by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("Invalid input proof record")
        relative = record["path"]
        if relative in by_path or relative not in required_files:
            raise ValueError("Duplicate/unexpected input proof path")
        if type(record.get("bytes")) is not int or record["bytes"] < 0:
            raise ValueError("Input proof must include a nonnegative byte size")
        by_path[relative] = record
    if set(by_path) != required_files:
        raise ValueError("Input proof must cover checkpoint, test cache, 84 motions and 84 features (170 files)")
    for filename, record in by_path.items():
        target = Path(filename)
        adapter.checkpoint_hash(target, record.get("sha256", ""))
        if target.stat().st_size != record["bytes"]:
            raise ValueError("Input proof byte size mismatch")
    if by_path[selection["checkpoint"]]["sha256"] != selection["sha256"]:
        raise ValueError("Proof checkpoint identity differs from explicit selection")
    python = Path(paths["python"])
    if not python.is_file() or proof.get("python_realpath") != python.resolve().as_posix():
        raise ValueError("Configured runtime executable is missing")
    # A venv interpreter is commonly an intentional symlink. Pin its resolved
    # executable separately; this exception never applies to assets/outputs.
    adapter.checkpoint_hash(python.resolve(), proof.get("python_sha256", ""))
    extensions = proof.get("native_extensions")
    if not isinstance(extensions, list) or len(extensions) != 3 or {item.get("module") for item in extensions if isinstance(item, dict)} != {"selective_scan_cuda", "causal_conv1d_cuda", "pytorch3d._C"}:
        raise ValueError("Native extension inventory must pin selective-scan, causal-conv1d and PyTorch3D .so bytes")
    for item in extensions:
        target = Path(item["path"])
        adapter.checkpoint_hash(target, item.get("sha256", ""))
        if target.stat().st_size != item.get("bytes"):
            raise ValueError("Native extension inventory size mismatch")
    adapter.validate_prepared_bundle(REPO_ROOT, Path(plan["prepared_root"]), paths, selection)
    resource = validate_gpu_resource(proof, paths)
    return {"hostname": hostname, "task_id": paths["task_id"], "input_files_sha_verified": len(by_path),
            "runtime_evidence_sha256": hashlib.sha256(proof_path.read_bytes()).hexdigest(),
            "native_runtime_inventory_checked": True, "native_forward_status_before_launch": "NOT_RUN",
            "native_model_imports_by_preflight": False, "resource": resource}


def process_identity(pid: int) -> dict[str, Any]:
    root = Path("/proc") / str(pid)
    tail = (root / "stat").read_text().rsplit(")", 1)[1].split()
    return {"pid": pid, "startticks": int(tail[19]), "ppid": int(tail[1]),
            "state": tail[0], "cwd": os.readlink(root / "cwd")}


def gpu_snapshot(device: str) -> dict[str, Any]:
    rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.total,memory.free",
                                    "--format=csv,noheader,nounits"], text=True)
    selected = None
    for line in rows.strip().splitlines():
        index, uuid, total, free = [field.strip() for field in line.split(",")]
        if index == device:
            selected = {"index": int(index), "uuid": uuid, "total_mib": int(total), "free_mib": int(free)}
    if selected is None:
        raise ValueError("Configured physical GPU was not found")
    applications = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory",
                                            "--format=csv,noheader,nounits"], text=True)
    owners = []
    for line in applications.strip().splitlines():
        if not line.strip():
            continue
        uuid, pid, used = [field.strip() for field in line.split(",")]
        if uuid == selected["uuid"]:
            owners.append({"identity": process_identity(int(pid)), "used_mib": int(used)})
    selected["owners"] = sorted(owners, key=lambda item: item["identity"]["pid"])
    return selected


def verify_inherited_shared_flock(expected_path: str) -> int:
    """Require the original lock's live open FD, held by this process/ancestor.

    This never creates/acquires a lock itself; an unavailable queue waits at
    the authorized supervisor. The open descriptor is inherited by the child.
    """
    original = "/tmp/t2m_loss_20260924_bubuyang-dmd-0924-102909-01-1x8-worker-0_gpu_7.lock"
    if expected_path != original:
        raise ValueError("Shared policy must use the exact original GPU7 flock")
    text_fd = os.environ.get("FLASHMOTION_SHARED_GPU_LOCK_FD", "")
    if not re.fullmatch(r"[0-9]+", text_fd) or int(text_fd) < 3:
        raise ValueError("Supervisor must pass its held flock FD via FLASHMOTION_SHARED_GPU_LOCK_FD")
    descriptor = int(text_fd)
    file_info = os.stat(expected_path, follow_symlinks=False)
    fd_info = os.fstat(descriptor)
    if not stat.S_ISREG(file_info.st_mode) or (file_info.st_dev, file_info.st_ino) != (fd_info.st_dev, fd_info.st_ino):
        raise ValueError("Inherited descriptor does not name the original lock inode")
    ancestors = set()
    pid = os.getpid()
    while pid > 0 and pid not in ancestors:
        ancestors.add(pid)
        identity = process_identity(pid)
        if identity["state"] == "Z":
            raise ValueError("Dead process in lock owner ancestry")
        pid = identity["ppid"]
    held = False
    for line in Path("/proc/locks").read_text().splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1] != "FLOCK" or fields[3] != "WRITE":
            continue
        device_inode = fields[5].split(":")
        if len(device_inode) != 3:
            continue
        if (int(device_inode[0], 16), int(device_inode[1], 16), int(device_inode[2])) == (
                os.major(file_info.st_dev), os.minor(file_info.st_dev), file_info.st_ino) and int(fields[4]) in ancestors:
            held = True
    if not held:
        raise ValueError("Original GPU7 FLOCK WRITE is not held by this process/live supervisor ancestry")
    return descriptor


def validate_gpu_resource(proof: dict[str, Any], paths: dict[str, Any]) -> dict[str, Any]:
    policy = proof.get("resource_policy", "dedicated_idle")
    current = gpu_snapshot(paths["visible_devices"][0])
    if current["free_mib"] < 30 * 1024:
        raise ValueError("Native eval requires >=30 GiB current GPU free memory")
    if policy == "dedicated_idle":
        if current["owners"]:
            raise ValueError("Dedicated policy requires a physically idle device, not merely free memory")
        return dict(current, policy=policy, gpu_idle_and_free30gib=True, pass_fds=[])
    if policy != "shared_gpu7_reserved_headroom" or paths["visible_devices"][0] != "7":
        raise ValueError("Unsupported resource policy/device")
    expected = proof.get("shared_owner_identities")
    if not isinstance(expected, list) or len(expected) != 2:
        raise ValueError("Shared GPU7 policy requires two exact known training owner identities")
    if sorted(expected, key=lambda item: item["pid"]) != SHARED_OWNER_IDENTITIES:
        raise ValueError("Shared owner proof differs from the explicitly approved DMD01 identities")
    actual = [{key: item["identity"][key] for key in ("pid", "startticks", "cwd")} for item in current["owners"]]
    if sorted(actual, key=lambda item: item["pid"]) != sorted(expected, key=lambda item: item["pid"]):
        raise ValueError("GPU7 has changed training owners or another inference; refuse rather than interfere")
    try:
        from . import adapter
    except ImportError:
        import adapter
    caps = proof.get("shared_training_cap_sources")
    if not isinstance(caps, list) or len(caps) != 2:
        raise ValueError("Two unchanged .40 training-cap source proofs are required")
    for item in caps:
        if item.get("caps") != [.40] or SHARED_CAP_SOURCES.get(item.get("path")) != item.get("sha256"):
            raise ValueError("Shared training cap is not the audited .40")
        adapter.checkpoint_hash(Path(item["path"]), item.get("sha256", ""))
    if {item["path"] for item in caps} != set(SHARED_CAP_SOURCES):
        raise ValueError("Both approved distinct training cap sources must be present")
    fraction = (6 * 1024**3) / (current["total_mib"] * 1024**2)
    if fraction >= .08 or .20 - fraction <= .12:
        raise ValueError("Capped6GiB eval cannot fit the approved .20 residual-cap headroom/margin")
    descriptor = verify_inherited_shared_flock(proof.get("shared_gpu_flock_path", ""))
    return dict(current, policy=policy, gpu_idle_and_free30gib=False, shared_gpu7_reserved_headroom=True,
                eval_cap_bytes=6 * 1024**3, eval_cap_fraction=fraction,
                residual_cap_headroom_fraction=.20 - fraction, pass_fds=[descriptor])


def execute_eval_plan(plan: dict[str, Any], paths: dict[str, Any]) -> int:
    if os.name != "posix":
        raise ValueError("Bounded native eval execution requires the configured independent Linux host")
    evidence = validate_eval_authorization(plan, paths)
    root = Path(plan["prepared_root"])
    # Refuse a second attempt/output reuse; a failed run needs a new namespace.
    target = root / "execution_result.json"
    with target.open("x", encoding="utf-8") as handle:
        receipt = {"status": "STARTING_NOT_VERIFIED", "launch_performed": False,
                   "selection": plan["eval_selection"], "evidence": evidence}
        json.dump(receipt, handle, indent=2, allow_nan=False)
    environment = os.environ.copy()
    environment.update(plan["environment"])
    result_code = 2
    try:
        result = subprocess.run(plan["argv"], cwd=root / "code", env=environment, check=False,
                                pass_fds=tuple(evidence["resource"]["pass_fds"]))
        receipt["launch_performed"] = True
        result_code = result.returncode
        receipt["exit_code"] = result_code
        receipt["status"] = "FAILED_NOT_VERIFIED"
        if result_code == 0:
            selection = plan["eval_selection"]
            metric_path = root / "results/metrics" / f"{selection['version']}_e{selection['epoch']}_{selection['n']}step.json"
            row = read_json(metric_path)
            for key, expected in (("coverage", 84), ("frames", 1020), ("epoch", selection["epoch"]),
                                  ("n", selection["n"]), ("version", selection["version"]),
                                  ("checkpoint", selection["checkpoint"]), ("protocol", "corrected_eval_20260930")):
                if row.get(key) != expected:
                    raise ValueError("Native result does not match the bounded eval contract")
            for key in ("FID_k", "FID_g", "Diversity_k", "Diversity_g", "BeatAlignment", "PFC"):
                if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]):
                    raise ValueError("Native metric result is missing/non-finite")
            receipt.update(status="BOUNDED_NATIVE_EVAL_COMPLETED", metric_path=metric_path.as_posix(),
                           metric_sha256=hashlib.sha256(metric_path.read_bytes()).hexdigest())
        # The wrapper still owns its inherited FD throughout launch and these
        # postchecks. It never unlocks/closes the supervisor's descriptor.
        proof_path = Path(paths["runtime_evidence"])
        if hashlib.sha256(proof_path.read_bytes()).hexdigest() != evidence["runtime_evidence_sha256"]:
            raise ValueError("Runtime proof changed during the bounded attempt")
        proof = read_json(proof_path)
        receipt["post_resource_check"] = validate_gpu_resource(proof, paths)
    except (OSError, ValueError, KeyError, TypeError) as error:
        receipt.update(status="FAILED_NOT_VERIFIED", error_type=type(error).__name__)
        result_code = 2
    # This is the new attempt's own receipt, not frozen/historical metadata.
    with target.open("w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    print(json.dumps(receipt, indent=2, allow_nan=False))
    return result_code


def self_test() -> int:
    """Stdlib static tests; adapter tests write only fresh temporary copies."""
    import copy
    import contextlib
    import io
    import tempfile
    import unittest
    from unittest import mock
    try:
        from . import adapter
    except ImportError:
        import adapter

    class PlannerTests(unittest.TestCase):
        def setUp(self) -> None:
            self.paths = read_json(REPO_ROOT / "configs/m2d.paths.example.json")

        def test_dry_run_contract(self) -> None:
            plan = build_plan("train", "genre", self.paths)
            self.assertFalse(plan["launch_performed"])
            self.assertFalse(plan["files_written"])
            self.assertFalse(plan["model_imports"])
            self.assertFalse(plan["ready_to_execute"])
            self.assertEqual(plan["argv"][-3:], ["train_pair.py", "genre", "train"])
            self.assertEqual(plan["contract"]["native_motion_dim"], 151)
            self.assertEqual(plan["contract"]["archived_total_updates"], 308000)
            self.assertEqual(plan["argv"][plan["argv"].index("--mixed_precision") + 1], "no")
            self.assertEqual(plan["frozen_source"]["formal_code_files_verified"], 30)

        def test_verify_no_genre_native_args(self) -> None:
            plan = build_plan("verify", "no_genre", self.paths)
            self.assertEqual(plan["argv"][-2:], ["no_genre", "verify"])
            self.assertFalse(plan["contract"]["genre_input"])
            self.assertFalse(plan["quality_eval"]["native_test_py_checkpoint_route_is_valid"])

        def test_historical_and_broad_paths_rejected(self) -> None:
            for path in (str(HISTORICAL_PROJECT), str(HISTORICAL_RUN / "new"), "/efs", "/", "/work/../efs", "/work//new", "C:/work"):
                with self.subTest(path=path):
                    paths = copy.deepcopy(self.paths)
                    paths["output_root"] = path
                    with self.assertRaises(ValueError):
                        validate_paths(paths)

        def test_source_and_input_output_overlap_rejected(self) -> None:
            for key, value in (("output_root", self.paths["release_root"] + "/new"),
                               ("evaluation_output_root", self.paths["output_root"] + "/eval"),
                               ("data_root", self.paths["output_root"] + "/data"),
                               ("generated_code_root", "/work/other/code"),
                               ("precheck_path", self.paths["output_root"] + "/proof.json")):
                with self.subTest(key=key):
                    paths = copy.deepcopy(self.paths)
                    paths[key] = value
                    with self.assertRaises(ValueError):
                        validate_paths(paths)

        def test_four_unique_devices_and_ports(self) -> None:
            for devices in (["0", "1"], ["0", "1", "2", "2"], [0, 1, 2, 3], ["0", "1", "2", "3;echo"]):
                with self.subTest(devices=devices):
                    paths = copy.deepcopy(self.paths)
                    paths["visible_devices"] = devices
                    with self.assertRaises(ValueError):
                        validate_paths(paths)
            paths = copy.deepcopy(self.paths)
            paths["main_process_ports"]["genre"] = paths["main_process_ports"]["no_genre"]
            with self.assertRaises(ValueError):
                validate_paths(paths)

        def test_unsafe_source_paths_and_duplicate_json(self) -> None:
            for path in ("../train_pair.py", "/tmp/train_pair.py", "a\\b.py", "a/../b.py", "a//b.py"):
                with self.subTest(path=path), self.assertRaises(ValueError):
                    safe_source_file(REPO_ROOT, path)
            with self.assertRaises(ValueError):
                json.loads('{"x":1,"x":2}', object_pairs_hook=_unique_object)

        def test_native_cli_only(self) -> None:
            with self.assertRaises(ValueError):
                build_plan("eval", "genre", self.paths)
            with self.assertRaises(ValueError):
                build_plan("train", "other", self.paths)

        def test_execute_refuses_without_launch(self) -> None:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["train", "--execute"]), 2)
            self.assertNotIn("torch", sys.modules)
            self.assertNotIn("numpy", sys.modules)

    class AdapterTests(unittest.TestCase):
        def setUp(self) -> None:
            self.paths = read_json(REPO_ROOT / "configs/m2d.paths.example.json")
            self.paths["expected_hostname"] = "independent-test-host"

        def source(self, name: str) -> bytes:
            return (REPO_ROOT / SNAPSHOT_RELATIVE / name).read_bytes()

        def test_training_only_S_changed(self) -> None:
            native = self.source("code/train_pair.py")
            generated, patches = adapter.adapt_train(native, "/datasets/test-m2d")
            self.assertEqual(len(patches), 1)
            self.assertEqual(generated, native.replace(b"'/efs/yangkaixing/FlowerDance_x0_ablation'", b"'/datasets/test-m2d'", 1))
            with self.assertRaises(ValueError):
                adapter.adapt_train(native + b"\n", "/datasets/test-m2d")

        def test_controller_guards_preserved(self) -> None:
            native = self.source("provenance/full_control.py")
            generated, patches = adapter.adapt_control(native, "/work/new-test", self.paths)
            self.assertEqual(len(patches), 6)
            before = [ast.dump(node) for node in ast.walk(ast.parse(native)) if isinstance(node, ast.Assert)]
            after = [ast.dump(node) for node in ast.walk(ast.parse(generated)) if isinstance(node, ast.Assert)]
            self.assertEqual(before[1:], after[1:])  # Only explicit host literal differs.
            self.assertIn(b"fcntl.LOCK_EX|fcntl.LOCK_NB", generated)
            self.assertIn(b"--mixed_precision", generated)
            self.assertIn(b"--dynamo_backend", generated)

        def test_eval_metric_functions_identical_no_scan_ranking(self) -> None:
            native = self.source("provenance/music_corrected_eval_20260930.py")
            selection = {"version": "genre", "epoch": 2250, "n": 1,
                         "checkpoint": "/models/test/train-2250.pt", "sha256": "a" * 64}
            generated, _ = adapter.adapt_eval(native, "/work/new-eval", "/datasets/test-m2d", selection)
            original_tree, generated_tree = ast.parse(native), ast.parse(generated)
            for name in ("kinetic", "geo_chunk", "extract", "fid", "diversity", "pfc"):
                before = next(node for node in original_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
                after = next(node for node in generated_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
                self.assertEqual(ast.dump(before), ast.dump(after), name)
            ranking = [node for node in ast.walk(generated_tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "rank_all"]
            self.assertEqual(ranking, [])
            self.assertNotIn(b"glob('train-*.pt')", generated)
            self.assertIn(b"free>=30*1024**3", generated)
            self.assertIn(b"fcntl.LOCK_EX|fcntl.LOCK_NB", generated)

        def test_exclusive_temp_copy_sha_receipt_and_source_unchanged(self) -> None:
            before = verify_frozen_source(REPO_ROOT)
            with tempfile.TemporaryDirectory(prefix="flashmotion-m2d-adapter-test-") as temporary:
                destination = Path(temporary) / "independent-run"
                receipt = adapter.prepare_bundle(REPO_ROOT, destination, self.paths)
                self.assertEqual(receipt["status"], "PREPARED_NOT_EXECUTED")
                self.assertFalse(receipt["launch_performed"])
                self.assertFalse(receipt["ready_to_execute"])
                self.assertEqual(len(receipt["files"]), 31)
                for item in receipt["files"]:
                    self.assertEqual(adapter.sha((destination / item["path"]).read_bytes()), item["generated_sha256"])
                with self.assertRaises(ValueError):
                    adapter.prepare_bundle(REPO_ROOT, destination, self.paths)
            self.assertEqual(before, verify_frozen_source(REPO_ROOT))

        def test_eval_sha_before_writes(self) -> None:
            with tempfile.TemporaryDirectory(prefix="flashmotion-m2d-eval-test-") as temporary:
                checkpoint = Path(temporary) / "fixture.pt"
                checkpoint.write_bytes(b"synthetic bytes, never deserialized")
                destination = Path(temporary) / "eval"
                selection = {"version": "genre", "epoch": 2250, "n": 1,
                             "checkpoint": str(checkpoint), "sha256": "a" * 64}
                with self.assertRaises(ValueError):
                    adapter.prepare_bundle(REPO_ROOT, destination, self.paths, selection)
                self.assertFalse(destination.exists())
                selection["sha256"] = adapter.sha(checkpoint.read_bytes())
                receipt = adapter.prepare_bundle(REPO_ROOT, destination, self.paths, selection)
                self.assertEqual(receipt["mode"], "one_checkpoint_eval")
                self.assertEqual(receipt["selection"], selection)

        def test_checkout_and_symlink_target_rejected(self) -> None:
            with self.assertRaises(ValueError):
                adapter.prepare_bundle(REPO_ROOT, REPO_ROOT / "must-not-create", self.paths)
            with tempfile.TemporaryDirectory(prefix="flashmotion-m2d-link-test-") as temporary:
                parent = Path(temporary)
                (parent / "real").mkdir()
                try:
                    (parent / "link").symlink_to(parent / "real", target_is_directory=True)
                except OSError:
                    self.skipTest("OS does not permit symlink fixture creation")
                with self.assertRaises(ValueError):
                    adapter.prepare_bundle(REPO_ROOT, parent / "link" / "unsafe", self.paths)

        def authorization_fixture(self, parent: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
            paths = copy.deepcopy(self.paths)
            paths.update(task_id="synthetic-test-only", python=Path(sys.executable).as_posix(),
                         data_root=(parent / "data").as_posix(), runtime_evidence=(parent / "runtime.json").as_posix())
            files = []
            for directory, suffix in (("motions_sliced", ".pkl"), ("baseline_feats", ".npy")):
                folder = parent / "data/data/test" / directory
                folder.mkdir(parents=True)
                for index in range(84):
                    path = folder / f"fixture_{index:03}{suffix}"
                    path.write_bytes(b"synthetic, not a dataset")
                    files.append(path)
            cache = parent / "data/data/dataset_backups_stable_v2/test_tensor_dataset.pkl"
            cache.parent.mkdir(parents=True)
            cache.write_bytes(b"synthetic, never unpickled")
            checkpoint = parent / "fixture.pt"
            checkpoint.write_bytes(b"synthetic, never torch-loaded")
            files += [cache, checkpoint]
            selection = {"version": "genre", "epoch": 2250, "n": 1, "checkpoint": checkpoint.as_posix(),
                         "sha256": adapter.sha(checkpoint.read_bytes())}
            destination = parent / "fresh-eval"
            adapter.prepare_bundle(REPO_ROOT, destination, paths, selection)
            plan = {"phase": "eval", "eval_selection": selection, "prepared_root": destination.as_posix()}
            proof = {"schema_version": 1, "stage": "m2d_eval", "status": "PASS", "task_id": paths["task_id"],
                     "expected_hostname": paths["expected_hostname"], "python": paths["python"],
                     "python_realpath": Path(sys.executable).resolve().as_posix(),
                     "python_sha256": adapter.sha(Path(sys.executable).resolve().read_bytes()),
                     "data_root": paths["data_root"], "checkpoint": selection["checkpoint"],
                     "checkpoint_sha256": selection["sha256"], "visible_device": paths["visible_devices"][0],
                     "validated_at_utc": datetime.now(timezone.utc).isoformat(),
                     "checks": {key: True for key in ("native_runtime_inventory", "finedance_written_private_permission",
                         "inherited_skeleton_license", "ordered_test84", "feature35_frame1020",
                         "checkpoint_normalizer", "gpu_idle_and_free30gib")},
                     "versions": {key: "synthetic-test-only" for key in (
                         "python", "torch", "numpy", "scipy", "mamba_ssm", "causal_conv1d", "pytorch3d")},
                     "files": [{"path": path.as_posix(), "sha256": adapter.sha(path.read_bytes()),
                                "bytes": path.stat().st_size} for path in files]}
            proof["native_extensions"] = []
            for index, name in enumerate(("selective_scan_cuda", "causal_conv1d_cuda", "pytorch3d._C")):
                extension = parent / f"synthetic-extension-{index}.so"
                extension.write_bytes(b"synthetic native inventory, never imported")
                proof["native_extensions"].append({"module": name, "path": extension.as_posix(),
                    "sha256": adapter.sha(extension.read_bytes()), "bytes": extension.stat().st_size})
            Path(paths["runtime_evidence"]).write_text(json.dumps(proof), encoding="utf-8")
            return plan, paths, proof

        def validate_fixture(self, plan: dict[str, Any], paths: dict[str, Any], hostname: str) -> dict[str, Any]:
            snapshot = {"index": 0, "uuid": "synthetic-test-only", "total_mib": 143771,
                        "free_mib": 47000, "owners": []}
            with mock.patch.object(sys.modules[__name__], "gpu_snapshot", return_value=snapshot):
                return validate_eval_authorization(plan, paths, actual_hostname=hostname)

        def test_current_host_eval_proof_and_all_input_hashes(self) -> None:
            with tempfile.TemporaryDirectory(prefix="flashmotion-m2d-proof-test-") as temporary:
                plan, paths, _ = self.authorization_fixture(Path(temporary))
                verified = self.validate_fixture(plan, paths, paths["expected_hostname"])
                self.assertEqual(verified["input_files_sha_verified"], 170)
                with self.assertRaises(ValueError):
                    self.validate_fixture(plan, paths, "wrong-host")
                checkpoint = Path(plan["eval_selection"]["checkpoint"])
                checkpoint.write_bytes(b"changed synthetic checkpoint")
                with self.assertRaises(ValueError):
                    self.validate_fixture(plan, paths, paths["expected_hostname"])

        def test_unchecked_runtime_and_changed_generated_copy_refused(self) -> None:
            with tempfile.TemporaryDirectory(prefix="flashmotion-m2d-proof-test-") as temporary:
                plan, paths, proof = self.authorization_fixture(Path(temporary))
                proof["checks"]["native_runtime_inventory"] = False
                Path(paths["runtime_evidence"]).write_text(json.dumps(proof), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.validate_fixture(plan, paths, paths["expected_hostname"])
                proof["checks"]["native_runtime_inventory"] = True
                Path(paths["runtime_evidence"]).write_text(json.dumps(proof), encoding="utf-8")
                generated = Path(plan["prepared_root"]) / "code/model/model.py"
                generated.write_bytes(generated.read_bytes() + b"\n# synthetic tamper\n")
                with self.assertRaises(ValueError):
                    self.validate_fixture(plan, paths, paths["expected_hostname"])

        def test_shared_policy_truthful_idle_false_and_known_owner_guard(self) -> None:
            with tempfile.TemporaryDirectory(prefix="flashmotion-m2d-shared-test-") as temporary:
                parent = Path(temporary)
                caps = []
                for index in range(2):
                    source = parent / f"synthetic-cap-{index}.py"
                    source.write_bytes(b"torch.cuda.set_per_process_memory_fraction(0.40)\n")
                    caps.append({"path": str(source), "sha256": adapter.sha(source.read_bytes()), "caps": [.40]})
                owners = [{"pid": 101, "startticks": 1001, "cwd": "/synthetic/train1"},
                          {"pid": 102, "startticks": 1002, "cwd": "/synthetic/train2"}]
                proof = {"resource_policy": "shared_gpu7_reserved_headroom", "shared_owner_identities": owners,
                         "shared_training_cap_sources": caps, "shared_gpu_flock_path": "/synthetic/lock"}
                paths = {"visible_devices": ["7", "0", "1", "2"]}
                current = {"index": 7, "total_mib": 143771, "free_mib": 47000,
                           "owners": [{"identity": owner} for owner in owners]}
                with mock.patch.object(sys.modules[__name__], "gpu_snapshot", return_value=current), mock.patch.object(
                        sys.modules[__name__], "verify_inherited_shared_flock", return_value=3), mock.patch.object(
                        sys.modules[__name__], "SHARED_OWNER_IDENTITIES", owners), mock.patch.object(
                        sys.modules[__name__], "SHARED_CAP_SOURCES", {item["path"]: item["sha256"] for item in caps}):
                    verified = validate_gpu_resource(proof, paths)
                    self.assertFalse(verified["gpu_idle_and_free30gib"])
                    self.assertTrue(verified["shared_gpu7_reserved_headroom"])
                    self.assertEqual(verified["pass_fds"], [3])
                    current["owners"].append({"identity": {"pid": 103, "startticks": 1003, "cwd": "/synthetic/other-eval"}})
                    with self.assertRaises(ValueError):
                        validate_gpu_resource(proof, paths)

    suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(test_case)
                               for test_case in (PlannerTests, AdapterTests)])
    return 0 if unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful() else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("phase", nargs="?", choices=("verify", "train", "eval"), default="train")
    result.add_argument("--version", choices=("genre", "no_genre"), default="genre")
    result.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/m2d.paths.example.json")
    mode = result.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Explicit plan-only mode (also the default)")
    mode.add_argument("--prepare", action="store_true", help="Linux only: create a fresh exclusive adapted copy and receipt; never launch")
    mode.add_argument("--execute", action="store_true", help="Eval only: validate fresh current-host input/runtime proof then launch bounded prepared scorer; training refused")
    result.add_argument("--checkpoint", help="Eval only: explicit absolute checkpoint under checkpoint_root")
    result.add_argument("--checkpoint-sha256", help="Eval only: verified checkpoint binary SHA256")
    result.add_argument("--epoch", type=int, help="Eval only: explicit checkpoint epoch")
    result.add_argument("--nfe", type=int, choices=(1, 4), help="Eval only: one native NFE")
    result.add_argument("--self-test", action="store_true", help="Run stdlib static tests; adapter fixtures only in fresh tempfile directories")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.self_test:
        if args.execute or args.prepare:
            arguments.error("--self-test cannot be combined with --execute/--prepare")
        return self_test()
    try:
        paths = read_json(args.paths)
        selection = {"checkpoint": args.checkpoint, "sha256": args.checkpoint_sha256,
                     "epoch": args.epoch, "n": args.nfe, "version": args.version} if args.phase == "eval" else None
        if args.phase != "eval" and any(value is not None for value in (args.checkpoint, args.checkpoint_sha256, args.epoch, args.nfe)):
            raise ValueError("Checkpoint/epoch/NFE options apply to the separate eval stage only")
        plan = build_plan(args.phase, args.version, paths, selection=selection)
        if args.prepare:
            if os.name != "posix":
                raise ValueError("--prepare requires the configured independent Linux host; planning/self-tests work on Windows")
            try:
                from .adapter import prepare_bundle
            except ImportError:
                from adapter import prepare_bundle
            receipt = prepare_bundle(REPO_ROOT, Path(plan["prepared_root"]), paths, selection)
            plan.update(status=receipt["status"], plan_only=False, files_written=True,
                        adaptation_receipt=str(Path(plan["prepared_root"]) / "adaptation_receipt.json"))
    except (OSError, ValueError, KeyError, TypeError, SyntaxError) as error:
        arguments.error(str(error))
    plan["execute_requested"] = args.execute
    print(json.dumps(plan, indent=2, allow_nan=False))
    if args.execute:
        if args.phase != "eval":
            print("M2D training refused: independent training controller/runtime proofs are not validated.", file=sys.stderr)
            return 2
        try:
            return execute_eval_plan(plan, paths)
        except (OSError, ValueError, KeyError, TypeError) as error:
            arguments.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
