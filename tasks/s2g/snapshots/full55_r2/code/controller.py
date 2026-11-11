"""One-host, one-shot scheduler; no queue, GPU takeover or automatic retry.

Run only on the newly authorized exact task hostname. --preflight performs
real eight-GPU BS probes; --launch serializes full55/body22 x three tested BS.
Both require a complete real PRECHECK gate. An interrupted run requires the
operator's explicit --resume; ordinary launch refuses existing partial runs.
"""
from __future__ import annotations

import argparse
try:
    import fcntl
except ImportError:
    fcntl = None  # --help/import is safe on Windows; actual launch is Linux-only.
import json
import math
import os
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

from data import atomic_json, sha256
from train import SOURCE_FILES, config_hash, load_config, require_precheck, require_bs_preflight, completed_bundle, EPOCHS, SAVE_EVERY


def start_ticks(pid):
    value = (Path("/proc")/str(pid)/"stat").read_text()
    # Fields after final ')' start at process-stat field 3. starttime is 22.
    return int(value[value.rfind(")")+2:].split()[19])


def validate_host(config):
    if not sys.platform.startswith("linux"):
        raise RuntimeError("Native Linux CUDA task only; this controller never starts on Windows")
    expected = config["task_name"]+"-worker-0"
    if config["expected_hostname"] != expected or socket.gethostname() != expected:
        raise RuntimeError("Refuse launch outside the sole newly authorized task hostname")
    if config["task_name"] != "bubuyang-motion3b-1007-134026-1x8":
        raise RuntimeError("This authorization does not cover another task")
    for key in ("JOB_NAME", "TENC_APP", "TAIJI_TASK_NAME"):
        if os.environ.get(key) and os.environ[key] != config["task_name"]:
            raise RuntimeError(f"Platform task identity disagrees through {key}")


def gpu_inventory_idle():
    csv = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                                   "--format=csv,noheader,nounits"], text=True)
    rows = [row.strip().split(",") for row in csv.splitlines() if row.strip()]
    if len(rows) != 8 or {int(row[0]) for row in rows} != set(range(8)):
        raise RuntimeError("Exactly eight physical GPUs are required")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                                         "--format=csv,noheader,nounits"], text=True).strip()
    if processes:
        raise RuntimeError("GPU compute processes already exist; refuse sharing, takeover or duplicate launch")
    for row in rows:
        if int(row[4].strip()) > 512 or int(row[5].strip()) > 10:
            raise RuntimeError("GPUs are not idle despite an empty compute-process list")
    return {"kind": "actual_nvidia_smi", "checked_at_unix": time.time(), "gpus_csv": csv,
            "compute_processes": []}


def run_child(command, output, state):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as log:
        child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        state.update(child_pid=child.pid, child_start_ticks=start_ticks(child.pid),
                     command=shlex.join(command), log=str(output), status="running")
        atomic_json(Path(state["output_root"])/"controller_state.json", state)
        code = child.wait()
    state.update(exit_code=code, finished_at_unix=time.time(), status="complete" if code == 0 else "failed")
    atomic_json(Path(state["output_root"])/"controller_state.json", state)
    if code:
        raise RuntimeError(f"Child failed with exit {code}; no automatic retry and no later runs started")


def validate_completed_run(run, config, mode, batch_size):
    """A completion marker alone cannot bypass retained-artifact checks."""
    run = Path(run)
    result = json.loads((run/"TRAINING_COMPLETE.json").read_text())
    metadata = json.loads((run/"run.json").read_text())
    if (result.get("status") != "complete" or result.get("config_sha256") != config_hash(config) or
            result.get("mode") != mode or result.get("batch_per_gpu") != batch_size or
            result.get("epochs_completed") != EPOCHS or result.get("historical_checkpoint_count") != EPOCHS//SAVE_EVERY or
            result.get("selection_uses_test") is not False or metadata.get("config_sha256") != config_hash(config) or
            metadata.get("mode") != mode or metadata.get("batch_per_gpu") != batch_size or
            metadata.get("world_size") != 8 or metadata.get("seed") != 42 or metadata.get("epochs") != EPOCHS):
        raise RuntimeError("Existing run completion/metadata is incompatible")
    if (metadata.get("manifest_sha256") != sha256(config["manifest"]) or
            metadata.get("normalizer_sha256") != sha256(Path(config["cache_root"])/f"stats_{mode}.npz") or
            metadata.get("source_hashes") != {name: sha256(Path(__file__).parent/name) for name in SOURCE_FILES}):
        raise RuntimeError("Completed run data/statistics/source changed")
    candidates = []
    for epoch in range(SAVE_EVERY, EPOCHS+1, SAVE_EVERY):
        checkpoint = run/"checkpoints"/f"epoch_{epoch:05d}.pt"
        bundle = completed_bundle(run/"validation"/f"epoch_{epoch:05d}"/"complete.json", config, checkpoint)
        if (not bundle or bundle.get("split") != "val" or bundle.get("mode") != mode or
                bundle.get("epoch") != epoch or not math.isfinite(bundle.get("selection_fgd_4step", float("nan")))):
            raise RuntimeError("A required historical val/render/prediction bundle is incomplete")
        candidates.append((bundle["selection_fgd_4step"], epoch, checkpoint))
    best_fgd, best_epoch, best_checkpoint = min(candidates, key=lambda value: (value[0], value[1]))
    if (result.get("selected_epoch") != best_epoch or result.get("independent_val_fgd_4step") != best_fgd or
            result.get("checkpoint") != str(best_checkpoint)):
        raise RuntimeError("Completed run selection no longer matches all 200 independent validations")
    test = completed_bundle(run/"test_selected"/"complete.json", config, best_checkpoint)
    if not test or test.get("split") != "test" or test.get("mode") != mode or test.get("epoch") != best_epoch or result.get("test") != test:
        raise RuntimeError("Completed run selected test/render/prediction bundle is incomplete")
    return result


def estimate_storage_budget(config, preflight, plan, completed_runs=()):
    """Lossless compression is optional savings, never assumed for clearance."""
    root = Path(config["output_root"])
    gate = json.loads(Path(config["precheck_path"]).read_text())
    budgets = json.loads(Path(gate["evaluation_budget_audit"]["evidence"]).read_text())
    val_frames = sum(item["evaluation_frames"] for item in budgets["splits"]["val"])
    test_frames = sum(item["evaluation_frames"] for item in budgets["splits"]["test"])
    # Poses165, translation3, expression100 float32; static betas300 and ZIP
    # metadata overhead. Count even the zero expression/body-only channels.
    prediction_frame_bytes, archive_overhead_bytes = 4*(165+3+100), 16384
    prediction_bytes_per_run = 3*((200*val_frames+test_frames)*prediction_frame_bytes+
                                    (200*106+265)*archive_overhead_bytes)
    reports, required = [], 0
    completed_runs = set(completed_runs)
    for mode, size in plan:
        if (mode, size) in completed_runs:
            reports.append({"mode": mode, "batch_per_gpu": size, "already_verified_complete": True, "remaining_bytes": 0})
            continue
        model_bytes = int(preflight["modes"][mode].get("model_state_bytes", 0))
        if model_bytes <= 0:
            raise RuntimeError("Actual BS preflight must record model-state bytes before storage clearance")
        checkpoint_bytes = 200*(model_bytes+4*1024**2)+11*model_bytes
        policies = 2 if mode == "full55" else 1
        render_bytes = 201*3*policies*32*1024**2  # 32 MiB per paired ten-second H.264/AAC video.
        logs_and_audits_bytes = 16*1024**3
        fresh_total = prediction_bytes_per_run+checkpoint_bytes+render_bytes+logs_and_audits_bytes
        run = root/f"{mode}_bs{size}"
        retained = sum(path.stat().st_size for path in run.rglob("*")
                       if path.is_file() and not path.is_symlink() and path.suffix in (".pt", ".npz", ".mp4")) if run.exists() else 0
        remaining = max(0, fresh_total-retained)
        required += remaining
        reports.append({"mode": mode, "batch_per_gpu": size, "actual_model_state_bytes": model_bytes,
            "all_200_model_checkpoints_plus_optimizer_atomic_bytes": checkpoint_bytes,
            "all_val_test_predictions_uncompressed_bytes": prediction_bytes_per_run,
            "paired_render_allowance_bytes": render_bytes, "logs_audits_allowance_bytes": logs_and_audits_bytes,
            "retained_known_artifact_bytes": retained, "remaining_bytes": remaining})
    reserve = max(16*1024**3, math.ceil(required*.2))
    usage = shutil.disk_usage(root)
    return {"passed": usage.free >= required+reserve, "kind": "conservative_offline_storage_estimate_not_GPU_test",
        "checked_at_unix": time.time(), "filesystem_free_bytes": usage.free,
        "filesystem_total_bytes": usage.total, "estimated_remaining_artifacts_bytes": required,
        "safety_reserve_bytes": reserve, "required_free_bytes": required+reserve, "runs": reports,
        "lossless_compression_savings_assumed": False, "deletions_performed": False,
        "limitations": "Filesystem free space is not a user quota guarantee; video/log allowances and concurrent EFS consumers require continued operator attention."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--launch", action="store_true")
    action.add_argument("--validate", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    gate = require_precheck(config)
    validate_host(config)
    output_root = Path(config["output_root"]).resolve()
    if output_root == Path("/") or output_root in (Path(config["raw_root"]).resolve(), Path(config["cache_root"]).resolve()):
        raise RuntimeError("Output must be an independent new experiment directory")
    if args.validate:
        print(json.dumps({"passed": True, "kind": "gate_and_host_only_not_gpu_training",
                          "task_name": config["task_name"], "hostname": socket.gethostname()}))
        return 0
    output_root.mkdir(parents=True, exist_ok=True)
    # All controller attempts for this node use the same fixed lock, regardless
    # of output directory. flock is held until every child has exited.
    lock_dir = Path("/local-ssd") if Path("/local-ssd").is_dir() else Path("/tmp")
    with (lock_dir/"music2gesture_semtalk_gpu.lock").open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("This node already has a Music2Gesture controller lock") from error
        inventory = gpu_inventory_idle()
        state = dict(task_name=config["task_name"], hostname=socket.gethostname(), output_root=str(output_root),
                     config_sha256=config_hash(config), pid=os.getpid(), start_ticks=start_ticks(os.getpid()),
                     controller_command=shlex.join(sys.argv), started_at_unix=time.time(),
                     source_hashes={name: sha256(Path(__file__).parent/name) for name in SOURCE_FILES},
                     initial_gpu_inventory=inventory, automatic_retry=False)
        command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                   str(Path(__file__).parent/"train.py"), "--config", str(args.config.resolve())]
        if args.preflight:
            target = output_root/"BS_PREFLIGHT_PASSED.json"
            if target.exists():
                raise RuntimeError("BS preflight already recorded; refusing overwrite")
            state["phase"] = "actual_bs_preflight"
            run_child(command+["--preflight"], output_root/"bs_preflight.log", state)
            return 0
        preflight = require_bs_preflight(config, gate, output_root/"BS_PREFLIGHT_PASSED.json")
        plan = [(mode, size) for mode in ("full55", "body22")
                for size in preflight["modes"][mode]["selected_batch_per_gpu"]]
        if len(plan) != 6 or any(len(set(preflight["modes"][mode]["selected_batch_per_gpu"])) != 3 for mode in ("full55", "body22")):
            raise RuntimeError("Plan must contain precisely two modes by three distinct tested batches")
        verified_complete = set()
        for mode, size in plan:
            run = output_root/f"{mode}_bs{size}"
            if (run/"TRAINING_COMPLETE.json").exists():
                if not args.resume:
                    raise RuntimeError("Existing run completion; ordinary launch is not repeatable")
                validate_completed_run(run, config, mode, size)
                verified_complete.add((mode, size))
        storage = estimate_storage_budget(config, preflight, plan, verified_complete)
        atomic_json(output_root/"storage_budget.json", storage)
        if not storage["passed"]:
            raise RuntimeError("Insufficient filesystem headroom for all required retained artifacts; no training child started")
        state.update(phase="six_serial_runs", plan=[dict(mode=mode, batch_per_gpu=size) for mode, size in plan])
        for index, (mode, size) in enumerate(plan):
            run = output_root/f"{mode}_bs{size}"
            complete = run/"TRAINING_COMPLETE.json"
            if (mode, size) in verified_complete:
                continue
            has_run = (run/"run.json").exists()
            if has_run and not args.resume:
                raise RuntimeError("Partial prior run exists; explicit resume required")
            if has_run and not (run/"latest_full.pt").is_file():
                raise RuntimeError("Partial run has no complete resume checkpoint; needs operator decision")
            gpu_inventory_idle()
            state.update(run_index=index, mode=mode, batch_per_gpu=size, manual_resume=has_run)
            child_command = command+["--mode", mode, "--batch-per-gpu", str(size)]
            if has_run:
                child_command += ["--resume"]
            run_child(child_command, output_root/f"controller_{mode}_bs{size}.log", state)
            if not complete.is_file():
                raise RuntimeError("Child exited without final val-selected test + render completion")
            validate_completed_run(run, config, mode, size)
        atomic_json(output_root/"ALL_RUNS_COMPLETE.json", {**state, "status": "complete", "completed_runs": 6,
            "results": [str(output_root/f"{mode}_bs{size}"/"TRAINING_COMPLETE.json") for mode, size in plan]})
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
