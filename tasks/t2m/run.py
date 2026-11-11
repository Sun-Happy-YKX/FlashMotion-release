#!/usr/bin/env python3
"""Portable, opt-in T2M commands. Planning uses only the Python standard library.

This is not a live-run controller. No resume, retry, GPU query, installation,
or subprocess is performed unless the caller explicitly supplies --execute.
The Full profile reads the bundled training source without modifying it.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys


RELEASE_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS = Path(__file__).resolve().parent / "snapshots"
LIVE_ROOT = "/efs/yangkaixing/FlowerDance_T2M_Long"
# profile: snapshot, manifest setting, historical port, worker parameter source
PROFILES = {
    "full": ("restart_20260922", None, 29893, None),
}
MANIFEST_SHA = {
    "restart_20260922": "c25678a56f1519582f705d129b86bf51221f79fe35e038c5acb471141e02620d",
}
# Configuration and source bindings from the archived training inventory.
# The Full manifest references restart_20260922/code and its source manifest.
FULL_BINDING = {
    "evidence_inventory": "FlashMotion_assets_inventory_20261010.json",
    "evidence_inventory_sha256": "cf7ba6bc1be1b7cba6df8ba5aa56a4e946d7e8cd8915d390002b52e9c9dafd78",
    "actual_manifest": LIVE_ROOT + "/runs/t2m_adain_selection_8000ep_20260923/manifest.json",
    "actual_manifest_sha256": "5f0f9850a78070ac5e80af6028668e8d644f56c608c0f811f7db81b0f59333c1",
    "actual_code": LIVE_ROOT + "/runs/t2m_restart_8000ep_20260922/code",
    "source_manifest": LIVE_ROOT + "/runs/t2m_restart_8000ep_20260922/manifest.json",
}
DISTRIBUTED_ENV = (
    "RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
    "ROLE_RANK", "MASTER_ADDR", "MASTER_PORT", "TORCHELASTIC_RUN_ID",
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def inside(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def new_output(path: Path, output_root: Path) -> None:
    # resolve() includes existing symlink/junction parents. Check again at execute.
    normalized = path.as_posix().lower()
    if LIVE_ROOT.lower() in normalized or "/efs/" in normalized:
        raise ValueError("Output under /efs or an original live T2M root is forbidden")
    if path == output_root or not inside(path, output_root):
        raise ValueError("Output must be a new child of configured output_root")
    if inside(path, SNAPSHOTS.resolve()):
        raise ValueError("Output cannot be inside immutable snapshots")
    if path.exists() or path.is_symlink():
        raise ValueError("Output already exists; no overwrite, reuse, resume, or retry")


def frozen_snapshot(profile: str) -> tuple[Path, dict, dict]:
    if profile not in PROFILES:
        raise ValueError("This release supports only the Full main-model profile")
    snapshot, setting, port, workers = PROFILES[profile]
    root = SNAPSHOTS / snapshot
    manifest_path = root / "provenance/manifest.json"
    if digest(manifest_path) != MANIFEST_SHA[snapshot]:
        raise ValueError(f"Frozen manifest SHA changed: {snapshot}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    record = manifest
    if profile == "full":
        binding_path = SNAPSHOTS / "full_selection_20260923/provenance/manifest.json"
        if digest(binding_path) != FULL_BINDING["actual_manifest_sha256"]:
            raise ValueError("Actual Full source-binding manifest SHA changed")
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
        if binding["code"] != FULL_BINDING["actual_code"] or binding["source_manifest"] != FULL_BINDING["source_manifest"]:
            raise ValueError("Full no longer binds to the exact shared restart source")
        matches = [row for row in binding["runs"] if row["setting"] == "adain_selection"]
        if len(matches) != 1:
            raise ValueError("Actual Full manifest needs one adain_selection run")
        record = matches[0]
    elif setting:
        matches = [row for row in manifest["runs"] if row["setting"] == setting]
        if len(matches) != 1:
            raise ValueError(f"Missing unique manifest setting {setting}")
        record = matches[0]
    collection = json.loads((RELEASE_ROOT / "manifests/frozen_source_collection.json").read_text(encoding="utf-8"))
    prefix = f"tasks/t2m/snapshots/{snapshot}/code/"
    pins = [row for row in collection["files"] if row["path"].startswith(prefix)]
    if not pins:
        raise ValueError("No actual frozen source hashes in collection receipt")
    for row in pins:
        source = (RELEASE_ROOT / row["path"]).resolve()
        if not inside(source, (root / "code").resolve()):
            raise ValueError("Collection pin escapes selected frozen code")
        if digest(source) != row["sha256"]:
            raise ValueError(f"Frozen source SHA changed: {row['path']}")
    config = dict(record)
    config["port"] = record.get("port", port)
    config["workers"] = workers
    evidence = {"snapshot": snapshot, "manifest_sha256": MANIFEST_SHA[snapshot], "verified_code_files": len(pins)}
    if profile == "full":
        evidence["full_source_binding"] = FULL_BINDING
    return root / "code", config, evidence


def parser_options(script: Path) -> tuple[set[str], dict]:
    """Read frozen argparse declarations without importing model/GPU code."""
    options: set[str] = set()
    defaults = {}
    tree = ast.parse(script.read_text(encoding="utf-8"), filename=str(script))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute) or node.func.attr != "add_argument":
            continue
        names = [arg.value for arg in node.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.startswith("--")]
        options.update(names)
        for keyword in node.keywords:
            if keyword.arg == "default" and isinstance(keyword.value, ast.Constant):
                for name in names:
                    defaults[name] = keyword.value.value
    return options, defaults


def train_arguments(profile: str, manifest: dict, paths: dict, output: Path, defaults: dict) -> list[str]:
    scan = manifest.get("condition_selection", manifest.get("scan", False))
    fusion = manifest.get("cross_modal_fusion", manifest.get("fusion", "adain"))
    if profile != "full":
        raise ValueError("This release supports only the Full main-model profile")
    objective = "tr_only"
    loss = manifest.get("loss", manifest.get("loss_weights", {"fm": 1, "rec": .636, "vel": .323, "acc": .2, "fk": .646, "floor": 0}))
    batch, world, steps = (int(manifest[key]) for key in ("batch_per_gpu", "world_size", "max_steps"))
    workers = manifest["workers"] if manifest["workers"] is not None else defaults["--workers"]
    args = ["--data-root", str(paths["data_root"]), "--embedding-root", str(paths["embedding_root"]),
            "--output", str(output), "--x0-variant", objective, "--cross-modal-fusion", fusion]
    if scan:
        args.append("--condition-selection")
    args += ["--direction-fusion", manifest.get("direction_fusion", "sum"), "--batch-size", str(batch),
             "--workers", str(workers), "--max-length", str(manifest.get("window", 200)),
             "--max-steps", str(steps), "--save-every-epochs", "50", "--save-every", "999999",
             "--log-every", "20", "--learning-rate", "2e-4", "--weight-decay", ".02",
             "--max-grad-norm", "1", "--min-t", ".05", "--min-interval", ".05",
             "--fm-time-weight-power", "2"]
    for name in ("fm", "rec", "vel", "acc", "fk", "floor"):
        args += [f"--{name}-weight", str(loss[name])]
    args += ["--mixed-precision", "bf16", "--ema-decay", ".9999", "--seed", str(manifest.get("seed", 42))]
    if batch * world != int(manifest.get("global_batch_size", manifest.get("global_batch", batch * world))):
        raise ValueError("Inconsistent frozen global batch")
    return args


def missing_inputs(paths: dict, mode: str, checkpoint: Path | None) -> list[str]:
    required = [paths["data_root"] / "HumanML3D/Mean.npy", paths["data_root"] / "HumanML3D/Std.npy",
                paths["embedding_root"] / "embeddings.npy", paths["embedding_root"] / "text_index.json"]
    if mode == "train":
        required += [paths["data_root"] / "HumanML3D/train.txt", paths["data_root"] / "HumanML3D/new_joint_vecs",
                     paths["data_root"] / "HumanML3D/texts", paths["data_root"] / "BABEL_streamed/train_processed.txt",
                     paths["data_root"] / "BABEL_streamed/motions", paths["data_root"] / "BABEL_streamed/texts"]
    else:
        required += [paths["data_root"] / "BABEL_streamed/val_processed.txt", paths["data_root"] / "BABEL_streamed/motions",
                     paths["data_root"] / "BABEL_streamed/texts", paths["flood_root"] / "metrics/tools/t2m_evaluator.py",
                     paths["flood_root"] / "metrics/tools/word_vectorizer.py", paths["flood_root"] / "deps/glove",
                     paths["flood_root"] / "deps/t2m/meta/mean.npy", paths["flood_root"] / "deps/t2m/meta/std.npy"]
        required += [paths["flood_root"] / f"deps/t2m/humanml3d/{name}.pt" for name in ("text_encoder", "movement_encoder", "motion_encoder")]
        if checkpoint:
            required.append(checkpoint)
    file_suffixes = {".npy", ".json", ".txt", ".pt", ".py"}
    errors = [f"Missing input or wrong type: {path}" for path in required
              if not (path.is_file() if path == checkpoint or path.suffix in file_suffixes else path.is_dir())]
    if mode == "eval":
        if checkpoint is None:
            errors.append("Evaluation requires explicit --checkpoint (trusted checkpoint pickle)")
        if paths["data_root"] != (paths["flood_root"] / "raw_data").resolve():
            errors.append("Frozen evaluator uses flood_root/raw_data; configured data_root must match")
        cohort_files = list(paths["cohort_root"].glob("*.npz"))
        ids = {p.stem.split("--", 1)[0] for p in cohort_files if "--" in p.stem}
        if len(cohort_files) != 311 or len(ids) != 311:
            errors.append("Canonical evaluation requires 311 unique NAME--HASH.npz cohort entries")
    return errors


def build_plan(args: argparse.Namespace) -> dict:
    if args.profile not in PROFILES:
        raise ValueError("This release supports only the Full main-model profile")
    config_path = Path(args.paths).expanduser().resolve()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Paths JSON must be an object")
    for key in ("data_root", "embedding_root", "flood_root", "cohort_root", "output_root"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ValueError(f"Paths JSON needs a nonempty string: {key}")
    paths = {key: resolve_path(raw[key], config_path.parent) for key in ("data_root", "embedding_root", "flood_root", "cohort_root", "output_root")}
    output = resolve_path(args.output, config_path.parent) if args.output else paths["output_root"] / f"{args.profile}_{args.mode}"
    output = output.resolve()
    new_output(output, paths["output_root"])
    code, manifest, evidence = frozen_snapshot(args.profile)
    python = args.python or raw.get("python", sys.executable)
    if not isinstance(python, str) or not python:
        raise ValueError("Python interpreter must be a nonempty string")
    if "/" in python or "\\" in python:
        python = str(resolve_path(python, config_path.parent))
    checkpoint = resolve_path(args.checkpoint, config_path.parent) if args.checkpoint else None
    strategy = args.eval_strategy or "nonar"
    if strategy != "nonar" or (args.mode == "train" and args.eval_strategy):
        raise ValueError("Requested eval strategy does not match the frozen profile")
    if args.mode == "train" and checkpoint:
        raise ValueError("Training from checkpoint is forbidden; no resume or pretrained loading")
    if args.mode == "eval" and args.port is not None:
        raise ValueError("--port is a training-only option")
    if args.mode == "train" and args.steps is not None:
        raise ValueError("--steps is an evaluation-only option")
    eval_steps = args.steps if args.steps is not None else [4]
    if not eval_steps or any(step not in (1, 4) for step in eval_steps) or len(set(eval_steps)) != len(eval_steps):
        raise ValueError("Main-model evaluation steps must be unique values from 1 and 4")
    script = code / ("train_t2m_long.py" if args.mode == "train" else "evaluate_t2m_ablation.py")
    options, defaults = parser_options(script)
    gpu_ids = args.gpus if args.gpus is not None else raw.get("training_gpus" if args.mode == "train" else "evaluation_gpus", "0,1,2,3,4,5,6,7" if args.mode == "train" else "0")
    if not isinstance(gpu_ids, str) or not gpu_ids:
        raise ValueError("GPU ids must be a nonempty comma-separated string")
    selected_gpus = gpu_ids.split(",")
    if any(not x.isdigit() for x in selected_gpus) or len(set(selected_gpus)) != len(selected_gpus):
        raise ValueError("GPU ids must be unique comma-separated nonnegative integers")
    if len(selected_gpus) != (int(manifest["world_size"]) if args.mode == "train" else 1):
        raise ValueError("Train requires frozen world_size GPUs; eval requires exactly one GPU")
    if args.mode == "train":
        entry_args = train_arguments(args.profile, manifest, paths, output, defaults)
        port = args.port if args.port is not None else int(manifest["port"])
        if not 1024 <= port <= 65535:
            raise ValueError("Invalid launch port")
        command = [python, "-m", "accelerate.commands.launch", "--multi_gpu", "--num_machines", "1", "--num_processes", str(manifest["world_size"]),
                   "--main_process_port", str(port), "--mixed_precision", "bf16", "--dynamo_backend", "no", str(script), *entry_args]
    else:
        entry_args = ["--checkpoint", str(checkpoint) if checkpoint else "<REQUIRED_TRUSTED_CHECKPOINT>", "--flood-root", str(paths["flood_root"]),
                      "--embedding-root", str(paths["embedding_root"]), "--cohort-dir", str(paths["cohort_root"]), "--output", str(output),
                      "--steps", *map(str, eval_steps), "--seed", "42", "--embedding-batch-size", "32", "--limit", "0"]
        command = [python, str(script), *entry_args]
    unknown = {value for value in entry_args if value.startswith("--")} - options
    if unknown or "--resume" in entry_args:
        raise ValueError(f"Unsupported or forbidden frozen CLI options: {sorted(unknown)}")
    return {"profile": args.profile, "mode": args.mode, "dry_run": not args.execute, "executed": False, "cwd": str(code), "output": str(output),
            "output_root": str(paths["output_root"]), "argv": command, "display_command": shlex.join(command), "entry_args": entry_args,
            "source": evidence, "eval_strategy": strategy if args.mode == "eval" else None,
            "environment_overrides": {"CUDA_VISIBLE_DEVICES": gpu_ids, "TOKENIZERS_PARALLELISM": "false", "PYTHONUNBUFFERED": "1",
                "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "4" if args.mode == "train" else "2", "TORCH_NCCL_ASYNC_ERROR_HANDLING": "1",
                "PYTHONPATH": os.pathsep.join((str(code), str(paths["flood_root"])))},
            "missing_inputs": missing_inputs(paths, args.mode, checkpoint), "runtime_verification": "not_run_GPU_dependencies_and_quality_unverified",
            "resume": False, "retry": False, "paper_full_profile": "full_proven_shared_restart_source_from_fresh_manifest",
            "workers_note": "Full uses the worker count from the source argparse default"}


def execute_plan(plan: dict) -> int:
    if plan["missing_inputs"]:
        raise ValueError("Execution refused until every missing input is supplied")
    python = plan["argv"][0]
    if not (Path(python).is_file() or shutil.which(python)):
        raise ValueError("Configured Python interpreter not found")
    output = Path(plan["output"])
    new_output(output.resolve(), Path(plan["output_root"]).resolve())
    # Exclusively reserve a new output before the frozen trainer's exist_ok mkdir.
    output.parent.mkdir(parents=True, exist_ok=True)
    new_output(output.resolve(), Path(plan["output_root"]).resolve())
    output.mkdir(exist_ok=False)
    (output / "portable_launch_plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    env = os.environ.copy()
    for key in DISTRIBUTED_ENV:
        env.pop(key, None)
    old_pythonpath = env.get("PYTHONPATH")
    env.update(plan["environment_overrides"])
    if old_pythonpath:
        env["PYTHONPATH"] += os.pathsep + old_pythonpath
    # Only this explicit execution boundary can create a child. No shell/controller.
    result = subprocess.run(plan["argv"], cwd=plan["cwd"], env=env, check=False)
    (output / "portable_exit.json").write_text(json.dumps({"executed": True, "exit_code": result.returncode, "automatic_restart": False, "automatic_retry": False}) + "\n", encoding="utf-8")
    return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, choices=tuple(PROFILES))
    parser.add_argument("--mode", choices=("train", "eval"), default="train")
    parser.add_argument("--paths", default=str(RELEASE_ROOT / "configs/t2m.paths.example.json"))
    parser.add_argument("--output", help="Fresh child of output_root; paths relative to paths JSON directory")
    parser.add_argument("--checkpoint", help="Evaluation only; explicit trusted checkpoint, never a training resume")
    parser.add_argument("--python", help="Interpreter override; no shell expression")
    parser.add_argument("--gpus", help="Comma-separated GPU ids; explicit execute uses these GPUs")
    parser.add_argument("--port", type=int, help="Optional new training port; budget/model/weights remain frozen")
    parser.add_argument("--steps", type=int, nargs="+", choices=(1, 4), help="Evaluation only: main-model NFE1/NFE4; default native4")
    parser.add_argument("--eval-strategy", choices=("nonar",))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--dry-run", action="store_true", help="Default: print plan only, no subprocess or output files")
    action.add_argument("--execute", action="store_true", help="Explicitly permit one new train/eval subprocess")
    args = parser.parse_args(argv)
    try:
        plan = build_plan(args)
        print(json.dumps(plan, indent=2))
        return execute_plan(plan) if args.execute else 0
    except (OSError, ValueError, KeyError, SyntaxError) as exc:
        print(f"T2M command refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
