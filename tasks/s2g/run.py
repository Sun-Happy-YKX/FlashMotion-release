#!/usr/bin/env python3
"""Plan S2G commands and optionally write a new native config; never launch.

Only the Python standard library is imported. Frozen source is hashed and its
CLI is inspected with AST, not imported. A generated training config remains
a template until its runtime and input checks are completed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import shlex
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_RELATIVE = "tasks/s2g/snapshots/full55_r2/code"
HISTORICAL_OUTPUT = PurePosixPath(
    "/efs/yangkaixing/music2gesture_semtalk_20261007_r1/results/r2"
)
ENVIRONMENT = {
    "OPENBLAS_CORETYPE": "Haswell",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
PATH_KEYS = (
    "python", "runtime_prefix", "raw_root", "cache_root", "hubert", "vocab",
    "smplx", "metric_assets", "evaluator_assets", "output_root", "native_config",
    "precheck_path", "numerical_environment_evidence", "bs_train_window_inventory",
    "predictions", "evaluation_output",
)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def validate_paths(paths: dict[str, Any]) -> None:
    if paths.get("schema_version") != 1:
        raise ValueError("paths JSON must have schema_version=1")
    for key in PATH_KEYS:
        value = paths.get(key)
        if not isinstance(value, str) or not PurePosixPath(value).is_absolute():
            raise ValueError(f"{key} must be an explicit absolute Linux path")
        if ".." in PurePosixPath(value).parts:
            raise ValueError(f"{key} must not contain unresolved '..' segments")
    for key in ("task_name", "task_id", "expected_hostname"):
        if not isinstance(paths.get(key), str) or not paths[key].strip():
            raise ValueError(f"Supply an independent {key}; placeholders remain drafts")
    for key in ("output_root", "native_config", "cache_root", "evaluation_output"):
        if PurePosixPath(paths[key]).is_relative_to(HISTORICAL_OUTPUT):
            raise ValueError(f"{key} targets the historical run; use independent outputs")
    if PurePosixPath(paths["raw_root"]) == PurePosixPath(paths["cache_root"]):
        raise ValueError("Preprocessing cache must differ from raw dataset root")


def verify_frozen_source(repo_root: Path) -> int:
    receipt = read_json(repo_root / "manifests/frozen_source_collection.json")
    entries = [item for item in receipt["files"]
               if item["path"].startswith(SOURCE_RELATIVE + "/")]
    if not entries:
        raise ValueError("No S2G source entries in the frozen collection receipt")
    for item in entries:
        payload = (repo_root / item["path"]).read_bytes()
        if len(payload) != item["bytes"] or hashlib.sha256(payload).hexdigest() != item["sha256"]:
            raise ValueError(f"Frozen source receipt mismatch: {item['path']}")
    return len(entries)


def cli_flags(source_path: Path) -> set[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    return {
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant) and isinstance(argument.value, str)
        and argument.value.startswith("--")
    }


def require_flags(source_path: Path, flags: list[str]) -> None:
    missing = set(flags) - cli_flags(source_path)
    if missing:
        raise ValueError(f"Native CLI does not support {sorted(missing)} in {source_path.name}")


def draft_native_config(paths: dict[str, Any], source: Path) -> dict[str, Any]:
    config = read_json(source / "config.target.json")
    for key in (
        "task_name", "task_id", "expected_hostname", "runtime_prefix", "raw_root",
        "cache_root", "hubert", "vocab", "smplx", "metric_assets", "evaluator_assets",
        "output_root", "precheck_path", "numerical_environment_evidence",
        "bs_train_window_inventory",
    ):
        config[key] = paths[key]
    config["namespace"] = paths.get("namespace", "independent-user-runtime")
    config["manifest"] = str(PurePosixPath(paths["cache_root"]) / "manifest.json")
    config["numerical_environment"] = {key: ENVIRONMENT[key] for key in (
        "OPENBLAS_CORETYPE", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"
    )}
    # Never reuse another host's proof hashes as fresh runtime evidence.
    config["numerical_environment_evidence_sha256"] = None
    config["bs_train_window_inventory_sha256"] = None
    config["template_only_not_launchable"] = True
    config["asset_verification"] = "Configuration template: validate runtime, assets and prechecks before training."
    config["batch_policy"] = "Use 240-frame windows; validate native and distributed batch behavior before training."
    config["bs_no_short_authorization"] = "Use the 240-frame training protocol; validate the training-window inventory before launch."
    config["archive_entrypoint"] = "plan-only; no process launch and no native safety gate removed"
    return config


def build_plan(args: argparse.Namespace, paths: dict[str, Any], repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    validate_paths(paths)
    source = repo_root / SOURCE_RELATIVE
    source_count = verify_frozen_source(repo_root)
    native_config = draft_native_config(paths, source)
    config_path = args.native_config or paths["native_config"]
    if not PurePosixPath(config_path).is_absolute() or ".." in PurePosixPath(config_path).parts:
        raise ValueError("--native-config must be an explicit absolute Linux path")
    if PurePosixPath(config_path).is_relative_to(HISTORICAL_OUTPUT):
        raise ValueError("Native config must use an independent output location")
    argv = [paths["python"]]
    if args.stage == "train":
        flags = ["--config", "--mode", "--batch-per-gpu"]
        require_flags(source / "train.py", flags)
        if args.batch_per_gpu < 1:
            raise ValueError("--batch-per-gpu must be positive")
        argv += ["-m", "torch.distributed.run", "--standalone", "--nproc_per_node=8",
                 (source / "train.py").as_posix(), "--config", config_path,
                 "--mode", args.mode, "--batch-per-gpu", str(args.batch_per_gpu)]
    elif args.stage == "preprocess":
        flags = ["--raw-root", "--cache-root", "--hubert", "--vocab", "--smplx",
                 "--device", "--shards", "--shard"]
        require_flags(source / "prepare.py", flags + (["--statistics-only"] if args.statistics_only else []))
        if args.shards < 1 or not 0 <= args.shard < args.shards:
            raise ValueError("Require shards>=1 and 0<=shard<shards")
        argv += [(source / "prepare.py").as_posix(),
                 "--raw-root", paths["raw_root"], "--cache-root", paths["cache_root"],
                 "--hubert", paths["hubert"], "--vocab", paths["vocab"], "--smplx", paths["smplx"],
                 "--device", args.device, "--shards", str(args.shards), "--shard", str(args.shard)]
        if args.statistics_only:
            argv.append("--statistics-only")
    else:
        flags = ["--predictions", "--ground-truth", "--audio", "--manifest", "--split",
                 "--expected-count", "--pose-policy", "--model-pose-policy", "--metric-assets",
                 "--encoder-dir", "--device", "--output"]
        require_flags(source / "evaluation.py", flags)
        raw = PurePosixPath(paths["raw_root"])
        argv += [(source / "evaluation.py").as_posix(), "--predictions", paths["predictions"],
                 "--ground-truth", str(raw / "smplxflame_30"), "--audio", str(raw / "wave16k"),
                 "--manifest", str(PurePosixPath(paths["cache_root"]) / "manifest.json"),
                 "--split", args.split, "--expected-count", str({"val": 106, "test": 265}[args.split]),
                 "--pose-policy", args.mode, "--model-pose-policy", args.mode,
                 "--metric-assets", paths["metric_assets"], "--encoder-dir", paths["evaluator_assets"],
                 "--device", args.device, "--output", paths["evaluation_output"]]
    return {
        "stage": args.stage,
        "plan_only": True,
        "launch_performed": False,
        "runtime_verified": False,
        "ready_to_train": False,
        "frozen_source_receipt_files_verified": source_count,
        "environment": ENVIRONMENT.copy(),
        "argv": argv,
        "shell_command": shlex.join(["env", *(f"{key}={value}" for key, value in ENVIRONMENT.items()), *argv]),
        "native_config_path": config_path,
        "native_config": native_config,
        "config_written": False,
        "pending": [
            "Supply independent Linux paths/host/resource identity and real licensed assets",
            "Collect/rebuild actual numerical/runtime/native-JVP/precheck/batch proofs for that identity",
            "Review generated config and native template marker; source gates are unchanged",
            "For eval, verify prediction provenance/full frame budgets and select a fresh output",
        ],
    }


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("stage", choices=("preprocess", "train", "eval"))
    result.add_argument("--paths", type=Path, default=REPO_ROOT / "configs/s2g.paths.example.json")
    result.add_argument("--dry-run", action="store_true", help="Explicitly forbid config writes; all modes only print commands")
    result.add_argument("--write-config", action="store_true", help="Write a new blocked native config using exclusive creation")
    result.add_argument("--native-config", help="Independent absolute Linux config output path")
    result.add_argument("--mode", choices=("full55", "body22"), default="full55")
    result.add_argument("--batch-per-gpu", type=int, default=32)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--split", choices=("val", "test"), default="test")
    result.add_argument("--shards", type=int, default=1)
    result.add_argument("--shard", type=int, default=0)
    result.add_argument("--statistics-only", action="store_true")
    return result


def main(argv: list[str] | None = None) -> int:
    arguments = parser()
    args = arguments.parse_args(argv)
    if args.dry_run and args.write_config:
        arguments.error("--dry-run cannot be combined with --write-config")
    try:
        plan = build_plan(args, read_json(args.paths))
        if args.write_config:
            target = Path(plan["native_config_path"])
            if target.resolve().is_relative_to((REPO_ROOT / SOURCE_RELATIVE).resolve()):
                raise ValueError("Generated config must not modify the frozen source directory")
            # No overwrite, no directories created, no claims/proofs synthesized.
            with target.open("x", encoding="utf-8") as handle:
                handle.write(json.dumps(plan["native_config"], indent=2, allow_nan=False) + "\n")
            plan["config_written"] = True
    except (OSError, ValueError, KeyError, TypeError) as error:
        arguments.error(str(error))
    print(json.dumps(plan, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
