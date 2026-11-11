"""Traceable M2D copy preparation using only stdlib; never imports/launches ML.

Source bytes are pinned. Only exact AST spans are replaced, with old/new text
recorded. A fresh destination and exclusive writes prevent old-run reuse.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any


SNAPSHOT = "tasks/m2d/snapshots/genre_20260929"
TRAIN_SHA = "03f1a2f6dc276c3aad600600fa29db3b5ca2c5d02b415773cb70c355a4ceb1b9"
CONTROL_SHA = "f91fd638c17cf0c255dd47caffea7c99771a8d5d6997e1bf32e9aa48e722cfeb"
EVAL_SHA = "bd28059623a1b00fc278e131f9664ae2fa78b9f4addec237d01c868c715dd3fb"
FORMAL_SHA = "6cc03bf74bd5fc822e161fcadc31b3da4423ccff44b733a5bc72003ab19de93f"
OLD_PROJECT = "/efs/yangkaixing/FlowerDance_x0_ablation"
OLD_RUN = OLD_PROJECT + "/runs/music_genre_ablation_20260929"
OMITTED_UTILITIES = {"music2dance_setup_20260929.py", "smoke_pair.py"}


def sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def no_links(path: Path) -> None:
    """Check all existing ancestors, including Windows junction/reparse points."""
    for component in (path, *path.parents):
        if component.exists() or component.is_symlink():
            info = component.lstat()
            if component.is_symlink() or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
                raise ValueError("Symlink/reparse target is not allowed")


def source_bytes(repo_root: Path, relative: str, expected: str) -> bytes:
    if relative.startswith("/") or "\\" in relative or any(part in ("", ".", "..") for part in relative.split("/")):
        raise ValueError("Unsafe source path")
    path = repo_root / relative
    no_links(path)
    if not path.resolve().is_relative_to(repo_root.resolve()) or not path.is_file():
        raise ValueError("Source file is missing/outside checkout")
    payload = path.read_bytes()
    if sha(payload) != expected:
        raise ValueError(f"Pinned source SHA mismatch: {relative}")
    return payload


def assignment(tree: ast.AST, name: str) -> ast.Assign:
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
               and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if len(matches) != 1:
        raise ValueError(f"Expected one native assignment to {name}")
    return matches[0]


def path_argument(node: ast.AST, expected: str) -> ast.Constant:
    if (not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name)
            or node.func.id != "Path" or len(node.args) != 1 or node.keywords
            or not isinstance(node.args[0], ast.Constant) or node.args[0].value != expected):
        raise ValueError("Native Path literal contract changed")
    return node.args[0]


def apply_spans(payload: bytes, changes: list[tuple[ast.AST, str, str]]) -> tuple[bytes, list[dict[str, Any]]]:
    lines = payload.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    patches = []
    for node, replacement, reason in changes:
        start = offsets[node.lineno - 1] + node.col_offset
        end = offsets[node.end_lineno - 1] + node.end_col_offset
        patches.append((start, end, replacement.encode("utf-8"), reason))
    patches.sort()
    if any(patches[index][1] > patches[index + 1][0] for index in range(len(patches) - 1)):
        raise ValueError("Overlapping AST transformations")
    result = payload
    for start, end, replacement, _ in reversed(patches):
        result = result[:start] + replacement + result[end:]
    ast.parse(result)
    records = [{"start_byte": start, "end_byte": end, "before": payload[start:end].decode("utf-8"),
                "after": replacement.decode("utf-8"), "reason": reason}
               for start, end, replacement, reason in patches]
    return result, records


def adapt_train(payload: bytes, data_root: str) -> tuple[bytes, list[dict[str, Any]]]:
    if sha(payload) != TRAIN_SHA:
        raise ValueError("Training source is not the pinned formal entry")
    tree = ast.parse(payload)
    return apply_spans(payload, [(path_argument(assignment(tree, "S").value, OLD_PROJECT),
                                 repr(data_root), "Relocate S; R remains derived from exclusive generated code directory")])


def adapt_control(payload: bytes, root: str, paths: dict[str, Any]) -> tuple[bytes, list[dict[str, Any]]]:
    if sha(payload) != CONTROL_SHA:
        raise ValueError("Controller source is not the pinned original")
    hostname = paths["expected_hostname"]
    if "REPLACE" in hostname.upper() or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", hostname):
        raise ValueError("Preparing a controller requires an explicit independent hostname")
    tree = ast.parse(payload)
    changes: list[tuple[ast.AST, str, str]] = [
        (path_argument(assignment(tree, "R").value, OLD_RUN), repr(root), "Independent run root"),
        (assignment(tree, "P").value, f"Path({paths['python']!r})", "Explicit independent Python; no historical .venv symlink"),
        (assignment(tree, "cards").value, repr([int(device) for device in paths["visible_devices"]]), "Configured physical device IDs; free-memory guard unchanged"),
    ]
    host_matches = [node for node in ast.walk(tree) if isinstance(node, ast.Constant)
                    and node.value == "bubuyang-dmd-0924-102244-1x8-worker-0"]
    cuda_matches = [node for node in ast.walk(tree) if isinstance(node, ast.keyword) and node.arg == "CUDA_VISIBLE_DEVICES"]
    cmd = assignment(tree, "cmd").value
    if len(host_matches) != 1 or len(cuda_matches) != 1 or not isinstance(cmd, ast.List):
        raise ValueError("Controller host/CUDA/command shape changed")
    port_index = [index for index, node in enumerate(cmd.elts) if isinstance(node, ast.Constant) and node.value == "--main_process_port"]
    if len(port_index) != 1:
        raise ValueError("Controller port contract changed")
    changes += [(host_matches[0], repr(hostname), "Preserve host-identity assertion with explicit independent host"),
                (cuda_matches[0].value, repr(",".join(paths["visible_devices"])), "Independent visible device mapping"),
                (cmd.elts[port_index[0] + 1], f"{str(paths['main_process_ports']['genre'])!r} if version=='genre' else {str(paths['main_process_ports']['no_genre'])!r}", "Independent configured ports; all native launch flags retained")]
    return apply_spans(payload, changes)


def adapt_eval(payload: bytes, root: str, data_root: str, selection: dict[str, Any]) -> tuple[bytes, list[dict[str, Any]]]:
    if sha(payload) != EVAL_SHA:
        raise ValueError("Evaluation source is not the repaired scorer")
    if (selection.get("version") not in ("genre", "no_genre") or type(selection.get("n")) is not int or selection.get("n") not in (1, 4)
            or type(selection.get("epoch")) is not int or selection["epoch"] < 1
            or not isinstance(selection.get("checkpoint"), str)
            or not re.fullmatch(r"[0-9a-f]{64}", selection.get("sha256", ""))):
        raise ValueError("One explicit checkpoint/version/epoch/NFE/SHA is required")
    tree = ast.parse(payload)
    changes: list[tuple[ast.AST, str, str]] = [
        (path_argument(assignment(tree, "C").value, OLD_RUN + "/code"), repr(root + "/code"), "Independent copied model source"),
        (path_argument(assignment(tree, "S").value, OLD_PROJECT), repr(data_root), "Independent exact test data root"),
        (assignment(tree, "OUT").value, f"R/'results'", "Fresh independent corrected-evaluation output"),
    ]
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Attribute)
                  and isinstance(node.test.func.value, ast.Name) and node.test.func.value.id == "planfile"
                  and node.test.func.attr == "exists"]
    if len(candidates) != 1:
        raise ValueError("Historical checkpoint-plan branch changed")
    entry = {key: selection[key] for key in ("version", "epoch", "n", "checkpoint")}
    replacement = "plan=" + repr([entry]) + "\n    dump(planfile,dict(plan=plan,fps=30,seed=42,batch=2,protocol='corrected_eval_20260930',selection='explicit_one_checkpoint_no_ranking',checkpoint_sha256=" + repr(selection["sha256"]) + ",source_code=str(C),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))"
    changes.append((candidates[0], replacement, "Bound scope to one explicit checkpoint; never scan or reuse historical pool/manifest"))
    ranking_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "rank_all"]
    if len(ranking_calls) != 2:
        raise ValueError("Expected exactly two historical reranking calls")
    changes += [(node, "None", "No reranking of the single checkpoint; metric kernels unchanged") for node in ranking_calls]
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main")
    expected_update = ast.parse(
        "y,_=FlowMatching._x_r_and_velocity(y,y0,t[:,None,None],r[:,None,None])"
    ).body[0]
    sampler_updates = [node for node in ast.walk(main) if isinstance(node, ast.Assign)
                       and ast.dump(node) == ast.dump(expected_update)]
    expected_grid = ast.parse("range(n)", mode="eval").body
    sampler_loops = [node for node in ast.walk(main) if isinstance(node, ast.For)
                     and isinstance(node.target, ast.Name) and node.target.id == "step"
                     and ast.dump(node.iter) == ast.dump(expected_grid)]
    if (len(sampler_updates) != 1 or len(sampler_loops) != 1
            or sampler_updates[0] not in sampler_loops[0].body
            or any(isinstance(node, ast.Name) and node.id == "ratio" for node in ast.walk(main))):
        raise ValueError("Expected one unchanged corrected-evaluation sampling update")
    # Only the scorer discarded the helper's velocity. Keep the training/JVP
    # helper and copied model bytes intact, and retain its exact x_r arithmetic.
    direct_update = (
        "ratio=(r[:,None,None]/t[:,None,None].clamp_min(1e-5)).clamp(0.0,1.0);"
        "y=ratio*y+(1.0-ratio)*y0"
    )
    changes.append((sampler_updates[0], direct_update,
                    "inference-only omit unused velocity; same direct interpolation"))
    first = main.body[0]
    original_first = ast.get_source_segment(payload.decode("utf-8"), first)
    if original_first != "OUT.mkdir(exist_ok=True)":
        raise ValueError("Evaluation main entry changed")
    changes.append((first, "import random\n    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)\n    " + original_first,
                    "Explicit independent Python/NumPy/Torch seed42 before model construction; native per-sampling torch seed42 retained"))
    return apply_spans(payload, changes)


def checkpoint_hash(path: Path, expected: str) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Checkpoint must be an explicit absolute file path")
    no_links(path)
    if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError("Checkpoint file/expected SHA missing")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError("Checkpoint SHA mismatch; no pickle/torch deserialization performed")


def bundle_payloads(repo_root: Path, destination: Path, paths: dict[str, Any], selection: dict[str, Any] | None) -> dict[str, tuple[bytes, dict[str, Any]]]:
    formal = json.loads(source_bytes(repo_root, SNAPSHOT + "/provenance/formal_code_sha256.json", FORMAL_SHA))
    files: dict[str, tuple[bytes, dict[str, Any]]] = {}
    root = destination.as_posix()
    for relative, expected in formal.items():
        original = source_bytes(repo_root, SNAPSHOT + "/code/" + relative, expected)
        if relative in OMITTED_UTILITIES:
            continue
        generated, patches = adapt_train(original, paths["data_root"]) if relative == "train_pair.py" else (original, [])
        files["code/" + relative] = (generated, {"source_path": SNAPSHOT + "/code/" + relative,
            "source_sha256": expected, "generated_sha256": sha(generated), "bytes": len(generated), "patches": patches})
    if selection is None:
        original = source_bytes(repo_root, SNAPSHOT + "/provenance/full_control.py", CONTROL_SHA)
        generated, patches = adapt_control(original, root, paths)
        relative = "full_control.py"
        source_relative, expected = SNAPSHOT + "/provenance/full_control.py", CONTROL_SHA
    else:
        checkpoint_hash(Path(selection["checkpoint"]), selection["sha256"])
        original = source_bytes(repo_root, SNAPSHOT + "/provenance/music_corrected_eval_20260930.py", EVAL_SHA)
        generated, patches = adapt_eval(original, root, paths["data_root"], selection)
        relative = "music_corrected_eval_20260930.py"
        source_relative, expected = SNAPSHOT + "/provenance/music_corrected_eval_20260930.py", EVAL_SHA
    files[relative] = (generated, {"source_path": source_relative, "source_sha256": expected,
        "generated_sha256": sha(generated), "bytes": len(generated), "patches": patches})
    for generated, _ in files.values():
        ast.parse(generated)
    return files


def validate_prepared_bundle(repo_root: Path, destination: Path, paths: dict[str, Any], selection: dict[str, Any]) -> dict[str, Any]:
    no_links(destination)
    receipt_path = destination / "adaptation_receipt.json"
    no_links(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (receipt.get("status") != "PREPARED_NOT_EXECUTED" or receipt.get("mode") != "one_checkpoint_eval"
            or receipt.get("selection") != selection or receipt.get("prepared_root") != destination.as_posix()):
        raise ValueError("Prepared eval receipt/selection mismatch")
    expected_files = bundle_payloads(repo_root, destination, paths, selection)
    if not isinstance(receipt.get("files"), list) or len(receipt["files"]) != len(expected_files):
        raise ValueError("Prepared file inventory mismatch")
    for item in receipt["files"]:
        if not isinstance(item, dict):
            raise ValueError("Malformed prepared file record")
        relative = item.get("path")
        if relative not in expected_files:
            raise ValueError("Unexpected prepared file")
        expected_payload, expected_record = expected_files.pop(relative)
        if item != dict(expected_record, path=relative):
            raise ValueError("Prepared transformation receipt was altered")
        target = destination / relative
        no_links(target)
        if target.read_bytes() != expected_payload:
            raise ValueError("Prepared generated file bytes changed")
    if expected_files:
        raise ValueError("Prepared files missing")
    entries = list(destination.rglob("*"))
    for path in entries:
        no_links(path)
    actual = {path.relative_to(destination).as_posix() for path in entries if path.is_file()}
    if actual != {item["path"] for item in receipt["files"]} | {"adaptation_receipt.json"}:
        raise ValueError("Prepared root is not fresh; unexpected files/results found")
    return receipt


def prepare_bundle(repo_root: Path, destination: Path, paths: dict[str, Any], selection: dict[str, Any] | None = None) -> dict[str, Any]:
    """Create one new copy and receipt; never execute. Partial failures retained."""
    if not destination.is_absolute() or ".." in destination.parts or destination.parent == destination:
        raise ValueError("Destination must be explicit, absolute and non-root")
    no_links(destination)
    resolved = destination.resolve()
    old = PurePosixPath(OLD_PROJECT)
    proposed = PurePosixPath(resolved.as_posix())
    if proposed.is_relative_to(old) or old.is_relative_to(proposed):
        raise ValueError("Destination overlaps historical project")
    if resolved.is_relative_to(repo_root.resolve()) or repo_root.resolve().is_relative_to(resolved):
        raise ValueError("Destination must not overlap code checkout")
    if destination.exists() or not destination.parent.is_dir():
        raise ValueError("Destination must not exist and its safe parent must already exist")
    files = bundle_payloads(repo_root, destination, paths, selection)
    root = destination.as_posix()
    receipt: dict[str, Any] = {"schema": "flashmotion.m2d.path_adapter.v1", "status": "PREPARATION_IN_PROGRESS",
        "prepared_root": root, "launch_performed": False, "model_imports": False,
        "runtime_verified": False, "ready_to_execute": False, "frozen_bytes_modified": True,
        "mode": "one_checkpoint_eval" if selection else "fresh_training",
        "selection": selection, "omitted_historical_unsafe_utilities": sorted(OMITTED_UTILITIES),
        "files": [], "guard_policy": "Controller host/locks/duplicate/verify/free-memory guards and evaluator output lock/memory/numerical assertions retained; independent runtime/asset/license proofs remain mandatory."}
    destination.mkdir()  # Exclusive: no exist_ok, no parent creation.
    try:
        for relative, (generated, record) in sorted(files.items()):
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            no_links(target.parent)
            with target.open("xb") as handle:
                handle.write(generated)
            if sha(target.read_bytes()) != record["generated_sha256"]:
                raise ValueError("Generated copy readback SHA mismatch")
            receipt["files"].append(dict(record, path=relative))
        receipt["status"] = "PREPARED_NOT_EXECUTED"
    except Exception:
        receipt["status"] = "PREPARATION_FAILED_PARTIAL_FILES_RETAINED"
        with (destination / "adaptation_receipt.json").open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, allow_nan=False)
        raise
    with (destination / "adaptation_receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
    return receipt
