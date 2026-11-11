"""Read-only environment report; native imports and a CUDA probe require opt-in."""
from __future__ import annotations

import argparse
import hashlib
import importlib
from importlib import metadata
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TORCH = {"torch": "2.7.1+cu128", "torchvision": "0.22.1+cu128", "torchaudio": "2.7.1+cu128"}
NATIVE = {"mamba-ssm": "2.2.6.post3", "causal-conv1d": "1.5.3.post1"}
NATIVE_COMMITS = {
    "mamba-ssm": "10b5d6358f27966f6a40e4bf0baa17a460688128",
    "causal-conv1d": "857842b10546d86aaf05c52570a51fba2eec04d2",
    "pytorch3d": "75ebeeaea0908c5527e7b1e305fbc7681382db47",
}
PROVIDER_SOURCE_HASHES = {
    "mamba_ssm.ops.selective_scan_interface": "2260a1486cea11fff63ba69d584175864f2687354b722fa9c16820de6a45158f",
    "causal_conv1d.causal_conv1d_interface": "221f41aa963a8f84683d9c6b2b4bf448692feb963fefcd778fa1b4ed2688c859",
    "causal_conv1d.cpp_functions": "80e75c8f761c88ede2ff73d244b0e9c0d2a974a780e2d3d8befeec1bd8aac29e",
}


def read_pins(path):
    pins = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)", line)
        if not match or match[1] in pins:
            raise ValueError(f"Unsupported/duplicate runtime pin at {path.name}:{number}")
        pins[match[1]] = match[2]
    return pins


def command_version(name, flag):
    executable = shutil.which(name)
    if executable is None:
        return {"present": False}
    try:
        result = subprocess.run([executable, flag], capture_output=True, text=True,
                                timeout=15, check=False)
        return {"present": True, "path": executable, "returncode": result.returncode,
                "output": (result.stdout + result.stderr).strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"present": True, "path": executable, "error": type(error).__name__}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_origin(name):
    """Read native pip provenance without emitting URLs, auth fields or local paths."""
    commit = None
    source_type = "unknown"
    try:
        raw = metadata.distribution(name).read_text("direct_url.json")
        data = json.loads(raw) if raw else {}
        if data.get("vcs_info", {}).get("vcs") == "git":
            source_type = "git"
            candidate = data["vcs_info"].get("commit_id", "")
            if isinstance(candidate, str) and re.fullmatch(r"[0-9a-fA-F]{40}", candidate):
                commit = candidate.lower()
        elif data.get("dir_info", {}).get("editable"):
            source_type = "editable_local_commit_not_in_metadata"
    except (metadata.PackageNotFoundError, ValueError, TypeError, AttributeError):
        pass
    return {"source_type": source_type, "expected_git_commit": NATIVE_COMMITS[name],
            "recorded_git_commit": commit,
            "git_commit_matches": None if commit is None else commit == NATIVE_COMMITS[name]}


def native_probe(task, cuda):
    modules = ["torch", "selective_scan_cuda", "causal_conv1d_cuda", "mamba_ssm", "causal_conv1d"]
    if task == "m2d":
        modules.extend(["pytorch3d._C", "pytorch3d.transforms"])
    result, loaded = {}, {}
    for name in modules:
        try:
            loaded[name] = module = importlib.import_module(name)
            path = Path(module.__file__) if getattr(module, "__file__", None) else None
            result[name] = {"imported": True, "path": str(path) if path else None,
                            "sha256": file_hash(path) if path and path.is_file() else None}
        except Exception as error:
            result[name] = {"imported": False, "error": type(error).__name__, "message": str(error)[:500]}
    source_checks = {}
    for name, expected in PROVIDER_SOURCE_HASHES.items():
        try:
            path = Path(importlib.import_module(name).__file__)
            actual = file_hash(path)
            source_checks[name] = {"sha256": actual, "expected_sha256": expected, "match": actual == expected}
        except Exception as error:
            source_checks[name] = {"match": False, "error": type(error).__name__}
    cuda_result = {"requested": cuda, "tensor_probe_passed": None}
    if cuda:
        try:
            torch = loaded["torch"]
            value = torch.ones(1, device="cuda")
            torch.cuda.synchronize()
            cuda_result.update(tensor_probe_passed=bool(value.cpu().item() == 1),
                               torch_cuda_build=torch.version.cuda,
                               device_name=torch.cuda.get_device_name())
            del value
        except Exception as error:
            cuda_result.update(tensor_probe_passed=False, error=type(error).__name__, message=str(error)[:500])
    return result, source_checks, cuda_result


def report_environment(task, native=False, cuda=False):
    expected = {**read_pins(ROOT / "requirements" / f"{task}.runtime.txt"), **TORCH, **NATIVE}
    if task == "m2d":
        expected["pytorch3d"] = "0.7.8"
    packages = {}
    for name, version in expected.items():
        try:
            installed = metadata.version(name)
        except metadata.PackageNotFoundError:
            installed = None
        packages[name] = {"expected": version, "installed": installed, "match": installed == version}
        if name in NATIVE_COMMITS:
            packages[name]["source_origin"] = source_origin(name)
    tools = {name: command_version(name, flag) for name, flag in
             (("nvcc", "--version"), ("ffmpeg", "-version"), ("ffprobe", "-version"))}
    imports, sources, cuda_result = ({}, {}, {"requested": False, "tensor_probe_passed": None})
    if native or cuda:
        imports, sources, cuda_result = native_probe(task, cuda)
    checks = {"linux_x86_64": sys.platform.startswith("linux") and platform.machine() == "x86_64",
              "python_3_10_12": platform.python_version() == "3.10.12",
              "package_versions": all(value["match"] for value in packages.values()),
              "known_native_source_revisions": all(row.get("source_origin", {}).get("git_commit_matches") is not False
                                                   for row in packages.values()),
              "nvcc_12_8_61": tools["nvcc"].get("returncode") == 0 and "V12.8.61" in tools["nvcc"].get("output", "")}
    if task in ("s2g", "m2d"):
        checks["audio_executables"] = all(tools[name].get("returncode") == 0 for name in ("ffmpeg", "ffprobe"))
    if native or cuda:
        checks["native_imports"] = all(value["imported"] for value in imports.values())
        checks["native_source_hashes"] = all(value["match"] for value in sources.values())
    if cuda:
        checks["cuda_tensor_probe"] = cuda_result["tensor_probe_passed"] is True
    return {"task": task, "status": "MATCH" if all(checks.values()) else "NEEDS_REVIEW",
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "checks": checks, "packages": packages, "executables": tools, "native_imports": imports,
            "native_source_checks": sources, "cuda": cuda_result,
            "model_or_dataset_inference": False, "installation_performed": False,
            "limitations": "Metadata/import/tiny CUDA checks do not validate training, JVP correctness, assets or paper metrics."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("t2m", "s2g", "m2d"), required=True)
    parser.add_argument("--native", action="store_true", help="Import native providers; their imports may initialize a GPU driver.")
    parser.add_argument("--cuda", action="store_true", help="Also allocate/synchronize one CUDA tensor on the visible device; implies --native.")
    args = parser.parse_args(argv)
    report = report_environment(args.task, args.native, args.cuda)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
