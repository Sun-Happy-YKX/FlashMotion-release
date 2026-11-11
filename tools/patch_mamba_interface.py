"""Inspect, or explicitly restore, the recorded Mamba interface in a new task env."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ASSET = ROOT / "third_party/native_patches/selective_scan_interface.observed.py"
COMMIT = "10b5d6358f27966f6a40e4bf0baa17a460688128"
UPSTREAM_SHA = "543edd077f11c80cccd5c5e2f926420d9105fd238d9e8960e53660cfa234dce8"
OBSERVED_SHA = "2260a1486cea11fff63ba69d584175864f2687354b722fa9c16820de6a45158f"
OLD = b"        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t(), out=dconv1d_out)\n"
NEW = (b"        # Avoid the out= variant here. MeanFlow computes a JVP through the\n"
       b"        # network, which requires differentiating this backward expression.\n"
       b"        # PyTorch forbids autograd through addmm(..., out=...), while the\n"
       b"        # functionally equivalent out-of-place form supports higher-order AD.\n"
       b"        dconv1d_out = torch.addmm(dconv1d_out, x_proj_weight.t(), dx_dbl.t())\n")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def restored_bytes(original):
    recorded = ASSET.read_bytes()
    if len(recorded) != 20311 or sha256(recorded) != OBSERVED_SHA:
        raise ValueError("Recorded provider asset does not match its byte provenance")
    if sha256(original) == OBSERVED_SHA:
        return original
    if len(original) != 20025 or sha256(original) != UPSTREAM_SHA or original.count(OLD) != 1:
        raise ValueError("Refusing to patch an unknown Mamba interface")
    repaired = original.replace(OLD, NEW, 1)
    if repaired != recorded:
        raise ValueError("Minimal patch does not reproduce the recorded provider bytes")
    return repaired


def patch_file(path, prefix, apply=False):
    """Hash-gate a single prefix-contained provider file; no imports or installs."""
    if path.is_symlink() or prefix.resolve() == Path(prefix.anchor):
        raise ValueError("Unsafe target/prefix")
    target = path.resolve(strict=True)
    if not target.is_relative_to(prefix.resolve()) or target.parts[-3:] != (
            "mamba_ssm", "ops", "selective_scan_interface.py"):
        raise ValueError("Provider target must stay inside the selected task environment")
    original = target.read_bytes()
    repaired = restored_bytes(original)
    report = {"status": "ALREADY_RECORDED" if original == repaired else "READY",
              "target": str(target), "before_sha256": sha256(original),
              "after_sha256": sha256(repaired), "applied": False,
              "native_imports": False, "installation_performed": False}
    if not apply or original == repaired:
        return report
    backup = target.with_name(target.name + ".flashmotion-upstream.bak")
    if backup.exists():
        if backup.is_symlink() or backup.read_bytes() != original:
            raise ValueError("Existing provider backup does not match upstream bytes")
    else:
        with backup.open("xb") as handle:
            handle.write(original)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".flashmotion-interface-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(repaired)
        os.chmod(temporary, target.stat().st_mode & 0o777)
        if target.is_symlink() or target.read_bytes() != original:
            raise ValueError("Provider changed during patch preparation")
        os.replace(temporary, target)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    if target.read_bytes() != repaired:
        raise ValueError("Provider post-write verification failed; upstream backup retained")
    report.update(status="RESTORED", applied=True, backup=str(backup))
    return report


def installed_target(task):
    prefix_text = os.environ.get("CONDA_PREFIX")
    if not prefix_text or not sys.platform.startswith("linux"):
        raise ValueError("Activate the newly created Linux task conda environment first")
    prefix = Path(prefix_text).resolve(strict=True)
    if (prefix != Path(sys.prefix).resolve() or prefix.name != f"flashmotion-{task}"
            or not (prefix / "conda-meta/history").is_file()):
        raise ValueError("Use the newly created, activated flashmotion task conda prefix")
    distribution = metadata.distribution("mamba-ssm")
    if distribution.version != "2.2.6.post3":
        raise ValueError("Mamba distribution version does not match the pinned provider")
    origin = json.loads(distribution.read_text("direct_url.json") or "{}")
    vcs = origin.get("vcs_info", {})
    if vcs.get("vcs") != "git" or vcs.get("commit_id") != COMMIT:
        raise ValueError("Install the pinned Mamba Git revision into the new prefix before patching")
    return Path(distribution.locate_file("mamba_ssm/ops/selective_scan_interface.py")), prefix


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("t2m", "s2g", "m2d"), required=True)
    parser.add_argument("--apply", action="store_true", help="Explicitly write the hash-gated repair and retain an upstream backup.")
    args = parser.parse_args(argv)
    try:
        target, prefix = installed_target(args.task)
        report = patch_file(target, prefix, args.apply)
    except (OSError, ValueError, metadata.PackageNotFoundError) as error:
        print(json.dumps({"status": "REFUSED", "error": type(error).__name__, "message": str(error)}))
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
