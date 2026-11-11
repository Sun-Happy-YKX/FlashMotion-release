"""Fetch SHA-pinned artifacts from an immutable private HF commit."""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import json
import os
import shutil

if __package__:
    from . import artifacts as a
else:
    import artifacts as a


def fetch_manifest(
    filename, revision, *, stage_root=None, dry_run=False, backend=None, code_root=None
):
    revision = a.revision_hash(revision)
    manifest = a.load_manifest(
        filename, code_root=code_root, verify_local=False, stage_root_override=stage_root
    )
    existing = set()
    # A corrupt existing destination is rejected before any network operation.
    for entry in manifest.files:
        if a.local_file(manifest, entry, must_exist=False).exists():
            with a.verified_open(manifest, entry):
                pass
            existing.add(entry.path)
    if dry_run:
        return a.dry_run_plan("fetch", manifest, revision)
    receipt = a.new_receipt("fetch", manifest)
    receipt["revision"] = revision
    receipt["partial_files"] = []
    phase = "authentication"
    try:
        backend = backend or a.hub_backend()
        a.require_owner(backend, manifest)
        phase = "private_revision_check"
        info = backend.api.repo_info(
            repo_id=manifest.repo_id, repo_type=manifest.repo_type,
            revision=revision, files_metadata=True,
        )
        a.require_private(info)
        if a.revision_hash(a.attr(info, "sha")) != revision:
            a.fail("The remote revision does not match the requested commit.", "revision_mismatch")
        remote = a.remote_entries(info)
        for entry in manifest.files:
            a.check_remote_metadata(entry, remote)
        phase = "local_destination_creation"
        a.reject_links(manifest.stage_root)
        manifest.stage_root.mkdir(parents=True, exist_ok=True)
        a.reject_links(manifest.stage_root)
        for entry in manifest.files:
            phase = "download"
            if entry.path in existing:
                # Recheck paths and bytes after the network preflight.
                with a.verified_open(manifest, entry):
                    pass
                method = "existing_local_sha256"
            else:
                with a.downloaded_verified(backend, manifest, entry, revision) as downloaded:
                    target = a.local_file(manifest, entry, must_exist=False)
                    a.reject_links(target.parent)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target = a.local_file(manifest, entry, must_exist=False)
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
                    descriptor = os.open(target, flags, 0o600)
                    receipt["partial_files"].append(entry.local_path)
                    with os.fdopen(descriptor, "wb") as output, downloaded.open("rb") as source:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
                        output.flush()
                        os.fsync(output.fileno())
                    with a.verified_open(manifest, entry):
                        pass
                    receipt["partial_files"].remove(entry.local_path)
                method = "download_sha256"
            receipt["verified_files"].append({
                "path": entry.path, "local_path": entry.local_path,
                "sha256": entry.sha256, "bytes": entry.bytes, "method": method,
            })
        phase = "final_private_check"
        a.require_private(backend.api.repo_info(
            repo_id=manifest.repo_id, repo_type=manifest.repo_type, revision=revision
        ))
        receipt["status"] = "verified"
        return receipt
    except Exception as error:
        receipt.update(
            status="failed", failure_phase=phase, error_type=type(error).__name__,
            error_code=error.code if isinstance(error, a.ArtifactError) else "hub_operation_failed",
        )
        raise a.ArtifactError(
            "Fetching was not fully verified; preserve the failure receipt.",
            code="fetch_failed", receipt=receipt,
        ) from None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest")
    parser.add_argument("--revision", required=True, help="Full immutable 40-hex commit SHA; branches/tags are rejected.")
    parser.add_argument("--stage-root", help="Optional safe absolute output root outside the code repository.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receipt", help="Required for live mode; new JSON file in an existing safe directory.")
    args = parser.parse_args(argv)
    if not args.dry_run and not args.receipt:
        parser.error("--receipt is required for live fetching")
    destination = None
    try:
        if not args.dry_run:
            destination = a.receipt_path(args.receipt)
        result = fetch_manifest(
            args.manifest, args.revision, stage_root=args.stage_root, dry_run=args.dry_run
        )
    except a.ArtifactError as error:
        if destination is not None:
            failure = error.receipt or {
                "schema_version": 1, "operation": "fetch", "status": "failed",
                "error_code": error.code, "remote_verified": False,
            }
            try:
                a.write_receipt(destination, failure)
            except (OSError, a.ArtifactError):
                print("Fetching failed and the failure receipt could not be written.", file=sys.stderr)
                return 1
        print(str(error), file=sys.stderr)
        return 1
    except Exception as error:
        print(f"Fetching failed ({type(error).__name__}); no success is claimed.", file=sys.stderr)
        return 1
    if args.dry_run:
        print(json.dumps(result, sort_keys=True))
        return 0
    try:
        a.write_receipt(destination, result)
    except (OSError, a.ArtifactError):
        print("Artifact verification completed, but writing its receipt failed; no receipt success is claimed.", file=sys.stderr)
        return 1
    print(f"Verified {len(result['verified_files'])} files from {result['repo_id']}@{result['revision']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
