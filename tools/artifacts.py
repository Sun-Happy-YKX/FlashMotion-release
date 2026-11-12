"""Strict local artifact manifests and shared, lazy HF integration."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO, Iterator

HF_ENDPOINT = "https://huggingface.co"
SHA256_RE = re.compile(r"[0-9a-fA-F]{64}\Z")
REVISION_RE = re.compile(r"[0-9a-fA-F]{40}\Z")
REPO_PART_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")
CODE_ROOT = Path(__file__).resolve().parents[1]


class ArtifactError(Exception):
    """An intentionally credential-free error, optionally carrying a receipt."""

    def __init__(self, message: str, *, code: str = "artifact_error", receipt=None):
        super().__init__(message)
        self.code = code
        self.receipt = receipt


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    local_path: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class ArtifactManifest:
    repo_id: str
    repo_type: str
    stage_root: Path
    code_root: Path
    files: tuple[ArtifactFile, ...]
    private: bool = True


@dataclass
class HubBackend:
    api: Any
    operation_add: Any
    download: Any
    repository_not_found_error: type[Exception]


def fail(message: str, code: str) -> None:
    raise ArtifactError(message, code=code)


def relative_path(value: Any) -> str:
    """Portable relative POSIX paths only; reject Windows aliases and ADS."""
    if (
        not isinstance(value, str) or not value
        or any(char in value for char in '\\:<>"|?*')
    ):
        fail("Artifact paths must be nonempty relative POSIX paths.", "unsafe_path")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        fail("Control characters are forbidden in artifact paths.", "unsafe_path")
    if PurePosixPath(value).is_absolute() or PureWindowsPath(value).is_absolute():
        fail("Absolute artifact paths are forbidden.", "unsafe_path")
    parts = value.split("/")
    for part in parts:
        if part in ("", ".", "..") or part.endswith((".", " ")):
            fail("Empty, dot, parent, or aliased path components are forbidden.", "unsafe_path")
        stem = part.split(".", 1)[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(r"(COM|LPT)[1-9]", stem):
            fail("Windows device names are forbidden in artifact paths.", "unsafe_path")
        if part.casefold() in {".git", ".hg", ".svn"}:
            fail("Version-control internal paths are forbidden.", "unsafe_path")
    return value


def revision_hash(value: Any) -> str:
    if not isinstance(value, str) or not REVISION_RE.fullmatch(value):
        fail("A full immutable 40-hex commit revision is required.", "invalid_revision")
    return value.lower()


def within(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def reject_links(path: Path) -> None:
    """Inspect each existing component, including Windows junctions."""
    for current in reversed((path, *path.parents)):
        try:
            mode = current.lstat().st_mode
        except FileNotFoundError:
            continue
        is_junction = getattr(current, "is_junction", lambda: False)
        if stat.S_ISLNK(mode) or is_junction():
            fail("Symlinks and junctions are forbidden in artifact paths.", "symlink")


def absolute_path(value: Any) -> Path:
    if not isinstance(value, (str, Path)):
        fail("A local filesystem path is required.", "unsafe_path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        fail("Stage roots must be absolute and contain no parent traversal.", "unsafe_root")
    if path == Path(path.anchor):
        fail("A filesystem root cannot be used as an artifact stage.", "unsafe_root")
    relative_path(path.relative_to(path.anchor).as_posix())
    reject_links(path)
    return path.resolve()


def stage_path(value: Any, code_root: Path, *, allow_missing: bool) -> Path:
    root = absolute_path(value)
    if within(root, code_root):
        fail("The stage root must be outside the code repository.", "stage_inside_code")
    if root.exists() and not root.is_dir():
        fail("The stage root must be a directory.", "unsafe_root")
    if not allow_missing and not root.is_dir():
        fail("The publishing stage root does not exist.", "missing_stage")
    return root


def local_file(manifest: ArtifactManifest, entry: ArtifactFile, *, must_exist=True) -> Path:
    candidate = manifest.stage_root.joinpath(*entry.local_path.split("/"))
    reject_links(candidate)
    resolved = candidate.resolve()
    if not within(resolved, manifest.stage_root) or within(resolved, manifest.code_root):
        fail("A staged artifact escaped its allowed root.", "unsafe_path")
    if candidate.exists() and not stat.S_ISREG(candidate.stat().st_mode):
        fail("Artifacts must be regular files.", "not_regular")
    if must_exist and not candidate.is_file():
        fail("A staged artifact is missing.", "missing_file")
    return candidate


def digest_stream(stream: BinaryIO) -> tuple[str, int]:
    digest = hashlib.sha256()
    count = 0
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
        count += len(block)
    return digest.hexdigest(), count


def digest_file(path: Path) -> tuple[str, int]:
    with path.open("rb") as stream:
        return digest_stream(stream)


@contextmanager
def verified_open(manifest: ArtifactManifest, entry: ArtifactFile) -> Iterator[BinaryIO]:
    path = local_file(manifest, entry)
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        current = local_file(manifest, entry)
        opened_stat, current_stat = os.fstat(stream.fileno()), current.stat()
        if not stat.S_ISREG(opened_stat.st_mode) or (
            opened_stat.st_dev, opened_stat.st_ino
        ) != (current_stat.st_dev, current_stat.st_ino):
            fail("A staged file changed while being opened.", "source_changed")
        actual_sha, actual_bytes = digest_stream(stream)
        if actual_sha != entry.sha256 or actual_bytes != entry.bytes:
            fail("A staged artifact SHA or size does not match its manifest.", "local_digest_mismatch")
        stream.seek(0)
        yield stream


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            fail("Duplicate JSON keys are forbidden.", "invalid_manifest")
        result[key] = value
    return result


def load_manifest(
    filename: str | Path,
    *,
    code_root: Path | None = None,
    verify_local: bool = True,
    stage_root_override: str | Path | None = None,
) -> ArtifactManifest:
    manifest_file = Path(filename).absolute()
    if ".." in manifest_file.parts:
        fail("Manifest parent traversal is forbidden.", "unsafe_path")
    relative_path(manifest_file.relative_to(manifest_file.anchor).as_posix())
    reject_links(manifest_file)
    if not manifest_file.is_file():
        fail("The artifact manifest is not a regular file.", "invalid_manifest")
    try:
        data = json.loads(manifest_file.read_text(encoding="utf-8-sig"), object_pairs_hook=unique_object)
    except (ValueError, UnicodeError):
        fail("The artifact manifest is not valid UTF-8 JSON.", "invalid_manifest")
    expected = {"schema_version", "repo_id", "repo_type", "private", "files", "stage_root"}
    if not isinstance(data, dict) or set(data) != expected:
        fail("The manifest must contain exactly the documented schema fields.", "invalid_manifest")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1:
        fail("Only manifest schema_version 1 is supported.", "invalid_manifest")
    if type(data["private"]) is not bool:
        fail("Repository visibility must explicitly be a boolean.", "invalid_manifest")
    repo_id = data["repo_id"]
    if not isinstance(repo_id, str) or len(repo_id.split("/")) != 2:
        fail("repo_id must have the form owner/name.", "invalid_repo")
    if any(
        not REPO_PART_RE.fullmatch(part) or ".." in part or "--" in part
        or part.endswith((".", "-")) for part in repo_id.split("/")
    ):
        fail("repo_id is not a safe owner/name identifier.", "invalid_repo")
    if data["repo_type"] not in ("model", "dataset"):
        fail("Only model and dataset repositories are supported.", "invalid_repo")
    root = (code_root or CODE_ROOT).resolve()
    stage = stage_path(
        stage_root_override if stage_root_override is not None else data["stage_root"],
        root, allow_missing=not verify_local,
    )
    rows = data["files"]
    if not isinstance(rows, list) or not rows:
        fail("The manifest must contain at least one file.", "invalid_manifest")
    entries, remote_names, local_names = [], set(), set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "local_path", "sha256", "bytes"}:
            fail("Each file must contain exactly path/local_path/sha256/bytes.", "invalid_manifest")
        remote, local = relative_path(row["path"]), relative_path(row["local_path"])
        if remote.casefold() in remote_names or local.casefold() in local_names:
            fail("Duplicate or case-aliased artifact paths are forbidden.", "duplicate_path")
        remote_names.add(remote.casefold())
        local_names.add(local.casefold())
        if not isinstance(row["sha256"], str) or not SHA256_RE.fullmatch(row["sha256"]):
            fail("Each artifact needs a 64-hex SHA256.", "invalid_manifest")
        if type(row["bytes"]) is not int or row["bytes"] < 0:
            fail("Each artifact needs a nonnegative integer byte count.", "invalid_manifest")
        entries.append(ArtifactFile(remote, local, row["sha256"].lower(), row["bytes"]))
    manifest = ArtifactManifest(repo_id, data["repo_type"], stage, root, tuple(entries), data["private"])
    for entry in manifest.files:
        local_file(manifest, entry, must_exist=verify_local)
        if verify_local:
            with verified_open(manifest, entry):
                pass
    return manifest


def attr(value: Any, name: str, default=None):
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def hub_backend(*, token=True) -> HubBackend:
    """Import the official SDK only for live operations; never read/print tokens."""
    try:
        from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_download
        from huggingface_hub.errors import RepositoryNotFoundError
    except ImportError:
        fail("Install huggingface_hub for live artifact operations.", "missing_dependency")
    return HubBackend(
        HfApi(endpoint=HF_ENDPOINT, token=token),
        CommitOperationAdd, hf_hub_download, RepositoryNotFoundError,
    )


def require_owner(backend: HubBackend, manifest: ArtifactManifest) -> None:
    expected = manifest.repo_id.split("/", 1)[0]
    if attr(backend.api.whoami(), "name") != expected:
        fail("The authenticated account must exactly match the repository owner.", "owner_mismatch")


def require_private(info: Any) -> None:
    if attr(info, "private") is not True:
        fail("The remote repository is not confirmed private; no visibility change is permitted.", "public_repository")


def require_visibility(info: Any, manifest: ArtifactManifest) -> None:
    if manifest.private:
        require_private(info)
    elif attr(info, "private") is not False:
        fail("The remote repository is not confirmed public.", "visibility_mismatch")


def is_missing_repository(backend: HubBackend, error: Exception) -> bool:
    return (
        isinstance(error, backend.repository_not_found_error)
        and attr(attr(error, "response"), "status_code") == 404
    )


def remote_entries(info: Any) -> dict[str, Any]:
    return {attr(item, "rfilename"): item for item in (attr(info, "siblings") or [])}


def check_remote_metadata(entry: ArtifactFile, remote: dict[str, Any]) -> str | None:
    item = remote.get(entry.path)
    if item is None:
        fail("An expected artifact is missing from the remote revision.", "remote_missing_file")
    lfs = attr(item, "lfs")
    if lfs is not None:
        sha = attr(lfs, "sha256")
        size = attr(lfs, "size", attr(item, "size"))
        if not isinstance(sha, str) or sha.lower() != entry.sha256 or size != entry.bytes:
            fail("Remote LFS SHA or size does not match the manifest.", "remote_digest_mismatch")
        return "remote_lfs_sha256"
    size = attr(item, "size")
    if size is not None and size != entry.bytes:
        fail("Remote artifact size does not match the manifest.", "remote_digest_mismatch")
    return None


@contextmanager
def downloaded_verified(
    backend: HubBackend, manifest: ArtifactManifest, entry: ArtifactFile, revision: str
) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix=".hf-readback-", dir=manifest.stage_root) as temp:
        temporary_root = Path(temp).resolve()
        filename = backend.download(
            repo_id=manifest.repo_id, repo_type=manifest.repo_type,
            filename=entry.path, revision=revision, local_dir=temp,
            token=True if manifest.private else False, endpoint=HF_ENDPOINT,
        )
        downloaded = Path(filename).absolute()
        reject_links(downloaded)
        if not within(downloaded.resolve(), temporary_root) or not downloaded.is_file():
            fail("The HF download did not produce a safe regular file.", "unsafe_download")
        if digest_file(downloaded) != (entry.sha256, entry.bytes):
            fail("Downloaded artifact SHA or size does not match the manifest.", "remote_digest_mismatch")
        yield downloaded


def new_receipt(operation: str, manifest: ArtifactManifest) -> dict[str, Any]:
    return {
        "schema_version": 1, "operation": operation, "status": "in_progress",
        "repo_id": manifest.repo_id, "repo_type": manifest.repo_type,
        "private_required": manifest.private, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "revision": None, "commits": [], "verified_files": [],
    }


def dry_run_plan(operation: str, manifest: ArtifactManifest, revision=None) -> dict[str, Any]:
    return {
        "status": "dry_run", "operation": operation, "repo_id": manifest.repo_id,
        "repo_type": manifest.repo_type, "private_required": manifest.private,
        "file_count": len(manifest.files), "stage_root": str(manifest.stage_root),
        "revision": revision, "authenticated": False,
        "remote_verified": False, "writes_performed": False,
    }


def receipt_path(value: str | Path) -> Path:
    path = Path(value).absolute()
    if ".." in path.parts:
        fail("Receipt parent traversal is forbidden.", "unsafe_receipt")
    relative_path(path.relative_to(path.anchor).as_posix())
    reject_links(path)
    if not path.parent.is_dir() or path.exists():
        fail("The receipt needs an existing safe parent and a new filename.", "unsafe_receipt")
    return path


def write_receipt(filename: str | Path, receipt: dict[str, Any]) -> None:
    path = receipt_path(filename)
    payload = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
