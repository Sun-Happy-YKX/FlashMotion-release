"""Offline safety tests. No HF authentication, repository creation, or upload."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from tools import artifacts as a
from tools import fetch_hf


INITIAL = "1" * 40
COMMITTED = "2" * 40


class MissingRepo(Exception):
    def __init__(self, status=404):
        self.response = SimpleNamespace(status_code=status)


class FakeAdd:
    def __init__(self, *, path_in_repo, path_or_fileobj):
        self.path = path_in_repo
        self.stream = path_or_fileobj


class FakeApi:
    def __init__(self, *, private=True, owner="owner", missing=False, lfs=True):
        self.private, self.owner, self.missing, self.lfs = private, owner, missing, lfs
        self.current_sha = INITIAL
        self.content = {}
        self.create_calls = []
        self.commit_calls = []
        self.info_calls = []
        self.whoami_calls = 0
        self.metadata_sha_override = None
        self.fail_commit = False
        self.flip_public_after_create = False

    def whoami(self):
        self.whoami_calls += 1
        return {"name": self.owner}

    def create_repo(self, **kwargs):
        self.create_calls.append(kwargs)
        self.missing = False
        if self.flip_public_after_create:
            self.private = False

    def repo_info(self, **kwargs):
        self.info_calls.append(kwargs)
        if self.missing:
            raise MissingRepo()
        siblings = []
        for name, data in self.content.items():
            lfs = None
            if self.lfs:
                lfs = {
                    "sha256": self.metadata_sha_override or hashlib.sha256(data).hexdigest(),
                    "size": len(data),
                }
            siblings.append(SimpleNamespace(rfilename=name, size=len(data), lfs=lfs))
        return SimpleNamespace(private=self.private, sha=self.current_sha, siblings=siblings)

    def create_commit(self, **kwargs):
        self.commit_calls.append(kwargs)
        if self.fail_commit:
            raise RuntimeError("SENSITIVE_TEST_SENTINEL")
        for operation in kwargs["operations"]:
            self.content[operation.path] = operation.stream.read()
        self.current_sha = COMMITTED
        return SimpleNamespace(oid=COMMITTED)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="artifact-tests-")
        self.root = Path(self.temp.name).resolve()
        self.code = self.root / "code"
        self.stage = self.root / "stage"
        self.code.mkdir()
        (self.stage / "weights").mkdir(parents=True)
        self.payload = b"verified dummy checkpoint"
        (self.stage / "weights" / "one.bin").write_bytes(self.payload)
        self.data = {
            "schema_version": 1, "repo_id": "owner/archive", "repo_type": "model",
            "private": True, "stage_root": str(self.stage),
            "files": [{
                "path": "checkpoints/one.bin", "local_path": "weights/one.bin",
                "sha256": hashlib.sha256(self.payload).hexdigest(), "bytes": len(self.payload),
            }],
        }
        self.filename = self.root / "manifest.json"
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        self.filename.write_text(json.dumps(self.data), encoding="utf-8")

    def backend(self, api=None, *, bad_download=False):
        api = api or FakeApi()

        def download(**kwargs):
            target = Path(kwargs["local_dir"]).joinpath(*kwargs["filename"].split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            payload = b"incorrect bytes" if bad_download else api.content[kwargs["filename"]]
            target.write_bytes(payload)
            return str(target)

        return a.HubBackend(api, FakeAdd, Mock(side_effect=download), MissingRepo)


    def test_valid_local_manifest(self):
        manifest = a.load_manifest(self.filename, code_root=self.code)
        self.assertEqual(manifest.files[0].bytes, len(self.payload))
        self.assertEqual(manifest.repo_type, "model")




    def test_paths_reject_traversal_absolute_and_windows_aliases(self):
        for value in ("../a", "x/../a", "/a", "C:/a", r"x\a", "x//a", "./a", "NUL.bin", "a.", "a:stream", "a?b", "a*b", ".git/config"):
            for field in ("path", "local_path"):
                with self.subTest(value=value, field=field):
                    data = copy.deepcopy(self.data)
                    data["files"][0][field] = value
                    self.filename.write_text(json.dumps(data), encoding="utf-8")
                    with self.assertRaises(a.ArtifactError):
                        a.load_manifest(self.filename, code_root=self.code)

















    def test_receipt_never_overwrites_existing_file(self):
        receipt = self.root / "receipt.json"
        receipt.write_text("preserve", encoding="utf-8")
        with self.assertRaises(a.ArtifactError):
            a.write_receipt(receipt, {"status": "verified"})
        self.assertEqual(receipt.read_text(), "preserve")

    def test_receipt_rejects_windows_alternate_stream(self):
        with self.assertRaises(a.ArtifactError):
            a.receipt_path(self.root / "receipt.json:stream")


    def test_fetch_rejects_mutable_revision_before_auth(self):
        for revision in ("main", "v1", "a" * 39, "g" * 40):
            with self.subTest(revision=revision):
                with patch.object(a, "hub_backend", side_effect=AssertionError("network")):
                    with self.assertRaises(a.ArtifactError):
                        fetch_hf.fetch_manifest(self.filename, revision, code_root=self.code)

    def test_fetch_dry_run_missing_root_is_not_created(self):
        target = self.root / "new" / "stage"
        before = {str(path) for path in self.root.rglob("*")}
        with patch.object(a, "hub_backend", side_effect=AssertionError("network")):
            result = fetch_hf.fetch_manifest(
                self.filename, COMMITTED, stage_root=target, dry_run=True, code_root=self.code
            )
        self.assertEqual(result["status"], "dry_run")
        self.assertFalse(target.exists())
        self.assertEqual(before, {str(path) for path in self.root.rglob("*")})

    def test_fetch_checks_download_and_immutable_revision(self):
        api = FakeApi()
        api.current_sha = COMMITTED
        api.content["checkpoints/one.bin"] = self.payload
        backend = self.backend(api)
        target = self.root / "fetched"
        receipt = fetch_hf.fetch_manifest(
            self.filename, COMMITTED, stage_root=target, backend=backend, code_root=self.code
        )
        self.assertEqual(receipt["status"], "verified")
        self.assertEqual((target / "weights" / "one.bin").read_bytes(), self.payload)
        self.assertEqual(backend.download.call_args.kwargs["revision"], COMMITTED)
        self.assertFalse(api.create_calls)
        self.assertFalse(api.commit_calls)

    def test_fetch_refuses_public_repo_without_local_write(self):
        target = self.root / "not-created"
        with self.assertRaises(a.ArtifactError):
            fetch_hf.fetch_manifest(
                self.filename, COMMITTED, stage_root=target,
                backend=self.backend(FakeApi(private=False)), code_root=self.code,
            )
        self.assertFalse(target.exists())

    def test_fetch_refuses_wrong_existing_bytes_before_network(self):
        target = self.root / "existing"
        (target / "weights").mkdir(parents=True)
        (target / "weights" / "one.bin").write_bytes(b"preserve this file")
        with patch.object(a, "hub_backend", side_effect=AssertionError("network")):
            with self.assertRaises(a.ArtifactError):
                fetch_hf.fetch_manifest(
                    self.filename, COMMITTED, stage_root=target, code_root=self.code
                )
        self.assertEqual((target / "weights" / "one.bin").read_bytes(), b"preserve this file")

    def test_fetch_bad_download_does_not_publish_unverified_target(self):
        api = FakeApi()
        api.current_sha = COMMITTED
        api.content["checkpoints/one.bin"] = self.payload
        target = self.root / "bad-fetch"
        with self.assertRaises(a.ArtifactError) as caught:
            fetch_hf.fetch_manifest(
                self.filename, COMMITTED, stage_root=target,
                backend=self.backend(api, bad_download=True), code_root=self.code,
            )
        self.assertEqual(caught.exception.receipt["status"], "failed")
        self.assertFalse((target / "weights" / "one.bin").exists())

    def test_fetch_rejects_revision_mismatch(self):
        api = FakeApi()
        api.content["checkpoints/one.bin"] = self.payload
        with self.assertRaises(a.ArtifactError):
            fetch_hf.fetch_manifest(
                self.filename, COMMITTED, stage_root=self.root / "revision-mismatch",
                backend=self.backend(api), code_root=self.code,
            )


if __name__ == "__main__":
    unittest.main()
