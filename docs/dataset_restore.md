# Restore private dataset packages

Restore is a byte-only operation. It does not load pickle/NPZ contents, import
models, start training/inference, or contact Hugging Face. Dataset access and
license permissions still apply; keep the FineDance license and written private
distribution scope. Restoring files does not authorize public redistribution.

First fetch the allowlisted files from `yangkaixing/FlashMotion-data` (dataset)
with `tools/fetch_hf.py`, using the full immutable 40-hex repository revision and
the complete `manifests/data_download_complete.json` artifact manifest. Keep its
verified fetch receipt. Current immutable revisions are in
[verified downloads](published_artifacts.md). Do not use
an unverified branch download or invent a member-manifest SHA.

The actual pack format is `flashmotion.dataset_tar_members.v1`:

```json
{
  "schema": "flashmotion.dataset_tar_members.v1",
  "archive_parts": ["m2d/canonical-stable-v2-part-001.tar"],
  "members": [
    {
      "path": "data/dataset_backups_stable_v2/test.pkl",
      "bytes": 123,
      "sha256": "<actual 64 lowercase hex SHA from the authenticated manifest>",
      "source": "/original/read-only/source/path"
    }
  ]
}
```

This is a structural illustration, not a usable manifest. `archive_parts` is a
list of strings relative to the fetched stage, not a list of file objects.
`source` is provenance only; the restore tool never opens it. One members JSON
covers **all** shards of one package. Fetch every shard it names.

Run with Python 3.10+ and stdlib only; replace all placeholders with verified
paths/hashes. Stage, manifest, and output must be absolute non-root paths. The
output parent must already exist, but the output directory must not exist.
Choose a destination outside the code checkout, the stage, and all inputs.

```text
python -B tools/restore_data.py --stage-root /safe/fetched-stage --manifest /safe/fetched-stage/m2d/canonical-stable-v2-members.json --manifest-sha256 VERIFIED_MEMBERS_JSON_SHA256 --output /safe/restored-canonical --dry-run
python -B tools/restore_data.py --stage-root /safe/fetched-stage --manifest /safe/fetched-stage/m2d/canonical-stable-v2-members.json --manifest-sha256 VERIFIED_MEMBERS_JSON_SHA256 --output /safe/restored-canonical --execute
```

The first command fully reads and verifies every member without writing. The
second repeats preflight and creates a new exclusive directory; it never resumes
or overwrites a prior result. Ensure free space for the payload plus at least
1 GiB. Verification reads entire shards repeatedly, so allow extra I/O time.

Only regular, non-sparse tar members are allowed. Directory entries, symlinks,
hardlinks, device/FIFO entries, traversal, ADS, Windows reserved names, trailing
dot/space aliases, case aliases (including directory names), file/directory
prefix collisions, duplicates, receipt namespace collisions, and incomplete
coverage are rejected. Existing symlinks, junctions, and Windows reparse points
in any input/output ancestry are rejected. Archive shard identity aliases are
also rejected.

Execution retains the same open input descriptors across both archive passes.
It pins entire-shard SHA, file identity and metadata, rechecks them before output
and after extraction, verifies exact member SHA/size/coverage and duplicates in
both passes, hashes each written file again, and verifies the complete output
file inventory/SHA once more before the success receipt. The pinned observed shard SHA
is a mutation guard; the authenticated member-manifest SHA plus the immutable
fetch manifest/receipt are the authenticity chain. Keep input/output parents
private to this operation; a concurrently hostile filesystem is not a sandbox.

Success returns exit code 0 and `EXTRACTED_AND_SHA_VERIFIED` in the new root's
`restoration_receipt.json`. A dry run returns `VERIFIED_NOT_EXTRACTED`, which is
not a restored dataset. Failures return nonzero with a truthful JSON diagnostic:
`FAILED_BEFORE_OUTPUT` or `FAILED_PARTIAL_RETAINED`. Partial files are retained,
with per-file confirmed `bytes_written` (a lower bound if a write fails), safely
observed file sizes when available, and `sha_verified`; they must not be used as a
complete dataset. A failure receipt is attempted only in the tool's own new
root. If writing it fails or its name already exists, the tool does not overwrite
it: the stdout diagnostic reports `receipt_written: false` and the receipt error.
Preserve that diagnostic too. Never retry into the partial root.

## Merge verified packages into a new task root

Restore each package into its own exclusive directory first. Their payload
layouts are intended to be merged, but their top-level restoration receipts have
the same filename and **must not be merged under that filename**.

| Package | Verified payload layout |
| --- | --- |
| T2M `babel-streamed` | `raw_data/BABEL_streamed/…` |
| T2M `humanml3d` | `raw_data/HumanML3D/…` |
| T2M `text-features` | `data/umt5_xxl_sentence_features/…` |
| T2M `cohort311` | `cohort311/…` |
| M2D `canonical-stable-v2` | `data/dataset_backups_stable_v2/…` |
| M2D `finedance-source` | `data_source/…` |
| S2G `beat2-selected-raw` | `raw/beat_english_v2.0.0/…` |
| S2G `cache-exact` | `cache/…`; separately owner-permitted, SHA/size verified and included in the complete data index |

A safe merge is a **byte-copy into a newly created exclusive root**, not an
overwrite/update command. Validate the union of all package member paths before
copying: reject duplicate/case-alias paths and file/directory prefix collisions
even if duplicate bytes happen to match. Shared exact-spelling directories are
allowed, existing files are not. Reject any input/destination link or reparse
point. Copy regular files with exclusive destination creation (`xb`/`O_EXCL`),
not hardlinks or symlinks, then check size and SHA against the package manifests.
Retain each original package receipt under a separate package-specific provenance
location outside the payload namespace and write a new merge receipt. Never use
`cp -f`, an overwriting `copytree`, or a tool that silently follows symlinks.
The restore utility itself does not perform or claim verification of this merge.

For M2D, merging the two packages yields the canonical test cache under
`data/dataset_backups_stable_v2` and raw FineDance under `data_source`. The native
corrected evaluator requires raw test files at **`data/test`**. Make a separate
byte-preserving, exclusive copy of the verified `data_source/test` regular files
to a new `data/test`, and verify every copied size/SHA. Do not replace this with a
symlink/junction to the source, the historical FlowerDance project, or another
run. The bounded quality scorer needs only its ordered test cache and the
allowlisted raw test inputs; training input requirements are separate. Set the
task's portable data-root config to the newly assembled root and use its normal
preflight/runtime proof rather than bypassing checks.

Offline adversarial tests:

```text
python -B -m unittest discover -s tests -p test_restore_data.py -v
```
