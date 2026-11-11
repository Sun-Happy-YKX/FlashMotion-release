"""Verify dataset tar shards and optionally restore into an exclusive new root.

Only stdlib byte reads are used: no pickle/NPZ deserialization or model imports.
Default is a write-free dry run. Fetch authenticated files at an immutable HF
revision first; the required member-manifest SHA is the local trust anchor.
"""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tarfile


SCHEMA = 'flashmotion.dataset_tar_members.v1'
RECEIPT_NAME = 'restoration_receipt.json'
BLOCK = 1024 * 1024
HEX64 = re.compile(r'[0-9a-f]{64}\Z')
DEVICES = {'con', 'prn', 'aux', 'nul', 'conin$', 'conout$'} | {
    prefix + suffix for prefix in ('com', 'lpt') for suffix in '123456789¹²³'
}


class RestoreError(ValueError):
    """Safe diagnostic with a truthful in-memory receipt, including failures."""
    def __init__(self, code, receipt=None):
        super().__init__(code)
        self.code = code
        self.receipt = receipt


def relative(value):
    """Reject non-portable paths before PurePath can normalize aliases away."""
    if not isinstance(value, str) or not value:
        raise RestoreError('unsafe_relative_path')
    if any(c in '\\:<>"|?*' or ord(c) < 32 or ord(c) == 127 for c in value):
        raise RestoreError('unsafe_relative_path')
    components = value.split('/')
    if any(p in ('', '.', '..') or p.endswith(('.', ' ')) for p in components):
        raise RestoreError('unsafe_relative_path')
    if any(p.split('.', 1)[0].rstrip(' .').casefold() in DEVICES for p in components):
        raise RestoreError('windows_reserved_path')
    path = PurePosixPath(value)
    if path.is_absolute():
        raise RestoreError('unsafe_relative_path')
    return path


def no_links(path):
    """Reject symlinks and *all* Windows reparse points, including junctions."""
    for item in (Path(path), *Path(path).parents):
        try:
            info = item.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, 'st_file_attributes', 0)
            & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0x400)
        ) or getattr(item, 'is_junction', lambda: False)():
            raise RestoreError('symlink_junction_or_reparse_point')


def absolute(value):
    path = Path(value)
    if not path.is_absolute() or '..' in path.parts or path == path.parent:
        raise RestoreError('absolute_nonroot_path_required')
    relative('/'.join(path.parts[1:]))
    no_links(path)
    return path.resolve(strict=False)


def _overlap(one, two):
    # Conservative on every OS: portable restoration must not rely on casing.
    a, b = tuple(p.casefold() for p in one.parts), tuple(p.casefold() for p in two.parts)
    return a[:len(b)] == b or b[:len(a)] == a


def _sha(value):
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise RestoreError('invalid_sha256')
    return value


def _unique_paths(names, reserve_receipt=False):
    """Check exact/case duplicates, directory aliases, and file/prefix collisions."""
    trie = {}
    for name in names:
        pieces = relative(name).parts
        if reserve_receipt and pieces[0].casefold() == RECEIPT_NAME.casefold():
            raise RestoreError('receipt_namespace_collision')
        node = trie
        for piece in pieces:
            if None in node:
                raise RestoreError('file_directory_prefix_collision')
            key = piece.casefold()
            if key in node:
                spelling, child = node[key]
                if spelling != piece:
                    raise RestoreError('case_alias_collision')
            else:
                child = {}
                node[key] = (piece, child)
            node = child
        if node:
            raise RestoreError('duplicate_or_file_directory_prefix_collision')
        node[None] = True


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise RestoreError('duplicate_json_key')
        result[key] = value
    return result


def _identity(info):
    # Windows Python versions can expose different ctime meanings through
    # stat(path) and fstat(fd). Pin ctime separately per API, never compare the
    # two API values to each other or discard modification/SHA checks.
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


def _stream_digest(handle):
    handle.seek(0)
    result = hashlib.sha256()
    for block in iter(lambda: handle.read(BLOCK), b''):
        result.update(block)
    handle.seek(0)
    return result.hexdigest()


def digest(path):
    with ExitStack() as stack:
        pinned = _pin(Path(path), stack)
        return pinned.sha256


@dataclass
class PinnedFile:
    path: Path
    handle: object
    identity: tuple
    sha256: str
    fd_ctime_ns: int
    path_ctime_ns: int

    def check(self, rehash=True):
        no_links(self.path)
        info = self.path.stat()
        fd_info = os.fstat(self.handle.fileno())
        if (_identity(info) != self.identity or _identity(fd_info) != self.identity
                or fd_info.st_ctime_ns != self.fd_ctime_ns or info.st_ctime_ns != self.path_ctime_ns):
            raise RestoreError('input_identity_or_metadata_changed')
        if rehash and _stream_digest(self.handle) != self.sha256:
            raise RestoreError('input_sha_changed')
        fd_info, info = os.fstat(self.handle.fileno()), self.path.stat()
        if (_identity(fd_info) != self.identity or _identity(info) != self.identity
                or fd_info.st_ctime_ns != self.fd_ctime_ns or info.st_ctime_ns != self.path_ctime_ns):
            raise RestoreError('input_changed_during_hash')


def _pin(path, stack, expected=None):
    no_links(path)
    flags = os.O_RDONLY | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    try:
        handle = os.fdopen(descriptor, 'rb')
    except BaseException:
        os.close(descriptor)
        raise
    stack.enter_context(handle)
    info = os.fstat(handle.fileno())
    if not stat.S_ISREG(info.st_mode):
        raise RestoreError('input_not_regular_file')
    path_info = path.stat()
    if _identity(path_info) != _identity(info):
        raise RestoreError('input_identity_or_metadata_changed')
    pinned = PinnedFile(path, handle, _identity(info), _stream_digest(handle),
                        info.st_ctime_ns, path_info.st_ctime_ns)
    pinned.check(rehash=False)
    if expected is not None and pinned.sha256 != expected:
        raise RestoreError('member_manifest_sha_mismatch')
    return pinned


def _plan(stage, manifest, expected_sha, stack):
    pinned_manifest = _pin(manifest, stack, _sha(expected_sha))
    try:
        value = json.loads(pinned_manifest.handle.read().decode('utf-8'), object_pairs_hook=_object)
    except (UnicodeError, json.JSONDecodeError):
        raise RestoreError('invalid_member_manifest_json') from None
    pinned_manifest.check()
    if not isinstance(value, dict) or set(value) != {'schema', 'archive_parts', 'members'} or value['schema'] != SCHEMA:
        raise RestoreError('unexpected_member_manifest_schema')
    if not isinstance(value['members'], list) or not value['members']:
        raise RestoreError('nonempty_member_list_required')
    members, names = {}, []
    for row in value['members']:
        if not isinstance(row, dict) or set(row) != {'path', 'bytes', 'sha256', 'source'}:
            raise RestoreError('unexpected_member_row_schema')
        name = str(relative(row['path']))
        if type(row['bytes']) is not int or row['bytes'] < 0:
            raise RestoreError('invalid_member_bytes')
        _sha(row['sha256'])
        # Source is provenance only: never opened or used as a destination.
        if not isinstance(row['source'], str) or not row['source']:
            raise RestoreError('invalid_member_source')
        names.append(name)
        members[name] = row
    _unique_paths(names, reserve_receipt=True)
    names = value['archive_parts']
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise RestoreError('archive_parts_must_be_nonempty_string_list')
    _unique_paths(names)
    parts = [_pin(stage.joinpath(*relative(n).parts), stack) for n in names]
    identities = {(p.identity[0], p.identity[1]) for p in parts}
    if len(identities) != len(parts) or (pinned_manifest.identity[0], pinned_manifest.identity[1]) in identities:
        raise RestoreError('input_file_identity_alias')
    plan = dict(manifest=pinned_manifest, parts=parts, part_names=names, members=members)
    _verify_archive_pass(plan)
    _check_pins(plan)
    return plan


def _check_pins(plan):
    for pinned in (plan['manifest'], *plan['parts']):
        pinned.check()


def _output_check(root, identity):
    no_links(root)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or (info.st_dev, info.st_ino) != identity:
        raise RestoreError('output_root_identity_changed')


def _parents(root, pieces, identity):
    _output_check(root, identity)
    current = root
    for piece in pieces:
        current = current / piece
        no_links(current)
        try:
            current.mkdir()
        except FileExistsError:
            if not current.is_dir():
                raise RestoreError('output_parent_not_directory') from None
        no_links(current)
    _output_check(root, identity)


def _exclusive_file(path):
    no_links(path)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0) | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        return os.fdopen(descriptor, 'wb')
    except BaseException:
        os.close(descriptor)
        raise


def _verify_archive_pass(plan, root=None, root_identity=None, files_written=None):
    seen = set()
    for pinned in plan['parts']:
        pinned.check(rehash=False)
        pinned.handle.seek(0)
        # Read past zero blocks so appended duplicate entries are not hidden
        # behind the first tar end marker.
        with tarfile.open(fileobj=pinned.handle, mode='r:', ignore_zeros=True) as archive:
            for entry in archive:
                name = str(relative(entry.name))
                row = plan['members'].get(name)
                if entry.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or entry.sparse is not None or entry.linkname:
                    raise RestoreError('tar_nonregular_or_sparse_member')
                if name in seen or row is None or entry.size != row['bytes']:
                    raise RestoreError('tar_duplicate_unapproved_or_size_mismatch')
                source = archive.extractfile(entry)
                if source is None:
                    raise RestoreError('tar_member_has_no_byte_stream')
                result, count, record = hashlib.sha256(), 0, None
                with ExitStack() as member_stack:
                    member_stack.enter_context(source)
                    target_handle = None
                    if root is not None:
                        pieces = relative(name).parts
                        _parents(root, pieces[:-1], root_identity)
                        target = root.joinpath(*pieces)
                        target_handle = member_stack.enter_context(_exclusive_file(target))
                        record = dict(path=name, bytes_written=0, sha_verified=False,
                                      bytes_written_is_confirmed_lower_bound=True)
                        files_written.append(record)
                    for block in iter(lambda: source.read(BLOCK), b''):
                        result.update(block)
                        count += len(block)
                        if target_handle is not None:
                            target_handle.write(block)
                            record['bytes_written'] = count
                    if count != row['bytes'] or result.hexdigest() != row['sha256']:
                        raise RestoreError('actual_member_sha_or_size_mismatch')
                    if target_handle is not None:
                        target_handle.flush()
                        os.fsync(target_handle.fileno())
                if root is not None:
                    _output_check(root, root_identity)
                    no_links(target)
                    if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
                        raise RestoreError('extracted_member_sha_or_size_mismatch')
                    record.update(sha_verified=True, sha256=row['sha256'])
                seen.add(name)
        pinned.check(rehash=False)
    if seen != set(plan['members']):
        raise RestoreError('incomplete_member_coverage')


def _check_output_payloads(plan, root, identity, files_written):
    _output_check(root, identity)
    records = {record['path']: record for record in files_written}
    seen = set()
    for directory, directories, filenames in os.walk(root, followlinks=False):
        parent = Path(directory)
        no_links(parent)
        for name in directories:
            no_links(parent / name)
        for name in filenames:
            target = parent / name
            no_links(target)
            member = target.relative_to(root).as_posix()
            row = plan['members'].get(member)
            if row is None or member in seen or not stat.S_ISREG(target.stat().st_mode):
                raise RestoreError('unapproved_final_output_member')
            if target.stat().st_size != row['bytes'] or digest(target) != row['sha256']:
                records[member]['sha_verified'] = False
                records[member]['final_verification_failed'] = True
                raise RestoreError('final_output_sha_or_size_mismatch')
            records[member]['observed_file_bytes'] = row['bytes']
            seen.add(member)
    if seen != set(plan['members']):
        raise RestoreError('incomplete_final_output_coverage')
    _output_check(root, identity)


def audit(stage, manifest, expected_sha):
    """Read-only compatibility audit; restore retains the same FDs across passes."""
    stage, manifest = absolute(stage), absolute(manifest)
    if not stage.is_dir():
        raise RestoreError('stage_not_directory')
    with ExitStack() as stack:
        plan = _plan(stage, manifest, expected_sha, stack)
        return [p.path for p in plan['parts']], plan['members']


def _write_receipt(root, identity, receipt):
    _output_check(root, identity)
    stored = dict(receipt, receipt_written=True)
    payload = (json.dumps(stored, indent=2, sort_keys=True) + '\n').encode('utf-8')
    with _exclusive_file(root / RECEIPT_NAME) as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _output_check(root, identity)


def restore(stage, manifest, sha, output, execute=False):
    receipt = dict(schema_version=1, operation='restore_dataset_tar',
                   status='FAILED_BEFORE_OUTPUT', output_created=False,
                   member_manifest_sha256=sha, deserialized_objects=False,
                   model_imports=False, files_written=[], receipt_written=False,
                   all_member_payloads_verified=False)
    root_identity = None
    try:
        stage, manifest, output = map(absolute, (stage, manifest, output))
        receipt['output'] = str(output)
        if not stage.is_dir() or not output.parent.is_dir() or output.exists():
            raise RestoreError('stage_directory_or_exclusive_output_required')
        if _overlap(stage, output) or _overlap(manifest, output):
            raise RestoreError('input_output_overlap')
        with ExitStack() as stack:
            plan = _plan(stage, manifest, sha, stack)
            for pinned in plan['parts']:
                if _overlap(pinned.path, output):
                    raise RestoreError('input_output_overlap')
            receipt.update(member_count=len(plan['members']),
                           payload_bytes=sum(r['bytes'] for r in plan['members'].values()),
                           archive_parts=[dict(path=n, bytes=p.identity[2], sha256=p.sha256)
                                          for n, p in zip(plan['part_names'], plan['parts'])])
            if not execute:
                receipt['status'] = 'VERIFIED_NOT_EXTRACTED'
                return receipt
            if shutil.disk_usage(output.parent).free < receipt['payload_bytes'] + 1024**3:
                raise RestoreError('insufficient_free_space')
            # Recheck pinned bytes before any output and again after extraction.
            _check_pins(plan)
            no_links(output)
            output.mkdir(mode=0o700, exist_ok=False)
            receipt['output_created'] = True
            info = output.stat()
            root_identity = (info.st_dev, info.st_ino)
            _verify_archive_pass(plan, output, root_identity, receipt['files_written'])
            _check_pins(plan)
            _check_output_payloads(plan, output, root_identity, receipt['files_written'])
            receipt.update(status='EXTRACTED_AND_SHA_VERIFIED', all_member_payloads_verified=True)
            _write_receipt(output, root_identity, receipt)
            receipt['receipt_written'] = True
            return receipt
    except Exception as error:
        receipt.update(status='FAILED_PARTIAL_RETAINED' if receipt['output_created'] else 'FAILED_BEFORE_OUTPUT',
                       error_code=error.code if isinstance(error, RestoreError) else type(error).__name__)
        if receipt['output_created'] and root_identity is not None:
            try:
                _output_check(output, root_identity)
                for record in receipt['files_written']:
                    target = output.joinpath(*relative(record['path']).parts)
                    try:
                        no_links(target)
                        info = target.stat()
                        if stat.S_ISREG(info.st_mode):
                            record['observed_file_bytes'] = info.st_size
                    except Exception as observation_error:
                        record['observation_error_code'] = (observation_error.code
                            if isinstance(observation_error, RestoreError) else type(observation_error).__name__)
                _write_receipt(output, root_identity, receipt)
                receipt['receipt_written'] = True
            except Exception as receipt_error:
                # Never overwrite a partial receipt or redirect it through a link.
                receipt['receipt_error_code'] = (receipt_error.code if isinstance(receipt_error, RestoreError)
                                                 else type(receipt_error).__name__)
        raise RestoreError(receipt['error_code'], receipt) from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('stage-root', 'manifest', 'manifest-sha256', 'output'):
        parser.add_argument('--' + name, required=True)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument('--execute', action='store_true')
    actions.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = restore(args.stage_root, args.manifest, args.manifest_sha256, args.output, args.execute)
    except RestoreError as error:
        print(json.dumps(error.receipt, indent=2, sort_keys=True))
        print('Restore failed; no completed restoration is claimed: ' + error.code, file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
