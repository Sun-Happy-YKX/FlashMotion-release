"""Synthetic, offline byte-only restore tests; no pickle/ML imports or network."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import restore_data as r


def sha(data):
    return hashlib.sha256(data).hexdigest()


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='restore-tests-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.stage = self.root / 'stage'
        self.stage.mkdir()
        self.manifest = self.stage / 'members.json'
        self.output = self.root / 'restored'
        self.payloads = {'data/a.pkl': b'not a pickle: NEVER DESERIALIZE',
                         'data/b.npz': b'not an NPZ: ordinary verified bytes'}
        self.value = dict(schema=r.SCHEMA, archive_parts=['part-1.tar', 'part-2.tar'],
                          members=[self.row(n, b) for n, b in self.payloads.items()])
        for name, (member, payload) in zip(self.value['archive_parts'], self.payloads.items()):
            self.tar(name, [(member, payload, tarfile.REGTYPE)])
        self.save()

    def row(self, name, data=b'x'):
        return dict(path=name, bytes=len(data), sha256=sha(data), source='/read-only/original/' + name)

    def tar(self, name, entries):
        with tarfile.open(self.stage / name, 'w', format=tarfile.PAX_FORMAT) as archive:
            for member, data, kind in entries:
                info = tarfile.TarInfo(member)
                info.type = kind
                info.size = len(data) if kind in (tarfile.REGTYPE, tarfile.AREGTYPE) else 0
                if kind in (tarfile.SYMTYPE, tarfile.LNKTYPE):
                    info.linkname = 'data/a.pkl'
                archive.addfile(info, io.BytesIO(data) if info.size else None)

    def save(self, raw=None):
        self.manifest.write_bytes(raw if raw is not None else json.dumps(self.value).encode('utf-8'))
        self.manifest_sha = sha(self.manifest.read_bytes())

    def restore(self, execute=False, **changes):
        args = dict(stage=self.stage, manifest=self.manifest, sha=self.manifest_sha,
                    output=self.output, execute=execute)
        args.update(changes)
        return r.restore(**args)

    def rejects(self, code=None, execute=False, **changes):
        with self.assertRaises(r.RestoreError) as caught:
            self.restore(execute, **changes)
        failure = caught.exception.receipt
        self.assertFalse(failure['all_member_payloads_verified'])
        if code:
            self.assertEqual(caught.exception.code, code)
        return failure

    def test_real_string_parts_schema_dry_run_writes_nothing(self):
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*'))
        receipt = self.restore()
        self.assertEqual(receipt['status'], 'VERIFIED_NOT_EXTRACTED')
        self.assertEqual(receipt['member_count'], 2)
        self.assertFalse(receipt['output_created'])
        self.assertFalse(receipt['deserialized_objects'])
        self.assertFalse(receipt['model_imports'])
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob('*')))
        self.assertEqual(receipt['archive_parts'][0]['sha256'], sha((self.stage/'part-1.tar').read_bytes()))

    def test_execute_exact_bytes_and_receipt(self):
        receipt = self.restore(True)
        self.assertEqual(receipt['status'], 'EXTRACTED_AND_SHA_VERIFIED')
        self.assertTrue(receipt['receipt_written'])
        self.assertTrue(receipt['all_member_payloads_verified'])
        self.assertEqual(len(receipt['files_written']), 2)
        for name, data in self.payloads.items():
            self.assertEqual((self.output/name).read_bytes(), data)
        stored = json.loads((self.output/r.RECEIPT_NAME).read_text())
        self.assertEqual(stored, receipt)
        self.assertTrue(all(row['sha_verified'] for row in stored['files_written']))

    def test_source_is_uninterpreted_provenance(self):
        self.value['members'][0]['source'] = '../DO-NOT-OPEN:this-source'
        self.save()
        self.assertEqual(self.restore()['status'], 'VERIFIED_NOT_EXTRACTED')

    def test_manifest_wrong_sha_before_output(self):
        self.rejects('member_manifest_sha_mismatch', sha='0'*64, execute=True)
        self.assertFalse(self.output.exists())

    def test_invalid_expected_sha(self):
        for value in ('A'*64, '0'*63, None, 123):
            with self.subTest(value=value):
                self.rejects('invalid_sha256', sha=value)

    def test_reject_unsafe_portable_paths(self):
        names = ('../escape', '/abs', 'a//b', './a', 'a/../b', 'a\\b', 'C:/abs',
                 'a:ads', 'a/file:stream', 'CON', 'nul.txt', 'COM1.bin', 'LPT9',
                 'COM¹.bin', 'LPT²', 'CON .txt', 'CONIN$', 'CONOUT$',
                 'a.', 'a ', 'a/b. ', 'a/\x00x', 'a/\nx', 'a/?', 'a/*', 'a/"x', 'a/|x')
        for name in names:
            with self.subTest(name=name), self.assertRaises(r.RestoreError):
                r.relative(name)

    def test_member_paths_checked_before_output(self):
        self.value['members'][0]['path'] = '../escape'
        self.save()
        self.rejects('unsafe_relative_path', execute=True)
        self.assertFalse(self.output.exists())

    def test_case_alias_and_prefix_collisions(self):
        for names in (('a', 'a'), ('A', 'a'), ('Dir/a', 'dir/b'), ('a', 'a/b'), ('a/b', 'a')):
            with self.subTest(names=names):
                self.value['members'] = [self.row(n) for n in names]
                self.save()
                self.rejects(execute=True)
                self.assertFalse(self.output.exists())

    def test_receipt_name_and_namespace_reserved(self):
        for name in (r.RECEIPT_NAME, r.RECEIPT_NAME.upper(), r.RECEIPT_NAME+'/child'):
            with self.subTest(name=name):
                self.value['members'] = [self.row(name)]
                self.save()
                self.rejects('receipt_namespace_collision', execute=True)

    def test_parts_must_be_string_list_with_unique_names(self):
        for value in ([{'path': 'part-1.tar'}], 'part-1.tar', [], ['part-1.tar', 'part-1.tar'],
                      ['part-1.tar', 'PART-1.tar'], ['part-1.tar', '../escape']):
            with self.subTest(value=value):
                self.value['archive_parts'] = value
                self.save()
                self.rejects()

    def test_exact_member_schema_bytes_and_sha(self):
        original = self.row('data/a.pkl', self.payloads['data/a.pkl'])
        changes = ({'bytes': True}, {'bytes': -1}, {'bytes': 1.5}, {'sha256': 'F'*64},
                   {'source': ''}, {'source': None}, {'unexpected': 'field'})
        for change in changes:
            with self.subTest(change=change):
                self.value['members'][0] = dict(original, **change)
                self.save()
                self.rejects()
        self.value['members'][0] = {k:v for k,v in original.items() if k != 'source'}
        self.save()
        self.rejects('unexpected_member_row_schema')

    def test_duplicate_json_keys(self):
        raw = json.dumps(self.value).encode().replace(b'"schema":', b'"schema":"duplicate", "schema":', 1)
        self.save(raw)
        self.rejects('duplicate_json_key')

    def test_unexpected_or_missing_schema(self):
        self.value['schema'] = 'another-schema'
        self.save()
        self.rejects('unexpected_member_manifest_schema')
        self.value['schema'] = r.SCHEMA
        self.value['extra'] = 1
        self.save()
        self.rejects('unexpected_member_manifest_schema')

    def test_invalid_json_and_missing_archive(self):
        self.save(b'{')
        self.rejects('invalid_member_manifest_json')
        self.save()
        (self.stage/'part-2.tar').unlink()
        self.rejects('FileNotFoundError')

    def test_nonregular_tar_members_rejected(self):
        for kind in (tarfile.DIRTYPE, tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE,
                     tarfile.CHRTYPE, tarfile.BLKTYPE, tarfile.GNUTYPE_SPARSE):
            with self.subTest(kind=kind):
                self.tar('part-1.tar', [('data/a.pkl', b'', kind)])
                self.rejects(execute=True)
                self.assertFalse(self.output.exists())

    def test_tar_traversal_is_rejected(self):
        self.tar('part-1.tar', [('../escape', b'x', tarfile.REGTYPE)])
        self.rejects('unsafe_relative_path')

    def test_tar_duplicate_across_shards(self):
        self.tar('part-2.tar', [('data/a.pkl', self.payloads['data/a.pkl'], tarfile.REGTYPE)])
        self.rejects('tar_duplicate_unapproved_or_size_mismatch')

    def test_appended_tar_duplicate_after_end_marker(self):
        duplicate = (self.stage/'part-1.tar').read_bytes()
        with (self.stage/'part-1.tar').open('ab') as handle:
            handle.write(duplicate)
        self.rejects('tar_duplicate_unapproved_or_size_mismatch')

    def test_full_coverage_and_unapproved_entry(self):
        self.value['members'].append(self.row('data/missing'))
        self.save()
        self.rejects('incomplete_member_coverage')
        self.value['members'].pop()
        self.save()
        self.tar('part-2.tar', [('data/extra', b'x', tarfile.REGTYPE)])
        self.rejects('tar_duplicate_unapproved_or_size_mismatch')

    def test_member_byte_sha_and_size_checked(self):
        self.value['members'][0]['sha256'] = '0'*64
        self.save()
        self.rejects('actual_member_sha_or_size_mismatch')
        self.value['members'][0] = self.row('data/a.pkl', b'short')
        self.save()
        self.rejects('tar_duplicate_unapproved_or_size_mismatch')

    def test_zero_byte_regular_member(self):
        self.payloads['data/a.pkl'] = b''
        self.value['members'][0] = self.row('data/a.pkl', b'')
        self.tar('part-1.tar', [('data/a.pkl', b'', tarfile.REGTYPE)])
        self.save()
        self.assertEqual(self.restore(True)['status'], 'EXTRACTED_AND_SHA_VERIFIED')

    def test_existing_output_never_overwritten(self):
        self.output.mkdir()
        sentinel = self.output/'sentinel'
        sentinel.write_bytes(b'keep')
        self.rejects('stage_directory_or_exclusive_output_required', execute=True)
        self.assertEqual(sentinel.read_bytes(), b'keep')

    def test_input_output_overlap_and_relative_roots(self):
        self.rejects('input_output_overlap', output=self.stage/'new')
        self.rejects('input_output_overlap', manifest=self.output/'future.json')
        self.rejects('absolute_nonroot_path_required', output='relative-output')
        self.rejects('absolute_nonroot_path_required', output=self.root/'..'/'elsewhere')

    def test_input_hardlink_alias_rejected(self):
        second = self.stage/'part-2.tar'
        second.unlink()
        try:
            os.link(self.stage/'part-1.tar', second)
        except OSError:
            self.skipTest('hardlinks unavailable on test filesystem')
        self.rejects('input_file_identity_alias')

    def symlink(self, source, destination, directory=False):
        try:
            destination.symlink_to(source, target_is_directory=directory)
        except OSError:
            self.skipTest('OS does not permit symlink creation for this account')

    def test_stage_symlink_rejected(self):
        alias = self.root/'stage-alias'
        self.symlink(self.stage, alias, True)
        self.rejects('symlink_junction_or_reparse_point', stage=alias)

    def test_input_symlink_rejected(self):
        second = self.stage/'part-2.tar'
        second.unlink()
        self.symlink(self.stage/'part-1.tar', second)
        self.rejects('symlink_junction_or_reparse_point')

    def test_output_parent_symlink_rejected(self):
        alias = self.root/'parent-alias'
        self.symlink(self.root, alias, True)
        self.rejects('symlink_junction_or_reparse_point', output=alias/'new')

    def test_windows_junction_reparse_flag_rejected_without_new_api(self):
        actual = Path.lstat
        def fake(path):
            if path == self.stage:
                return SimpleNamespace(st_mode=stat.S_IFDIR, st_file_attributes=0x400)
            return actual(path)
        with patch.object(Path, 'lstat', fake):
            self.rejects('symlink_junction_or_reparse_point')

    def test_part_modified_between_passes_fails_before_output(self):
        original = r._plan
        def mutate(*args):
            plan = original(*args)
            with (self.stage/'part-1.tar').open('r+b') as handle:
                handle.seek(-1, os.SEEK_END)
                handle.write(b'X')
            return plan
        with patch.object(r, '_plan', mutate):
            self.rejects(execute=True)
        self.assertFalse(self.output.exists())

    def test_sha_rechecked_even_if_metadata_identity_is_mocked_unchanged(self):
        original = r._plan
        def mutate(*args):
            plan = original(*args)
            with (self.stage/'part-1.tar').open('r+b') as handle:
                handle.seek(-1, os.SEEK_END)
                handle.write(b'X')
            # Model a metadata-spoofing attack. Hash pin must independently fail.
            plan['parts'][0].identity = r._identity((self.stage/'part-1.tar').stat())
            plan['parts'][0].fd_ctime_ns = os.fstat(plan['parts'][0].handle.fileno()).st_ctime_ns
            plan['parts'][0].path_ctime_ns = (self.stage/'part-1.tar').stat().st_ctime_ns
            return plan
        with patch.object(r, '_plan', mutate):
            self.rejects('input_sha_changed', execute=True)
        self.assertFalse(self.output.exists())

    def test_manifest_modified_between_passes(self):
        original = r._plan
        def mutate(*args):
            plan = original(*args)
            with self.manifest.open('ab') as handle:
                handle.write(b' ')
            return plan
        with patch.object(r, '_plan', mutate):
            self.rejects(execute=True)
        self.assertFalse(self.output.exists())

    def test_part_replaced_between_passes(self):
        original = r._plan
        def replace(*args):
            plan = original(*args)
            new = self.stage/'replacement.tar'
            new.write_bytes((self.stage/'part-1.tar').read_bytes())
            try:
                os.replace(new, self.stage/'part-1.tar')
            except PermissionError:
                # Windows sharing restrictions can themselves prevent replacement.
                raise r.RestoreError('os_prevented_input_replacement') from None
            return plan
        with patch.object(r, '_plan', replace):
            self.rejects(execute=True)
        self.assertFalse(self.output.exists())

    def test_partial_failure_receipt_and_verified_first_file_preserved(self):
        original = r._exclusive_file
        def fail_second(path):
            if path.name == 'b.npz':
                raise OSError('synthetic failure')
            return original(path)
        with patch.object(r, '_exclusive_file', fail_second):
            receipt = self.rejects('OSError', execute=True)
        self.assertEqual(receipt['status'], 'FAILED_PARTIAL_RETAINED')
        self.assertTrue(receipt['receipt_written'])
        self.assertEqual((self.output/'data/a.pkl').read_bytes(), self.payloads['data/a.pkl'])
        self.assertFalse((self.output/'data/b.npz').exists())
        self.assertEqual(len(receipt['files_written']), 1)
        self.assertTrue(receipt['files_written'][0]['sha_verified'])
        self.assertEqual(json.loads((self.output/r.RECEIPT_NAME).read_text()), receipt)

    def test_final_part_sha_rechecked_after_extraction(self):
        original = r._verify_archive_pass
        def mutate(plan, *args, **kwargs):
            original(plan, *args, **kwargs)
            if args and args[0] is not None:
                with (self.stage/'part-1.tar').open('r+b') as handle:
                    handle.seek(-1, os.SEEK_END)
                    handle.write(b'X')
        with patch.object(r, '_verify_archive_pass', mutate):
            receipt = self.rejects(execute=True)
        self.assertEqual(receipt['status'], 'FAILED_PARTIAL_RETAINED')
        self.assertEqual(len(receipt['files_written']), 2)
        self.assertTrue(receipt['receipt_written'])

    def test_second_pass_checks_full_coverage_not_only_each_entry(self):
        original = r._verify_archive_pass
        def expand_allowlist(plan, *args, **kwargs):
            if args and args[0] is not None:
                plan['members']['data/absent'] = self.row('data/absent')
            return original(plan, *args, **kwargs)
        with patch.object(r, '_verify_archive_pass', expand_allowlist):
            receipt = self.rejects('incomplete_member_coverage', execute=True)
        self.assertEqual(len(receipt['files_written']), 2)
        self.assertTrue(receipt['receipt_written'])

    def test_final_output_hash_rechecked_before_success_receipt(self):
        original = r._verify_archive_pass
        def tamper(plan, *args, **kwargs):
            original(plan, *args, **kwargs)
            if args and args[0] is not None:
                (self.output/'data/a.pkl').write_bytes(b'tampered')
        with patch.object(r, '_verify_archive_pass', tamper):
            receipt = self.rejects('final_output_sha_or_size_mismatch', execute=True)
        self.assertTrue(receipt['receipt_written'])
        self.assertFalse(receipt['files_written'][0]['sha_verified'])
        self.assertEqual(receipt['files_written'][0]['observed_file_bytes'], len(b'tampered'))

    def test_unapproved_extra_output_does_not_get_success_receipt(self):
        original = r._verify_archive_pass
        def add_extra(plan, *args, **kwargs):
            original(plan, *args, **kwargs)
            if args and args[0] is not None:
                (self.output/'unexpected').write_bytes(b'extra')
        with patch.object(r, '_verify_archive_pass', add_extra):
            receipt = self.rejects('unapproved_final_output_member', execute=True)
        self.assertTrue(receipt['receipt_written'])

    def test_receipt_write_failure_truthful_and_does_not_overwrite(self):
        original = r._exclusive_file
        def fail_second(path):
            if path.name == 'b.npz':
                (self.output/r.RECEIPT_NAME).write_bytes(b'existing sentinel')
                raise OSError('synthetic failure')
            return original(path)
        with patch.object(r, '_exclusive_file', fail_second):
            receipt = self.rejects('OSError', execute=True)
        self.assertFalse(receipt['receipt_written'])
        self.assertEqual(receipt['receipt_error_code'], 'FileExistsError')
        self.assertEqual((self.output/r.RECEIPT_NAME).read_bytes(), b'existing sentinel')

    def test_insufficient_space_is_write_free(self):
        with patch.object(r.shutil, 'disk_usage', return_value=SimpleNamespace(free=0)):
            self.rejects('insufficient_free_space', execute=True)
        self.assertFalse(self.output.exists())

    def test_cli_failure_nonzero_and_json_truthful(self):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = r.main(['--stage-root', str(self.stage), '--manifest', str(self.manifest),
                           '--manifest-sha256', '0'*64, '--output', str(self.output), '--execute'])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(out.getvalue())['status'], 'FAILED_BEFORE_OUTPUT')
        self.assertIn('no completed restoration is claimed', err.getvalue())


if __name__ == '__main__':
    unittest.main()
