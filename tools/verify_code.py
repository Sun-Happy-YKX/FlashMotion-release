"""CPU-only byte provenance and syntax checks. Never imports model code."""
from __future__ import annotations

import ast
import hashlib
import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SECRET_PATTERNS = (
    re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
    re.compile(rb'\bgh[pousr]_[A-Za-z0-9]{30,}\b'),
    re.compile(rb'\bhf_[A-Za-z0-9]{30,}\b'),
)

def main():
    receipt = json.loads((ROOT / 'manifests/frozen_source_collection.json').read_text())
    verified = 0
    for item in receipt['files']:
        path = ROOT / item['path']
        if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
            raise RuntimeError('Unsafe/missing source file: ' + item['path'])
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != item['sha256'] or len(data) != item['bytes']:
            raise RuntimeError('Byte provenance mismatch: ' + item['path'])
        verified += 1
    additional_verified = 0
    for filename in ('third_party_source_collection.20261010.json',):
        extra = json.loads((ROOT / 'manifests' / filename).read_text())
        for item in extra['files']:
            relative = item['destination'] if 'destination' in item else item['path']
            path = ROOT / relative
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(ROOT):
                raise RuntimeError('Unsafe/missing additional source: ' + relative)
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != item['sha256'] or len(data) != item['bytes']:
                raise RuntimeError('Additional byte provenance mismatch: ' + relative)
            additional_verified += 1
    checked = 0
    for path in ROOT.rglob('*'):
        if '.git' in path.parts or '__pycache__' in path.parts or not path.is_file():
            continue
        relative = path.relative_to(ROOT).as_posix()
        if path.is_symlink():
            raise RuntimeError('Symlink in code release: ' + str(path.relative_to(ROOT)))
        data = path.read_bytes()
        if path.stat().st_size > 10_000_000:
            raise RuntimeError('Unexpected large file in code release: ' + str(path.relative_to(ROOT)))
        if any(pattern.search(data) for pattern in SECRET_PATTERNS):
            raise RuntimeError('Credential-like content found; inspect locally: ' + str(path.relative_to(ROOT)))
        if path.suffix in {'.pt', '.pth', '.ckpt', '.pkl', '.npz', '.npy', '.pem', '.key'}:
            raise RuntimeError('Asset/credential file belongs outside Git: ' + str(path.relative_to(ROOT)))
        if path.suffix == '.py':
            ast.parse(data, filename=str(path))
            checked += 1
    print(json.dumps({'status': 'PASS', 'source_files_sha256_verified': verified,
                      'delivery_scope': 'six main task weights; no noFK/uncond',
                      'additional_sources_and_licenses_verified': additional_verified,
                      'python_files_syntax_checked': checked, 'model_imports': False,
                      'gpu_runtime_verified': False, 'uploads_verified': False}, indent=2))

if __name__ == '__main__':
    main()
