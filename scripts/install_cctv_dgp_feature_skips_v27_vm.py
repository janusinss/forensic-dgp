"""Thin immutable-parent installation on the existing VM; no neural imports."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import sys
import time
import traceback


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 ** 2), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe(root, name):
    relative = PurePosixPath(name)
    assert not relative.is_absolute() and '..' not in relative.parts and '\\' not in name and ':' not in name and relative.as_posix() == name
    path = (root / name).resolve()
    assert path.is_relative_to(root.resolve()) and not (root / name).is_symlink(), name
    return path


def verify(root, pin):
    assert sha(root / 'protocol.json') == pin, 'V27 protocol changed'
    p = read(root / 'protocol.json')
    assert p['format'] == 'dgp-direct-feature-skips-capacity-v27'
    assert len(p['transfer_assets_sha256']) == 6 and len(p['inherited_assets']) == 240 and len(p['assets_sha256']) == 246
    for name, digest in p['transfer_assets_sha256'].items():
        assert sha(safe(root, name)) == digest, 'Transfer file changed: ' + name
    return p


def install(root, p, pin):
    assert sys.platform == 'linux' and platform.node().split('.')[0] == 'forensic-dgp-thesis', 'Existing Linux VM only'
    assert root.is_relative_to((Path.home() / 'forensic-dgp').resolve()), '~/forensic-dgp only'
    assert not any((root / name).exists() for name in ['installation_receipt.json', 'installation_failure.json', 'outputs', 'trainer.log']), 'Preserve earlier/partial work'
    started = time.monotonic(); linked = []
    parent = Path.home() / 'forensic-dgp/cctv_dgp_batchmatched_identity_vm_v26'
    try:
        assert sha(parent / 'protocol.json') == p['closed_V26_protocol_sha256'], 'Preserved V26 missing/changed'
        old = read(parent / 'protocol.json')
        assert len(old['assets_sha256']) == 240
        assert sha(parent / 'outputs/early_structure_stop.json') == p['closed_V26_early_failure_sha256']
        assert read(parent / 'outputs/early_structure_stop.json')['pass'] is False
        assert sha(parent / 'outputs/update0/head.pth') == p['closed_V26_initial_head_sha256']
        expected_bytes = 0
        for name, row in p['inherited_assets'].items():
            assert old['assets_sha256'][row['source']] == row['sha256'] == p['assets_sha256'][name]
            source = safe(parent, row['source'])
            assert sha(source) == row['sha256'], 'Preserved V26 asset differs: ' + name
            assert not safe(root, name).exists(), 'Preserve partial inherited file: ' + name
            expected_bytes += source.stat().st_size
        assert shutil.disk_usage(root).free >= 3 * 1024 ** 3, 'Need3GiB free; original research is preserved'
        for name, row in p['inherited_assets'].items():
            assert time.monotonic() - started < 60, 'Installation cap60s'
            source = safe(parent, row['source']); destination = safe(root, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, destination)
            assert source.stat().st_ino == destination.stat().st_ino and source.stat().st_dev == destination.stat().st_dev
            assert sha(destination) == row['sha256']
            linked.append(name)
        for name, digest in p['assets_sha256'].items():
            assert sha(safe(root, name)) == digest
        assert time.monotonic() - started < 60
        receipt = {'complete': True, 'protocol_sha256': pin,
            'inherited_assets_verified_and_hardlinked': 240, 'assets_verified': 246,
            'inherited_logical_bytes': expected_bytes, 'inherited_data_bytes_copied': 0,
            'old_V26_protocol_sha256': p['closed_V26_protocol_sha256'],
            'old_V26_failure_preserved': True, 'old_V26_initial_head_preserved': True,
            'seconds': time.monotonic() - started, 'cap_seconds': 60,
            'model_or_gradient_calls': 0, 'optimizer_updates': 0, 'original_files_changed': False,
            'copy_fallback_permitted': False, 'quality_acceptance_not_implied': True}
        write(root / 'installation_receipt.json', receipt)
        print(json.dumps(receipt))
    except BaseException as exc:
        write(root / 'installation_failure.json', {'complete': False, 'protocol_sha256': pin,
            'linked': linked, 'cause': str(exc), 'traceback': traceback.format_exc(),
            'seconds': time.monotonic() - started, 'resume_permitted': False, 'original_files_changed': False})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path); parser.add_argument('--protocol-sha', required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--verify-transfer', action='store_true'); modes.add_argument('--install', action='store_true')
    args = parser.parse_args(); root = args.root.resolve(); p = verify(root, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps({'complete': True, 'thin_assets': 6, 'inherited_assets': 240, 'model_or_gradient_calls': 0})); return
    if sys.platform == 'linux':
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('Install cap60s'))); signal.alarm(60)
    install(root, p, args.protocol_sha)


if __name__ == '__main__':
    main()
