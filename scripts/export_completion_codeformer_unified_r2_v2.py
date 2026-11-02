"""Bounded evidence export for the distinct manual CodeFormer unified-r2-v2 pilot.
No model imports, neural work, cleanup, overwrite or automatic training action.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
import shutil
import stat
import tarfile
import time
from pathlib import Path, PurePosixPath

STEM = 'cctv-completion-codeformer-unified-r2-v2'
VM_NAME = 'cctv_completion_codeformer_unified_r2_v2_vm'
MEMBER_ROOT = 'cctv_completion_codeformer_unified_r2_v2_results'
MAX_BYTES = 2816 * 1024 ** 2
OUTPUT_BYTES = 2560 * 1024 ** 2
MAX_SECONDS = 900
STOP_FREE = 2 * 1024 ** 3


def need(ok, label):
    if not ok:
        raise ValueError(label)


def sha(path, guard=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            if guard:
                guard()
            digest.update(block)
    return digest.hexdigest()


def read(path):
    def reject(x):
        raise ValueError('NonfiniteJSON ' + x)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def regular(path):
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not path.is_symlink(), 'Only regular evidence files may be exported')
    return info


def member(root, relative):
    need(
        isinstance(relative, str) and relative and '\\' not in relative and ':' not in relative,
        'Invalid member spelling',
    )
    name = PurePosixPath(relative)
    need(not name.is_absolute() and '..' not in name.parts, 'Member traversal')
    path = root.joinpath(*name.parts)
    for item in [path, *path.parents]:
        need(not item.is_symlink(), 'Evidence symlink refused')
        if item == root:
            break
    need(path.resolve().is_relative_to(root.resolve()), 'Evidence escaped root')
    return path


class CountedWriter:
    def __init__(self, stream, guard, *, disk_directory=None):
        self.stream = stream
        self.guard = guard
        self.disk_directory = disk_directory
        self.count = 0

    def write(self, data):
        self.guard()
        need(self.count + len(data) <= MAX_BYTES, 'Raw/compressed return exceeds 2816 MiB')
        if self.disk_directory is not None:
            need(
                shutil.disk_usage(self.disk_directory).free - len(data) >= STOP_FREE,
                'Preserve 2 GiB free during export',
            )
        result = self.stream.write(data)
        need(result == len(data), 'Short evidence archive write')
        self.count += result
        return result

    def flush(self):
        return self.stream.flush()


def export(root, expected, home, *, enforce_vm=True):
    root = Path(root).resolve(strict=True)
    home = Path(home).resolve(strict=True)
    started = time.monotonic()
    if enforce_vm:
        need(
            os.name == 'posix' and root == (home / 'forensic-dgp' / VM_NAME).resolve(),
            'Export only named manual VM root',
        )

    def guard():
        need(time.monotonic() - started < MAX_SECONDS, 'Export 900 second cooperative stop')

    need(shutil.disk_usage(home).free >= STOP_FREE, 'Need 2 GiB export free-space floor')
    ppath = member(root, 'protocol.json')
    regular(ppath)
    need(sha(ppath, guard) == expected, 'Protocol SHA differs')
    protocol = read(ppath)
    need(protocol.get('format') == 'completion-codeformer-unified-r2-v2', 'Wrong pilot protocol')
    need(
        isinstance(protocol.get('files'), list) and len(protocol['files']) > 0,
        'Empty source manifest refused',
    )
    need(
        protocol.get('manual_execution_only') is True
        and protocol.get('automatic_resume_or_promotion') is False,
        'Manual-only protocol differs',
    )
    contract = protocol.get('contract', {})
    need(
        contract.get('output_bytes') == OUTPUT_BYTES and contract.get('stop_free_bytes') == STOP_FREE,
        'Export/output/disk contract differs',
    )
    seen = set()
    for record in protocol['files']:
        guard()
        relative = record['path']
        need(relative not in seen, 'Duplicate source manifest member')
        seen.add(relative)
        path = member(root, relative)
        info = regular(path)
        need(
            info.st_size == record['bytes'] and sha(path, guard) == record['sha256'],
            'Installed source/data drift ' + relative,
        )
    files = []
    for name in ('protocol.json', 'supervisor.json', 'trainer.log'):
        path = member(root, name)
        regular(path)
        files.append(path)
    supervisor = read(root / 'supervisor.json')
    need(
        supervisor.get('complete') is True and type(supervisor.get('worker_exit')) is int,
        'Terminal supervisor receipt required',
    )
    output = member(root, 'outputs')
    if output.exists():
        need(output.is_dir() and not output.is_symlink(), 'Invalid output root')
        for path in sorted(output.rglob('*')):
            guard()
            need(not path.is_symlink(), 'Output symlink refused')
            need(path.resolve().is_relative_to(output), 'Output escaped evidence root')
            if path.is_dir():
                continue
            regular(path)
            files.append(path)
    need(len(files) >= 3 and len(set(files)) == len(files), 'Empty/duplicate return manifest refused')
    need(len(files) + 1 <= 1000, 'Return regular member cap 1000 including manifest')
    need(
        all(len(path.relative_to(root).as_posix().encode('utf-8')) <= 240 for path in files),
        'Return relative path 240 UTF8 byte cap',
    )
    need(
        sum('.npz' in [s.lower() for s in path.suffixes] for path in files) <= 468,
        'Numeric archive cap 468 including 80 cached teacher copies and failed partials',
    )
    checkpoint_files = [
        path for path in files if any(s.lower() in ('.pth', '.pt') for s in path.suffixes)
    ]
    need(
        all(
            path.relative_to(root).as_posix()
            in (
                'outputs/terminal_training_state.pth',
                'outputs/terminal_training_state.pth.partial',
            )
            for path in checkpoint_files
        ),
        'Unexpected checkpoint path',
    )
    checkpoint_aliases = []
    if len(checkpoint_files) == 2:
        committed_path = output / 'terminal_training_state.pth'
        partial_path = output / 'terminal_training_state.pth.partial'
        need(
            set(checkpoint_files) == {committed_path, partial_path},
            'Unexpected checkpoint pair',
        )
        left = regular(committed_path)
        right = regular(partial_path)
        need(
            (left.st_dev, left.st_ino, left.st_uid, left.st_size)
            == (right.st_dev, right.st_ino, right.st_uid, right.st_size)
            and left.st_nlink >= 2
            and right.st_nlink >= 2,
            'Distinct checkpoint payloads violate one-save invariant; preserve both at source',
        )
        digest = sha(committed_path, guard)
        need(sha(partial_path, guard) == digest, 'Same-inode checkpoint alias bytes differ')
        checkpoint_aliases.append(
            {
                'path': partial_path.relative_to(root).as_posix(),
                'canonical_payload_path': committed_path.relative_to(root).as_posix(),
                'device': right.st_dev,
                'inode': right.st_ino,
                'uid': right.st_uid,
                'nlink_at_source': right.st_nlink,
                'bytes': right.st_size,
                'sha256': digest,
                'source_alias_preserved': True,
                'archived_payload_duplicated': False,
                'reason': 'Interrupted exclusive hardlink commit window; exact same inode only',
            }
        )
        files.remove(partial_path)
        checkpoint_files = [committed_path]
    need(len(checkpoint_files) <= 1, 'At most one bounded terminal payload')
    need(
        sum(path.stat().st_size for path in checkpoint_files) <= 1024 ** 3,
        'Terminal payload cumulative 1 GiB cap',
    )
    terminal = output / 'terminal_training_state.pth'
    results = output / 'results.json'
    if supervisor['worker_exit'] in (0, 2):
        need(
            terminal.is_file() and regular(terminal).st_size > 0 and results.is_file(),
            'Completed/quality-stopped run needs terminal state and results',
        )
        summary = read(results)
        need(
            summary.get('complete') is True
            and summary.get('protocol_sha256') == expected
            and summary.get('model_qualified') is False,
            'Results/protocol scope differs',
        )
    entries = []
    for path in files:
        guard()
        before = regular(path)
        digest = sha(path, guard)
        after = regular(path)
        need(
            (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
            'Evidence changed while hashing',
        )
        entries.append(
            {
                'path': path.relative_to(root).as_posix(),
                'bytes': after.st_size,
                'sha256': digest,
            }
        )
    payload_bytes = sum(row['bytes'] for row in entries)
    need(payload_bytes <= OUTPUT_BYTES, 'Uncompressed whole output exceeds 2560 MiB')
    partials = [r['path'] for r in entries if r['path'].endswith('.partial')]
    committed = [
        r for r in entries if r['path'] == 'outputs/terminal_training_state.pth'
    ]
    manifest = {
        'format': 'completion-codeformer-unified-r2-v2-return-manifest-v1',
        'protocol_sha256': expected,
        'files': entries,
        'worker_exit': supervisor['worker_exit'],
        'worker_timeout': supervisor.get('timeout'),
        'worker_interruption': supervisor.get('interruption'),
        'terminal_state_committed': bool(committed),
        'terminal_state_pin': committed[0] if committed else None,
        'partial_files_retained': partials,
        'retained_teacher_label_arrays': len(
            [
                r
                for r in entries
                if r['path'].startswith('outputs/teacher_labels/')
                and r['path'].endswith('.npz')
            ]
        ),
        'original_base_weights_exported': False,
        'parent_training_state_exported': False,
        'state_recoverability': 'Independent safe-load/state/Adam/RNG audit required; failed .partials are not usable checkpoints.',
        'model_qualified': False,
        'training_success_not_implied': True,
        'raw_and_compressed_cap_bytes': MAX_BYTES,
        'whole_output_cap_bytes': OUTPUT_BYTES,
        'same_inode_checkpoint_alias_policy': 'Exact same device/inode/uid/size/hash and nlink>=2 only; committed payload exported once; source partial alias preserved and recorded. Distinct payload pair refuses.',
        'retained_checkpoint_source_aliases': checkpoint_aliases,
        'export_cooperative_seconds': MAX_SECONDS,
        'minimum_free_bytes': STOP_FREE,
    }
    body = (json.dumps(manifest, indent=2, allow_nan=False) + '\n').encode('utf-8')
    need(len(body) <= 2 * 1024 ** 2, 'Return manifest metadata cap')
    prebound = (
        sum(((row['bytes'] + 511) // 512) * 512 + 2048 for row in entries)
        + ((len(body) + 511) // 512) * 512
        + 2048
        + 10240
    )
    need(prebound <= MAX_BYTES, 'Payload plus conservative tar/header bound exceeds 2816 MiB')
    archive = home / (STEM + '-results.tar.gz')
    partial = Path(str(archive) + '.partial')
    checksum = Path(str(archive) + '.sha256')
    receipt_path = home / (STEM + '-export.json')
    for path in (archive, partial, checksum, receipt_path):
        need(not path.exists(), 'Preserve existing export; refusing overwrite')
    guard()
    raw_count = 0
    compressed_count = 0
    try:
        with partial.open('xb') as raw:
            compressed = CountedWriter(raw, guard, disk_directory=home)
            with gzip.GzipFile(
                filename='', mode='wb', fileobj=compressed, compresslevel=1, mtime=0
            ) as zipped:
                uncompressed = CountedWriter(zipped, guard)
                with tarfile.open(
                    fileobj=uncompressed,
                    mode='w|',
                    bufsize=1024 * 1024,
                    format=tarfile.PAX_FORMAT,
                ) as tar:
                    for path, row in zip(files, entries):
                        guard()
                        need(
                            regular(path).st_size == row['bytes']
                            and sha(path, guard) == row['sha256'],
                            'Evidence changed before tar member',
                        )
                        info = tarfile.TarInfo(MEMBER_ROOT + '/' + row['path'])
                        info.size = row['bytes']
                        info.mode = 0o644
                        info.mtime = 0
                        with path.open('rb') as stream:
                            tar.addfile(info, stream)
                        need(
                            sha(path, guard) == row['sha256'],
                            'Evidence changed during tar member',
                        )
                    info = tarfile.TarInfo(MEMBER_ROOT + '/export_manifest.json')
                    info.size = len(body)
                    info.mode = 0o644
                    info.mtime = 0
                    tar.addfile(info, io.BytesIO(body))
                raw_count = uncompressed.count
            compressed_count = compressed.count
            raw.flush()
            os.fsync(raw.fileno())
        guard()
        need(
            raw_count <= MAX_BYTES and compressed_count == partial.stat().st_size <= MAX_BYTES,
            'Final raw/compressed cap',
        )
        for alias in checkpoint_aliases:
            alias_path = member(root, alias['path'])
            canonical = member(root, alias['canonical_payload_path'])
            a = regular(alias_path)
            b = regular(canonical)
            need(
                (a.st_dev, a.st_ino, a.st_uid, a.st_size)
                == (alias['device'], alias['inode'], alias['uid'], alias['bytes'])
                == (b.st_dev, b.st_ino, b.st_uid, b.st_size)
                and a.st_nlink >= 2
                and b.st_nlink >= 2
                and sha(alias_path, guard) == sha(canonical, guard) == alias['sha256'],
                'Checkpoint source alias drift during export',
            )
        digest = sha(partial, guard)
        need(shutil.disk_usage(home).free >= STOP_FREE, 'Export free space below 2 GiB')
        guard()
        need(not archive.exists(), 'Archive appeared before publication')
        os.link(partial, archive)
        need(
            archive.stat().st_ino == partial.stat().st_ino,
            'Archive publication identity differs',
        )
        with checksum.open('x', encoding='ascii', newline='\n') as stream:
            stream.write(digest + '  ' + archive.name + '\n')
        receipt = {
            'format': 'completion-codeformer-unified-r2-v2-export-receipt-v1',
            'complete': True,
            'archive_sha256': digest,
            'bytes': archive.stat().st_size,
            'raw_tar_bytes': raw_count,
            'payload_bytes': payload_bytes,
            'seconds': time.monotonic() - started,
            'protocol_sha256': expected,
            'worker_exit': supervisor['worker_exit'],
            'terminal_state_committed': bool(committed),
            'partial_files_retained': partials,
            'successful_archive_partial_hardlink_retained': str(partial),
            'state_recoverability_pending_independent_audit': True,
            'training_success_not_implied': True,
            'model_qualified': False,
            'payload_files': len(entries),
            'export_only_neural_calls': 0,
            'retained_checkpoint_source_aliases': checkpoint_aliases,
        }
        with receipt_path.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(receipt, stream, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(receipt), flush=True)
        return receipt
    except BaseException as exc:
        failure = home / (STEM + '-export-failure.json')
        if not failure.exists():
            with failure.open('x', encoding='utf-8', newline='\n') as stream:
                json.dump(
                    {
                        'complete': False,
                        'error': repr(exc),
                        'partial_archive': str(partial),
                        'partial_retained': partial.exists(),
                        'seconds': time.monotonic() - started,
                        'raw_tar_bytes_observed': raw_count,
                        'compressed_bytes_observed': (
                            partial.stat().st_size if partial.exists() else 0
                        ),
                        'training_success_not_implied': True,
                    },
                    stream,
                    indent=2,
                )
                stream.write('\n')
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    export(args.root, args.protocol_sha, Path.home())


if __name__ == '__main__':
    main()
