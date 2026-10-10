"""Verify/restore the portable output backup; no inference, training or overwrite."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile


def digest(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024**2), b''):
        h.update(chunk)
    return h.hexdigest()


def safe_destination(root, relative):
    rel = PurePosixPath(relative)
    assert not rel.is_absolute() and '..' not in rel.parts and ':' not in relative and '\\' not in relative
    assert rel.as_posix() == relative and len(rel.parts) >= 4 and rel.parts[1] == 'outputs'
    assert rel.suffix in {'.png', '.npy', '.npz'}
    target = root.joinpath(*rel.parts)
    assert target.resolve().is_relative_to(root)
    for parent in [target] + list(target.parents):
        if parent == root.parent:
            break
        assert not parent.is_symlink(), 'Refuse link in restoration target'
    return target


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--root', type=Path, default=Path.home() / 'forensic-dgp')
    parser.add_argument('--restore', action='store_true', help='Restore absent files after the full archive audit; default is verification only')
    parser.add_argument('--group', action='append', help='Restore only this experiment root; repeat for several roots')
    args = parser.parse_args()
    with args.archive.open('rb') as stream:
        assert digest(stream) == args.expected_sha, 'Recovery archive SHA256 differs'
    root = args.root.resolve()
    assert root.is_dir() and not args.root.is_symlink()
    restored, retained = 0, 0
    with tarfile.open(args.archive, 'r:') as tar:
        members = tar.getmembers()
        names = [member.name for member in members]
        assert len(names) == len(set(names)) and all(member.isfile() and not member.issym() and not member.islnk() for member in members)
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and '..' not in path.parts and path.as_posix() == name
            assert ':' not in name and '\\' not in name
        with tar.extractfile('_recovery/recovery_manifest.json') as stream:
            manifest = json.load(stream)
        assert manifest['complete'] and manifest['format'] == 'exact-image-output-recovery-v1'
        rows = manifest['candidates']
        assert len(rows) == manifest['candidate_count']
        selected_groups = set(args.group or manifest['groups'])
        assert selected_groups and selected_groups.issubset(manifest['groups'])
        expected = {row['recovery_member'] for row in rows} | {
                    '_recovery/recovery_manifest.json', '_recovery/restore_cctv_dgp_vm_output_offload_v1.py'}
        assert set(names) == expected
        with tar.extractfile('_recovery/restore_cctv_dgp_vm_output_offload_v1.py') as stream:
            assert digest(stream) == manifest['restore_script_sha256']
        for n, row in enumerate(rows, 1):
            assert row['recovery_member'] == 'forensic-dgp/' + row['relative_path']
            member = tar.getmember(row['recovery_member'])
            assert member.size == row['metadata'][2]
            with tar.extractfile(member) as stream:
                assert digest(stream) == row['sha256'], row['relative_path']
            path = safe_destination(root, row['relative_path'])
            if path.exists():
                assert path.is_file() and stat.S_ISREG(path.lstat().st_mode)
                with path.open('rb') as stream:
                    assert digest(stream) == row['sha256'], 'Refuse differing existing output: ' + row['relative_path']
                retained += 1
            if n % 10000 == 0:
                print(json.dumps(dict(verified_outputs=n, of=len(rows))), flush=True)
        if args.restore:
            missing = [row for row in rows if PurePosixPath(row['relative_path']).parts[0] in selected_groups
                       and not safe_destination(root, row['relative_path']).exists()]
            required = sum(row['metadata'][2] for row in missing) + 1024**3
            assert shutil.disk_usage(root).free >= required, 'Insufficient restore space plus1GiB reserve; choose fewer groups or a larger disk'
            for row in rows:
                if PurePosixPath(row['relative_path']).parts[0] not in selected_groups:
                    continue
                path = safe_destination(root, row['relative_path'])
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(descriptor, 'wb') as target, tar.extractfile(row['recovery_member']) as source:
                    for chunk in iter(lambda: source.read(1024**2), b''):
                        target.write(chunk)
                    target.flush(); os.fsync(target.fileno())
                with path.open('rb') as stream:
                    assert digest(stream) == row['sha256']
                os.chmod(path, stat.S_IMODE(row['metadata'][1]) & 0o666)
                os.utime(path, ns=(row['metadata'][3], row['metadata'][3]))
                restored += 1
    print(json.dumps(dict(complete=True, archive_sha256=args.expected_sha, checked_outputs=len(rows),
          restored_absent_outputs=restored, existing_identical_outputs_retained=retained,
          selected_groups=sorted(selected_groups), overwrite=False, model_gradient_or_training_calls=0)), flush=True)


if __name__ == '__main__':
    main()
