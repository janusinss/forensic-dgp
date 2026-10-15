"""Prepare two exact cold-backup batches; never upload or remove research files."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/google_drive_offload_20261012_v1'
IMAGE = ROOT / 'outputs/cctv_dgp_vm_output_offload_20261010_v1_r2'
CACHE = ROOT / 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1'
PACKET = ROOT / 'outputs/cctv_dgp_bank_group_step_check_v1_vm'
PARENT = ROOT / 'outputs/cctv_dgp_bank_photometric_guard_v1_vm'
ARCHIVE_SHA = '5fe41326fa0ccd6399babf18f4f45582b0fa7d9815d506b998ca29fa637a4c1c'
SOURCE_SHA = 'af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291'
PACKET_SHA = '72ad815c6514cfb54115e7f5f519ec99ad5f968cc682b3875168149ca9f2f215'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def regular(path):
    rel = path.relative_to(ROOT).as_posix()
    assert path.resolve(strict=True).is_relative_to(ROOT.resolve())
    for part in [path, *path.parents]:
        if part == ROOT.parent:
            break
        assert not part.is_symlink() and not getattr(part.lstat(), 'st_file_attributes', 0) & 0x400, rel
    info = path.lstat()
    assert stat.S_ISREG(info.st_mode), rel
    return info


def row(path, expected=None, role='upload companion retained locally'):
    before = regular(path)
    value = sha(path) if expected is None else expected
    after = regular(path)
    assert (before.st_size, before.st_mtime_ns, before.st_ino) == (after.st_size, after.st_mtime_ns, after.st_ino)
    return dict(path=path.relative_to(ROOT).as_posix(), bytes=before.st_size,
                sha256=value, mtime_ns=before.st_mtime_ns, inode=before.st_ino,
                nlink=before.st_nlink, role=role)


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Retain previous preparation; do not overwrite'
    assert sha(CACHE / 'cache_source_inventory.json') == SOURCE_SHA
    cache_audit = read(CACHE / 'drive_preflight_20261012_v1.json')
    assert cache_audit['complete'] and cache_audit['manifest_sha256'] == SOURCE_SHA
    assert cache_audit['verified_cache_files'] == 4431 and cache_audit['verified_cache_bytes'] == 39448585279
    assert sha(PACKET / 'protocol.json') == PACKET_SHA
    protocol = read(PACKET / 'protocol.json')
    protected = {}
    missing_vm_parent_bindings = {}
    all_current_expected_paths = set()
    for base, bindings in [(PACKET, protocol['assets_sha256']), (PARENT, protocol['parent_required_sha256'])]:
        for name, expected in bindings.items():
            path = base / name
            relative = path.relative_to(ROOT).as_posix()
            all_current_expected_paths.add(relative)
            if not path.exists():
                assert base == PARENT, 'The prepared local transfer packet itself must be complete'
                missing_vm_parent_bindings[relative] = expected
                continue
            regular(path)
            assert sha(path) == expected, str(path)
            protected[relative] = expected
    for name in ['dgp_face_workflow_v3.py', 'face_workflow.py', 'face_workflow_web.py',
                 'dgp_frozen_inference_v2.py', 'dgp_face_restoration.py', 'cctv_dgp_frozen_norm.py',
                 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth',
                 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth',
                 'checkpoints/codeformer_inpainting.pth',
                 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth']:
        path = ROOT / name
        regular(path)
        protected[name] = sha(path)
    assert protected['outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'] == '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    archive = IMAGE / 'dgp-image-output-recovery-20261010-v1.tar'
    assert sha(archive) == ARCHIVE_SHA
    assert read(IMAGE / 'independent_backup_audit.json')['archive_sha256'] == ARCHIVE_SHA
    image_rows = [row(archive, ARCHIVE_SHA, 'inactive portable historical output backup; removal requires separate cloud audit')]
    for path in [IMAGE / 'dgp-image-output-recovery-20261010-v1.tar.sha256',
                 IMAGE / 'recovery_export.json', IMAGE / 'independent_backup_audit.json',
                 IMAGE / 'recovery_manifest.json', ROOT / 'scripts/restore_cctv_dgp_vm_output_offload_v1.py',
                 ROOT / 'VM_IMAGE_OUTPUT_BACKUP_RESTORE_20261010_V1.md']:
        image_rows.append(row(path))
    source = read(CACHE / 'cache_source_inventory.json')
    cache_rows = []
    cache_paths = set()
    for item in source['files']:
        linux = PurePosixPath(item['path'])
        assert linux.is_absolute() and '..' not in linux.parts
        path = CACHE / 'cache_files' / Path(*linux.relative_to('/').parts)
        info = regular(path)
        assert info.st_size == item['bytes']
        cache_rows.append(row(path, item['sha256'], 'inactive local VM-cache recovery copy; removal requires separate cloud audit'))
        cache_paths.add(path.relative_to(CACHE / 'cache_files').as_posix())
    actual_cache_paths = set()
    for folder, directories, names in os.walk(CACHE / 'cache_files', followlinks=False):
        for name in directories:
            target = Path(folder) / name
            assert not target.is_symlink() and not getattr(target.lstat(), 'st_file_attributes', 0) & 0x400
        for name in names:
            actual_cache_paths.add((Path(folder) / name).relative_to(CACHE / 'cache_files').as_posix())
    assert actual_cache_paths == cache_paths and len(cache_rows) == 4431
    for name in ['cache_source_inventory.json', 'complete.json', 'independent_local_cache_audit.json',
                 'drive_preflight_20261012_v1.json', 'closure_manifest.json', 'independent_closure_audit.json']:
        cache_rows.append(row(CACHE / name))
    for path in sorted((CACHE / 'recovery_metadata').rglob('*')):
        if path.is_file():
            cache_rows.append(row(path))
    for records in [image_rows, cache_rows]:
        assert len({item['path'] for item in records}) == len(records)
        assert not set(item['path'] for item in records) & set(protected)
        assert not set(item['path'] for item in records) & all_current_expected_paths
        assert all(not item['path'].startswith(('outputs/cctv_dgp_bank_group_step_check_v1_vm/',
                    'outputs/cctv_dgp_bank_photometric_guard_v1_vm/')) for item in records)
        assert all(not any(word in item['path'].lower() for word in ['client_secret', 'rclone.conf', '.env']) for item in records)
    OUT.mkdir()
    write(OUT / 'protected_active_bindings.json', protected)
    write(OUT / 'vm_parent_bindings_not_duplicated_locally.json', missing_vm_parent_bindings)
    for name, records in [('image', image_rows), ('cache', cache_rows)]:
        (OUT / f'{name}_files.txt').write_text(''.join(item['path'] + '\n' for item in records), encoding='utf-8', newline='\n')
        (OUT / f'{name}_sha256.txt').write_text(''.join(item['sha256'] + '  ' + item['path'] + '\n' for item in records), encoding='utf-8', newline='\n')
    profile = Path(os.environ['APPDATA']) / 'rclone/forensic-dgp-drive-manual.conf'
    import configparser
    private = configparser.ConfigParser(interpolation=None)
    private.read(profile, encoding='utf-8-sig')
    settings = private['thesis_archive']
    assert settings['type'] == 'drive' and settings['scope'] == 'drive.file'
    assert not settings.get('team_drive', '').strip() and not settings.get('root_folder_id', '').strip()
    assert json.loads(settings['token'])['access_token']
    client_binding = hashlib.sha256(settings['client_id'].encode()).hexdigest()
    plan = dict(version='exact-cold-backups-v1', prepared_utc=datetime.now(timezone.utc).isoformat(),
                root=str(ROOT), connection_verified=True, remote='thesis_archive', scope='drive.file',
                client_id_sha256=client_binding,
                destination='thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261012_v1',
                private_config_outside_project=True, drive_total_bytes=5497558138880,
                drive_free_bytes_at_connection_check=5477794986957,
                local_free_bytes_at_preparation=shutil.disk_usage(ROOT).free,
                protected_active_bindings=len(protected),
                required_vm_parent_bindings_not_duplicated_at_local_path=len(missing_vm_parent_bindings),
                missing_parent_files_are_not_offload_candidates=True,
                VM_parent_preflight_has_not_been_rerun=True,
                protected_manifest_sha256=sha(OUT / 'protected_active_bindings.json'),
                driver_sha256=sha(ROOT / 'scripts/run_google_drive_offload_20261012_v1.ps1'),
                batches={name: dict(files=records, file_count=len(records),
                    upload_bytes=sum(item['bytes'] for item in records),
                    removal_candidate_bytes=sum(item['bytes'] for item in records if item['role'].startswith('inactive')),
                    file_list_sha256=sha(OUT / f'{name}_files.txt'),
                    sha256_list_sha256=sha(OUT / f'{name}_sha256.txt'))
                    for name, records in [('image', image_rows), ('cache', cache_rows)]},
                active_policy=['Keep app/source/design, environments and original checkpoints at existing paths.',
                    'Keep current signed VM packet and all declared parent dependencies.',
                    'Keep frozen native/synthetic data, final identities, provenance, splits and current audits.',
                    'Keep backup metadata/verifiers, instructions and failure records locally.',
                    'All other outputs remain local pending a separate dependency review.'],
                files_uploaded=0, files_removed=0, training_calls=0,
                goal_complete=False, seconds=time.monotonic() - started)
    write(OUT / 'plan.json', plan)
    (OUT / 'plan.sha256').write_text(sha(OUT / 'plan.json') + '  plan.json\n', encoding='utf-8', newline='\n')
    print(json.dumps(dict(prepared=True, protected_bindings=len(protected),
        plan_sha256=sha(OUT / 'plan.json'), batches={k: {'files': v['file_count'], 'upload_bytes': v['upload_bytes'],
        'removal_candidate_bytes': v['removal_candidate_bytes']} for k, v in plan['batches'].items()},
        files_uploaded=0, files_removed=0, seconds=plan['seconds'])), flush=True)


if __name__ == '__main__':
    main()
