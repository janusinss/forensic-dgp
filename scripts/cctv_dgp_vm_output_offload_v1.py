"""Prepare and execute a backed-up image-output offload; no model work."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shlex
import shutil
import stat
import tarfile
import time
import zipfile

import cctv_dgp_bank_archive_cleanup_v1 as t

ROOT = t.ROOT
OUT = ROOT / 'outputs/cctv_dgp_vm_output_offload_20261010_v1_r2'
REMOTE_OUT = '/home/janusdominic0/dgp_output_offload_20261010_v1_receipts'
WORKER = ROOT / 'scripts/cctv_dgp_vm_output_offload_v1_vm.py'
RESTORE = ROOT / 'scripts/restore_cctv_dgp_vm_output_offload_v1.py'
GROUPS = {
    'cctv_dgp_head4_capacity_vm_v1': 'cctv_dgp_head4_capacity_vm_v1_return',
    'cctv_dgp_actual_step_review_v1_vm': 'cctv_dgp_actual_step_review_v1_vm_return',
    'cctv_dgp_loss_cone_probe_v33_vm': 'cctv_dgp_loss_cone_probe_v33_return',
    'cctv_dgp_profile_batches_vm_v31': 'cctv_dgp_profile_batches_v31_return',
    'cctv_dgp_feature_fusion_vm_v32_r2': 'cctv_dgp_feature_fusion_v32_r2_return',
    'cctv_dgp_pcgrad_fit_vm_v41': 'cctv_dgp_pcgrad_fit_v41_return',
    'cctv_dgp_spatial_fit_vm_v40': 'cctv_dgp_spatial_fit_v40_return',
}
BACKUP_NAME = 'dgp-image-output-recovery-20261010-v1.tar'
sha, read, write = t.sha, t.read, t.write


def bind_transport():
    t.transport.OUT = OUT
    t.OUT = OUT


def numpy_header(stream):
    import numpy as np
    version = np.lib.format.read_magic(stream)
    assert version in {(1, 0), (2, 0)}, 'Retain unsupported NumPy header version'
    reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
    shape, fortran, dtype = reader(stream)
    assert not dtype.hasobject
    return dict(shape=list(shape), fortran_order=bool(fortran), dtype=str(dtype))


def payload_kind(path):
    if path.suffix == '.png':
        from PIL import Image
        with Image.open(path) as image:
            assert image.format == 'PNG' and image.width > 0 and image.height > 0
            return 'PNG_image', dict(size=[image.width, image.height], mode=image.mode)
    if path.suffix == '.npy':
        with path.open('rb') as stream:
            header = numpy_header(stream)
        if header['shape'] not in ([256, 256, 3], [3, 256, 256]):
            return None, header
        assert header['dtype'] == 'float32'
        return 'raw_RGB_array', header
    assert path.suffix == '.npz'
    with zipfile.ZipFile(path) as archive:
        headers = {}
        for name in archive.namelist():
            assert '/' not in name and name.endswith('.npy')
            with archive.open(name) as stream:
                headers[name[:-4]] = numpy_header(stream)
        if 'rgb' in headers:
            assert headers['rgb']['shape'] == [256, 256, 3] and headers['rgb']['dtype'] == 'float32'
        else:
            if not {'planes', 'shape', 'xor', 'ids', 'embeddings'}.issubset(headers):
                return None, headers
            assert headers['planes']['dtype'] == 'uint8' and headers['planes']['shape'][0] == 4
            assert headers['shape']['shape'] == [4]
    return 'lossless_RGB_output_pack', headers


def prepare():
    started = time.monotonic()
    assert not OUT.exists(), 'Keep previous preparation evidence'
    assert shutil.disk_usage(ROOT).free > 30 * 1024**3
    OUT.mkdir(); (OUT / 'transport').mkdir()
    bind_transport()
    original = ROOT / 'outputs/cctv_dgp_bank_archive_cleanup_v1'
    assert read(original / 'independent_audit.json')['complete']
    snapshot = original / 'remote_receipts/protected_after.json.gz'
    with gzip.open(snapshot, 'rb') as stream:
        previous = json.load(stream)
    protocol_path = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_vm/protocol.json'
    protocol = read(protocol_path)
    current_sources = {PurePosixPath(name).relative_to('outputs').as_posix()
                       for name in protocol['local_sources'] if name.startswith('outputs/')}
    metadata_groups = {name: [] for name in GROUPS}
    for row in previous['metadata']:
        rel = PurePosixPath(row[0])
        if len(rel.parts) < 4 or rel.parts[0] not in GROUPS or rel.parts[1] != 'outputs':
            continue
        if not stat.S_ISREG(row[1]) or row[6:8] != [1, 1001] or rel.suffix not in {'.png', '.npy', '.npz'}:
            continue
        if 'embedding' in rel.name or any(any(word in part.lower() for word in ('cache', 'gradient', 'direction')) for part in rel.parts[2:]):
            continue
        assert row[0] not in current_sources, 'Current packet source must stay on VM'
        metadata_groups[rel.parts[0]].append(row)
    candidates, evidence, summaries, retained_arrays = [], {}, {}, []
    for vm, local in GROUPS.items():
        folder = ROOT / 'outputs' / local
        manifest_path = folder / 'export_manifest.json'
        manifest = read(manifest_path)
        assert manifest['complete'], 'Partial export cannot authorize offload'
        for rel, expected in manifest['files_sha256'].items():
            path = folder.joinpath(*PurePosixPath(rel).parts)
            assert path.resolve().is_relative_to(folder.resolve()) and path.is_file() and sha(path) == expected, rel
            evidence[path.relative_to(ROOT).as_posix()] = expected
        evidence[manifest_path.relative_to(ROOT).as_posix()] = sha(manifest_path)
        evidence[(folder / 'protocol.json').relative_to(ROOT).as_posix()] = sha(folder / 'protocol.json')
        assert evidence[(folder / 'protocol.json').relative_to(ROOT).as_posix()] == manifest['protocol_sha256']
        group_rows = []
        for row in metadata_groups[vm]:
            rel = PurePosixPath(row[0])
            exported_rel = PurePosixPath(*rel.parts[1:]).as_posix()
            expected = manifest['files_sha256'][exported_rel]
            path = folder.joinpath(*rel.parts[1:])
            assert path.stat().st_size == row[2] and evidence[path.relative_to(ROOT).as_posix()] == expected
            kind, header = payload_kind(path)
            if kind is None:
                retained_arrays.append(dict(relative_path=row[0], metadata=row, sha256=expected,
                                           local_backup=path.relative_to(ROOT).as_posix(), payload_header=header))
                continue
            group_rows.append(dict(relative_path=row[0], path='/home/janusdominic0/forensic-dgp/' + row[0],
                       metadata=row, sha256=expected, local_backup=path.relative_to(ROOT).as_posix(),
                       local_backup_sha256=expected, local_complete_return_verified=True,
                       payload_kind=kind, payload_header=header, recovery_member='forensic-dgp/' + row[0]))
        assert group_rows
        candidates.extend(group_rows)
        summaries[vm] = dict(files=len(group_rows), payload_bytes=sum(row['metadata'][2] for row in group_rows),
                             complete_local_return=folder.relative_to(ROOT).as_posix(),
                             exported_scientific_qualification_not_implied=True)
        print(dict(verified_local_return=local, image_output_copies=len(group_rows),
                   payload_GiB=summaries[vm]['payload_bytes']/1024**3), flush=True)
    candidates.sort(key=lambda row: row['relative_path'])
    assert len({row['path'] for row in candidates}) == len(candidates)
    assert len({row['metadata'][5] for row in candidates}) == len(candidates)
    assert sum(row['metadata'][2] for row in candidates) > 11 * 1024**3
    backup_manifest = dict(format='exact-image-output-recovery-v1', complete=True,
        research_root='/home/janusdominic0/forensic-dgp', instance_id=t.INSTANCE_ID,
        UTC=datetime.now(timezone.utc).isoformat(), groups=summaries,
        candidate_count=len(candidates), candidates=candidates, restored_model_training_calls=0,
        retained_non_image_arrays=retained_arrays,
        scopes='Exact output payloads only; original returned archives/checkpoints/failures are separately retained',
        restore_script_sha256=sha(RESTORE))
    manifest_path = OUT / 'recovery_manifest.json'
    write(manifest_path, backup_manifest)
    write(OUT / 'local_return_bindings.json', evidence)
    archive = OUT / BACKUP_NAME
    with tarfile.open(archive, 'x', format=tarfile.PAX_FORMAT) as tar:
        for n, row in enumerate(candidates, 1):
            file = ROOT / row['local_backup']
            info = tarfile.TarInfo(row['recovery_member'])
            info.size = row['metadata'][2]; info.mode = stat.S_IMODE(row['metadata'][1])
            info.mtime = row['metadata'][3] // 1000000000
            info.uid = info.gid = 1001; info.uname = info.gname = 'janusdominic0'
            with file.open('rb') as stream:
                tar.addfile(info, stream)
            if n % 10000 == 0:
                print(dict(recovery_payload_files=n, of=len(candidates)), flush=True)
        for file, name in [(manifest_path, '_recovery/recovery_manifest.json'),
                           (RESTORE, '_recovery/restore_cctv_dgp_vm_output_offload_v1.py')]:
            tar.add(file, arcname=name, recursive=False)
    receipt = dict(complete=True, archive_sha256=sha(archive), bytes=archive.stat().st_size,
                   recovery_manifest_sha256=sha(manifest_path), payload_bytes=sum(row['metadata'][2] for row in candidates),
                   candidate_count=len(candidates), full_member_audit_required=True,
                   source_returns_already_downloaded=True, newly_created_portable_backup=True,
                   files_removed=0, model_gradient_or_training_calls=0, seconds=time.monotonic()-started)
    write(OUT / 'recovery_export.json', receipt)
    (OUT / (BACKUP_NAME + '.sha256')).write_text(receipt['archive_sha256'] + '  ' + BACKUP_NAME + '\n', encoding='ascii')
    print(dict(complete=True, stage='local_backup_prepared', image_output_copies=len(candidates),
               payload_GiB=receipt['payload_bytes']/1024**3, files_removed=0), flush=True)


def freeze():
    assert not (OUT / 'plan.json').exists()
    recovery = read(OUT / 'recovery_manifest.json')
    export = read(OUT / 'recovery_export.json')
    independent = read(OUT / 'independent_backup_audit.json')
    assert independent['complete'] and independent['archive_sha256'] == export['archive_sha256']
    assert independent['candidate_count'] == recovery['candidate_count']
    assert sha(OUT / BACKUP_NAME) == export['archive_sha256']
    plan0 = read(ROOT / 'outputs/cctv_dgp_bank_archive_cleanup_v1/plan.json')
    local_protected = ROOT / 'outputs/cctv_dgp_bank_archive_cleanup_v1/local_protected_sha256.json'
    plan = dict(complete=True, instance_id=t.INSTANCE_ID, research_root='/home/janusdominic0/forensic-dgp',
        authorization_scope='inactive image-output copies offloaded to verified Windows recovery backup',
        remote_script_sha256=sha(WORKER), local_driver_sha256=sha(Path(__file__)),
        recovery_manifest_sha256=sha(OUT / 'recovery_manifest.json'), portable_backup_sha256=export['archive_sha256'],
        portable_backup_independently_verified=True, local_backup_audit_sha256=sha(OUT / 'independent_backup_audit.json'),
        candidate_count=recovery['candidate_count'], candidates=recovery['candidates'], groups=recovery['groups'],
        retained_non_image_arrays=recovery['retained_non_image_arrays'],
        current_packet_VM_source_paths=sorted(PurePosixPath(name).relative_to('outputs').as_posix()
                       for name in read(ROOT / 'outputs/cctv_dgp_bank_comparison_v1_vm/protocol.json')['local_sources']
                       if name.startswith('outputs/')),
        current_packet_protocol_sha256=plan0['current_packet_protocol_sha256'],
        current_packet_archive_sha256=plan0['current_packet_archive_sha256'],
        local_protected_manifest=local_protected.relative_to(ROOT).as_posix(), local_protected_manifest_sha256=sha(local_protected),
        install_reservation_bytes=plan0['install_reservation_bytes'],
        post_install_free_requirement_bytes=plan0['post_install_free_requirement_bytes'],
        scientific_cache_bytes_not_selected=True, checkpoints_sources_splits_gates_logs_not_selected=True,
        per_phase_seconds_cap=900, model_gradient_or_training_calls=0, VM_started=False)
    write(OUT / 'plan.json', plan)
    print(dict(complete=True, plan_sha256=sha(OUT / 'plan.json'), output_copies=plan['candidate_count'], files_removed=0), flush=True)


def execute(phase):
    started = time.monotonic(); bind_transport()
    plan_path = OUT / 'plan.json'; plan = read(plan_path); pin = sha(plan_path)
    assert plan['complete'] and sha(WORKER) == plan['remote_script_sha256']
    assert not (OUT / (phase + '_execution.json')).exists(), 'Retain earlier execution evidence'
    if phase == 'apply':
        verified = read(OUT / 'verify_readback.json')
        assert verified['complete'] and verified['files_removed'] == 0 and verified['plan_sha256'] == pin
        assert sha(OUT / BACKUP_NAME) == plan['portable_backup_sha256']
        assert sha(OUT / 'independent_backup_audit.json') == plan['local_backup_audit_sha256']
    for rel, expected in read(ROOT / plan['local_protected_manifest']).items():
        assert sha(ROOT / rel) == expected, rel
    write(OUT / ('instance_' + phase + '.json'), t.api('api_' + phase))
    remote_script = '/home/janusdominic0/cctv_dgp_vm_output_offload_v1_vm.py'
    remote_plan = '/home/janusdominic0/dgp_output_offload_20261010_v1_plan.json'
    if phase == 'verify':
        for file, dest, stem in [(WORKER, remote_script, 'worker_upload'), (plan_path, remote_plan, 'plan_upload')]:
            t.invoke([str(t.CLOUD), 'compute', 'scp'] + t.BASE + ['--scp-flag=' + flag for flag in t.FLAGS] +
                     [str(file), 'janusdominic0@forensic-dgp-thesis:' + dest], stem, 180, False)
    if phase in {'verify', 'apply'}:
        command = 'python3 -B ' + shlex.quote(remote_script) + ' --plan ' + shlex.quote(remote_plan) + ' --plan-sha ' + pin + ' --phase ' + phase
        t.invoke([str(t.CLOUD), 'compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + t.BASE +
                 ['--ssh-flag=' + flag for flag in t.FLAGS] + ['--command=' + command], phase, 960, False)
        if phase == 'verify':
            src = 'from pathlib import Path;print(Path(' + repr(REMOTE_OUT + '/verify_receipt.json') + ').read_text())'
            verified = read(t.ssh_source(src, 'verify_readback', 60))
            assert verified['complete'] and verified['plan_sha256'] == pin
            write(OUT / 'verify_readback.json', verified)
        else:
            for name in ['dgp-output-offload-20261010-v1-receipts.tar.gz', 'dgp-output-offload-20261010-v1-export.json']:
                t.invoke([str(t.CLOUD), 'compute', 'scp'] + t.BASE + ['--scp-flag=' + flag for flag in t.FLAGS] +
                         ['janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + name, str(OUT / name)],
                         'download_' + name.replace('.', '_'), 240, False)
    else:
        assert read(OUT / 'independent_cleanup_audit.json')['complete']
        # The extended inventory is sent as a file to stay below PuTTY's command length limit.
        source = t.INVENTORY_SOURCE.replace('files_removed=0,model_gradient_or_training_calls=0',
                    'files_removed=' + str(plan['candidate_count']) + ',model_gradient_or_training_calls=0')
        inventory_file = OUT / 'post_inventory_source.py'
        inventory_file.write_text(source, encoding='utf-8', newline='\n')
        remote_inventory = '/home/janusdominic0/dgp_output_offload_20261010_v1_inventory.py'
        t.invoke([str(t.CLOUD), 'compute', 'scp'] + t.BASE + ['--scp-flag=' + flag for flag in t.FLAGS] +
                 [str(inventory_file), 'janusdominic0@forensic-dgp-thesis:' + remote_inventory], 'post_inventory_upload', 120, False)
        raw = t.invoke([str(t.CLOUD), 'compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + t.BASE +
                 ['--ssh-flag=' + flag for flag in t.FLAGS] + ['--command=python3 -B ' + shlex.quote(remote_inventory)], 'post_inventory', 240, True)
        guest = read(raw)
        assert guest['gpu_processes']['exit_code'] == 0 and not guest['gpu_processes']['stdout'].strip()
        assert not guest['current_bank_installed']
        guest.update(dict(free_GiB=guest['disk']['free_bytes']/1024**3,
            conservative_post_install_GiB=(guest['disk']['free_bytes']-plan['install_reservation_bytes'])/1024**3,
            install_reservation_bytes=plan['install_reservation_bytes'], hostkey_pinned=t.HOSTKEY,
            manual_run_space_requirement_met=guest['disk']['free_bytes']-plan['install_reservation_bytes'] >= plan['post_install_free_requirement_bytes'],
            snapshot_not_future_launch_guarantee=True, outputs_recoverable_locally=True))
        assert guest['manual_run_space_requirement_met'], 'Retain cleanup evidence; manual pilot still needs more disk'
        write(OUT / 'inventory_after.json', guest)
        print(dict(complete=True, free_GiB=guest['free_GiB'], conservative_post_install_GiB=guest['conservative_post_install_GiB'],
                   manual_run_space_requirement_met=True, model_gradient_or_training_calls=0), flush=True)
    write(OUT / (phase + '_execution.json'), dict(complete=True, phase=phase, plan_sha256=pin,
        driver_sha256=sha(Path(__file__)), remote_script_sha256=sha(WORKER), VM_started=False,
        model_gradient_or_training_calls=0, seconds=time.monotonic()-started))
    print(dict(complete=True, maintenance_phase=phase, model_gradient_or_training_calls=0), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['prepare', 'freeze', 'verify', 'apply', 'post'], required=True)
    args = parser.parse_args()
    if args.phase == 'prepare':
        prepare()
    elif args.phase == 'freeze':
        freeze()
    else:
        execute(args.phase)


if __name__ == '__main__':
    main()
