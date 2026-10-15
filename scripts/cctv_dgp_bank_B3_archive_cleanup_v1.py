"""Authorized exact B3 archive-copy cleanup and conservative B4 reuse space check."""
import argparse
from pathlib import Path
import shlex
import time

import cctv_dgp_bank_archive_cleanup_v1 as t
import cctv_dgp_multiscale_archive_cleanup_v1 as transport

ROOT = t.ROOT
OUT = ROOT / 'outputs/cctv_dgp_bank_B3_archive_cleanup_v1'
RETURN = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_B_return'
ANALYSIS = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_B_analysis'
REMOTE = '/home/janusdominic0/dgp_bank_B3_archive_cleanup_v1'
BANK = '/home/janusdominic0/forensic-dgp/cctv_dgp_bank_comparison_v1_r3_B_vm'
WORKER = ROOT / 'scripts/cctv_dgp_bank_B3_archive_cleanup_v1_vm.py'


def cloud_copy(source, destination, stem):
    return t.invoke([str(t.CLOUD), 'compute', 'scp'] + t.BASE +
        ['--scp-flag='+flag for flag in t.FLAGS] + [str(source), str(destination)], stem, 180, False)


def vm_phase(plan, phase, stem):
    command = 'python3 -B '+shlex.quote(REMOTE+'_vm.py')
    command += ' --plan '+shlex.quote(REMOTE+'_plan.json')+' --plan-sha '+t.sha(plan)+' --phase '+phase
    return t.invoke([str(t.CLOUD), 'compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + t.BASE +
        ['--ssh-flag='+flag for flag in t.FLAGS] + ['--command='+command], stem, 960, phase == 'inventory')


def backups(plan):
    for row in plan['candidates']:
        file = ROOT / row['local_backup']
        assert file.resolve().is_relative_to(ROOT.resolve()) and file.is_file() and not file.is_symlink()
        assert file.stat().st_size == row['bytes'] and t.sha(file) == row['sha256']
    for name, expected in t.read(OUT / 'local_protected_sha256.json').items():
        assert t.sha(ROOT / name) == expected, name


def prepare():
    assert not OUT.exists()
    audit, imported = [t.read(ANALYSIS / name) for name in ['independent_audit.json', 'import.json']]
    execution = t.read(ANALYSIS / 'failure_execution_audit.json')
    transfer_path = ROOT / 'outputs/cctv_dgp_bank_comparison_B4_transfer_v1_audit.json'
    transfer = t.read(transfer_path)
    assert all(record['complete'] for record in [audit, imported, execution, transfer])
    assert audit['optimizer_updates_in_VM'] == execution['actual_optimizer_updates'] == 0
    assert audit['saved_paired_raw_PNG_cases_checked'] == audit['native_compositions_checked'] == 0
    assert execution['missing_hash_API_reproduced'] and transfer['real_transfer_CLI_passed_with_loader_file_digest_absent']
    assert audit['manifest_sha256'] == imported['export_manifest_sha256']
    OUT.mkdir(); (OUT / 'transport').mkdir()
    t.write(OUT / 'instance_before.json', t.api('api_before'))
    prepared = t.read(ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_B_preparation.json')
    result = t.read(ROOT / 'outputs/cctv-dgp-bank-comparison-v1-r3-B-export.json')
    entries = []
    for name, expected, size in [
        ('cctv-dgp-bank-comparison-v1-r3-B-execution.tar.gz', prepared['archive_sha256'], prepared['archive_bytes']),
        ('cctv-dgp-bank-comparison-v1-r3-B-results.tar.gz', imported['archive_sha256'], result['bytes'])]:
        file = ROOT / 'outputs' / name
        assert file.stat().st_size == size and t.sha(file) == expected
        entries.append(dict(path='/home/janusdominic0/'+name, bytes=size, sha256=expected,
                            local_backup=file.relative_to(ROOT).as_posix()))
    manifest = t.read(RETURN / 'export_manifest.json')
    keep = {(RETURN / name).relative_to(ROOT).as_posix(): expected for name, expected in manifest['files'].items()}
    keep.update(t.read(ROOT / 'scratch/dgp-bank-app-preflight-20261010-v1/flow.json')['sources'])
    keep['outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'] = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    t.write(OUT / 'local_protected_sha256.json', keep)
    basic = dict(complete=True, instance_id=t.INSTANCE_ID,
        authorization_scope='two completed or superseded B3 home archives only; full verified local backups; all packet research assets retained',
        worker_sha256=t.sha(WORKER), candidates=entries,
        independent_return_audit_sha256=t.sha(ANALYSIS / 'independent_audit.json'),
        original_return_import_sha256=t.sha(ANALYSIS / 'import.json'),
        failure_execution_audit_sha256=t.sha(ANALYSIS / 'failure_execution_audit.json'),
        B4_transfer_audit_sha256=t.sha(transfer_path),
        local_protected_sha256=t.sha(OUT / 'local_protected_sha256.json'),
        manual_training_pending=True, model_gradient_or_training_calls=0, maintenance_cap_seconds=900)
    backups(basic)
    request = OUT / 'candidate_request.json'; t.write(request, basic)
    cloud_copy(WORKER, 'janusdominic0@forensic-dgp-thesis:'+REMOTE+'_vm.py', 'worker_upload')
    cloud_copy(request, 'janusdominic0@forensic-dgp-thesis:'+REMOTE+'_plan.json', 'request_upload')
    inventory = t.read(vm_phase(request, 'inventory', 'candidate_inventory'))
    assert inventory['complete'] and inventory['files_removed'] == 0 and len(inventory['candidates']) == 2
    t.write(OUT / 'candidate_inventory.json', inventory)
    plan = {**basic, 'candidates': inventory['candidates'],
        'candidate_inventory_sha256': t.sha(OUT / 'candidate_inventory.json'),
        'allocated_reclaim_bytes': sum(row['allocated_bytes'] for row in inventory['candidates'])}
    t.write(OUT / 'plan.json', plan)
    tails = OUT / 'post_export_logs'; tails.mkdir()
    for name in ['export.log', 'export_exit_code.txt']:
        cloud_copy('janusdominic0@forensic-dgp-thesis:'+BANK+'/'+name, tails/name, 'tail_'+name.replace('.', '_'))
    t.write(tails / 'manifest.json', dict(complete=True, files_sha256={name: t.sha(tails/name) for name in ['export.log', 'export_exit_code.txt']},
        original_archive_not_overwritten=True))
    print(dict(complete=True, phase='prepare', candidates=2, reclaim_GiB=plan['allocated_reclaim_bytes']/1024**3,
        free_before_GiB=inventory['free_bytes']/1024**3, files_removed=0), flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--phase', choices=['prepare', 'verify', 'apply', 'post'], required=True)
    phase = parser.parse_args().phase; start = time.monotonic()
    transport.OUT = OUT; t.OUT = OUT
    if phase == 'prepare':
        prepare(); return
    assert not (OUT / (phase+'_execution.json')).exists()
    plan_path = OUT / 'plan.json'; plan = t.read(plan_path)
    assert plan['worker_sha256'] == t.sha(WORKER)
    if phase in {'verify', 'apply'}:
        backups(plan)
    t.write(OUT / ('instance_'+phase+'.json'), t.api('api_'+phase))
    if phase == 'verify':
        cloud_copy(plan_path, 'janusdominic0@forensic-dgp-thesis:'+REMOTE+'_plan.json', 'bound_plan_upload')
        result = t.read(vm_phase(plan_path, 'verify', 'verify'))
        assert result['complete'] and result['files_removed'] == 0 and result['candidate_count'] == 2
        t.write(OUT / 'verify_readback.json', result)
    elif phase == 'apply':
        assert t.read(OUT / 'verify_readback.json')['plan_sha256'] == t.sha(plan_path)
        result = t.read(vm_phase(plan_path, 'apply', 'apply'))
        assert result['complete'] and result['files_removed'] == 2
        t.write(OUT / 'apply_readback.json', result)
        for name in ['dgp-bank-B3-archive-cleanup-v1-receipts.tar.gz', 'dgp-bank-B3-archive-cleanup-v1-export.json']:
            cloud_copy('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+name, OUT/name, 'download_'+name.replace('.', '_'))
    else:
        assert t.read(OUT / 'independent_audit.json')['complete']
        guest = t.read(t.ssh_source(t.INVENTORY_SOURCE, 'inventory_after'))
        assert guest['gpu_processes']['exit_code'] == 0 and not guest['gpu_processes']['stdout'].strip()
        assert not ({row['path'] for row in guest['archives']} & {row['path'] for row in plan['candidates']})
        route = t.read(ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r4_B_preparation.json')
        packet = ROOT / 'outputs' / route['name']; protocol = t.read(packet / 'protocol.json')
        assert t.sha(packet / 'protocol.json') == route['protocol_sha256']
        install = route['transfer_archive_bytes'] + sum((packet/name).stat().st_size for name in protocol['assets_sha256'])
        install += (packet / 'protocol.json').stat().st_size + 64*1024**2
        guest.update(free_GiB=guest['disk']['free_bytes']/1024**3, install_reservation_bytes=install,
            conservative_B4_reuse_post_install_GiB=(guest['disk']['free_bytes']-install)/1024**3,
            manual_B4_space_requirement_met=guest['disk']['free_bytes']-install >= 16*1024**3,
            archive_reuse_upload_required=True, current_snapshot_not_future_launch_guarantee=True)
        t.write(OUT / 'inventory_after.json', guest)
        print(dict(complete=True, free_GiB=guest['free_GiB'],
            conservative_B4_reuse_post_install_GiB=guest['conservative_B4_reuse_post_install_GiB'],
            manual_B4_space_requirement_met=guest['manual_B4_space_requirement_met']), flush=True)
    t.write(OUT / (phase+'_execution.json'), dict(complete=True, phase=phase, plan_sha256=t.sha(plan_path),
        driver_sha256=t.sha(__file__), worker_sha256=t.sha(WORKER), model_gradient_or_training_calls=0,
        VM_started=False, seconds=time.monotonic()-start))


if __name__ == '__main__':
    main()
