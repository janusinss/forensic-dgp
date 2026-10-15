"""Audited A-result offload through the authorized maintenance-only connection."""
import argparse
from pathlib import Path
import shlex
import time

import cctv_dgp_bank_archive_cleanup_v1 as t
import cctv_dgp_multiscale_archive_cleanup_v1 as transport

ROOT = t.ROOT
OUT = ROOT / 'outputs/cctv_dgp_bank_A_return_offload_v1'
RETURN = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_A_return'
ANALYSIS = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_A_analysis'
REMOTE = '/home/janusdominic0/dgp_bank_A_return_offload_v1'
BANK = '/home/janusdominic0/forensic-dgp/cctv_dgp_bank_comparison_v1_r3_A_vm'
WORKER = ROOT / 'scripts/cctv_dgp_bank_A_return_offload_v1_vm.py'


def cloud_copy(source, destination, stem):
    return t.invoke([str(t.CLOUD), 'compute', 'scp'] + t.BASE +
                    ['--scp-flag=' + f for f in t.FLAGS] + [str(source), str(destination)], stem, 180, False)


def vm_phase(plan, phase, stem):
    command = 'python3 -B ' + shlex.quote(REMOTE + '_vm.py')
    command += ' --plan ' + shlex.quote(REMOTE + '_plan.json')
    command += ' --plan-sha ' + t.sha(plan) + ' --phase ' + phase
    return t.invoke([str(t.CLOUD), 'compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + t.BASE +
                    ['--ssh-flag=' + f for f in t.FLAGS] + ['--command=' + command], stem, 960, phase == 'inventory')


def backups(plan):
    for row in plan['candidates']:
        f = ROOT / row['local_backup']
        assert f.resolve().is_relative_to(ROOT.resolve()) and f.is_file() and not f.is_symlink()
        assert f.stat().st_size == row['bytes'] and t.sha(f) == row['sha256']
    for name, expected in t.read(OUT / 'local_protected_sha256.json').items():
        assert t.sha(ROOT / name) == expected, name


def prepare():
    assert not OUT.exists()
    audit = t.read(ANALYSIS / 'independent_audit.json')
    imported = t.read(ANALYSIS / 'import.json')
    execution = t.read(ANALYSIS / 'execution_audit.json')
    visual = t.read(ANALYSIS / 'visual_review.json')
    assert audit['complete'] and imported['complete'] and execution['complete'] and visual['complete']
    assert audit['optimizer_updates_in_VM'] == execution['actual_optimizer_updates'] == 50
    assert audit['saved_paired_raw_PNG_cases_checked'] == 7810 and visual['viewed_sheets'] == 26
    assert audit['manifest_sha256'] == imported['export_manifest_sha256']
    OUT.mkdir(); (OUT / 'transport').mkdir()
    t.write(OUT / 'instance_before.json', t.api('api_before'))
    manifest = t.read(RETURN / 'export_manifest.json')
    entries = []
    candidate_names = set()
    for name, expected in manifest['files'].items():
        parts = Path(name).parts
        if len(parts) != 5 or parts[:2] != ('outputs', 'bank_comparison_v1'):
            continue
        if parts[2] not in {'baseline', 'A50'} or parts[3] not in {'packs', 'previews', 'native'}:
            continue
        if Path(name).suffix not in {'.npz', '.npy', '.png'}:
            continue
        f = RETURN / name
        assert t.sha(f) == expected
        entries.append(dict(path=BANK + '/' + name, bytes=f.stat().st_size, sha256=expected,
                            local_backup=f.relative_to(ROOT).as_posix()))
        candidate_names.add(name)
    assert len(entries) == 1858
    archive = ROOT / 'outputs/cctv-dgp-bank-comparison-v1-r3-A-results.tar.gz'
    exported = t.read(ROOT / 'outputs/cctv-dgp-bank-comparison-v1-r3-A-export.json')
    assert exported['complete'] and t.sha(archive) == imported['archive_sha256'] == exported['archive_sha256']
    assert archive.stat().st_size == exported['bytes']
    entries.append(dict(path='/home/janusdominic0/' + archive.name, bytes=archive.stat().st_size,
                        sha256=exported['archive_sha256'], local_backup=archive.relative_to(ROOT).as_posix()))
    legacy = ROOT / 'outputs/cctv-dgp-bank-comparison-v1-execution.tar.gz'
    legacy_prepared = t.read(ROOT / 'outputs/cctv_dgp_bank_comparison_v1_preparation.json')
    legacy_audit = t.read(ROOT / 'outputs/cctv_dgp_bank_comparison_v1_transfer_audit.json')
    assert legacy_prepared['complete'] and legacy_audit['complete']
    assert legacy.stat().st_size == legacy_prepared['bytes'] and t.sha(legacy) == legacy_prepared['archive_sha256']
    entries.append(dict(path='/home/janusdominic0/' + legacy.name, bytes=legacy.stat().st_size,
                        sha256=legacy_prepared['archive_sha256'], local_backup=legacy.relative_to(ROOT).as_posix()))
    keep = {str((RETURN / n).relative_to(ROOT).as_posix()): h for n, h in manifest['files'].items() if n not in candidate_names}
    keep.update(t.read(ROOT / 'scratch/dgp-bank-app-preflight-20261010-v1/flow.json')['sources'])
    keep['outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'] = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    t.write(OUT / 'local_protected_sha256.json', keep)
    basic = dict(complete=True, instance_id=t.INSTANCE_ID,
        authorization_scope='inactive A/baseline RGB packs plus returned A and historical v1 execution home archives; full verified local backup',
        worker_sha256=t.sha(WORKER), candidates=entries, independent_return_audit_sha256=t.sha(ANALYSIS / 'independent_audit.json'),
        original_return_import_sha256=t.sha(ANALYSIS / 'import.json'), execution_audit_sha256=t.sha(ANALYSIS / 'execution_audit.json'),
        visual_review_sha256=t.sha(ANALYSIS / 'visual_review.json'), local_protected_sha256=t.sha(OUT / 'local_protected_sha256.json'),
        manual_training_pending=True, model_gradient_or_training_calls=0, maintenance_cap_seconds=900)
    backups(basic)
    request = OUT / 'candidate_request.json'; t.write(request, basic)
    cloud_copy(WORKER, 'janusdominic0@forensic-dgp-thesis:' + REMOTE + '_vm.py', 'worker_upload')
    cloud_copy(request, 'janusdominic0@forensic-dgp-thesis:' + REMOTE + '_plan.json', 'request_upload')
    inventory = t.read(vm_phase(request, 'inventory', 'candidate_inventory'))
    assert inventory['complete'] and inventory['files_removed'] == 0 and len(inventory['candidates']) == 1860
    t.write(OUT / 'candidate_inventory.json', inventory)
    plan = {**basic, 'candidates': inventory['candidates'], 'candidate_inventory_sha256': t.sha(OUT / 'candidate_inventory.json'),
            'allocated_reclaim_bytes': sum(r['allocated_bytes'] for r in inventory['candidates'])}
    t.write(OUT / 'plan.json', plan)
    # Keep both post-export finalizer files independently of the sealed archive.
    tails = OUT / 'post_export_logs'; tails.mkdir()
    for name in ['export.log', 'export_exit_code.txt']:
        cloud_copy('janusdominic0@forensic-dgp-thesis:' + BANK + '/' + name, tails / name, 'tail_' + name.replace('.', '_'))
    t.write(tails / 'manifest.json', dict(complete=True, files_sha256={n: t.sha(tails / n) for n in ['export.log', 'export_exit_code.txt']},
        original_archive_not_overwritten=True))
    print(dict(complete=True, phase='prepare', candidates=1860, reclaim_GiB=plan['allocated_reclaim_bytes']/1024**3,
               free_before_GiB=inventory['free_bytes']/1024**3, files_removed=0), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['prepare', 'verify', 'apply', 'post'], required=True)
    phase = parser.parse_args().phase
    started = time.monotonic()
    transport.OUT = OUT; t.OUT = OUT
    if phase == 'prepare':
        prepare(); return
    assert not (OUT / (phase + '_execution.json')).exists()
    plan_path = OUT / 'plan.json'; plan = t.read(plan_path)
    assert plan['worker_sha256'] == t.sha(WORKER)
    if phase in {'verify', 'apply'}:
        backups(plan)
    t.write(OUT / ('instance_' + phase + '.json'), t.api('api_' + phase))
    if phase == 'verify':
        cloud_copy(plan_path, 'janusdominic0@forensic-dgp-thesis:' + REMOTE + '_plan.json', 'bound_plan_upload')
        r = t.read(vm_phase(plan_path, 'verify', 'verify'))
        assert r['complete'] and r['files_removed'] == 0 and r['candidate_count'] == 1860
        t.write(OUT / 'verify_readback.json', r)
    elif phase == 'apply':
        assert t.read(OUT / 'verify_readback.json')['plan_sha256'] == t.sha(plan_path)
        r = t.read(vm_phase(plan_path, 'apply', 'apply'))
        assert r['complete'] and r['files_removed'] == 1860
        t.write(OUT / 'apply_readback.json', r)
        for name in ['dgp-bank-A-return-offload-v1-receipts.tar.gz', 'dgp-bank-A-return-offload-v1-export.json']:
            cloud_copy('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + name, OUT / name, 'download_' + name.replace('.', '_'))
    else:
        assert t.read(OUT / 'independent_audit.json')['complete']
        guest = t.read(t.ssh_source(t.INVENTORY_SOURCE, 'inventory_after'))
        assert guest['gpu_processes']['exit_code'] == 0 and not guest['gpu_processes']['stdout'].strip()
        assert not ({r['path'] for r in guest['archives']} & {r['path'] for r in plan['candidates']})
        route, = [r for r in t.read(ROOT / 'outputs/cctv_dgp_bank_comparison_v1_r3_preparation.json')['routes'] if r['route'] == 'B']
        packet = ROOT / 'outputs' / route['name']; protocol = t.read(packet / 'protocol.json')
        assert t.sha(packet / 'protocol.json') == route['protocol_sha256']
        install = route['archive_bytes'] + sum((packet/n).stat().st_size for n in protocol['assets_sha256'])
        install += (packet / 'protocol.json').stat().st_size + 32*1024**2
        guest.update(free_GiB=guest['disk']['free_bytes']/1024**3, install_reservation_bytes=install,
                     conservative_B_post_install_GiB=(guest['disk']['free_bytes']-install)/1024**3,
                     manual_B_space_requirement_met=guest['disk']['free_bytes']-install >= 16*1024**3,
                     current_snapshot_not_future_launch_guarantee=True)
        t.write(OUT / 'inventory_after.json', guest)
        print(dict(complete=True, free_GiB=guest['free_GiB'], conservative_B_post_install_GiB=guest['conservative_B_post_install_GiB'],
                   manual_B_space_requirement_met=guest['manual_B_space_requirement_met']), flush=True)
    t.write(OUT / (phase + '_execution.json'), dict(complete=True, phase=phase, plan_sha256=t.sha(plan_path),
        driver_sha256=t.sha(__file__), worker_sha256=t.sha(WORKER), model_gradient_or_training_calls=0,
        VM_started=False, seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
