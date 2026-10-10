"""Execute the new exact maintenance plan only; manual model work is separate."""
import argparse
from pathlib import Path
import shlex
import time
import cctv_dgp_multiscale_archive_cleanup_v1 as t


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', choices=['verify', 'apply', 'post'], required=True)
    args = p.parse_args()
    started = time.monotonic()
    plan_path = t.OUT / 'plan.json'
    plan = t.read(plan_path)
    pin = t.sha(plan_path)
    script = t.ROOT / 'scripts/cctv_dgp_multiscale_archive_cleanup_v1_vm.py'
    assert t.sha(script) == plan['remote_script_sha256'] and len(plan['candidates']) == 2
    assert not (t.OUT / (args.phase + '_execution.json')).exists(), 'No repeat mutation or previous evidence overwrite'
    if args.phase != 'post':
        for row in plan['candidates']:
            assert row['full_gzip_CRC_verified'] and t.sha(t.ROOT / row['local_backup']) == row['sha256']
        for name, digest in t.read(t.OUT / 'local_protected_sha256.json').items():
            assert t.sha(t.ROOT / name) == digest
    t.write(t.OUT / ('instance_' + args.phase + '.json'), t.api('api_' + args.phase))
    remote_script = '/home/janusdominic0/cctv_dgp_multiscale_archive_cleanup_v1_vm.py'
    remote_plan = '/home/janusdominic0/dgp_multiscale_archive_cleanup_v1_plan.json'
    if args.phase == 'verify':
        for file, dest, stem in [(script, remote_script, 'worker_upload'), (plan_path, remote_plan, 'plan_upload')]:
            t.invoke([str(t.CLOUD), 'compute', 'scp'] + t.BASE + ['--scp-flag=' + flag for flag in t.FLAGS] +
                     [str(file), 'janusdominic0@forensic-dgp-thesis:' + dest], stem, 120, False)
    if args.phase in {'verify', 'apply'}:
        if args.phase == 'apply':
            receipt = t.read(t.OUT / 'verify_readback.json')
            assert receipt['complete'] and receipt['plan_sha256'] == pin and receipt['files_removed'] == 0
            assert receipt['candidate_count'] == 2 and receipt['protected_metadata_entries'] > 100 and receipt['protected_byte_hashes'] > 100
        command = 'python3 -B ' + shlex.quote(remote_script) + ' --plan ' + shlex.quote(remote_plan) + ' --plan-sha ' + pin + ' --phase ' + args.phase
        t.invoke([str(t.CLOUD), 'compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + t.BASE +
                 ['--ssh-flag=' + flag for flag in t.FLAGS] + ['--command=' + command], args.phase, 960, False)
        if args.phase == 'verify':
            source = 'from pathlib import Path;print(Path(' + repr(t.REMOTE_OUT + '/verify_receipt.json') + ').read_text())'
            receipt = t.read(t.ssh_source(source, 'verify_readback', 60))
            assert receipt['complete'] and receipt['plan_sha256'] == pin
            t.write(t.OUT / 'verify_readback.json', receipt)
        else:
            for name in ['dgp-multiscale-archive-cleanup-v1-receipts.tar.gz', 'dgp-multiscale-archive-cleanup-v1-export.json']:
                t.invoke([str(t.CLOUD), 'compute', 'scp'] + t.BASE + ['--scp-flag=' + flag for flag in t.FLAGS] +
                         ['janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' + name, str(t.OUT / name)],
                         'download_' + name.replace('.', '_'), 180, False)
    else:
        assert t.read(t.OUT / 'independent_audit.json')['complete']
        guest = t.read(t.ssh_source(t.INVENTORY_SOURCE, 'inventory_after'))
        removed = {row['path'] for row in plan['candidates']}
        assert removed.isdisjoint({row['path'] for row in guest['archives']})
        assert guest['gpu_processes']['exit_code'] == 0 and not guest['gpu_processes']['stdout'].strip()
        assert not guest['current_multiscale_installed']
        guest.update(dict(free_GiB=guest['disk']['free_bytes']/1024**3,
              conservative_post_install_GiB=(guest['disk']['free_bytes']-plan['install_reservation_bytes'])/1024**3,
              installation_reservation_bytes=plan['install_reservation_bytes'],
              hostkey_pinned=t.HOSTKEY, target_archives_remaining=0, snapshot_not_future_launch_guarantee=True))
        assert guest['disk']['free_bytes']-plan['install_reservation_bytes'] >= plan['post_install_free_requirement_bytes']
        t.write(t.OUT / 'inventory_after.json', guest)
        print(dict(complete=True, free_GiB=guest['free_GiB'], conservative_post_install_GiB=guest['conservative_post_install_GiB'],
                   target_archives_remaining=0, model_gradient_or_training_calls=0), flush=True)
    t.write(t.OUT / (args.phase + '_execution.json'), dict(complete=True, phase=args.phase, plan_sha256=pin,
            driver_sha256=t.sha(Path(__file__)), remote_script_sha256=t.sha(script), VM_started=False,
            model_gradient_or_training_calls=0, seconds=time.monotonic()-started))
    print(dict(complete=True, maintenance_phase=args.phase, model_gradient_or_training_calls=0), flush=True)


if __name__ == '__main__':
    main()
