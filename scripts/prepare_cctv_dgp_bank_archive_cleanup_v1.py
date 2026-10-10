"""Bind four inactive home archives to complete retained Windows backups."""
import gzip
from pathlib import Path
import time
import cctv_dgp_bank_archive_cleanup_v1 as t


def main():
    start = time.monotonic()
    assert not (t.OUT / 'plan.json').exists()
    inventory = t.read(t.OUT / 'inventory_before.json')
    assert inventory['instance_id'] == t.INSTANCE_ID and not inventory['current_bank_installed']
    assert inventory['gpu_processes']['exit_code'] == 0 and not inventory['gpu_processes']['stdout'].strip()
    backups = {
        'cctv-dgp-multiscale-calibration-v1-execution.tar.gz': 'outputs/cctv-dgp-multiscale-calibration-v1-execution.tar.gz',
        'cctv-dgp-multiscale-calibration-v1-results.tar.gz': 'outputs/cctv-dgp-multiscale-calibration-v1-results.tar.gz',
        'dgp-head4-archive-cleanup-v1-receipts.tar.gz': 'outputs/cctv_dgp_head4_archive_cleanup_v1/dgp-head4-archive-cleanup-v1-receipts.tar.gz',
        'dgp-multiscale-archive-cleanup-v1-receipts.tar.gz': 'outputs/cctv_dgp_multiscale_archive_cleanup_v1/dgp-multiscale-archive-cleanup-v1-receipts.tar.gz',
    }
    rows = [row for row in inventory['archives'] if Path(row['path']).name in backups]
    assert len(rows) == 4
    audit = t.read(t.ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_independent_audit_r2.json')
    assert audit['complete']
    candidates = []
    for row in rows:
        name = Path(row['path']).name
        assert row['path'] == '/home/janusdominic0/' + name
        assert row['regular_file'] and not row['symlink'] and row['uid'] == 1001 and row['nlink'] == 1
        local = t.ROOT / backups[name]
        assert local.is_file() and not local.is_symlink() and local.stat().st_size == row['bytes']
        digest = t.sha(local)
        if name.startswith('cctv-'):
            assert Path(str(local) + '.sha256').read_text().split() == [digest, name]
            if '-results.' in name:
                export = t.read(t.ROOT / 'outputs/cctv-dgp-multiscale-calibration-v1-export.json')
                assert export['complete'] and export['archive_sha256'] == audit['archive_sha256'] == digest
                assert export['bytes'] == row['bytes']
        else:
            export = t.read(local.parent / name.replace('-receipts.tar.gz', '-export.json'))
            assert export['complete'] and export['archive_sha256'] == digest and export['bytes'] == row['bytes']
        count = 0
        with gzip.open(local, 'rb') as stream:
            for block in iter(lambda: stream.read(1024**2), b''):
                count += len(block)
                assert count < 30 * 1024**3 and time.monotonic() - start < 600
        candidates.append(dict(**row, sha256=digest, local_backup=backups[name],
                               full_gzip_CRC_verified=True, uncompressed_stream_bytes=count))
        print(dict(full_backup_verified=name, bytes=row['bytes']), flush=True)
    packet = t.ROOT / 'outputs/cctv_dgp_bank_comparison_v1_vm'
    p = t.read(packet / 'protocol.json')
    prepared = t.read(t.ROOT / 'outputs/cctv_dgp_bank_comparison_v1_preparation.json')
    archive = t.ROOT / 'outputs/cctv-dgp-bank-comparison-v1-execution.tar.gz'
    assert t.sha(archive) == prepared['archive_sha256']
    assert t.sha(packet / 'protocol.json') == prepared['protocol_sha256']
    protected = dict(p['local_sources'])
    protected[(packet / 'protocol.json').relative_to(t.ROOT).as_posix()] = prepared['protocol_sha256']
    protected[archive.relative_to(t.ROOT).as_posix()] = prepared['archive_sha256']
    protected['CCTV_DGP_BANK_COMPARISON_V1_VM.md'] = t.read(t.ROOT / 'outputs/cctv_dgp_bank_comparison_v1_closure_checks.json')['guide_sha256']
    flow = t.read(t.ROOT / 'scratch/dgp-bank-app-preflight-20261010-v1/flow.json')
    protected.update(flow['sources'])
    asset_bytes = 0
    for name, digest in p['assets_sha256'].items():
        file = packet / name
        assert t.sha(file) == digest
        asset_bytes += file.stat().st_size
        protected[file.relative_to(t.ROOT).as_posix()] = digest
    for name, digest in protected.items():
        assert t.sha(t.ROOT / name) == digest, name
    t.write(t.OUT / 'local_protected_sha256.json', protected)
    install_bytes = archive.stat().st_size + asset_bytes + (packet / 'protocol.json').stat().st_size + 32 * 1024**2
    plan = dict(complete=True, instance_id=t.INSTANCE_ID, research_root='/home/janusdominic0/forensic-dgp',
                authorization_scope='inventory-first hash-bound inactive archive copies only',
                inventory_sha256=t.sha(t.OUT / 'inventory_before.json'),
                remote_script_sha256=t.sha(t.ROOT / 'scripts/cctv_dgp_bank_archive_cleanup_v1_vm.py'),
                preparation_script_sha256=t.sha(Path(__file__)), candidates=candidates,
                estimated_allocated_reclaim_bytes=sum(row['allocated_bytes'] for row in candidates),
                local_protected_manifest_sha256=t.sha(t.OUT / 'local_protected_sha256.json'),
                local_protected_bindings=len(protected), current_packet_assets=len(p['assets_sha256']),
                current_packet_archive_sha256=prepared['archive_sha256'],
                current_packet_protocol_sha256=prepared['protocol_sha256'],
                install_reservation_bytes=install_bytes, post_install_free_requirement_bytes=16 * 1024**3,
                maintenance_cap_seconds=900, caches_checkpoints_splits_failures_current_packet_excluded=True,
                model_gradient_or_training_calls=0, files_removed=0, seconds=time.monotonic()-start)
    t.write(t.OUT / 'plan.json', plan)
    print(dict(complete=True, candidates=4,
               reclaim_GiB=plan['estimated_allocated_reclaim_bytes']/1024**3,
               conservative_post_install_GiB=(inventory['disk']['free_bytes']+plan['estimated_allocated_reclaim_bytes']-install_bytes)/1024**3,
               plan_sha256=t.sha(t.OUT / 'plan.json'), files_removed=0), flush=True)


if __name__ == '__main__':
    main()
