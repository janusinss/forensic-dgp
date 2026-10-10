"""Freeze exact archive-only deletion after complete local backup validation."""
import gzip
from pathlib import Path
import time
import cctv_dgp_multiscale_archive_cleanup_v1 as t

ROOT = t.ROOT
OUT = t.OUT
NAMES = {'cctv-dgp-head4-capacity-v1-execution.tar.gz', 'cctv-dgp-head4-capacity-v1-results.tar.gz'}


def main():
    started = time.monotonic()
    assert not (OUT / 'plan.json').exists()
    g = t.read(OUT / 'inventory_before.json')
    api = t.read(OUT / 'instance_before.json')
    assert g['complete'] and g['instance_id'] == api['id'] == t.INSTANCE_ID and api['status'] == 'RUNNING'
    assert g['gpu_processes']['exit_code'] == 0 and not g['gpu_processes']['stdout'].strip()
    assert not g['current_multiscale_installed']
    rows = [row for row in g['archives'] if Path(row['path']).name in NAMES]
    assert len(rows) == 2
    head4_audit = ROOT / 'outputs/cctv_dgp_head4_capacity_v1_independent_audit.json'
    audit = t.read(head4_audit)
    assert audit['complete'] and audit['failure_preserved'] and not audit['early_capacity_pass']
    binding = t.read(ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_milestone/closure.json')
    for name, digest in binding['protected_sha256'].items():
        assert t.sha(ROOT / name) == digest, 'Protected local binding differs: ' + name
    prepared = t.read(ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_preparation.json')
    assert prepared['complete'] and prepared['manual_run_pending']
    current_archive = ROOT / 'outputs/cctv-dgp-multiscale-calibration-v1-execution.tar.gz'
    assert t.sha(current_archive) == prepared['archive_sha256']
    packet = ROOT / 'outputs/cctv_dgp_multiscale_calibration_vm_v1'
    protocol = t.read(packet / 'protocol.json')
    assert t.sha(packet / 'protocol.json') == prepared['protocol_sha256']
    asset_bytes = 0
    for name, digest in protocol['assets_sha256'].items():
        assert t.sha(packet / name) == digest
        asset_bytes += (packet / name).stat().st_size
    candidates = []
    for row in rows:
        name = Path(row['path']).name
        local = ROOT / 'outputs' / name
        assert row['regular_file'] and not row['symlink'] and row['uid'] == 1001 and row['nlink'] == 1
        assert local.is_file() and not local.is_symlink() and local.stat().st_size == row['bytes']
        digest = t.sha(local)
        assert Path(str(local) + '.sha256').read_text().split() == [digest, name]
        export_digest = None
        if name.endswith('-results.tar.gz'):
            export = ROOT / 'outputs' / name.replace('-results.tar.gz', '-export.json')
            e = t.read(export)
            assert e['complete'] and e['archive_sha256'] == digest == audit['archive_sha256'] and e['bytes'] == row['bytes']
            export_digest = t.sha(export)
        total = 0
        with gzip.open(local, 'rb') as stream:
            for chunk in iter(lambda: stream.read(1024**2), b''):
                total += len(chunk)
                assert total < 30*1024**3 and time.monotonic()-started < 600
        candidates.append(dict(**row, sha256=digest, local_backup=local.relative_to(ROOT).as_posix(),
                               full_gzip_CRC_verified=True, uncompressed_stream_bytes=total,
                               export_sha256=export_digest, unpacked_failure_and_research_tree_preserved=True))
        print(dict(local_backup_verified=name, full_gzip_CRC=True), flush=True)
    t.write(OUT / 'local_protected_sha256.json', binding['protected_sha256'])
    plan = dict(complete=True, instance_id=t.INSTANCE_ID, research_root='/home/janusdominic0/forensic-dgp',
                authorization_scope='inventory-first hash-bound inactive archive copies only',
                inventory_sha256=t.sha(OUT / 'inventory_before.json'),
                remote_script_sha256=t.sha(ROOT / 'scripts/cctv_dgp_multiscale_archive_cleanup_v1_vm.py'),
                preparation_script_sha256=t.sha(Path(__file__)), candidates=candidates,
                estimated_allocated_reclaim_bytes=sum(row['allocated_bytes'] for row in candidates),
                local_protected_manifest_sha256=t.sha(OUT / 'local_protected_sha256.json'),
                local_protected_bindings=len(binding['protected_sha256']),
                current_packet_archive_sha256=prepared['archive_sha256'], current_packet_protocol_sha256=prepared['protocol_sha256'],
                current_packet_assets=len(protocol['assets_sha256']), current_packet_preserved=True,
                install_reservation_bytes=current_archive.stat().st_size+asset_bytes+(packet/'protocol.json').stat().st_size+32*1024**2,
                post_install_free_requirement_bytes=7*1024**3, maintenance_cap_seconds=900,
                caches_checkpoints_splits_failures_current_packet_excluded=True,
                model_gradient_or_training_calls=0, files_removed=0, seconds=time.monotonic()-started)
    t.write(OUT / 'plan.json', plan)
    print(dict(complete=True, candidates=2, reclaim_GiB=plan['estimated_allocated_reclaim_bytes']/1024**3,
               plan_sha256=t.sha(OUT / 'plan.json'), files_removed=0), flush=True)


if __name__ == '__main__':
    main()
