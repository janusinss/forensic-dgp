"""Close the audited local backup; remove only its redundant transfer blocks."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_local_research_cache_backup_20261007_v1'
PIN = 'af7a583f06cbc9a245e9c93815231be53d53c76d4a529578b4551c8bc318e291'
TOTAL = 39448585279


def sha(p):
    with p.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save(p, data):
    with p.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, indent=2) + '\n')


def close():
    started = time.monotonic()
    assert not (OUT / 'download.lock').exists(), 'Wait for the downloader to exit'
    assert not (OUT / 'closure_manifest.json').exists(), 'Preserve completed closure'
    completed = json.loads((OUT / 'complete.json').read_text())
    independent = json.loads((OUT / 'independent_local_cache_audit.json').read_text())
    source = json.loads((OUT / 'source_final/independent_source_resource_audit.json').read_text())
    assert completed['complete'] and completed['actual_cache_files'] == 4431
    assert independent['complete'] and independent['verified_cache_files'] == 4431
    assert completed['actual_cache_bytes'] == independent['verified_cache_bytes'] == TOTAL
    assert completed['manifest_sha256'] == independent['manifest_sha256'] == PIN
    assert source['complete'] and source['original_source_disk_preserved']
    assert source['unwanted_snapshot_no_longer_active'] and source['source_cache_stamps'] == 4431
    manifest = OUT / 'cache_source_inventory.json'
    assert sha(manifest) == PIN
    rows = json.loads(manifest.read_text())['files']
    raw_root = OUT / 'cache_files'
    stamps = {}
    for row in rows:
        p = raw_root.joinpath(*row['path'].lstrip('/').split('/'))
        assert p.resolve().is_relative_to(raw_root.resolve()) and not p.is_symlink()
        s = p.stat()
        assert p.is_file() and s.st_size == row['bytes']
        stamps[p] = (s.st_size, s.st_mtime_ns, s.st_ino)
    chunk_root = (OUT / 'chunks').resolve()
    chunks = sorted(chunk_root.glob('*.bin'))
    assert len(chunks) == 147 and sum(p.stat().st_size for p in chunks) == TOTAL
    plan = []
    for i, p in enumerate(chunks):
        assert time.monotonic() - started < 2400, 'Finite transfer-block closure limit'
        assert p.name == f'{i:04d}.bin' and not p.is_symlink()
        assert p.resolve().parent == chunk_root and p.is_file()
        s = p.stat()
        assert s.st_nlink == 1
        proof = json.loads(p.with_suffix('.json').read_text())
        assert proof['manifest_sha256'] == PIN and proof['start'] == i * 256 * 1024**2
        assert proof['bytes'] == s.st_size and sha(p) == proof['sha256']
        plan.append({'path': str(p), 'bytes': s.st_size, 'mtime_ns': s.st_mtime_ns,
                     'inode': s.st_ino, 'sha256': proof['sha256']})
        if (i + 1) % 20 == 0:
            print(json.dumps({'verified_disposable_blocks': i + 1, 'total': 147}), flush=True)
    save(OUT / 'transfer_block_cleanup_plan.json', {'raw_cache_audit_passed': True, 'files': plan})
    free_before = shutil.disk_usage(OUT).free
    with (OUT / 'transfer_block_deletions.jsonl').open('x', encoding='utf-8', newline='\n') as ledger:
        for row in plan:
            p = Path(row['path'])
            assert p.resolve().parent == chunk_root and not p.is_symlink()
            s = p.stat()
            assert (s.st_size, s.st_mtime_ns, s.st_ino, s.st_nlink) == (
                row['bytes'], row['mtime_ns'], row['inode'], 1)
            p.unlink()
            assert not p.exists()
            ledger.write(json.dumps(row) + '\n'); ledger.flush()
    assert not list(chunk_root.glob('*.bin'))
    for p, before in stamps.items():
        s = p.stat()
        assert (s.st_size, s.st_mtime_ns, s.st_ino) == before
    save(OUT / 'transfer_block_cleanup_receipt.json', {
        'complete': True, 'deleted_redundant_blocks': 147, 'deleted_bytes': TOTAL,
        'actual_cache_files_preserved': 4431, 'source_VM_files_deleted': 0,
        'failed_transfer_partials_preserved': True,
        'Windows_free_bytes_before': free_before, 'Windows_free_bytes_after': shutil.disk_usage(OUT).free})
    guide = ROOT / 'VM_CACHE_BACKUP_RESTORE_20261007_V1.md'
    prior_guide = guide.read_bytes()
    with (OUT / 'runbook_before_completion.md').open('xb') as stream:
        stream.write(prior_guide)
    old_status = ('Status: actual cache transfer is in progress. Completion requires `complete.json`\n'
                  'and `independent_local_cache_audit.json` in that folder, both reporting\n'
                  '`complete: true`, 4,431 files and 39,448,585,279 bytes. A checksum inventory or\n'
                  'successful transfer probe alone does not establish a complete backup.')
    text = prior_guide.decode('utf-8').replace('\r\n', '\n')
    assert old_status in text
    text = text.replace(old_status, 'Status: **complete**. Both `complete.json` and\n'
                        '`independent_local_cache_audit.json` confirm all **4,431 actual cache files /\n'
                        '39,448,585,279 bytes**, with every file SHA256 matching the frozen source.\n'
                        'The 147 redundant transfer blocks were removed after that audit; original\n'
                        'VM files, failed-transfer evidence and the earlier partial remain.', 1)
    guide.write_text(text, encoding='utf-8', newline='\n')
    metadata = OUT / 'recovery_metadata'
    shutil.copyfile(guide, metadata / guide.name)
    for name in ('complete.json', 'independent_local_cache_audit.json', 'transfer_block_cleanup_receipt.json'):
        shutil.copyfile(OUT / name, metadata / name)
    old_docs = ROOT / 'outputs/cctv_dgp_vm_migration_backup_20261007_v1/before_docs'
    doc_receipts = []
    addition = ('\n**Latest maintenance — 7 October 2026: actual research caches backed up locally and audited.**\n\n'
                'The user chooses a Windows backup in preparation for a possible later Google Cloud VM.\n'
                'All three historical caches are now copied as actual files: 4,431 files /\n'
                '39,448,585,279 bytes (36.74 GiB), including both expanded-feature caches and the\n'
                'V16 r2 cache. Every reconstructed file matches its frozen VM SHA256; a separate\n'
                'read-only verifier re-reads all 4,431 local files and passes. The source caches\n'
                'retain their original sizes, inodes and timestamps. No source cache is removed.\n\n'
                'The initially selected cloud snapshot was an agent destination error. It is deleted;\n'
                'independent live checks confirm its absence and the original running VM/disk.\n'
                'The earlier migration recovery ZIP contains metadata only and its cloud restore\n'
                'instructions are obsolete; its original bytes and the rollback evidence remain.\n'
                'The actual local files and current restore guide supersede that route.\n\n'
                'The installed Windows SDK cannot carry the interactive binary control stream\n'
                'through its automatic PuTTY stdin reply. The corrected transfer uses the SDK-generated\n'
                'console SSH connection with cached host-key checking. A demonstrated Windows\n'
                'temporary-path limit was fixed using short names; completed blocks were retained\n'
                'through an owned local-transfer pause. The early failures, original partial and\n'
                'source revisions remain. Only 147 redundant new Windows transfer blocks are\n'
                'removed after the independent full-file audit, reclaiming 36.74 GiB locally.\n\n'
                'This is a three-cache backup, not a boot-disk image or proof that every separate\n'
                'checkpoint, dataset and environment is bundled. Existing model, split, provenance\n'
                'and gate-failure files remain. The GPU is idle; latest source free space is\n'
                f"{source['VM_free_GiB']:.2f} GiB, still above the unchanged 6 GiB V30 requirement.\n"
                'Actual training remains human/manual on the existing L4 at ~/forensic-dgp.\n'
                'No new VM, optimizer, pilot, application change or source shutdown occurs.\n'
                'V29 remains rejected for application promotion; useful DGP restoration, all\n'
                'covering families, independent final review and the full app flow remain required.\n'
                'Goal active/incomplete.\n\n'
                '[Local backup and restore guide](VM_CACHE_BACKUP_RESTORE_20261007_V1.md) ·\n'
                '[Independent local audit](outputs/cctv_dgp_local_research_cache_backup_20261007_v1/independent_local_cache_audit.json) ·\n'
                '[Source/resource audit](outputs/cctv_dgp_local_research_cache_backup_20261007_v1/source_final/independent_source_resource_audit.json)\n\n'
                'Earlier complete document bodies remain preserved history.\n\n').encode('utf-8')
    for name in ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md'):
        p = ROOT / name
        before = p.read_bytes()
        assert before == (old_docs / name).read_bytes(), 'Preserve intervening user edits'
        split = before.index(b'\n') + 1
        after = before[:split] + addition + before[split:]
        p.write_bytes(after)
        assert p.read_bytes() == after and after[:split] + after[split + len(addition):] == before
        doc_receipts.append({'path': str(p), 'before_sha256': hashlib.sha256(before).hexdigest(),
                             'after_sha256': sha(p), 'complete_previous_body_preserved': True})
    copies = json.loads((metadata / 'copied_metadata.json').read_text())['copies']
    assert all(sha(Path(r['copy'])) == r['sha256'] for r in copies)
    result = {'complete': True, 'scope': 'Actual local research-cache backup and source preservation',
              'actual_cache_files': 4431, 'actual_cache_bytes': TOTAL, 'manifest_sha256': PIN,
              'local_cache_root': str(raw_root), 'source_cache_files_deleted': 0,
              'unwanted_snapshot_deleted': True, 'original_VM_and_disk_preserved': True,
              'training_started': False, 'restoration_goal_complete': False,
              'source_recovery_metadata_copies_verified': len(copies), 'documents': doc_receipts,
              'guide_sha256': sha(guide), 'seconds': time.monotonic() - started}
    save(OUT / 'closure_manifest.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--close', action='store_true', required=True)
    parser.parse_args()
    close()
