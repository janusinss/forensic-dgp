"""Independent backup, deletion-ledger and retained-state maintenance audit."""
import gzip
import hashlib
import json
from pathlib import Path,PurePosixPath
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_bank_split_offload_v1_r1'
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def main():
    started=time.monotonic();receipt=OUT/'independent_audit.json';assert not receipt.exists()
    plan=read(OUT/'plan.json');pin=sha(OUT/'plan.json')
    exp=read(OUT/'dgp-bank-split-offload-v1-export.json');archive=OUT/'dgp-bank-split-offload-v1-receipts.tar.gz'
    assert exp['complete'] and archive.stat().st_size==exp['bytes'] and sha(archive)==exp['archive_sha256']
    dest=OUT/'remote_receipts';assert not dest.exists();dest.mkdir();seen=set()
    allowed={'protected_before.json.gz','protected_after.json.gz','verify_receipt.json','cleanup_receipt.json','deletion_ledger.jsonl','manifest.json'}
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            p=PurePosixPath(member.name)
            assert member.isfile() and len(p.parts)==2 and p.parts[0]=='receipts' and p.name in allowed and p.name not in seen
            with tar.extractfile(member) as source,(dest/p.name).open('xb') as sink:
                for b in iter(lambda:source.read(1024**2),b''):sink.write(b)
            seen.add(p.name)
    assert seen==allowed
    manifest=read(dest/'manifest.json');assert manifest['complete'] and manifest['plan_sha256']==pin
    assert set(manifest['files_sha256'])==allowed-{'manifest.json'}
    for n,h in manifest['files_sha256'].items():assert sha(dest/n)==h
    vr=read(dest/'verify_receipt.json');cr=read(dest/'cleanup_receipt.json')
    assert vr['complete'] and cr['complete'] and vr['plan_sha256']==cr['plan_sha256']==pin
    assert vr['worker_sha256']==cr['worker_sha256']==plan['worker_sha256']==sha(ROOT/'scripts/cctv_dgp_bank_split_offload_v1_vm.py')
    rows=[json.loads(line) for line in (dest/'deletion_ledger.jsonl').read_text().splitlines()]
    expected={r['path']:r for r in plan['candidates']}
    assert len(rows)==cr['files_removed']==vr['candidate_count']==len(expected)==930
    assert {r['path'] for r in rows}==set(expected)==set(cr['removed_paths'])
    for r in rows:
        p=expected[r['path']];assert r['backup_verified'] and r['nlink']==p['nlink']==1 and p['uid']==1001
        assert all(r[k]==p[k] for k in ['sha256','inode','allocated_bytes','local_backup'])
        backup=ROOT/r['local_backup'];assert backup.is_file() and not backup.is_symlink() and backup.stat().st_size==p['bytes'] and sha(backup)==r['sha256']
    with gzip.open(dest/'protected_before.json.gz','rb') as f:before=json.load(f)
    with gzip.open(dest/'protected_after.json.gz','rb') as f:after=json.load(f)
    assert before==after and canonical(before)==cr['protected_canonical_before']==cr['protected_canonical_after']==vr['protected_canonical_sha256']
    assert len(before['metadata'])==cr['protected_metadata_entries'] and len(before['bank_retained_bytes_sha256'])==cr['retained_bank_hashes']
    assert cr['model_gradient_or_training_calls']==0 and not cr['VM_started'] and cr['checkpoints_inputs_splits_logs_metrics_provenance_and_failures_retained']
    for n,h in read(OUT/'local_protected_sha256.json').items():assert sha(ROOT/n)==h
    old=ROOT/'outputs/cctv_dgp_bank_comparison_v1_return'
    # These two finalizer files are written after the returned tar is sealed.
    # Preserve the archive's original bytes and bind fresh local tail copies to
    # the equal VM before/after hashes, instead of substituting archived logs.
    tails=read(OUT/'post_export_logs/manifest.json')
    assert tails['complete'] and set(tails['files_sha256'])=={'export.log','export_exit_code.txt'}
    for n,h in before['bank_retained_bytes_sha256'].items():
        relative=PurePosixPath(n).relative_to('cctv_dgp_bank_comparison_v1_vm').as_posix()
        if relative in tails['files_sha256']:
            assert tails['files_sha256'][relative]==h==sha(OUT/'post_export_logs'/relative)
        else:assert sha(old/relative)==h
    allocated=sum(p['allocated_bytes'] for p in expected.values());assert allocated==plan['allocated_reclaim_bytes']==cr['allocated_reclaim_bytes']
    assert cr['free_after_bytes']-cr['free_before_bytes']==cr['recovered_bytes']
    result=dict(complete=True,checker_sha256=sha(__file__),plan_sha256=pin,receipt_archive_sha256=exp['archive_sha256'],exact_deletions_checked=930,complete_local_backup_hashes_rechecked=930,retained_metadata_entries=len(before['metadata']),retained_bank_byte_hashes=len(before['bank_retained_bytes_sha256']),post_export_log_files_preserved_locally=2,post_export_log_manifest_sha256=sha(OUT/'post_export_logs/manifest.json'),original_return_archive_unchanged=True,local_app_and_original_packet_bindings_unchanged=True,all_scientific_cache_bytes_rehashed=False,retained_state_exactly_equal=True,allocated_reclaim_bytes=allocated,measured_recovered_bytes=cr['recovered_bytes'],free_after_GiB=cr['free_after_bytes']/1024**3,model_gradient_or_training_calls=0,manual_training_pending=True,goal_complete=False,seconds=time.monotonic()-started)
    with receipt.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(dict(complete=True,exact_deletions_checked=930,retained_metadata_entries=result['retained_metadata_entries'],retained_bank_byte_hashes=result['retained_bank_byte_hashes'],free_after_GiB=result['free_after_GiB']),flush=True)

if __name__=='__main__':main()
