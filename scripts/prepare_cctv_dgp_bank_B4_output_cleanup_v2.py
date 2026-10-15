"""Prepare an exact backup-bound plan; no remote execution or deletion."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
DEST = OUT/'cctv_dgp_bank_B4_output_cleanup_v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(1024**2),b''): h.update(value)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    source = ROOT/'scripts/cctv_dgp_bank_B4_output_cleanup_v1_vm.py'
    worker = ROOT/'scripts/cctv_dgp_bank_B4_output_cleanup_v2_vm.py'
    text = source.read_text(encoding='utf-8')
    text = text.replace("OUT = HOME/'dgp_bank_B4_output_cleanup_v1_receipts'", "OUT = HOME/'dgp_bank_B4_output_cleanup_v2_receipts'")
    text = text.replace("LABELS = {'baseline','B50','B50_bank_disabled'}", "LABELS = {'baseline','B50','B50_bank_disabled'}\nLEGACY = ROOT/'cctv_dgp_residual_epochs_vm_v42'")
    text = text.replace("    stopped = read(BANK/'outputs/bank_comparison_v1/quality_stop_B.json')",
        "    assert sha(LEGACY/'protocol.json') == 'c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d'\n"
        "    assert (LEGACY/'outputs/failure.json').is_file() and (LEGACY/'outputs/capacity_update50.json').is_file()\n"
        "    stopped = read(BANK/'outputs/bank_comparison_v1/quality_stop_B.json')")
    text = text.replace("    relative = path.relative_to(BANK/'outputs/bank_comparison_v1')",
        "    if path.is_relative_to(LEGACY):\n"
        "        relative = path.relative_to(LEGACY/'outputs')\n"
        "        assert len(relative.parts) == 2 and relative.parts[0] in {'update0','update50'} and path.suffix == '.png'\n"
        "        return\n"
        "    relative = path.relative_to(BANK/'outputs/bank_comparison_v1')")
    text = text.replace("if path.is_relative_to(BANK) and stat.S_ISREG(s.st_mode): hashes[relative] = sha(path)",
        "if (path.is_relative_to(BANK) or path.is_relative_to(LEGACY)) and stat.S_ISREG(s.st_mode): hashes[relative] = sha(path)")
    text = text.replace("        paths += [path for path in sorted(ARCHIVES) if path.exists()]",
        "        paths += [path for folder in ['update0','update50'] for path in sorted((LEGACY/'outputs'/folder).glob('*.png')) if path.is_file()]\n"
        "        paths += [path for path in sorted(ARCHIVES) if path.exists()]")
    scope = 'completed B4 RGB output packs including archived measurement fields, backed-up home archives, and completed V42 update0/update50 PNG copies; all checkpoints, caches, raw V42 arrays, splits, instructions, logs and stop records retained'
    text = text.replace('completed B4 RGB output packs including archived measurement fields and backed-up home archives; all checkpoints, caches, splits, instructions, logs and stop records retained',scope)
    text = text.replace('dgp-bank-B4-output-cleanup-v1-','dgp-bank-B4-output-cleanup-v2-')
    if worker.exists():
        assert worker.read_bytes() == text.encode('utf-8'), 'Retain a different worker'
    else:
        with worker.open('x',encoding='utf-8',newline='\n') as stream: stream.write(text)
    assert DEST.is_dir() and (DEST/'discovery.json').is_file()
    assert not (DEST/'plan.json').exists()
    discovery = read(DEST/'discovery.json')
    assert discovery['complete'] and discovery['worker_sha256'] == sha(worker)
    assert discovery['runtime']['GPU_compute_idle'] and not discovery['runtime']['active_research_processes']
    b = OUT/'cctv_dgp_bank_comparison_v1_r4_B_return'
    b_manifest = read(b/'export_manifest.json')['files']
    audit = read(OUT/'cctv_dgp_bank_comparison_v1_r4_B_analysis/independent_audit.json')
    imp = read(OUT/'cctv_dgp_bank_comparison_v1_r4_B_analysis/import.json')
    assert audit['complete'] and imp['complete'] and imp['every_member_path_and_byte_hash_checked']
    legacy = OUT/'cctv_dgp_residual_epochs_vm_v42_return'
    old_manifest = read(legacy/'export_manifest.json')
    old_import = read(OUT/'cctv_dgp_residual_epochs_v42_return_import.json')
    assert old_manifest['complete'] and old_import['complete'] and old_import['members'] == len(old_manifest['files_sha256'])+1
    # Recheck the entire retained V42 payload, including checkpoint/failure files.
    expected = set(old_manifest['files_sha256']) | {'export_manifest.json'}
    actual = {path.relative_to(legacy).as_posix() for path in legacy.rglob('*') if path.is_file()}
    assert actual == expected and not any(path.is_symlink() for path in legacy.rglob('*'))
    for name,expected_sha in old_manifest['files_sha256'].items(): assert sha(legacy/name) == expected_sha, name
    rows = []
    prefix = '/home/janusdominic0/forensic-dgp/'
    for row in discovery['prospective_candidates']:
        if row['nlink'] != 1: continue
        path = row['path']
        if path == '/home/janusdominic0/cctv-dgp-bank-comparison-v1-r4-B-results.tar.gz':
            backup = OUT/'cctv-dgp-bank-comparison-v1-r4-B-results.tar.gz'; expected_sha = imp['archive_sha256']
        elif path.startswith(prefix+'cctv_dgp_bank_comparison_v1_r4_B_vm/'):
            name = path[len(prefix+'cctv_dgp_bank_comparison_v1_r4_B_vm/'):]
            backup = b/name; expected_sha = b_manifest[name]
        elif path.startswith(prefix+'cctv_dgp_residual_epochs_vm_v42/'):
            name = path[len(prefix+'cctv_dgp_residual_epochs_vm_v42/'):]
            assert name.startswith(('outputs/update0/','outputs/update50/')) and name.endswith('.png')
            backup = legacy/name; expected_sha = old_manifest['files_sha256'][name]
        else: raise AssertionError('Not an independently backed-up exact candidate: '+path)
        assert backup.is_file() and not backup.is_symlink() and backup.stat().st_size == row['bytes'] and sha(backup) == expected_sha
        rows.append({**row,'sha256':expected_sha,'local_backup':str(backup)})
    assert len(rows) == len(discovery['prospective_candidates'])
    record = dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),instance_id='4410777042005672095',
        scope=scope,worker_sha256=sha(worker),discovery_sha256=sha(DEST/'discovery.json'),candidates=rows,
        local_backups_completely_verified=True,V42_entire_retained_payload_rehashed=len(old_manifest['files_sha256']),
        B4_prior_full_import_and_independent_audit_complete=True,
        B4_import_sha256=sha(OUT/'cctv_dgp_bank_comparison_v1_r4_B_analysis/import.json'),
        B4_audit_sha256=sha(OUT/'cctv_dgp_bank_comparison_v1_r4_B_analysis/independent_audit.json'),
        V42_manifest_sha256=sha(legacy/'export_manifest.json'),V42_import_sha256=sha(OUT/'cctv_dgp_residual_epochs_v42_return_import.json'),
        exact_candidate_bytes=sum(row['bytes'] for row in rows),files_removed=0,remote_apply_pending=True,
        no_directory_checkpoint_cache_split_log_stop_or_raw_V42_array_deletion=True,
        model_gradient_or_training_calls=0,goal_complete=False)
    write(DEST/'plan.json',record)
    print(dict(complete=True,candidates=len(rows),candidate_GiB=record['exact_candidate_bytes']/1024**3,
        plan_sha256=sha(DEST/'plan.json'),worker_sha256=sha(worker),V42_retained_payload_rehashed=record['V42_entire_retained_payload_rehashed']),flush=True)


if __name__ == '__main__': main()
