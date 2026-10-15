"""Delete only inventory-bound, fully backed-up historical result archives.

All native Windows file operations stay inside the checked local workspace.
Exclusive archive handles prevent removal while another process uses an archive.
"""
import argparse
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
import gzip
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from prepare_local_archive_cleanup_20261011_v1 import ROOT, OUT, bounded, metadata, sha, write

GENERIC_READ = 0x80000000
DELETE = 0x00010000
FILE_SHARE_READ = 1
OPEN_EXISTING = 3
FILE_ATTRIBUTE_NORMAL = 0x80
FILE_FLAG_SEQUENTIAL_SCAN = 0x08000000
SHARED_DOCUMENTS = {'PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md',
                    'PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md'}


def processes(target_names):
    command = ("Get-CimInstance Win32_Process | "
               "Select-Object Name,ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3")
    r = subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',command],
                       capture_output=True,text=True,check=True,timeout=30)
    rows = json.loads(r.stdout) if r.stdout.strip() else []
    if isinstance(rows,dict):
        rows = [rows]
    by_pid = {r['ProcessId']: r for r in rows}
    binding = json.loads((OUT/'resume_binding.json').read_text())
    maintenance_source = ROOT/'scripts/cctv_dgp_bank_B_return_offload_v1.py'
    assert sha(maintenance_source) == binding['disjoint_bank_maintenance_driver_sha256']
    maintenance_pids = set()
    for row in rows:
        command_text = (row.get('CommandLine') or '').replace('\\','/')
        if row['Name'].lower().startswith('python') and re.search(
                r'scripts/cctv_dgp_bank_B_return_offload_v1\.py\s+--phase\s+(?:prepare|verify|apply|post)\b', command_text):
            assert not any(name.casefold() in command_text.casefold() for name in target_names)
            maintenance_pids.add(row['ProcessId'])
    rows = [r for r in rows if re.match(r'^(python|pythonw|tar|7z|gzip|gcloud|pscp|php|httpd)',r['Name'],re.I)]
    for row in rows:
        text = row.get('CommandLine') or ''
        assert not any(name.casefold() in text.casefold() for name in target_names), ('Active task names a cleanup target',row)
        ancestor = row['ProcessId']
        belongs_to_maintenance = False
        visited = set()
        while ancestor in by_pid and ancestor not in visited:
            visited.add(ancestor)
            if ancestor in maintenance_pids:
                belongs_to_maintenance = True
                break
            ancestor = by_pid[ancestor]['ParentProcessId']
        if belongs_to_maintenance and (row['ProcessId'] in maintenance_pids or
                ('gcloud.py' in text and row['Name'].lower().startswith('python')) or
                row['Name'].lower()=='pscp.exe'):
            row['allowed_disjoint_user_clarified_bank_VM_maintenance'] = True
            row['maintenance_driver_sha256'] = binding['disjoint_bank_maintenance_driver_sha256']
            continue
        # The user is independently downloading the current B return. It is
        # disjoint from all52 historical archive paths and must remain active.
        known_source = re.search(r'/home/janusdominic0/(cctv-dgp-bank-comparison-v1-r2-B-(?:results\.tar\.gz(?:\.sha256)?|export\.json))', text)
        transfer_program = ('gcloud.py' in text and 'compute scp' in text) or row['Name'].lower()=='pscp.exe'
        destination_is_same_basename = text.strip().endswith((' .', ' "."'))
        if known_source and transfer_program and destination_is_same_basename:
            assert not any(name.casefold() in text.casefold() for name in target_names), ('Transfer overlaps cleanup target',row)
            row['allowed_disjoint_user_bank_download']=True
            continue
        bank_review_script = re.search(r'(?:scripts[\\/])(?:import|audit|review|close|publish)_[A-Za-z0-9_]*bank[A-Za-z0-9_]*\.py\b', text)
        if row['Name'].lower().startswith('python') and bank_review_script:
            assert not any(name.casefold() in text.casefold() for name in target_names), ('Bank review overlaps cleanup target',row)
            script_name = bank_review_script.group().replace('\\','/').split('/')[-1]
            script_path = ROOT/'scripts'/script_name
            assert script_path.is_file() and not script_path.is_symlink()
            row['allowed_disjoint_local_bank_return_review']=True
            row['bank_review_source_sha256']=sha(script_path)
            continue
        if row['Name'].lower().startswith('python'):
            assert ' -B app.py' in text or 'apply_local_archive_cleanup_20261011_v1.py' in text, ('Competing Python task',row)
        else:
            assert not row['Name'].lower().startswith(('tar','7z','gzip','gcloud','pscp')), ('Competing transfer/archive task',row)
    return rows


def critical_bindings():
    paths = set()
    for directory in ['checkpoints','weights','models','templates','static','docs']:
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and not p.is_symlink():
                paths.add(p.relative_to(ROOT).as_posix())
    paths.update(p.name for p in ROOT.iterdir() if p.is_file() and p.suffix.lower() in {'.py','.md','.txt','.json','.bat','.html'})
    paths.add('outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth')
    prepared = json.loads((ROOT/'outputs/cctv_dgp_bank_comparison_v1_r3_preparation.json').read_text())
    for route in prepared['routes']:
        packet = ROOT/'outputs'/route['name']
        protocol = json.loads((packet/'protocol.json').read_text())
        assert sha(packet/'protocol.json') == route['protocol_sha256']
        for name,expected in protocol['assets_sha256'].items():
            relative = (packet/name).relative_to(ROOT).as_posix()
            assert sha(bounded(relative)) == expected
            paths.add(relative)
        paths.add((packet/'protocol.json').relative_to(ROOT).as_posix())
        stem = route['stem']+'-execution.tar.gz'
        for ending in ['', '.sha256']:
            paths.add('outputs/'+stem+ending)
    app = json.loads((ROOT/'scratch/dgp-bank-app-preflight-20261010-v1/flow.json').read_text())
    paths.update(app['sources'])
    hashes = {name:sha(bounded(name)) for name in sorted(paths)}
    assert hashes['outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'] == '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    return hashes


def open_checked(path, access, share):
    k32 = ctypes.WinDLL('kernel32',use_last_error=True)
    create = k32.CreateFileW
    create.argtypes = [wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,
                       wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE]
    create.restype = wintypes.HANDLE
    handle = create(str(path),access,share,None,OPEN_EXISTING,
                    FILE_ATTRIBUTE_NORMAL|FILE_FLAG_SEQUENTIAL_SCAN,None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def verify_then_remove(row):
    archive = bounded(row['archive'])
    assert archive.parent == ROOT/'outputs' and archive.name.endswith('-results.tar.gz')
    assert 'bank-comparison' not in archive.name
    assert metadata(archive) == row['metadata'] and row['metadata']['nlink'] == 1
    manifest = bounded(row['member_manifest'])
    assert sha(manifest) == row['member_manifest_sha256']
    members = json.load(gzip.open(manifest,'rt',encoding='utf-8'))
    assert len(members) == row['complete_regular_members']
    held = []
    try:
        for member in members:
            p = bounded(member['retained_path'])
            assert p != archive and metadata(p) == member['retained_metadata']
            # Hold read-only access with no write/delete sharing until archive removal.
            held.append(open_checked(p,GENERIC_READ,FILE_SHARE_READ))
            assert sha(p) == member['sha256']
        handle = open_checked(archive,GENERIC_READ|DELETE,0)
        fd = msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
        with os.fdopen(fd,'rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest() == row['archive_sha256']
            assert metadata(archive) == row['metadata']
            class FileDispositionInfo(ctypes.Structure):
                _fields_ = [('DeleteFile',ctypes.c_ubyte)]
            info = FileDispositionInfo(1)
            k32 = ctypes.WinDLL('kernel32',use_last_error=True)
            disposition = k32.SetFileInformationByHandle
            disposition.argtypes = [wintypes.HANDLE,ctypes.c_int,wintypes.LPVOID,wintypes.DWORD]
            disposition.restype = wintypes.BOOL
            assert disposition(handle,4,ctypes.byref(info),ctypes.sizeof(info)), ctypes.WinError(ctypes.get_last_error())
        assert not archive.exists()
    finally:
        close = ctypes.WinDLL('kernel32',use_last_error=True).CloseHandle
        close.argtypes=[wintypes.HANDLE]
        close.restype=wintypes.BOOL
        for handle in held:
            assert close(handle)
    return dict(path=row['archive'],archive_sha256=row['archive_sha256'],
        former_bytes=row['metadata']['bytes'],former_inode=row['metadata']['inode'],
        complete_member_backup_reverified=True, retained_tree=row['retained_tree'],
        retained_members=len(members),member_manifest=row['member_manifest'],
        member_manifest_sha256=row['member_manifest_sha256'],exclusive_archive_handle=True,
        retained_sources_held_read_only_during_removal=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--resume',action='store_true')
    a=parser.parse_args()
    start=time.monotonic()
    assert sha(OUT/'plan.json') == a.plan_sha
    plan=json.loads((OUT/'plan.json').read_text())
    assert plan['complete'] and plan['files_removed']==0 and plan['source_directories_will_not_be_removed']
    assert a.resume, 'Original stopped execution is retained; use the verified resume only'
    assert not (OUT/'apply_receipt.json').exists()
    binding=json.loads((OUT/'resume_binding.json').read_text())
    clarification=json.loads((OUT/'coordination_clarification.json').read_text())
    stopped=json.loads((OUT/'apply_guard_stop_03.json').read_text())
    assert binding['complete'] and clarification['complete'] and clarification['local_and_remote_maintenance_scopes_separate']
    assert binding['plan_sha256']==a.plan_sha
    assert sha(OUT/'worker_before_resume.py')==binding['first_deletion_worker_sha256']==stopped['worker_sha256']
    original_ledger=(OUT/'deletion_ledger_until_stop_03.jsonl').read_bytes()
    assert sha(OUT/'deletion_ledger_until_stop_03.jsonl')==binding['first_deletion_ledger_sha256']
    assert (OUT/'deletion_ledger.jsonl').read_bytes()==original_ledger
    removed=[json.loads(line) for line in original_ledger.decode('utf-8').splitlines()]
    assert len(removed)==clarification['existing_deletions']==stopped['actual_archives_removed']==15
    for actual,planned in zip(removed,plan['candidates']):
        assert actual['path']==planned['archive'] and not (ROOT/actual['path']).exists()
        assert actual['archive_sha256']==planned['archive_sha256']
        assert actual['member_manifest_sha256']==planned['member_manifest_sha256']
        assert actual['former_bytes']==planned['metadata']['bytes']
    assert len(plan['candidates'])-len(removed)==clarification['remaining_local_archives']==37
    initial_count=len(removed)
    target_names={Path(r['archive']).name for r in plan['candidates']}
    assert not any('bank-comparison' in name for name in target_names)
    active=processes(target_names)
    write(OUT/'active_processes_at_resume.json',active)
    before=json.loads((OUT/'critical_protected_before.json').read_text())
    assert before['complete']
    critical=before['files_sha256']
    documents_before=OUT/'shared_documents_before'
    for name in SHARED_DOCUMENTS:
        assert sha(documents_before/name)==critical[name]
    inventory=json.loads((OUT/'complete_inventory_before.json').read_text())
    assert inventory['complete']
    free_before=inventory['disk_free_bytes']
    resume_free_before=shutil.disk_usage(ROOT).free
    with (OUT/'deletion_ledger.jsonl').open('a',encoding='utf-8') as ledger:
        for number,row in enumerate(plan['candidates'][initial_count:],initial_count+1):
            assert row['verified']
            if number%5==1:
                processes(target_names)
            result=verify_then_remove(row)
            result['UTC']=datetime.now(timezone.utc).isoformat()
            ledger.write(json.dumps(result)+'\n');ledger.flush();os.fsync(ledger.fileno())
            removed.append(result)
            print(dict(removed=number,of=len(plan['candidates']),archive=Path(row['archive']).name,
                reclaimed_GiB=sum(r['former_bytes'] for r in removed)/1024**3),flush=True)
            assert time.monotonic()-start<3600,'Finite cleanup stop; preserve completed ledger'
    observed={name:sha(bounded(name)) for name in critical if name not in SHARED_DOCUMENTS}
    concurrent_documents=[]
    documents_after=OUT/'shared_documents_after'
    documents_after.mkdir()
    for name,expected in critical.items():
        if name in SHARED_DOCUMENTS:
            content=(ROOT/name).read_bytes()
            (documents_after/name).write_bytes(content)
            observed[name]=hashlib.sha256(content).hexdigest()
            assert sha(documents_after/name)==observed[name]
            if observed[name]!=expected:
                concurrent_documents.append(dict(path=name,before_sha256=expected,after_sha256=observed[name]))
        else:
            assert observed[name]==expected, name
    write(OUT/'critical_protected_after.json',dict(complete=True,files_sha256=observed))
    write(OUT/'concurrent_document_observations.json',dict(complete=True,
        allowed_shared_document_names=sorted(SHARED_DOCUMENTS),documents_changed=concurrent_documents,
        before_after_full_bytes_preserved=True,cleanup_did_not_edit_these_documents=True))
    free_after=shutil.disk_usage(ROOT).free
    record=dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),plan_sha256=a.plan_sha,
        worker_sha256=sha(__file__),root=str(ROOT),files_removed=len(removed),
        resumed_from_verified_partial_ledger=True,first_execution_complete=False,
        first_worker_sha256=binding['first_deletion_worker_sha256'],
        first_execution_deletion_ledger_sha256=binding['first_deletion_ledger_sha256'],
        first_execution_files_removed=initial_count,resume_files_removed=len(removed)-initial_count,
        resume_binding_sha256=sha(OUT/'resume_binding.json'),
        coordination_clarification_sha256=sha(OUT/'coordination_clarification.json'),
        removed_paths=[r['path'] for r in removed],archive_logical_bytes_removed=sum(r['former_bytes'] for r in removed),
        free_before_bytes=free_before,free_after_bytes=free_after,observed_net_free_increase_bytes=free_after-free_before,
        free_space_measurement_basis='complete_inventory_snapshot_to_cleanup_end_includes_concurrent_writes',
        resume_free_before_bytes=resume_free_before,resume_free_after_bytes=free_after,
        resume_observed_net_free_increase_bytes=free_after-resume_free_before,
        critical_protected_files=len(critical),immutable_critical_hashes_unchanged=True,
        concurrent_shared_document_changes=concurrent_documents,
        extracted_payloads_checkpoints_data_caches_splits_logs_and_failures_retained=True,
        current_bank_transfers_and_migration_backups_preserved=True,
        no_directories_removed=True,VM_connections_or_training_calls=0,
        independent_closure_pending=True,maintenance_cap_seconds=3600,seconds=time.monotonic()-start)
    write(OUT/'apply_receipt.json',record)
    print(dict(complete=True,files_removed=len(removed),free_GiB=free_after/1024**3,
        archive_bytes_removed_GiB=sum(r['former_bytes'] for r in removed)/1024**3,
        net_free_increase_GiB=(free_after-free_before)/1024**3),flush=True)


if __name__=='__main__':
    main()
