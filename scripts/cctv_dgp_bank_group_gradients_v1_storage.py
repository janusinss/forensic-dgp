"""Pinned transport and full local backup checks for one authorized archive."""
import argparse
import ast
import base64
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import cctv_dgp_multiscale_archive_cleanup_v1 as transport

ROOT = transport.ROOT
OUT = ROOT / 'outputs/cctv_dgp_bank_group_gradients_v1_storage_cleanup'
HOME = '/home/janusdominic0'
WORKER = ROOT / 'scripts/cctv_dgp_bank_group_gradients_v1_storage_vm.py'
PLAN = OUT / 'plan.json'
TARGET_NAME = 'cctv-dgp-bank-photometric-guard-v1-results.tar.gz'
transport.OUT = OUT
sha, read, write = transport.sha, transport.read, transport.write


def prepare():
    inventory = read(OUT / 'inventory_before.json')
    assert inventory['complete'] and inventory['gpu_processes']['stdout'].strip() == ''
    candidate = next(row for row in inventory['archives'] if row['path'] == HOME+'/'+TARGET_NAME)
    assert candidate['regular_file'] and not candidate['symlink'] and candidate['nlink'] == 1 and candidate['uid'] == 1001
    backup = ROOT / 'outputs' / TARGET_NAME
    checksum = Path(str(backup)+'.sha256')
    exported_path = ROOT / 'outputs/cctv-dgp-bank-photometric-guard-v1-export.json'
    exported = read(exported_path)
    line = checksum.read_text(encoding='ascii').strip().split()
    assert len(line) == 2 and line[1] == TARGET_NAME and exported['complete']
    digest = sha(backup)
    assert digest == line[0] == exported['archive_sha256']
    assert backup.stat().st_size == exported['bytes'] == candidate['bytes']
    parent = ROOT / 'outputs/cctv_dgp_bank_photometric_guard_v1_return'
    old_import = read(ROOT / 'outputs/cctv_dgp_bank_photometric_guard_v1_analysis/import.json')
    assert old_import['complete'] and old_import['archive_sha256'] == digest
    p = read(ROOT / 'outputs/cctv_dgp_bank_group_gradients_v1_vm/protocol.json')
    critical = {'cctv_dgp_bank_photometric_guard_v1_vm/'+n:h for n,h in p['parent_required_sha256'].items()}
    assert len(critical) == 5852 and all(sha(parent/n) == h for n,h in p['parent_required_sha256'].items())
    packet = ROOT / 'outputs/cctv_dgp_bank_group_gradients_v1_return'
    manifest = read(packet/'export_manifest.json')
    # The archive seals export.log while it is still open. Its finalized VM
    # bytes are preserved by the full before/after packet snapshot, without
    # falsely equating them to the earlier archived empty log.
    critical.update({'cctv_dgp_bank_group_gradients_v1_vm/'+n:h for n,h in manifest['files'].items() if n!='export.log'})
    critical['cctv_dgp_bank_group_gradients_v1_vm/export_manifest.json'] = sha(packet/'export_manifest.json')
    for n,h in read(ROOT/'outputs/cctv_dgp_bank_group_gradients_v1_analysis/local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT/n) == h
    ast.parse(WORKER.read_text(encoding='utf-8'), feature_version=(3,10))
    plan = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), instance_id=transport.INSTANCE_ID,
        authorization_scope='one exact completed home archive; full hash-verified local backup; all research tree assets retained',
        worker_sha256=sha(WORKER), inventory_sha256=sha(OUT/'inventory_before.json'),
        candidates=[dict(**{k:candidate[k] for k in ['path','bytes','allocated_bytes','uid','nlink','inode','mtime_ns']},
            sha256=digest, local_backup=backup.relative_to(ROOT).as_posix(),
            local_backup_mtime_ns=backup.stat().st_mtime_ns, local_checksum_sha256=sha(checksum),
            local_export_receipt_sha256=sha(exported_path), local_import_sha256=sha(ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_analysis/import.json'))],
        critical_sha256=critical, allocated_reclaim_bytes=candidate['allocated_bytes'],
        only_one_home_archive_unlink=True, no_research_tree_deletion=True,
        finalized_export_tail_checked_in_live_packet_snapshot_not_archived_log=True,
        all_scientific_cache_bytes_rehashed=False, model_gradient_or_training_calls=0)
    write(PLAN,plan)
    write(OUT/'local_backup_verified.json',dict(complete=True,plan_sha256=sha(PLAN),
        full_archive_sha256=digest,full_archive_bytes=backup.stat().st_size,
        parent_bindings_checked=5852,returned_failure_bindings_retained=len(manifest['files'])+1,
        local_protected_bindings_rechecked=True,files_removed=0,model_gradient_or_training_calls=0))
    print(dict(complete=True,backup_GiB=backup.stat().st_size/1024**3,files_planned=1),flush=True)


def invoke_ssh(source,label,cap,read_only):
    encoded=base64.b64encode(source.encode()).decode()
    command='python3 -B -c '+shlex.quote('import base64;exec(base64.b64decode('+repr(encoded)+'))')
    arguments=[str(transport.CLOUD),'compute','ssh','janusdominic0@forensic-dgp-thesis']+transport.BASE
    arguments+=['--ssh-flag='+flag for flag in transport.FLAGS]+['--command='+command]
    assert len(subprocess.list2cmdline(arguments))<8000
    return transport.invoke(arguments,label,cap,read_only)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','upload','verify','apply','return','audit'])
    parser.add_argument('--file',choices=['dgp-bank-group-gradients-v1-storage-receipts.tar.gz','dgp-bank-group-gradients-v1-storage-export.json','live_audit.json'])
    args=parser.parse_args()
    if args.phase=='prepare':
        prepare();return
    plan=read(PLAN);assert plan['complete'] and sha(WORKER)==plan['worker_sha256']
    pins={HOME+'/'+WORKER.name:sha(WORKER),HOME+'/dgp_bank_group_gradients_v1_storage_plan.json':sha(PLAN)}
    if args.phase=='upload':
        source="import os,socket;from pathlib import Path\nassert socket.gethostname().split('.')[0]=='forensic-dgp-thesis'\n"
        source+='names='+repr(list(pins)+[HOME+'/dgp_bank_group_gradients_v1_storage_receipts'])+'\n'
        source+="assert all(not os.path.lexists(name) for name in names)\nprint('Fresh maintenance destinations; no deletion or model work')\n"
        invoke_ssh(source,'fresh_maintenance_destinations',90,True)
        for path,remote in [(WORKER,HOME+'/'+WORKER.name),(PLAN,HOME+'/dgp_bank_group_gradients_v1_storage_plan.json')]:
            command=[str(transport.CLOUD),'compute','scp']+transport.BASE+['--scp-flag='+flag for flag in transport.FLAGS]
            command+=[str(path),'janusdominic0@forensic-dgp-thesis:'+remote]
            transport.invoke(command,'upload_'+path.name,120,False)
        write(OUT/'upload.json',dict(complete=True,remote_sha256=pins,model_gradient_or_training_calls=0))
        print(dict(complete=True,uploaded_maintenance_files=2,files_removed=0),flush=True);return
    assert read(OUT/'upload.json')['remote_sha256']==pins
    if args.phase=='return':
        assert args.file
        remote=HOME+'/'+args.file if args.file!='live_audit.json' else HOME+'/dgp_bank_group_gradients_v1_storage_receipts/live_audit.json'
        local=OUT/args.file;assert not local.exists()
        command=[str(transport.CLOUD),'compute','scp']+transport.BASE+['--scp-flag='+flag for flag in transport.FLAGS]
        command+=['janusdominic0@forensic-dgp-thesis:'+remote,str(local)]
        transport.invoke(command,'return_'+args.file,360,True)
        print(dict(complete=True,returned=args.file),flush=True);return
    if args.phase=='apply':
        row=plan['candidates'][0];backup=ROOT/row['local_backup']
        assert backup.is_file() and not backup.is_symlink() and backup.stat().st_size==row['bytes']
        assert backup.stat().st_mtime_ns==row['local_backup_mtime_ns'] and sha(backup)==row['sha256']
        write(OUT/'local_preapply_backup_recheck.json',dict(complete=True,plan_sha256=sha(PLAN),
            full_archive_sha256=sha(backup),full_archive_bytes=backup.stat().st_size,files_removed=0))
    source="import hashlib,json,socket,subprocess;from pathlib import Path\nassert socket.gethostname().split('.')[0]=='forensic-dgp-thesis' and Path.home().resolve()==Path('/home/janusdominic0')\n"
    source+='pins='+repr(pins)+'\n'
    source+="for name,h in pins.items():\n p=Path(name);assert p.is_file() and not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==h\n"
    argv=['python3','-B',HOME+'/'+WORKER.name,'--plan',HOME+'/dgp_bank_group_gradients_v1_storage_plan.json','--plan-sha',sha(PLAN),'--phase',args.phase]
    source+='subprocess.run('+repr(argv)+',check=True,timeout=930)\n'
    print(dict(stage=args.phase,started=True,cap_seconds=960,model_gradient_or_training_calls=0),flush=True)
    stdout=invoke_ssh(source,args.phase,960,args.phase!='apply')
    result=read(stdout);assert result['complete']
    write(OUT/(args.phase+'_transport_result.json'),result)
    print(dict(stage=args.phase,complete=True,files_removed=result.get('files_removed',0),
        seconds=result.get('seconds'),free_GiB=result.get('free_GiB',result.get('free_bytes',0)/1024**3)),flush=True)


if __name__=='__main__':
    main()
