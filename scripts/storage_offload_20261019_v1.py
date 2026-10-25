"""Exact inactive result-copy offload. No VM, training or recursive deletion."""
from __future__ import annotations
import argparse, gzip, hashlib, json, os, shutil, stat, subprocess, time
from pathlib import Path
import storage_offload_20261018_v1 as base

ROOT = base.ROOT
OUT = ROOT / 'outputs/storage_offload_20261019_v1'
DEST = 'thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261019_v1'
NAMES = [
('codeformer-adain-gradient-isolation-v1','codeformer_adain_gradient_isolation_v1'),
('codeformer-adaptation-v1','codeformer_adaptation_v1'),
('codeformer-adaptation-v2','codeformer_adaptation_v2'),
('codeformer-bounded-fusion-feasibility-v1','codeformer_bounded_fusion_feasibility_v1'),
('codeformer-filter-fusion-training-v1','codeformer_filter_fusion_training_v1'),
('codeformer-fp32-comparison-v1','codeformer_fp32_comparison_v1'),
('codeformer-loss-response-v1','codeformer_loss_response_v1'),
('codeformer-normal-tone-training-v1','codeformer_normal_tone_training_v1'),
('codeformer-role-preserving-training-v1-r2','codeformer_role_preserving_training_v1_r2'),
('codeformer-tone-control-training-v1','codeformer_tone_control_training_v1'),
('dgp-bank-path-ablation-v1','cctv_dgp_bank_path_ablation_v1'),
('dgp-conditioning-learning-v1-r4','cctv_dgp_conditioning_learning_v1_r4'),
('dgp-conditioning-response-v1','cctv_dgp_conditioning_response_v1'),
('dgp-decoder-hinge-repair-v1','cctv_dgp_decoder_hinge_repair_v1'),
('dgp-live-conditioning-v1-r1','cctv_dgp_live_conditioning_v1_r1'),
('dgp-nearby-gradient-learning-v1-r1','cctv_dgp_nearby_gradient_learning_v1_r1'),
('dgp-raw-conditioning-learning-v1-r1','cctv_dgp_raw_conditioning_learning_v1_r1'),
('dgp-spatial-path-replay-v1','cctv_dgp_spatial_path_replay_v1'),
('dgp-spatial-readout-v1','cctv_dgp_spatial_readout_v1')]
ARCHIVES = {'outputs/cctv-'+a+'-results.tar.gz': {'retained_root':'outputs/'+b+'_return'} for a,b in NAMES}
require, read, save, pin, sha = base.require, base.read, base.save, base.pin, base.sha

def guard():
    # Reuse the reviewed full-CIM guard with this helper's exact marker/scope.
    import inspect
    source = inspect.getsource(base.process_guard).replace('20261018_v1','20261019_v1')
    namespace = dict(vars(base)); namespace['ARCHIVES'] = ARCHIVES
    exec(compile(source, str(__file__)+'::process_guard', 'exec'), namespace)
    return namespace['process_guard']()

def snapshot(label):
    target = OUT/(label+'_namespace.jsonl.gz')
    start=time.monotonic(); stack=[ROOT]; count=0; models=[]; links=[]
    with target.open('xb') as raw, gzip.GzipFile(filename='',fileobj=raw,mode='wb',mtime=0) as out:
        while stack:
            require(time.monotonic()-start<600 and count<2_000_000,'Inventory finite cap')
            folder=stack.pop()
            with os.scandir(folder) as entries:
                for e in entries:
                    p=Path(e.path)
                    if p==OUT: continue
                    s=e.stat(follow_symlinks=False); rel=p.relative_to(ROOT).as_posix()
                    if stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&0x400:
                        links.append(rel); continue
                    if stat.S_ISDIR(s.st_mode): stack.append(p)
                    elif stat.S_ISREG(s.st_mode):
                        row={'path':rel,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}
                        out.write((json.dumps(row,separators=(',',':'))+'\n').encode()); count+=1
                        if p.suffix.lower() in ('.pth','.pt','.ckpt','.safetensors'):
                            # Hash every retained weight; shared-link reads are allowed.
                            before=p.stat(); h=hashlib.sha256()
                            with p.open('rb') as f:
                                for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
                            after=p.stat()
                            require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),'Protected weight changed')
                            models.append({**row,'sha256':h.hexdigest()})
    result={'complete':True,'files':count,'models':models,'links_not_followed':links,
            'namespace':pin(target),'seconds':time.monotonic()-start}
    save(OUT/(label+'_snapshot.json'),result)
    print(json.dumps({'snapshot':label,'files':count,'protected_models':len(models),'seconds':result['seconds']}),flush=True)
    return result

def command(args, name, cap=28800):
    require(not (OUT/(name+'_receipt.json')).exists(),'Operation already has a receipt; preserve it')
    start=time.monotonic()
    with (OUT/(name+'_stdout.txt')).open('x',encoding='utf-8') as stdout, (OUT/(name+'_stderr.txt')).open('x',encoding='utf-8') as stderr:
        child=subprocess.Popen([str(base.EXE),'--config',str(base.CONFIG),*args],stdout=stdout,stderr=stderr,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        print(json.dumps({'operation':name,'pid':child.pid,'cap_seconds':cap}),flush=True)
        timeout=False
        try: code=child.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            child.kill(); code=child.wait(); timeout=True
    receipt={'complete':code==0 and not timeout,'exit_code':code,'timeout':timeout,'seconds':time.monotonic()-start,
             'stdout':pin(OUT/(name+'_stdout.txt')),'stderr':pin(OUT/(name+'_stderr.txt'))}
    save(OUT/(name+'_receipt.json'),receipt)
    require(receipt['complete'],'Rclone operation failed; inspect retained '+name+' logs')
    return receipt

def plan(expected):
    p=OUT/'plan.json'; require(sha(p)==expected,'Exact plan SHA required')
    result=read(p); require(result['destination']==DEST and {r['path'] for r in result['files']}==set(ARCHIVES),'Exact scope differs')
    return result

def local_check(p):
    for row in p['files']:
        require(pin(ROOT/row['path'])==row,'Candidate identity/content changed: '+row['path'])
        require((ROOT/ARCHIVES[row['path']]['retained_root']).is_dir(),'Retained research root missing')
    for row in p['critical']:
        require(pin(ROOT/row['path'])==row,'Active packet or document changed')

def prepare():
    OUT.mkdir(exist_ok=False); g=guard(); conn=base.connection(); free=shutil.disk_usage(ROOT).free
    files=[]
    for name, retained in ARCHIVES.items():
        require((ROOT/retained['retained_root']).is_dir(),'Retained root absent '+name)
        files.append(pin(ROOT/name)); print(json.dumps({'hashed':name,'bytes':files[-1]['bytes']}),flush=True)
    require(sum(r['bytes'] for r in files)==38608921227,'Fresh candidate total differs')
    critical=[pin(ROOT/name) for name in ['outputs/cctv-codeformer-skip-feature-adapter-v1-execution.tar.gz',
        'outputs/cctv-codeformer-skip-feature-adapter-v1-execution.tar.gz.sha256',
        'CCTV_CODEFORMER_SKIP_FEATURE_ADAPTER_V1_VM.md','PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md']]
    p={'version':'exact-inactive-results-20261019-v1','destination':DEST,'files':files,'critical':critical,
       'retained_roots':ARCHIVES,'free_bytes_at_plan':free,'connection':conn,'process_guard':g,
       'source':pin(Path(__file__)),'training_or_VM_operations':0}
    save(OUT/'plan.json',p)
    (OUT/'files.txt').write_text(''.join(r['path']+'\n' for r in files),encoding='utf-8',newline='\n')
    (OUT/'sha256.txt').write_text(''.join(r['sha256']+'  '+r['path']+'\n' for r in files),encoding='ascii',newline='\n')
    print(json.dumps({'plan_sha256':sha(OUT/'plan.json'),'files':len(files),'bytes':sum(r['bytes'] for r in files)}),flush=True)

def remote(name):
    command(['lsjson',DEST,'--recursive','--files-only','--hash','--max-depth','2'],name,600)
    rows=json.loads((OUT/(name+'_stdout.txt')).read_text(encoding='utf-8-sig'))
    require(len(rows)==19 and len({r['Path'] for r in rows})==19,'Remote namespace differs')
    return {r['Path']:r for r in rows}

def check_remote(p, rows):
    for row in p['files']:
        r=rows.get(row['path'],{}); hashes={k.lower():v for k,v in r.get('Hashes',{}).items()}
        require(r.get('Size')==row['bytes'] and hashes.get('sha256')==row['sha256'] and r.get('ID'),'Remote payload hash/ID differs')
    require(len({r['ID'] for r in rows.values()})==19,'Duplicate remote IDs')

def run(mode,expected):
    if mode=='prepare': return prepare()
    p=plan(expected)
    if mode=='snapshot': return snapshot('before')
    guard(); base.connection()
    if mode=='copy':
        local_check(p)
        command(['copy',str(ROOT),DEST,'--files-from-raw',str(OUT/'files.txt'),'--immutable','--checksum',
            '--transfers','2','--checkers','4','--drive-chunk-size','32M','--tpslimit','6','--retries','3',
            '--low-level-retries','5','--timeout','2m','--contimeout','20s','--max-duration','8h','--cutoff-mode','hard',
            '--stats','60s','--log-level','INFO','--log-file',str(OUT/'copy.log')],'copy')
    elif mode=='verify':
        require(read(OUT/'copy_receipt.json')['complete'],'Copy not complete')
        command(['check',str(OUT/'sha256.txt'),DEST,'--checkfile','SHA256','--download','--one-way',
            '--combined',str(OUT/'combined.txt'),'--checkers','2','--tpslimit','6','--retries','1',
            '--low-level-retries','3','--timeout','2m','--contimeout','20s','--stats','60s'],'stream_verify')
        require(sorted((OUT/'combined.txt').read_text().splitlines())==sorted('= '+r['path'] for r in p['files']),'Full streamed verification list differs')
        check_remote(p,remote('remote_verified'))
    elif mode=='apply':
        require(read(OUT/'stream_verify_receipt.json')['complete'],'Full streamed SHA verification missing')
        require(not (OUT/'deletion_journal.jsonl').exists(),'Deletion journal exists; audit partial state instead of replay')
        check_remote(p,remote('remote_preapply')); local_check(p)
        before=read(OUT/'before_snapshot.json'); require(before['complete'],'Protection inventory absent')
        guard(); free_before=shutil.disk_usage(ROOT).free
        with (OUT/'deletion_journal.jsonl').open('x',encoding='utf-8',newline='\n') as journal:
            for row in p['files']:
                path=(ROOT/row['path']).absolute()
                require(path.parent==ROOT/'outputs' and path.is_relative_to(ROOT) and pin(path)==row,'Exact local removal bound failed')
                journal.write(json.dumps({'intent':row})+'\n'); journal.flush(); os.fsync(journal.fileno())
                path.unlink()
                journal.write(json.dumps({'removed':row['path'],'bytes':row['bytes']})+'\n'); journal.flush(); os.fsync(journal.fileno())
        free_after=shutil.disk_usage(ROOT).free
        save(OUT/'apply.json',{'complete':True,'removed_files':19,'logical_bytes_removed':sum(r['bytes'] for r in p['files']),
            'free_bytes_before':free_before,'free_bytes_after':free_after,'observed_free_bytes_increase':free_after-free_before,
            'remote':DEST,'plan_sha256':expected,'training_or_VM_operations':0})
        print(json.dumps(read(OUT/'apply.json')),flush=True)
    elif mode=='audit':
        after=snapshot('after'); before=read(OUT/'before_snapshot.json')
        require(before['models']==after['models'],'Protected model content/metadata differs')
        def namespace(label):
            with gzip.open(OUT/(label+'_namespace.jsonl.gz'),'rt',encoding='utf-8') as stream:return {r['path']:r for r in map(json.loads,stream)}
        b=namespace('before'); a=namespace('after')
        missing=sorted(set(b)-set(a)); added=sorted(set(a)-set(b)); changed=sorted(k for k in set(a)&set(b) if a[k]!=b[k])
        require(missing==sorted(ARCHIVES),'Unexpected local file disappearance')
        local_critical=[]
        for row in p['critical']:
            require(pin(ROOT/row['path'])==row,'Protected active file differs');local_critical.append(row)
        require(not changed,'Retained namespace metadata changed; independent review required')
        save(OUT/'closure.json',{'complete':True,'missing_exact_planned_archives':missing,'added':added,'changed':changed,
            'protected_models':len(after['models']),'retained_files_checked':len(a),'active_files':local_critical,
            'apply':pin(OUT/'apply.json'),'plan_sha256':expected,'only_archive_copies_removed':True})
        print(json.dumps({'complete':True,'protected_models':len(after['models']),'retained_files':len(a),'removed':len(missing),'added':added}),flush=True)
    else: raise ValueError('Unknown operation')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','snapshot','copy','verify','apply','audit']);p.add_argument('--plan-sha',default='')
    a=p.parse_args();run(a.mode,a.plan_sha)
