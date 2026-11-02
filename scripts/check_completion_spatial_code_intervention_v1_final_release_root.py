"""Root full-stream archive and exact published manual-guide check; zero NN/VM."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,ast,gzip,hashlib,json,os,shlex,tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/completion_codeformer_spatial_code_intervention_v1_packet/draft_004'
NAME='cctv_completion_codeformer_spatial_code_intervention_v1_vm'
STEM='cctv-completion-codeformer-spatial-code-intervention-v1'
EXPECTED='b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123'
TRANSFER=Path('C:/xampp/htdocs/YEAR 4/Testing/outputs')

def need(ok,message):
    if not ok:raise ValueError(message)
def physical(path):
    p=Path(path).resolve()
    return Path('\\\\?\\'+str(p)) if os.name=='nt' and not str(p).startswith('\\\\?\\') else p
def sha(path):
    h=hashlib.sha256()
    with physical(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':physical(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(physical(path).read_text(encoding='utf-8-sig'))

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive-sha',required=True);parser.add_argument('--guide-sha',required=True);a=parser.parse_args()
    dest=BASE/'root_final_archive_audit.json';need(not dest.exists(),'Preserve prior archive audit')
    archive=BASE/(STEM+'-execution.tar.gz');stage=BASE/NAME
    need(sha(archive)==a.archive_sha,'Archive identity')
    ppath=stage/'protocol.json';need(sha(ppath)==EXPECTED,'Exact reviewed protocol');p=read(ppath)
    expected={NAME+'/protocol.json':{'bytes':physical(ppath).stat().st_size,'sha256':EXPECTED}}
    expected.update({NAME+'/'+r['path']:r for r in p['files']});seen=[];total=0
    with physical(archive).open('rb') as raw,gzip.GzipFile(fileobj=raw) as gz:
        with tarfile.open(fileobj=gz,mode='r|') as tar:
            for m in tar:
                need(m.isreg() and m.name in expected and m.name not in seen,'Unsafe/extra/duplicate archive member')
                need(seen or m.name==NAME+'/protocol.json','Protocol first member')
                r=expected[m.name];need(m.size==r['bytes'],'Member bytes')
                h=hashlib.sha256();count=0
                with tar.extractfile(m) as f:
                    for b in iter(lambda:f.read(8*1024**2),b''):h.update(b);count+=len(b)
                need(count==r['bytes'] and h.hexdigest()==r['sha256'],'Full member SHA')
                seen.append(m.name);total+=count
        while gz.read(8*1024**2):pass
    need(set(seen)==set(expected) and len(seen)==167 and total==19491897,'Closed167member scope')
    checksum=(a.archive_sha+'  '+archive.name+'\n').encode('ascii')
    need(physical(BASE/(archive.name+'.sha256')).read_bytes()==checksum,'Exact LF checksum')
    need(sha(TRANSFER/archive.name)==a.archive_sha and physical(TRANSFER/(archive.name+'.sha256')).read_bytes()==checksum,'Published exact transfer copy')
    guide='COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_VM.md';guides=[ROOT/guide,TRANSFER/guide]
    need(all(sha(q)==a.guide_sha for q in guides),'Both guide identities')
    body=physical(guides[0]).read_text(encoding='utf-8')
    need(a.archive_sha in body and EXPECTED in body and not any(v in body for v in ('UNBOUND_PROTOCOL','UNPREPARED_ARCHIVE','NOT READY — SOURCE-ONLY')),'Published guide source/protocol binding')
    scp=[line for line in body.splitlines() if line.startswith('gcloud compute scp ')]
    need(len(scp)==5,'Two uploads/three downloads')
    remotes=[]
    for line in scp:
        t=shlex.split(line);need(t[:3]==['gcloud','compute','scp'] and '--project=forensic-dgp-thesis' in t and '--zone=us-central1-a' in t,'Exact project/zone SDK command')
        rr=[v for v in t if v.startswith('janusdominic0@forensic-dgp-thesis:')];need(len(rr)==1,'One remote source per PuTTY command');remotes+=rr
    need(remotes[-3:]==['janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+STEM+x for x in ('-results.tar.gz','-results.tar.gz.sha256','-export.json')],'Exact downloadable return filenames')
    need('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"' in body,'Windows SDK working folder')
    need('tmux new-session -A -s completion_spatial_code_intervention_v1' in body,'Named tmux')
    need('bash scripts/run_completion_codeformer_spatial_code_intervention_v1.sh '+EXPECTED in body,'Exact manual tmux protocol launch')
    need('quiet' in body and '6GiB' in body,'Actual quiet-worker behavior/storage described')
    install=body.split("python3 - <<'PY'\n",1)[1].split('\nPY',1)[0];ast.parse(install,feature_version=(3,10))
    builder=stage/'scripts/prepare_completion_codeformer_spatial_code_intervention_v1.py';tree=ast.parse(physical(builder).read_text(encoding='utf-8-sig'))
    literal=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INSTALL_CODE' for t in n.targets))
    need(install==literal.replace('__PROTOCOL_SHA__',EXPECTED).rstrip('\n'),'Guide literal equals actual nine-fixture installer')
    need(hashlib.sha256(literal.encode()).hexdigest()=='305777c2e412e145bd2059aebc5830e9ff06f374df2ae1f2079b0507136584a3','Root tested installer SHA')
    for row in p['files']:
        if row['path'].endswith('.py'):ast.parse(physical(stage/row['path']).read_text(encoding='utf-8-sig'),feature_version=(3,10))
    app=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json');pins=[*app['source_files'],app['candidate_manifest'],app['checkpoint']]
    for r in pins:need(physical(r['path']).stat().st_size==r['bytes'] and sha(r['path'])==r['sha256'],'R1 current protected pin')
    need(len(pins)==22,'All22 R1 app/model sources')
    report={'format':'root-spatial-code-intervention-v1-final-archive-guide-audit-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'archive':pin(archive),'protocol':pin(ppath),'archive_members':167,'pinned_payload_files':166,'payload_bytes_including_protocol':total,'full_stream_all167_SHA_and_gzip_CRC':True,'published_exact_LF_checksum_and_archive':True,'guides':[pin(q) for q in guides],'SCP_commands':scp,'single_remote_per_download':True,'installer_matches_nine_actual_root_fixtures':True,'Python310_sources_parse':True,'manual_tmux_exact_protocol':True,'all22_R1_pins_unchanged':True,'actual_new_model_or_VM_preflight':False,'new_model_created':False,'neural_calls':0,'optimizer_steps':0,'VM_calls':0,'model_qualified':False,'goal_complete':False}
    with physical(dest).open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(pin(dest)))

if __name__=='__main__':main()
