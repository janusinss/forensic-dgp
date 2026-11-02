"""Root full archive/guide verification. No extraction, model, VM or training."""
from pathlib import Path
from datetime import datetime,timezone
import ast,gzip,hashlib,json,shlex,tarfile
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001'
NAME='cctv_completion_codeformer_predictor_ce_v1_vm'
STEM='cctv-completion-codeformer-predictor-ce-v1'
EXPECTED='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b'
ARCHIVE='1052d09627cdd69f5e652a6996cee9909e5498ba516cb2b5900981b066a458c3'
GUIDE='0c35dd81e95dd96b72b127e7d58a91e40fec76ed04930113d9ca91f92b1e1749'
TRANSFER=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 archive=BASE/(STEM+'-execution.tar.gz');stage=BASE/NAME
 need(archive.stat().st_size==73748641 and sha(archive)==ARCHIVE,'Archive identity')
 ppath=stage/'protocol.json';need(sha(ppath)==EXPECTED,'Protocol identity');p=read(ppath)
 expected={NAME+'/protocol.json':{'bytes':ppath.stat().st_size,'sha256':EXPECTED}}
 expected.update({NAME+'/'+r['path']:r for r in p['files']});seen=[];total=0
 with archive.open('rb') as raw,gzip.GzipFile(fileobj=raw) as gz:
  with tarfile.open(fileobj=gz,mode='r|') as tar:
   for m in tar:
    need(m.isreg() and m.name in expected and m.name not in seen,'Unsafe/extra/duplicate tar member')
    need(seen or m.name==NAME+'/protocol.json','First member must be protocol')
    row=expected[m.name];need(m.size==row['bytes'],'Tar member size')
    h=hashlib.sha256();n=0
    with tar.extractfile(m) as f:
     for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b);n+=len(b)
    need(n==row['bytes'] and h.hexdigest()==row['sha256'],'Tar payload SHA')
    seen.append(m.name);total+=n
  while gz.read(8*1024*1024):pass
 need(set(seen)==set(expected) and len(seen)==671 and total==79488873,'Closed archive scope')
 checksum=(ARCHIVE+'  '+archive.name+'\n').encode('ascii')
 need((BASE/(archive.name+'.sha256')).read_bytes()==checksum,'Exact LF checksum')
 need(sha(TRANSFER/archive.name)==ARCHIVE and (TRANSFER/(archive.name+'.sha256')).read_bytes()==checksum,'Published transfer drift')
 guides=[ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md',TRANSFER/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md']
 need(all(q.stat().st_size==10265 and sha(q)==GUIDE for q in guides),'Published guide identity')
 body=guides[0].read_text(encoding='utf-8');need(ARCHIVE in body and EXPECTED in body,'Guide artifact bindings')
 need(not any(v in body for v in ('UNBOUND_PROTOCOL','UNPREPARED_ARCHIVE','NOT_PREPARED','NOT READY — SOURCE-ONLY')),'Unreleased placeholder in guide')
 scp=[line for line in body.splitlines() if line.startswith('gcloud compute scp ')]
 need(len(scp)==5,'Expected two uploads / three single-source downloads')
 remotes=[]
 for line in scp:
  tokens=shlex.split(line);need(tokens[:3]==['gcloud','compute','scp'],'SCP command prefix')
  need('--project=forensic-dgp-thesis' in tokens and '--zone=us-central1-a' in tokens,'SCP VM project/zone')
  remote=[v for v in tokens if v.startswith('janusdominic0@forensic-dgp-thesis:')];need(len(remote)==1,'PuTTY one-remote-source contract');remotes+=remote
 need(remotes[-3:]==['janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+STEM+x for x in ('-results.tar.gz','-results.tar.gz.sha256','-export.json')],'Exact return filenames')
 need('tmux new-session -A -s completion_predictor_ce_v1' in body,'Tmux session')
 need('bash scripts/run_completion_codeformer_predictor_ce_v1.sh '+EXPECTED in body,'Exact manual launch')
 install=body.split("python3 - <<'PY'\n",1)[1].split('\nPY',1)[0]
 ast.parse(install,feature_version=(3,10))
 builder=stage/'scripts/prepare_completion_codeformer_predictor_ce_v1.py';tree=ast.parse(builder.read_text(encoding='utf-8-sig'))
 literal=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INSTALL_CODE' for t in n.targets))
 need(install==literal.replace('__PROTOCOL_SHA__',EXPECTED).rstrip('\n'),'Guide installer differs from tested literal')
 need(hashlib.sha256(literal.encode()).hexdigest()=='5e851c5bfe26b94b70c4adcd89f96671423a5e5f157959719fd6cd7fe7fbfb4b','Tested installer binding')
 for rel in ('scripts/cctv_completion_codeformer_predictor_ce_v1_vm.py','scripts/export_completion_codeformer_predictor_ce_v1.py','research/completion_codeformer_predictor_ce_v1/terminal_admission.py'):
  ast.parse((stage/rel).read_text(encoding='utf-8-sig'),feature_version=(3,10))
 admission=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json')
 r1=admission['source_files']+[admission['candidate_manifest'],admission['checkpoint']]
 for r in r1:
  q=Path(r['path']);need(q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'R1 source/model drift')
 need(len(r1)==22,'Exact R1 assets')
 receipt={'format':'root-predictor-CE-full-archive-and-published-guide-audit-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'archive':pin(archive),'protocol':pin(ppath),'archive_members':671,'pinned_payload_files':670,'uncompressed_payload_bytes':total,'all_archive_members_fullstream_hashes_pass':True,'whole_gzip_CRC_consumed':True,'exact_LF_checksums_and_published_copy':True,'guides':[pin(q) for q in guides],'SCP_commands_checked':scp,'each_download_single_remote_source':True,'installer_exact_tested_literal_and_Python310_parse':True,'manual_tmux_protocol_bound':True,'all22_R1_model_app_pins_exact':True,'no_extraction_or_model_imports':True,'neural_calls':0,'training_steps':0,'VM_calls':0,'actual_VM_preflight_or_training_result':False,'new_model_created':False,'model_qualified':False,'goal_complete':False}
 out=BASE/'root_final_archive_audit.json'
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(receipt,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(pin(out)))
if __name__=='__main__':main()
