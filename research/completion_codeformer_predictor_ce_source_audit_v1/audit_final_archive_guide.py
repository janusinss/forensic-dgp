"""Independent complete archive/guide admission; no extraction, VM or training."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import ast,gzip,hashlib,json,re,tarfile,time
ROOT=Path(__file__).resolve().parents[2];FOLDER=Path(__file__).resolve().parent
BASE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001'
STAGE=BASE/'cctv_completion_codeformer_predictor_ce_v1_vm'
STEM='cctv-completion-codeformer-predictor-ce-v1'
PROTOCOL='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b'
ARCHIVE='1052d09627cdd69f5e652a6996cee9909e5498ba516cb2b5900981b066a458c3'
checks=[]
def need(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def main():
 started=time.monotonic();publication=read(BASE/'archive_publication_receipt.json');archive=BASE/(STEM+'-execution.tar.gz');need(archive.stat().st_size==73748641 and sha(archive)==ARCHIVE,'Whole released archive SHA/size exact')
 p=read(STAGE/'protocol.json');need(sha(STAGE/'protocol.json')==PROTOCOL and len(p['files'])==670,'Sealed670source protocol exact');expected={r['path']:r for r in p['files']};expected['protocol.json']={'bytes':(STAGE/'protocol.json').stat().st_size,'sha256':PROTOCOL};seen=set();payload=0;memberroot='cctv_completion_codeformer_predictor_ce_v1_vm'
 with gzip.open(archive,'rb') as gz:
  with tarfile.open(fileobj=gz,mode='r|') as tar:
   for ordinal,row in enumerate(tar):
    name=PurePosixPath(row.name);need(row.isreg() and not name.is_absolute() and '..' not in name.parts and ':' not in row.name and '\\' not in row.name and name.parts[0]==memberroot and str(name)==row.name,'Safe regular archive member '+row.name)
    relative='/'.join(name.parts[1:]);need(relative in expected and relative not in seen,'Exact unique archive namespace '+relative)
    if ordinal==0:need(relative=='protocol.json','Protocol first for safe streamed installer')
    wanted=expected[relative];need(row.size==wanted['bytes'],'Exact archive extent '+relative);h=hashlib.sha256();copied=0
    with tar.extractfile(row) as f:
     for b in iter(lambda:f.read(1024*1024),b''):copied+=len(b);need(copied<=wanted['bytes'],'Archive member bounded '+relative);h.update(b)
    need(copied==wanted['bytes'] and h.hexdigest()==wanted['sha256'],'Exact entire archived bytes '+relative);seen.add(relative);payload+=copied
  trailing=0
  for block in iter(lambda:gz.read(1024*1024),b''):need(not any(block),'Only TAR zero padding after closed namespace');trailing+=len(block)
 need(seen==set(expected) and len(seen)==671 and payload==79488873,'Closed671archive namespace and79488873payload bytes')
 need(publication['protocol_sha256']==PROTOCOL and publication['archive_sha256']==ARCHIVE and publication['payload_files']==671 and publication['payload_bytes']==payload,'Publication receipt exact artifact linkage')
 release=read(BASE/'root_manual_transfer_release.json');need(release['released_for_manual_vm_transfer'] and not release['automatic_vm_launch'] and release['protocol_sha256']==PROTOCOL and not release['training_launched'],'Manual-only root release')
 for key,row in release['evidence'].items():q=Path(row['path']);need(q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Exact release evidence '+key)
 checksum=BASE/(STEM+'-execution.tar.gz.sha256');line=(ARCHIVE+'  '+archive.name+'\n').encode('ascii');need(checksum.read_bytes()==line and b'\r' not in line,'Exact LF checksum compatible with VMsha256sum')
 for row in publication['published']+[publication['manual_VM_guide'],publication['root_release'],publication['independent_source_admission']]:
  q=Path(row['path']);need(q.is_file() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Exact published/release copy '+q.name)
 guide=ROOT/'COMPLETION_CODEFORMER_PREDICTOR_CE_V1_VM.md';need(guide.stat().st_size==10265 and sha(guide)=='0c35dd81e95dd96b72b127e7d58a91e40fec76ed04930113d9ca91f92b1e1749' and guide.read_bytes()==Path(publication['manual_VM_guide']['path']).read_bytes(),'Canonical and Windows transfer guide byteexact')
 text=guide.read_text();builder=ROOT/'scripts/prepare_completion_codeformer_predictor_ce_v1.py';tree=ast.parse(builder.read_text());install=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INSTALL_CODE' for t in n.targets));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='manual_guide');scope={'INSTALL_CODE':install};exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),'<exact-guide-builder>','exec'),scope);need(text==scope['manual_guide'](PROTOCOL,ARCHIVE,73748641,670),'Final guide generated exactly from sealed builder')
 fixtures=read(ROOT/'research/completion_codeformer_predictor_ce_design_v1/root_installer_fixtures.json');need(hashlib.sha256(install.encode()).hexdigest()==fixtures['installer_literal_sha256']=='5e851c5bfe26b94b70c4adcd89f96671423a5e5f157959719fd6cd7fe7fbfb4b' and fixtures['complete'] and len(fixtures['fixtures'])==9,'Exact installer literal retains independent9negative/positive fixtures')
 code=text.split("python3 - <<'PY'\n",1)[1].split('\nPY\n',1)[0]+'\n';need(code==install.replace('__PROTOCOL_SHA__',PROTOCOL),'Pasted Python installer is exact admitted literal');ast.parse(code,feature_version=(3,10));need('assert ' not in code,'Installer checks remain active under optimizedPython')
 commands=[line for line in text.splitlines() if line.startswith('gcloud compute scp ')];need(len(commands)==5 and all(line.count('janusdominic0@forensic-dgp-thesis:')==1 for line in commands),'Five commands each have one remote source/destination')
 need(len([line for line in text.splitlines() if line.startswith('## ')])==5 and 'UNRELEASED' not in text and 'PENDING_ROOT_RELEASE' not in text and 'RELEASED FOR MANUAL TRANSFER ONLY' in text,'Five clear released manual steps without placeholders')
 for value in ('32 Adam updates /128 backwards','Only 114','383','384 network calls','168 empty','teacher NPZ copies','1800 seconds hard/1680','2.75 GiB','8 GiB','Final 32 must pass','1%','No optimizer reset','no automatic resume'):
  need(value in text,'Guide scientific/execution statement '+value)
 previous=BASE/'prior_guide_worktree.md';need(sha(previous)=='eac7c2f54af1e55d151c0505c0fca698402aebbd111cb1e01a28d7b4212a79a4' and previous.read_bytes()==(BASE/'manual_guide_DRAFT.md').read_bytes(),'Exact unreleased guide preserved')
 alias=Path(publication['successful_archive_partial_hardlink_retained']);need(alias.is_file() and alias.stat().st_ino==archive.stat().st_ino and alias.stat().st_dev==archive.stat().st_dev and alias.stat().st_size==archive.stat().st_size and sha(alias)==ARCHIVE,'Successful archive partial is same-inode alias,not duplicate bytes')
 need(sha(archive)==ARCHIVE and sha(STAGE/'protocol.json')==PROTOCOL,'Final archive/protocol unchanged by read-only admission')
 out=FOLDER/'independent_final_archive_guide_admission.json';need(not out.exists(),'Never overwrite final admission');report={'format':'independent-predictor-CE-final-archive-guide-admission-v1','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'archive':pin(archive),'protocol_sha256':PROTOCOL,'member_count':len(seen),'payload_bytes':payload,'whole_gzip_stream_CRC_and_tar_member_hashes_verified':True,'closed_namespace':True,'source_packet_admission':pin(FOLDER/'independent_source_packet_admission.json'),'actual_guarded_transfer_CLI':pin(FOLDER/'independent_pure_transfer_CLI_fixture_v1.json'),'root_release':pin(BASE/'root_manual_transfer_release.json'),'publication_receipt':pin(BASE/'archive_publication_receipt.json'),'guide':pin(guide),'prior_unreleased_guide':pin(previous),'checks_passed':len(checks),'checks':checks,'released_for_manual_transfer_only':True,'new_model_created':False,'training_launched':False,'VM_calls':0,'NN_calls':0,'optimizer_calls':0,'gradient_calls':0,'archive_extracted':False,'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started};out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'complete':True,'path':str(out),'sha256':sha(out),'checks':len(checks),'seconds':report['seconds']}))
if __name__=='__main__':main()
