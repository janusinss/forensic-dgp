"""Independent final full archive stream, checksum and published guide admission; no extraction/NN."""
import ast,gzip,hashlib,importlib.util,json,os,re,shlex,tarfile
from pathlib import Path,PurePosixPath
B=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION');H=B/'research/completion_spatial_code_intervention_v1_transport_review';T=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs');D=B/'outputs/completion_codeformer_spatial_code_intervention_v1_packet/draft_004';V='cctv_completion_codeformer_spatial_code_intervention_v1_vm';STEM='cctv-completion-codeformer-spatial-code-intervention-v1';EXPECTED='b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123';CAP=128*1024**2;checks=[]
def xp(p):return '\\\\?\\'+str(Path(p).absolute())
def digest(p):
 h=hashlib.sha256()
 with open(xp(p),'rb') as f:
  for v in iter(lambda:f.read(1024**2),b''):h.update(v)
 return h.hexdigest()
def pin(p):return {'path':str(p),'bytes':os.stat(xp(p)).st_size,'sha256':digest(p)}
def read(p):
 with open(xp(p),'r',encoding='utf-8-sig') as f:return json.load(f)
def need(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
p=read(D/V/'protocol.json');need(digest(D/V/'protocol.json')==EXPECTED,'Final staged protocol exact');expected={r['path']:r for r in p['files']};expected['protocol.json']={'path':'protocol.json','bytes':os.stat(xp(D/V/'protocol.json')).st_size,'sha256':EXPECTED}
receipt=read(D/'archive_publication_receipt.json');need(receipt['complete'] is True and receipt['protocol_sha256']==EXPECTED and receipt['training_launched'] is False and receipt['VM_calls']==0 and receipt['model_qualified'] is False,'Actual manual publication scope')
archive=T/(STEM+'-execution.tar.gz');local=D/archive.name;need(pin(local)==receipt['archive'],'Published canonical archive receipt pin');need(os.stat(xp(archive)).st_size==os.stat(xp(local)).st_size and digest(archive)==digest(local)==receipt['archive_sha256'],'Original transfer and canonical archive exactbytes');need(os.stat(xp(archive)).st_size<=CAP,'Execution compressed cap128MiB')
checksum=T/(archive.name+'.sha256');canonical_checksum=D/checksum.name;expectedline=(receipt['archive_sha256']+'  '+archive.name+'\n').encode('ascii');need(checksum.read_bytes()==canonical_checksum.read_bytes()==expectedline,'LF checksum exactfilename/hash/publishedcopies')
class Counted:
 def __init__(self,stream):self.stream=stream;self.count=0
 def read(self,size=-1):
  body=self.stream.read(1024**2 if size<0 else size);self.count+=len(body)
  if self.count>CAP:raise ValueError('Raw archive exceeds128MiB')
  return body
 def tell(self):return self.count
seen={};payload=0
with open(xp(archive),'rb') as raw,gzip.GzipFile(fileobj=raw,mode='rb') as gz:
 counted=Counted(gz)
 with tarfile.open(fileobj=counted,mode='r|',bufsize=1024**2) as tar:
  for member in tar:
   need(member.name.startswith(V+'/'),'Exact execution archive root');relative=member.name[len(V)+1:];name=PurePosixPath(relative);need(str(name)==relative and not name.is_absolute() and '..' not in name.parts and '\\' not in relative and ':' not in relative and relative.casefold() not in {v.casefold() for v in seen},'Canonical unique execution namespace');need(member.isreg() and relative in expected,'Regular exact protocol manifest member');need(member.size==expected[relative]['bytes'],'Exact member size '+relative)
   if not seen:need(relative=='protocol.json','Installer protocol member first')
   h=hashlib.sha256();length=0
   with tar.extractfile(member) as f:
    for block in iter(lambda:f.read(1024**2),b''):h.update(block);length+=len(block);need(length<=member.size,'Bounded archive payload')
   need(length==member.size and h.hexdigest()==expected[relative]['sha256'],'Actual complete member hash '+relative);seen[relative]={'bytes':length,'sha256':h.hexdigest()};payload+=length
 while True:
  block=counted.read(1024**2)
  if not block:break
  need(not any(block),'Only zero raw-tar trailing padding')
need(set(seen)==set(expected) and len(seen)==167,'All166payloads+protocol exactclosednamespace');need(payload==receipt['payload_bytes']==19491897,'Exact payload sum');need(counted.count==receipt['raw_tar_bytes'] and counted.count<=CAP,'Full gzipCRC/EOF and exact physical rawtar length')
guide_name='COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_VM.md';guidefile=T/guide_name;guidecanonical=B/guide_name;body=guidefile.read_bytes();need(body==guidecanonical.read_bytes() and pin(guidefile)==receipt['manual_VM_guide'],'Published guide copies/receipt exact');need(b'\r' not in body,'Published guide LFonly');guide=body.decode('utf-8');builderpath=B/'scripts/prepare_completion_codeformer_spatial_code_intervention_v1.py';spec=importlib.util.spec_from_file_location('independent_spatial_final_builder',builderpath);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
need(guide==builder.manual_guide(EXPECTED,receipt['archive_sha256'],os.stat(xp(archive)).st_size,166),'Published guide exact current template/actual archive pins')
install=guide.split("python3 - <<'PY'\n",1)[1].split('\nPY\n',1)[0]+'\n';need(install==builder.INSTALL_CODE.replace('__PROTOCOL_SHA__',EXPECTED),'Actual pasted installer literal exact');need(hashlib.sha256(install.replace(EXPECTED,'__PROTOCOL_SHA__').encode()).hexdigest()=='305777c2e412e145bd2059aebc5830e9ff06f374df2ae1f2079b0507136584a3','Actual installer equals root9fixtureliteral');ast.parse(install,feature_version=(3,10))
commands=[line for line in guide.splitlines() if line.startswith('gcloud compute scp ')];need(len(commands)==5,'Actual published fiveSCPcommands')
for i,line in enumerate(commands):
 args=[v for v in shlex.split(line)[3:] if not v.startswith('--')];need(len(args)==2 and sum('janusdominic0@forensic-dgp-thesis:' in v for v in args)==1 and '--project=forensic-dgp-thesis' in line and '--zone=us-central1-a' in line,'Actual SCP one remote exact project/zone')
 if i<2:need(args[1]=='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' and args[0]==STEM+('-execution.tar.gz' if i==0 else '-execution.tar.gz.sha256'),'Actual uploadfiles/direction')
 else:need(args[1]=='.' and args[0]=='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+STEM+('-results.tar.gz' if i==2 else '-results.tar.gz.sha256' if i==3 else '-export.json'),'Actual downloadfiles/direction')
need(guide.count('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"')==2 and 'tmux new-session -A -s completion_spatial_code_intervention_v1' in guide and guide.count('--protocol-sha '+EXPECTED)==1,'Actual Windowsfolder/tmux/verifyprotocol')
need('bash scripts/run_completion_codeformer_spatial_code_intervention_v1.sh '+EXPECTED in guide,'Actual launchprotocol');need('PREPARED, UNRUN' in guide and 'UNAVAILABLE CLEAN INFORMATION' in guide and 'No model is trained.' in guide,'Actual scope truth')
release=receipt['root_release'];need(pin(Path(release['path']))==release,'Actual root release pin');r=read(Path(release['path']));need(r['released_for_manual_vm_transfer'] is True and r['automatic_vm_launch'] is False and r['protocol_sha256']==EXPECTED,'Actual manual rootrelease');ind=receipt['independent_source_admission'];need(pin(Path(ind['path']))==ind and read(Path(ind['path']))['protocol_sha256']==EXPECTED,'Actual separate independent source/CLI admission pin')
need(digest(archive)==receipt['archive_sha256'],'Archive stable after complete independent read')
result={'format':'spatial-code-intervention-v1-independent-final-archive-guide-admission-v1','complete':True,'independent':True,'protocol_sha256':EXPECTED,'archive':pin(archive),'canonical_archive':pin(local),'checksum':pin(checksum),'guide':pin(guidefile),'regular_members':len(seen),'payload_bytes':payload,'physical_raw_tar_bytes':counted.count,'full_stream_member_SHA_gzip_CRC_verified':True,'all167_manifest_members_match':True,'actual_guide_installer_and_single_remote_commands_verified':True,'root_release':release,'independent_source_admission':ind,'earlier_staged_admission':pin(H/'independent_staged_transport_admission_v004.json'),'transport_source_admission':pin(H/'independent_transport_source_review_v004.json'),'checks_passed':len(checks),'checks':checks,'files':seen,'manual_packet_ready':True,'actual_VM_execution_verified':False,'new_model_created':False,'model_qualified':False,'runtime_semantics_certified_by_other_independent_agent':True,'neural_calls':0,'VM_calls':0,'model_construction':False,'actual_training_calls':0}
out=H/'independent_final_archive_guide_admission_v004.json'
with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'pin':pin(out),'checks':len(checks),'archive':pin(archive)},indent=2))
