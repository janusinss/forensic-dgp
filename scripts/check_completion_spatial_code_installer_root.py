"""Run the exact installer literal on confined toy archives, without VM or models."""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import ast,hashlib,io,json,tarfile,uuid
ROOT=Path(__file__).resolve().parents[1]
NAME='cctv_completion_codeformer_spatial_code_intervention_v1_vm'
STEM='cctv-completion-codeformer-spatial-code-intervention-v1'
def digest(body):return hashlib.sha256(body).hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())}
def need(ok,message):
 if not ok:raise ValueError(message)
def main():
 builder=ROOT/'scripts/prepare_completion_codeformer_spatial_code_intervention_v1.py'
 parsed=ast.parse(builder.read_text(encoding='utf-8-sig'))
 literal=next(ast.literal_eval(n.value) for n in parsed.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='INSTALL_CODE' for t in n.targets))
 need(digest(literal.encode())=='305777c2e412e145bd2059aebc5830e9ff06f374df2ae1f2079b0507136584a3','Installer literal drift')
 ast.parse(literal,feature_version=(3,10))
 base=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')/('spatial_installer_root_fixture_'+uuid.uuid4().hex);base.mkdir()
 checks=[]
 for scenario in ('valid','already_installed','wrong_protocol_SHA','missing_payload','payload_hash_drift','duplicate_payload','symlink_payload','unsafe_member','insufficient_payload_plus6GiB'):
  home=base/scenario;home.mkdir();(home/'forensic-dgp').mkdir()
  data=b'toy regular payload only';row={'path':'assets/toy.bin','bytes':len(data),'sha256':digest(data)}
  p={'format':'completion-codeformer-spatial-code-intervention-v1','files':[row]};body=(json.dumps(p)+'\n').encode();protocol_sha=digest(body)
  archive=home/(STEM+'-execution.tar.gz')
  with tarfile.open(archive,'w:gz') as tar:
   m=tarfile.TarInfo(NAME+'/protocol.json');m.size=len(body);tar.addfile(m,io.BytesIO(body))
   if scenario!='missing_payload':
    m=tarfile.TarInfo(NAME+'/assets/toy.bin');m.size=len(data)
    if scenario=='symlink_payload':m.type=tarfile.SYMTYPE;m.linkname='../escape';m.size=0;tar.addfile(m)
    else:tar.addfile(m,io.BytesIO(b'x'*len(data) if scenario=='payload_hash_drift' else data))
   if scenario=='duplicate_payload':
    m=tarfile.TarInfo(NAME+'/assets/toy.bin');m.size=len(data);tar.addfile(m,io.BytesIO(data))
   if scenario=='unsafe_member':
    m=tarfile.TarInfo(NAME+'/../escape');m.size=1;tar.addfile(m,io.BytesIO(b'x'))
  before=pin(archive);destination=home/'forensic-dgp'/NAME
  if scenario=='already_installed':destination.mkdir();(destination/'owned.marker').write_bytes(b'preserve existing')
  expected='0'*64 if scenario=='wrong_protocol_SHA' else protocol_sha
  code=literal.replace('__PROTOCOL_SHA__',expected);error=None
  free=6*1024**3 if scenario=='insufficient_payload_plus6GiB' else 20*1024**3
  try:
   with patch.object(Path,'home',return_value=home),patch('shutil.disk_usage',return_value=type('Disk',(),{'free':free})()):
    exec(compile(code,'<exact-toy-installer>','exec'),{})
  except (ValueError,FileExistsError,tarfile.TarError) as exc:error=repr(exc)
  need((error is None)==(scenario=='valid'),'Wrong verdict '+scenario)
  need(pin(archive)==before,'Installer changed source archive')
  if scenario=='valid':need((destination/'assets/toy.bin').read_bytes()==data and (destination/'protocol.json').read_bytes()==body,'Successful toy install not exact')
  elif scenario=='already_installed':need((destination/'owned.marker').read_bytes()==b'preserve existing','Existing target changed')
  else:need(not destination.exists(),'Refused fixture published a target')
  need(not (home/'forensic-dgp/escape').exists(),'Archive path escaped')
  checks.append({'fixture':scenario,'expected_accept':scenario=='valid','accepted':error is None,'error':error,'source_archive_unchanged':True,'target_preservation_or_closed_failure_pass':True,'archive':before})
 out=ROOT/'research/completion_codeformer_spatial_code_intervention_v1_design/root_installer_fixtures.json'
 result={'format':'root-spatial-code-intervention-exact-installer-toy-fixtures-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'builder_source':pin(builder),'installer_literal_sha256':digest(literal.encode()),'Python310_parse':True,'fixtures':checks,'all9_expected_verdicts':True,'fixture_home_and_disk_usage_patched_only_in_own_test_process':True,'fixture_namespace':str(base),'actual_VM_or_gcloud_calls':0,'scientific_imports':0,'NN_calls':0,'optimizer_objects':0,'real_model_or_protocol_execution':False,'runtime_or_packet_admitted':False}
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(pin(out)))
if __name__=='__main__':main()
