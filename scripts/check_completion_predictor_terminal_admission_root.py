"""Independent tiny-file terminal admission fixtures. No actual checkpoint or model."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,importlib.util,json,struct,uuid,zipfile
ROOT=Path(__file__).resolve().parents[1]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def pin(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def need(ok,message):
 if not ok:raise ValueError(message)
def main():
 source=ROOT/'research/completion_codeformer_predictor_ce_v1_development/terminal_admission.py'
 spec=importlib.util.spec_from_file_location('tiny_terminal_admission_fixture',source)
 helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
 target=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')/('predictor_terminal_root_fixture_'+uuid.uuid4().hex)
 target.mkdir();checks=[]
 def create(name,extra=(),picklebytes=b'not-executed-metadata'):
  path=target/(name+'.pth.partial')
  with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_STORED) as z:
   for key,data in [('archive/data.pkl',picklebytes),('archive/data/0',b'01234567'),('archive/version',b'3'),('archive/byteorder',b'little'),*extra]:z.writestr(key,data)
  return path
 def run(name,path,good,guard=lambda:None):
  before=pin(path);error=None;value=None
  try:value=helper.admit_terminal_zip(path,guard=guard)
  except (ValueError,zipfile.BadZipFile,TimeoutError) as exc:error=repr(exc)
  need((error is None)==good,'Unexpected fixture verdict '+name)
  need(pin(path)==before,'Fixture source mutated '+name)
  checks.append({'fixture':name,'source':before,'expected_accept':good,'accepted':error is None,'error':error,'source_preserved_exact':True,'verdict':value})
 path=create('valid_small_stored');calls=[0]
 def guard():calls[0]+=1
 run('valid_closed_partial_CRC',path,True,guard);need(calls[0]>8,'Guard not used throughout archive')
 run('unsafe_parent_path',create('unsafe_parent_path',[('archive/../escaped',b'x')]),False)
 run('executable_member',create('executable_member',[('archive/plugin.py',b'x')]),False)
 run('pickle_over1MiB',create('large_pickle',picklebytes=b'x'*(1024**2+1)),False)
 run('too_many_members',create('too_many',[('archive/data/'+str(i+1),b'x') for i in range(1612)]),False)
 corrupt=create('bad_CRC')
 with zipfile.ZipFile(corrupt) as z:entry=z.getinfo('archive/data/0')
 with corrupt.open('r+b') as f:
  f.seek(entry.header_offset);header=f.read(30);name_len,extra_len=struct.unpack('<HH',header[26:30]);f.seek(entry.header_offset+30+name_len+extra_len);first=f.read(1);f.seek(-1,1);f.write(bytes([first[0]^1]))
 run('stored_member_CRC_corruption',corrupt,False)
 def deadline():raise TimeoutError('Fixture save deadline reached; retain partial')
 run('deadline_refusal_preserves_partial',path,False,deadline)
 need(len(checks)==7,'Fixture count')
 out=ROOT/'research/completion_codeformer_predictor_ce_design_v1/root_terminal_admission_fixtures.json'
 result={'format':'root-independent-predictor-terminal-closed-partial-fixtures-v1','complete':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'tested_helper':pin(source),'fixture_directory':str(target),'actual_checkpoint_used':False,'fixtures':checks,'all7_expected_verdicts':True,'all_source_bytes_preserved':True,'guard_invocations_valid_fixture':calls[0],'Torch_imported':False,'NN_calls':0,'optimizer_objects':0,'VM_calls':0,'runtime_or_packet_admitted':False}
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(pin(out)))
if __name__=='__main__':main()
