"""Independent passive HQ512 teacher compatibility; no model/NN/optimizer/VM."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import hashlib,json,pickletools,time,zipfile
ROOT=Path(__file__).resolve().parents[2];ASSET=ROOT/'outputs/completion_codeformer_hq512_teacher_assets_v1';TEACHER=ASSET/'vqgan_code512.pth';ORIGINAL=Path(r'C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth')
SAFE={'collections OrderedDict','torch._utils _rebuild_tensor_v2','torch FloatStorage','torch ByteStorage','torch LongStorage'}
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def passive(path,expected):
 p=pin(path);need(p['sha256']==expected,'Checkpoint SHA differs');need(zipfile.is_zipfile(path),'Not complete ZIP')
 with zipfile.ZipFile(path) as z:
  rows=z.infolist();names=[r.filename for r in rows];need(len(set(names))==len(names) and all(not r.is_dir() and not r.flag_bits&1 for r in rows),'Unsafe ZIP entries')
  need(all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and ':' not in n and '\\' not in n and '/code/' not in n for n in names),'Unsafe namespace');need(z.testzip() is None,'ZIP CRC failed');metadata=[n for n in names if n.endswith('/data.pkl')];need(len(metadata)==1,'Metadata count');raw=z.read(metadata[0]);need(len(raw)<=2*1024**2,'Metadata cap');found=set();count=0
  for op,arg,pos in pickletools.genops(raw):
   count+=1;need(op.name!='STACK_GLOBAL','Dynamic global refused')
   if op.name=='GLOBAL':need(arg in SAFE,'Unexpected pickle global '+str(arg));found.add(arg)
 return {'file':p,'ZIP_entries':len(rows),'CRC_all_members_pass':True,'pickle_opcodes':count,'pickle_globals':sorted(found),'safe_globals_added':False}
def main():
 start=time.monotonic();root_report=ASSET/'root_passive_compatibility.json';need(sha(root_report)=='0ad1d25ae32f29df18ad11e6109c5c56cf9241e270a5220d76a4eba91e923d15','Root prior comparison changed')
 admissions={'teacher':passive(TEACHER,'22a1bb403c9ba0e5b85b52a85badced1cb88ac9ccf0790873b5b2e6690fbb906'),'original_inpainting':passive(ORIGINAL,'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797')}
 import torch
 torch.set_num_threads(1);teacher=torch.load(TEACHER,map_location='cpu',weights_only=True);original=torch.load(ORIGINAL,map_location='cpu',weights_only=True)
 need(set(teacher)==set(original)=={'params_ema'},'Checkpoint object schema');t=teacher['params_ema'];o=original['params_ema'];need(len(t)==329 and len(o)==497,'Model state cardinality');need(set(t)=={n for n in o if n.startswith(('encoder.','quantize.','generator.'))},'Teacher state namespaces differ')
 def logical(m):
  h=hashlib.sha256()
  for n,v in sorted(m.items()):a=v.contiguous();h.update(n.encode());h.update(str(tuple(a.shape)).encode());h.update(str(a.dtype).encode());h.update(a.numpy().tobytes())
  return h.hexdigest()
 comparisons=[]
 for n,v in t.items():
  other=o[n];need(v.device.type==other.device.type=='cpu' and v.dtype==other.dtype==torch.float32 and v.shape==other.shape and torch.isfinite(v).all().item() and torch.isfinite(other).all().item(),'Tensor schema/finite differs '+n);delta=v.double()-other.double();comparisons.append({'key':n,'shape':list(v.shape),'dtype':str(v.dtype),'bitexact':torch.equal(v,other),'maximum_absolute_difference':float(delta.abs().max()),'L2_difference':float(torch.linalg.vector_norm(delta))})
 namespaces={}
 for label,prefix in [('encoder','encoder.'),('codebook','quantize.'),('generator','generator.')]:
  selected=[r for r in comparisons if r['key'].startswith(prefix)];tm={n:v for n,v in t.items() if n.startswith(prefix)};om={n:v for n,v in o.items() if n.startswith(prefix)};namespaces[label]={'tensors':len(selected),'equal_tensors':sum(r['bitexact'] for r in selected),'different_tensors':sum(not r['bitexact'] for r in selected),'teacher_logical_sha256':logical(tm),'original_logical_sha256':logical(om),'all_bitexact':all(r['bitexact'] for r in selected)}
 need(tuple(t['quantize.embedding.weight'].shape)==(512,256),'Teacher codebook dimension differs');need(namespaces['encoder']['different_tensors']==164 and namespaces['codebook']['all_bitexact'] is False and namespaces['generator']['all_bitexact'] is False,'Expected incompatibility not independently confirmed')
 parent=json.loads(root_report.read_text());prior=[r for r in comparisons if r['key'].startswith(('quantize.','generator.'))];need(len(prior)==165 and parent['prior_all_bitexact'] is False and parent['codebook_bitexact'] is False and parent['encoder_changed']==164,'Root finding differs')
 actual={r['key']:r for r in comparisons}
 for row in parent['tensor_comparisons']:need(actual[row['key']]['bitexact']==row['bitexact_to_original'] and actual[row['key']]['maximum_absolute_difference']==row['maximum_absolute_difference'],'Root per-tensor comparison differs')
 result={'format':'independent-official-HQ512-teacher-passive-compatibility-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'acquisition':pin(ASSET/'acquisition.json'),'root_comparison':pin(root_report),'passive_admission':admissions,'safe_CPU_weights_only':True,'all_passive_reviews_before_loading':True,'teacher_tensors':329,'original_inpainting_tensors':497,'all329_state_names_shapes_dtypes_compatible':True,'codebook_shape':[512,256],'namespace_comparison':namespaces,'all165_original_prior_bitexact':False,'root_all329_tensor_results_independently_match':True,'tensor_comparisons':comparisons,'decision':'NOT_A_DROP_IN_TEACHER_CODE_SPACE_OR_PRIOR','interpretation':['Same512x256dimensions do not establish matching latent vectors, nearest-code boundaries, codeIDs or decoder semantics.','Official HQ512 code IDs must not be supplied as direct CE labels to the fixed inpainting decoder/codebook; no replacement or mapping is justified by shape compatibility.','A future bounded representation/code-oracle diagnostic must distinguish teacher codes/latents from actual inference, avoid claiming oracle hidden-region recovery and retain all parent/prior gates.','Neither codebook permutation equivalence nor a viable mapping is proven or ruled out by unequal bytes alone.','No neural teacher reconstruction, code prediction, generated outputs or useful training behavior were tested.'],'local_actions':{'CPU_weights_only_loads':2,'neural_modules_or_forwards':0,'gradients':0,'optimizer_operations':0,'VM_calls':0,'asset_replacement_or_adoption':False},'new_training_packet':False,'model_qualified':False,'seconds':time.monotonic()-start}
 out=ASSET/'independent_passive_compatibility.json'
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(out),'namespace_comparison':namespaces,'decision':result['decision'],'seconds':result['seconds']},indent=2))
if __name__=='__main__':main()
