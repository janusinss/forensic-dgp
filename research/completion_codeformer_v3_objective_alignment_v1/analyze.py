"""Saved-array-only HQ training-loss versus delivered endpoint diagnosis.
No Torch, model, neural, autograd, optimizer, VM or generated-output edits.
"""
from pathlib import Path
from datetime import datetime,timezone
import ast,hashlib,json,sys,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2]
RETURN=ROOT/'outputs/completion_codeformer_fusion_only_v3_return/cctv_completion_codeformer_fusion_only_v3_results'
STAGE=ROOT/'outputs/completion_codeformer_fusion_only_v3_packet/draft_002/cctv_completion_codeformer_fusion_only_v3_vm'
AUDIT=ROOT/'outputs/completion_codeformer_fusion_only_v3_return/independent_saved_output_audit.json'
OUT=ROOT/'outputs/completion_codeformer_v3_objective_alignment_v1'
EXPECTED_PROTOCOL='3dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033'
EXPECTED_AUDIT='9788881c412173e9f564aaf512859b07179a2977089d9ffd878cb044a3da49a0'
KINDS=('lower_face','eye_band','image_left_lens','cheek_object')
HQ=('HQ512_hole_L1_unclamped','HQ512_boundary_L1_unclamped','HQ512_training_sum','HQ512_hole_L1_clipped','HQ512_boundary_L1_clipped')
DELIVERED=('raw_hole_L1','raw_boundary_L1','PNG_hole_L1','PNG_boundary_L1')
AUX=('clipped_channel_fraction_hole512','clipped_pixel_fraction_hole512','target256_vs_downsample512_hole_L1','target256_vs_downsample512_boundary_L1')

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def read(path):
 def reject(value):raise ValueError('Nonfinite JSON '+value)
 return json.loads(path.read_text(encoding='utf-8-sig'),parse_constant=reject)
def rgb(path,size):
 with Image.open(path) as im:
  need(im.format=='PNG' and im.mode=='RGB' and im.size==(size,size),'Target PNG geometry/mode');return np.array(im,copy=True)
def ring(mask):
 padded=np.pad(mask,((3,3),(3,3)),constant_values=True)
 return mask & ~np.lib.stride_tricks.sliding_window_view(padded,(7,7)).all(axis=(-1,-2))
def down2(a):return ((a[:,:,0::2,0::2]+a[:,:,0::2,1::2])+(a[:,:,1::2,0::2]+a[:,:,1::2,1::2]))*np.float32(.25)
def group(rows,condition=False):
 def avg(part):
  need(part,'Missing metric group');return {name:float(np.mean([r[name] for r in part],dtype=np.float64)) for name in (*HQ,*DELIVERED,*AUX)}
 result={'macro':avg(rows),'geometry':{name:avg([r for r in rows if r['mask_kind']==name]) for name in KINDS}}
 if condition:result['condition']={name:avg([r for r in rows if r['condition']==name]) for name in sorted({r['condition'] for r in rows})}
 return result

def changes(now,old):
 def compare(a,b):
  d={key:{'baseline':b[key],'candidate':a[key],'delta':a[key]-b[key],'gain_percent':100*(b[key]-a[key])/max(abs(b[key]),1e-12)} for key in (*HQ,*DELIVERED,*AUX)}
  d['HQ_training_sum_improves_while_delivered_PNG_hole_worsens']=a['HQ512_training_sum']<b['HQ512_training_sum']-1e-12 and a['PNG_hole_L1']>b['PNG_hole_L1']+1e-12
  d['HQ_hole_improves_while_delivered_PNG_hole_worsens']=a['HQ512_hole_L1_unclamped']<b['HQ512_hole_L1_unclamped']-1e-12 and a['PNG_hole_L1']>b['PNG_hole_L1']+1e-12
  d['HQ_hole_and_delivered_PNG_hole_both_worsen']=a['HQ512_hole_L1_unclamped']>b['HQ512_hole_L1_unclamped']+1e-12 and a['PNG_hole_L1']>b['PNG_hole_L1']+1e-12
  d['HQ_training_sum_and_delivered_PNG_hole_both_worsen']=a['HQ512_training_sum']>b['HQ512_training_sum']+1e-12 and a['PNG_hole_L1']>b['PNG_hole_L1']+1e-12
  return d
 result={'macro':compare(now['macro'],old['macro'])}
 for category in ('geometry','condition'):
  if category in now:result[category]={name:compare(now[category][name],old[category][name]) for name in now[category]}
 return result

def main():
 start=time.monotonic();need(not OUT.exists(),'Preserve previous analysis output')
 need(sha(RETURN/'protocol.json')==sha(STAGE/'protocol.json')==EXPECTED_PROTOCOL,'Protocol differs')
 need(sha(AUDIT)==EXPECTED_AUDIT,'Independent delivered audit differs')
 p=read(RETURN/'protocol.json');audit=read(AUDIT);manifest={r['path']:r for r in read(RETURN/'export_manifest.json')['files']};files={r['path']:r for r in p['files']}
 worker=STAGE/'scripts/cctv_completion_codeformer_fusion_only_v3_vm.py';need(sha(worker)==files[worker.relative_to(STAGE).as_posix()]['sha256'],'Worker source differs')
 source=worker.read_text(encoding='utf-8');tree=ast.parse(source)
 required=["prediction=(raw+1)/2", "rgb_loss=(abs(prediction-target)*m512).sum()/(m512.sum()*3)","seam=(abs(prediction-target)*b).sum()/(b.sum()*3)","loss=(rgb_loss+seam)/4","m512=F.interpolate(m[active],size=(512,512),mode='nearest')","def boundary(m):return (m-(1-F.max_pool2d(1-m,7,1,3))).clamp(0,1)","generated=((raw+1)/2).clamp(0,1)","generated=F.interpolate(generated,size=rgb.shape[-2:],mode='bilinear',align_corners=False)"]
 need(all(fragment in source for fragment in required),'Inspected formulas differ')
 assetpins={};evidencepins={};targets={}
 def asset(name):
  path=STAGE/name;row=files[name]
  if name not in assetpins:need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Target asset differs');assetpins[name]=pin(path)
  return path
 def evidence(path):
  name=path.relative_to(RETURN).as_posix();row=manifest[name]
  if name not in evidencepins:need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Returned source differs');evidencepins[name]=pin(path)
  return path
 delivered={(r['cycle'],r['kind'],r['id']):r for r in audit['all_outputs']}
 records=[];summary={};strict_gray=[]
 for kind,key,cycles in [('monitor','monitor_cases',(0,16,32,64)),('challenge','challenge_cases',(0,64))]:
  for cycle in cycles:
   cohort=[]
   for case in p[key]:
    ident=case['id'];path=evidence(RETURN/('outputs/u%04d'%cycle)/kind/(ident+'.npz'))
    with np.load(path,allow_pickle=False) as arrays:
     raw=arrays['raw512'].copy();m256=arrays['mask256'][0,0].copy();result=arrays['result256'].copy();observed=arrays['input256'].copy()
    need(raw.dtype==np.float32 and raw.shape==(1,3,512,512) and np.isfinite(raw).all(),'Raw512 FP32 finite');need(np.isin(m256,[0,1]).all(),'Mask binary')
    support=m256.astype(bool);m512=np.repeat(np.repeat(support,2,axis=0),2,axis=1);b512=ring(m512);b256=ring(support)
    tkey=(case['clean256'],case['clean512'])
    if tkey not in targets:
     t256=rgb(asset(case['clean256']),256).astype(np.float32).transpose(2,0,1)[None]/np.float32(255);t512=rgb(asset(case['clean512']),512).astype(np.float32).transpose(2,0,1)[None]/np.float32(255)
     targets[tkey]=(t256,t512)
    target256,target512=targets[tkey];unclamped=(raw+np.float32(1))/np.float32(2);clamped=np.clip(unclamped,0,1)
    error=np.abs(unclamped-target512)[0].transpose(1,2,0).astype(np.float64);cerror=np.abs(clamped-target512)[0].transpose(1,2,0).astype(np.float64)
    saturation=(unclamped<0)|(unclamped>1);satHWC=saturation[0].transpose(1,2,0)
    discrepancy=np.abs(down2(target512)-target256)[0].transpose(1,2,0).astype(np.float64)
    row={'kind':kind,'cycle':cycle,'id':ident,'reference_id':case['reference_id'],'mask_kind':case['mask_kind'],'condition':case['condition'],'style':case['style'],'mask256_pixels':int(support.sum()),'mask512_pixels':int(m512.sum()),'boundary256_pixels':int(b256.sum()),'boundary512_pixels':int(b512.sum()),'HQ512_hole_L1_unclamped':float(error[m512].mean()),'HQ512_boundary_L1_unclamped':float(error[b512].mean()),'HQ512_hole_L1_clipped':float(cerror[m512].mean()),'HQ512_boundary_L1_clipped':float(cerror[b512].mean()),'clipped_channel_fraction_hole512':float(satHWC[m512].mean()),'clipped_pixel_fraction_hole512':float(satHWC[m512].any(axis=-1).mean()),'target256_vs_downsample512_hole_L1':float(discrepancy[support].mean()),'target256_vs_downsample512_boundary_L1':float(discrepancy[b256].mean()),**{name:delivered[(cycle,kind,ident)][name] for name in DELIVERED}}
    row['HQ512_training_sum']=row['HQ512_hole_L1_unclamped']+row['HQ512_boundary_L1_unclamped']
    if case['style']=='GRAY':
     row['HQ512_raw_mean_channel_spread_hole']=float(np.ptp(unclamped[0].transpose(1,2,0),axis=-1)[m512].mean());row['delivered_RAW256_mean_channel_spread_hole']=float(np.ptp(result[0].transpose(1,2,0),axis=-1)[support].mean())
     pngPath=evidence(path.with_suffix('.png'));png=rgb(pngPath,256);row['delivered_PNG256_mean_channel_spread_hole']=float(np.ptp(png.astype(np.int16),axis=-1)[support].mean());strict_gray.append(row)
    records.append(row);cohort.append(row)
   summary[(kind,cycle)]=group(cohort,condition=kind=='challenge')
 final_changes={kind:changes(summary[(kind,64)],summary[(kind,0)]) for kind in ('monitor','challenge')}
 all_changes={kind:{str(cycle):changes(summary[(kind,cycle)],summary[(kind,0)]) for cycle in ((16,32,64) if kind=='monitor' else (64,))} for kind in ('monitor','challenge')}
 case_changes=[]
 for kind in ('monitor','challenge'):
  old={r['id']:r for r in records if r['kind']==kind and r['cycle']==0}
  for r in records:
   if r['kind']!=kind or r['cycle']!=64:continue
   delta=changes({'macro':r},{'macro':old[r['id']]})['macro'];case_changes.append({'kind':kind,'id':r['id'],'mask_kind':r['mask_kind'],'condition':r['condition'],'changes':delta})
 classifications={kind:{name:sum(bool(r['changes'][name]) for r in case_changes if r['kind']==kind) for name in ('HQ_training_sum_improves_while_delivered_PNG_hole_worsens','HQ_hole_improves_while_delivered_PNG_hole_worsens','HQ_hole_and_delivered_PNG_hole_both_worsen','HQ_training_sum_and_delivered_PNG_hole_both_worsen')} for kind in ('monitor','challenge')}
 training=[]
 for cycle in range(1,65):
  row=read(evidence(RETURN/'outputs'/('training_cycle_%04d.json'%cycle)));need(row['cycle']==cycle and row['code_route_frozen'] and row['code_losses']==[] and row['code_preclip_L2'] is None,'Fusion route receipt differs')
  norm=row['fusion_preclip_L2'];need(np.isfinite(norm) and norm>=0,'Invalid gradient receipt norm')
  losses=row['fusion_losses'];need(len(losses)==4 and {r['id'] for r in losses}==set(p['schedule'][cycle-1]),'Training case exposure differs')
  training.append({'cycle':cycle,'ids':[r['id'] for r in losses],'preclip_L2':norm,'clipping_applied':norm>1.,'mean_HQ512_hole_L1':float(np.mean([r['mask_RGB512_L1'] for r in losses])),'mean_HQ512_boundary_L1':float(np.mean([r['inner_boundary_RGB512_L1'] for r in losses]))})
 gray_by_id={r['id']:{r['cycle']:r for r in strict_gray if r['id']==ident} for ident in {r['id'] for r in strict_gray}}
 gray_summary={'cases':len(gray_by_id),'PNG_chroma_cases_reduced':sum(rows[64]['delivered_PNG256_mean_channel_spread_hole']<rows[0]['delivered_PNG256_mean_channel_spread_hole'] for rows in gray_by_id.values()),'initial_PNG_mean_channel_spread':float(np.mean([v[0]['delivered_PNG256_mean_channel_spread_hole'] for v in gray_by_id.values()])),'terminal_PNG_mean_channel_spread':float(np.mean([v[64]['delivered_PNG256_mean_channel_spread_hole'] for v in gray_by_id.values()])),'scope':'Eight exact-gray paired challenge cases; channel spread is not anatomy or identity accuracy.'}
 # Unpaired strict-gray photos provide only a style diagnostic, never HQ error.
 public_gray=[]
 for case in p['development_cases']:
  inputPNG=rgb(asset(case['input256']),256)
  if not (np.array_equal(inputPNG[:,:,0],inputPNG[:,:,1]) and np.array_equal(inputPNG[:,:,1],inputPNG[:,:,2])):continue
  row={'id':case['id'],'condition':case['condition'],'kind':'public_development_unpaired','no_clean_target':True}
  for cycle in (0,64):
   rawpath=evidence(RETURN/('outputs/u%04d'%cycle)/'public_development'/(case['id']+'.npz'))
   with np.load(rawpath,allow_pickle=False) as z:mask=z['mask256'][0,0].astype(bool);result=z['result256'].copy()
   if mask.any():
    observed=rgb(evidence(rawpath.with_suffix('.png')),256);row[str(cycle)]={'RAW256_channel_spread':float(np.ptp(result[0].transpose(1,2,0),axis=-1)[mask].mean()),'PNG256_channel_spread':float(np.ptp(observed.astype(np.int16),axis=-1)[mask].mean())}
  public_gray.append(row)
 OUT.mkdir()
 report={'format':'fusion-only-v3-saved-objective-alignment-diagnosis-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'protocol':pin(RETURN/'protocol.json'),'source_formula_worker':pin(worker),'source_delivered_audit':pin(AUDIT),'source_formula_contract':{'training':'(raw512+1)/2 WITHOUT clamp, maskedRGBL1 + kernel7innerboundaryL1, 1+1, four-microbatchmean.','delivery':'clamp((raw512+1)/2,0,1), bilinear512->256 (2x2average at this exact scale), active-indexcompose into source, rint*255 PNG.','training_boundary':'kernel7 at512, approximatelyhalf the physical width of kernel7 at256; no silent boundary rescaling applied.','target_pyramid':'Independent frozen target256 and styledHQ512 target; measure their downsample mismatch rather than assume equality.','no_NEW_training_loss_or_gate':True},'paired_saved_cases':len(records),'cohort_metrics':{f'{k[0]}:{k[1]}':v for k,v in summary.items()},'terminal_changes':final_changes,'all_snapshot_changes':all_changes,'terminal_case_classifications':classifications,'terminal_case_changes':case_changes,'all_case_measurements':records,'gray_challenge_summary':gray_summary,'unpaired_strict_gray_style':public_gray,'training_receipt_clipping':{'cycles':64,'clipped_cycles':sum(r['clipping_applied'] for r in training),'minimum_preclip_L2':min(r['preclip_L2'] for r in training),'maximum_preclip_L2':max(r['preclip_L2'] for r in training),'cycle_records':training,'loss_trend_limit':'All256FITpairs occur once. First/last training receipts concern different pairs and are not matched improvement evidence.','gradient_norm_scope':'Recorded preclip scalar only; no new gradients or causal explanation inferred.'},'frozen_target_assets':assetpins,'returned_payload_pins':evidencepins,'interpretation_limits':['These HQ512 diagnostics are recomputed on adaptive monitor/challenge saved outputs, not a new evaluation of the256FITtraining predictions.','The two resolutions, clipping and kernel7boundaryphysicalwidth differ; observing disagreement identifies objective alignment, not solecausal attribution.','Both R1 and V3 use the same fixed support/targets; no mask enlargement, postprocessing or threshold relaxation.','Publicphotos remain unpaired; no hidden HQtarget or identity metric invented.','No next training packet or model adoption follows this diagnostic.'],'local_execution':{'Torch_imported':False,'model_calls':0,'neural_calls':0,'gradients':0,'optimizer_objects_or_steps':0,'VM_calls':0,'return_edited':False},'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-start}
 need(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'Torch unexpectedly imported')
 path=OUT/'analysis.json';path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
 print(json.dumps({'analysis':pin(path),'terminal_macro_changes':{k:{'HQsum_gain_percent':v['macro']['HQ512_training_sum']['gain_percent'],'HQhole_gain_percent':v['macro']['HQ512_hole_L1_unclamped']['gain_percent'],'PNGhole_gain_percent':v['macro']['PNG_hole_L1']['gain_percent']} for k,v in final_changes.items()},'classifications':classifications,'gray':gray_summary,'fusion_clipped_cycles':sum(r['clipping_applied'] for r in training),'seconds':report['seconds']}),flush=True)
if __name__=='__main__':main()
