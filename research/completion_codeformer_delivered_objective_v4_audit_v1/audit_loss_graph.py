"""Independent exact V4 delivery/loss CPU-tensor value and autograd audit.
No neural model, learned weight load, optimizer object/step or VM operation.
"""
import argparse,ast,hashlib,importlib.util,json,math,sys,time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pin(path):return {'path':str(path.resolve()),'bytes':path.stat().st_size,'sha256':sha(path)}
def main():
 a=argparse.ArgumentParser();a.add_argument('--helper',type=Path,required=True);a.add_argument('--expected-sha',required=True);a.add_argument('--output',type=Path,required=True);args=a.parse_args();started=time.monotonic();helper=args.helper.resolve(strict=True);need(sha(helper)==args.expected_sha,'Helper source changed')
 tree=ast.parse(helper.read_text(),feature_version=(3,10));need(not any(isinstance(n,(ast.Import,ast.ImportFrom)) for n in ast.walk(tree)),'Helper imports libraries');need(not any(isinstance(n,ast.Attribute) and n.attr in ('backward','step','round','rint','numpy') for n in ast.walk(tree)),'Hidden non-differentiable or training operation')
 spec=importlib.util.spec_from_file_location('independent_V4_delivered_loss',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 need(not any(n=='torch' or n.startswith('torch.') for n in sys.modules),'Helper imported Torch at module load')
 import torch
 import torch.nn.functional as F
 torch.set_num_threads(1);checks=[];backwards=0;finite_difference_calls=0
 def check(ok,label):need(ok,label);checks.append(label)
 def interpolate_spy(value,*,size,mode,align_corners):
  check(mode=='bilinear' and align_corners is False,'Exact bilinear align_corners=False injected operator')
  return F.interpolate(value,size=size,mode=mode,align_corners=align_corners)
 source=torch.linspace(.02,.98,48,dtype=torch.float64).reshape(1,3,4,4);target=torch.full_like(source,.37);mask=torch.zeros((1,1,4,4),dtype=torch.float64);mask[0,0,0,1]=1;mask[0,0,1,1:3]=1;mask[0,0,2,3]=1
 raw=torch.linspace(-1.8,1.6,192,dtype=torch.float64).reshape(1,3,8,8).clone().requires_grad_(True)
 app,hole,scaled=m.delivered_hole_loss(raw,source,mask,target,interpolate=interpolate_spy,microbatches=4)
 pred=((raw+1)/2).clamp(0,1);manual_generated=(pred[:,:,0::2,0::2]+pred[:,:,0::2,1::2]+pred[:,:,1::2,0::2]+pred[:,:,1::2,1::2])/4;manual=source*(1-mask)+manual_generated*mask;manual_hole=(abs(manual-target)*mask).sum()/(3*mask.sum())
 check(torch.allclose(app,manual,rtol=0,atol=1e-15),'Independent2x2-average delivery values match')
 check(torch.equal(app*(1-mask),source*(1-mask)),'Outside source pixels exactly preserved')
 check(torch.allclose(hole,manual_hole,rtol=0,atol=1e-15) and torch.allclose(scaled,hole/4,rtol=0,atol=1e-15),'RGB-hole normalization and once-only /4')
 check(scaled.requires_grad and scaled.grad_fn is not None,'Actual returned loss retains autograd graph')
 grad=torch.autograd.grad(scaled,raw)[0];backwards+=1
 low_sign=torch.sign(manual_generated.detach()-target)*mask;expanded=low_sign.repeat_interleave(2,dim=2).repeat_interleave(2,dim=3);analytic=expanded*(.5*.25/(3*float(mask.sum())*4));analytic=analytic*((raw.detach()>-1)&(raw.detach()<1))
 check(torch.allclose(grad,analytic,rtol=0,atol=2e-17),'Exact analytic gradient includes clamp, /2, downsample1/4, RGBpixels and microbatch/4')
 check(torch.equal(grad*(1-mask.repeat_interleave(2,2).repeat_interleave(2,3)),torch.zeros_like(grad)),'No outside-mask output gradients')
 saturated=(raw.detach()<-1)|(raw.detach()>1);check(torch.equal(grad[saturated],torch.zeros_like(grad[saturated])),'Strictly saturated raw channels have zero gradient; no STE')
 check(float(grad.abs().sum())>0 and torch.isfinite(grad).all().item(),'Finite nonzero interior gradient exists')
 # Finite differences on all types: active/interior, inactive and saturated.
 selected=[tuple(int(v) for v in idx) for idx in torch.nonzero((analytic!=0),as_tuple=False)[:3]];selected.extend([(0,0,0,0),(0,2,7,7)])
 for index in selected:
  eps=1e-6;a=raw.detach().clone();b=raw.detach().clone();a[index]+=eps;b[index]-=eps
  la=m.delivered_hole_loss(a,source,mask,target,interpolate=F.interpolate)[2];lb=m.delivered_hole_loss(b,source,mask,target,interpolate=F.interpolate)[2];finite_difference_calls+=2
  numeric=float((la-lb)/(2*eps));check(math.isclose(numeric,float(analytic[index]),rel_tol=1e-6,abs_tol=1e-10),'Finite-difference gradient '+str(index))
 # Changing unobserved targets outside fixed support cannot change the loss.
 target_alt=target.clone();target_alt[mask.expand_as(target)==0]=.91
 alt=m.delivered_hole_loss(raw.detach(),source,mask,target_alt,interpolate=F.interpolate)[2];check(torch.equal(alt,scaled.detach()),'Target outside support excluded')
 # Deliberately mark interpolation inputs to demonstrate clamp-before-resize.
 observed=[]
 def inspect_interp(v,**kw):observed.append(v.detach().clone());return F.interpolate(v,**kw)
 m.delivered_float(raw.detach(),source,mask,interpolate=inspect_interp)
 check(len(observed)==1 and float(observed[0].min())>=0 and float(observed[0].max())<=1,'Interpolation receives already-clamped RGB')
 # Empty bypass must neither inspect raw nor call interpolation.
 def refuse_interp(*a,**kw):raise ValueError('Empty support attempted interpolation')
 empty=m.delivered_float(None,source,torch.zeros_like(mask),interpolate=refuse_interp)
 check(torch.equal(empty,source) and empty.data_ptr()!=source.data_ptr(),'Empty bypass returns exact source clone without raw/interpolation')
 # Active-index semantics for mixed source batch and compact active-only raw.
 mixed_source=torch.cat((source,source*.7),dim=0);mixed_mask=torch.cat((torch.zeros_like(mask),mask),dim=0);mixed=m.delivered_float(raw.detach(),mixed_source,mixed_mask,interpolate=F.interpolate)
 expected_second=m.delivered_float(raw.detach(),mixed_source[1:],mask,interpolate=F.interpolate)
 check(torch.equal(mixed[:1],mixed_source[:1]) and torch.equal(mixed[1:],expected_second),'Mixed batch preserves inactive source and uses compact active raw')
 # Full actual512->256 geometry with plain tensors only, including FP32 delivery parity.
 rgb=torch.linspace(.01,.99,3*256*256,dtype=torch.float32).reshape(1,3,256,256);gt=rgb.flip(-1);mk=torch.zeros((1,1,256,256));mk[:,:,70:90,110:140]=1;actual_raw=torch.linspace(-1.2,1.2,3*512*512).reshape(1,3,512,512).clone().requires_grad_(True)
 out,lh,ls=m.delivered_hole_loss(actual_raw,rgb,mk,gt,interpolate=F.interpolate)
 old=rgb.clone();old[(mk.sum((1,2,3))>0)]=rgb*(1-mk)+F.interpolate(((actual_raw+1)/2).clamp(0,1),size=(256,256),mode='bilinear',align_corners=False)*mk
 check(torch.equal(out,old),'Actual512/256FP32 output bitexact to sealed V3 delivery operator')
 g=torch.autograd.grad(ls,actual_raw)[0];backwards+=1;check(torch.isfinite(g).all().item() and bool((g!=0).any()),'Actual geometry finite nonzero loss gradient')
 # Diagnostics are detached, honest counters and do not modify raw/graph.
 raw_before=raw.detach().clone();ms512=mask.repeat_interleave(2,2).repeat_interleave(2,3);diagnostic=m.clipping_diagnostic(raw,ms512);support=ms512.expand_as(raw).bool();norm=(raw.detach()+1)/2
 check(diagnostic['raw512_hole_channel_count']==int(support.sum()) and math.isclose(diagnostic['below0_channel_fraction_hole512'],float((norm[support]<0).double().mean()),abs_tol=1e-15) and math.isclose(diagnostic['above1_channel_fraction_hole512'],float((norm[support]>1).double().mean()),abs_tol=1e-15),'Detached clipping fractions correct on support only')
 check(torch.equal(raw.detach(),raw_before) and all(not isinstance(v,torch.Tensor) for v in diagnostic.values()) and diagnostic['surrogate_gradient_used'] is False,'Clipping metadata neither changes graph nor tensor values')
 for kwargs,label in [({'microbatches':0},'Zero divisor'),({'microbatches':2},'Wrong frozen divisor'),({'microbatches':True},'Boolean divisor')]:
  try:m.delivered_hole_loss(raw,source,mask,target,interpolate=F.interpolate,**kwargs)
  except ValueError:check(True,label+' refused')
  else:raise ValueError(label+' was accepted')
 try:m.delivered_hole_loss(raw,source,torch.zeros_like(mask),target,interpolate=F.interpolate)
 except ValueError:check(True,'Empty training support refused before optimizer work')
 else:raise ValueError('Empty training support accepted')
 check(sha(helper)==args.expected_sha,'Helper final hash unchanged')
 report={'format':'independent-delivered-objective-v4-plain-CPU-loss-graph-audit-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'helper':pin(helper),'torch':str(torch.__version__),'checks':checks,'checks_passed':len(checks),'synthetic_shapes':['1x3x8x8->1x3x4x4 float64','mixed2source/1active float64','1x3x512x512->1x3x256x256 float32'],'analytic_gradient_max_absolute_error':float((grad-analytic).abs().max()),'finite_difference_scalar_forward_calls':finite_difference_calls,'plain_tensor_autograd_calls':backwards,'loss_graph_training_semantics':'Clamp-before-bilinear; source-composite exact outside; clean256 supplied directly; mean RGB inside fixed support; divide4exactlyonce; no512boundary/PNG/STE/detach on loss path.','clamp_diagnostic':diagnostic,'scope_limit':'Synthetic exact operator tests, not a CodeFormer model forward, learned-model gradient proof, actual VM preflight or usefulness evidence.','local_actions':{'Torch_imported':True,'neural_model_construction':0,'checkpoint_loads':0,'neural_forwards':0,'optimizer_objects_or_steps':0,'VM_calls':0,'actual_training':False},'model_qualified':False,'seconds':time.monotonic()-started}
 need(not args.output.exists(),'Preserve prior graph audit');args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'complete':True,'report':pin(args.output),'checks':len(checks),'plain_tensor_autograd_calls':backwards,'neural_calls':0,'seconds':report['seconds']},indent=2))
if __name__=='__main__':main()
