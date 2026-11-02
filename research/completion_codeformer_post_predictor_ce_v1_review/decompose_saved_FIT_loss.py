"""Plain-array decomposition of retained FIT logits; no model/optimizer/autograd."""
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
from PIL import Image
BASE=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION');HERE=BASE/'research/completion_codeformer_post_predictor_ce_v1_review'
P=BASE/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm'
R=BASE/'outputs/completion_codeformer_predictor_ce_v1_return/cctv_completion_codeformer_predictor_ce_v1_results';OUT=R/'outputs'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def pin(p):return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def need(ok,label):
    if not ok:raise ValueError(label)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
protocol=read(P/'protocol.json');need(sha(P/'protocol.json')=='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b','Protocol pin')
files={r['path']:r for r in protocol['files']};cases={r['id']:r for r in protocol['fit_cases']};rows=[];pins=[]
for cycle in range(1,33):
    reportpath=OUT/f'training_cycle_{cycle:04d}.json';report=read(reportpath);pins.append(pin(reportpath));need(report['cycle']==cycle and report['microbatch_ids']==protocol['schedule'][cycle-1],'Exact schedule')
    for micro,entry in enumerate(report['predictor_losses'],1):
        case=cases[entry['id']];path=OUT/entry['FIT_logits'];need(sha(path)==entry['FIT_logits_sha256'],'Actual FIT logit pin')
        with zipfile.ZipFile(path) as z:
            infos=z.infolist();need(set(z.namelist())=={n+'.npy' for n in ('logits','labels','case_index','cycle','microbatch')} and len(infos)==5,'Exact FIT archive keys');need(all(not i.is_dir() and not i.flag_bits&1 and i.file_size<1024**2 and i.compress_type in (0,8) for i in infos) and z.testzip() is None,'Bounded FIT archive/CRC')
        with np.load(path,allow_pickle=False,max_header_size=512) as a:
            logits=a['logits'];labels=a['labels'];need(logits.dtype==np.float32 and logits.shape==(1,256,512) and np.isfinite(logits).all(),'FIT logits schema');need(labels.dtype==np.int64 and labels.shape==(1,256) and labels.min()>=0 and labels.max()<512,'FIT labels schema');need(a['cycle'].shape==() and int(a['cycle'])==cycle and int(a['microbatch'])==micro,'FIT scalar binding');x=logits[0].astype(np.float64);y=labels[0].copy()
        maskpath=P/case['mask'];need(sha(maskpath)==files[case['mask']]['sha256'],'Mask pin')
        with Image.open(maskpath) as im:need(im.mode=='L' and im.size==(256,256),'Mask exact geometry');mask=np.array(im)>0
        need(int(mask.sum())==case['mask_pixels'],'Mask pixel count');fraction=mask.reshape(16,16,16,16).mean(axis=(1,3)).reshape(256)
        maximum=x.max(1);logsum=maximum+np.log(np.exp(x-maximum[:,None]).sum(1));ce=logsum-x[np.arange(256),y];prob=np.exp(-ce);correct=x.argmax(1)==y
        teacher_logits=x[np.arange(256),y];other=x.copy();other[np.arange(256),y]=-np.inf;margin=teacher_logits-other.max(1)
        regions={'all':np.ones(256,bool),'touched':fraction>0,'fully_hidden':fraction==1,'visible':fraction==0,'crossing':(fraction>0)&(fraction<1)}
        region_data={}
        for name,select in regions.items():
            region_data[name]={'tokens':int(select.sum()),'CE_sum':float(ce[select].sum()),'CE_mean':float(ce[select].mean()) if select.any() else None,'correct':int(correct[select].sum()),'teacher_probability_sum':float(prob[select].sum()),'mean_teacher_logit_margin':float(margin[select].mean()) if select.any() else None,'positive_teacher_margin_tokens':int((margin[select]>0).sum())}
        weighted_hole=float((ce*fraction).sum());weighted_visible=float((ce*(1-fraction)).sum())
        row={'id':case['id'],'reference_id':case['reference_id'],'cycle':cycle,'microbatch':micro,'condition':case['condition'],'style':case['style'],'mask_kind':case['mask_kind'],'mask_pixels':int(mask.sum()),'geometric_coverage_fraction':float(fraction.mean()),'uniform_CE':float(ce.mean()),'uniform_CE_worker':entry['CE_mean_all256'],'uniform_CE_absolute_roundoff':abs(float(ce.mean())-entry['CE_mean_all256']),'fractional_hole_CE_sum':weighted_hole,'fractional_visible_CE_sum':weighted_visible,'fractional_hole_CE_share':weighted_hole/float(ce.sum()),'fractional_visible_CE_share':weighted_visible/float(ce.sum()),'regions':region_data,'logits_pin':pin(path),'mask_pin':pin(maskpath)};need(row['uniform_CE_absolute_roundoff']<2e-6,'Independent full CE replay tolerance');rows.append(row)
need(len(rows)==128 and len({r['id'] for r in rows})==128,'128 distinct FIT observations, not repeated measurement')
def aggregate(items):
    result={'cases':len(items),'mean_case_geometric_hole_fraction':float(np.mean([r['geometric_coverage_fraction'] for r in items])),'mean_case_uniform_CE':float(np.mean([r['uniform_CE'] for r in items])),'pooled_fractional_hole_CE_share':sum(r['fractional_hole_CE_sum'] for r in items)/sum(r['fractional_hole_CE_sum']+r['fractional_visible_CE_sum'] for r in items),'mean_case_fractional_hole_CE_share':float(np.mean([r['fractional_hole_CE_share'] for r in items])),'regions':{}}
    for region in rows[0]['regions']:
        n=sum(r['regions'][region]['tokens'] for r in items);s=sum(r['regions'][region]['CE_sum'] for r in items);correct=sum(r['regions'][region]['correct'] for r in items);regionresult={'tokens':n,'CE_sum':s,'pooled_CE_mean':s/n if n else None,'pooled_code_agreement':correct/n if n else None,'pooled_teacher_probability':sum(r['regions'][region]['teacher_probability_sum'] for r in items)/n if n else None,'fraction_all_CE':s/sum(r['regions']['all']['CE_sum'] for r in items)};result['regions'][region]=regionresult
    return result
aggregates={'all':aggregate(rows)}
for key in ('style','mask_kind','condition'):
    aggregates[key]={value:aggregate([r for r in rows if r[key]==value]) for value in sorted({r[key] for r in rows})}
result={'format':'post-predictor-ce-v1-FIT-loss-geometric-decomposition-v1','complete':True,'protocol':pin(P/'protocol.json'),'worker':pin(P/'scripts/cctv_completion_codeformer_predictor_ce_v1_vm.py'),'actual_return_transport':pin(R.parent/'transport_audit.json'),'method':'Stable float64 CE from all128 savedFP32 FIT logits, original exact labels and masks.16x16 gridcell occupancy is exact area proxy only, not convolutional receptivefield or causalgradient attribution. Each observation differs, so cycleordered CE is not a controlled learningcurve. No before/after FIT comparison or recommendation proof.','no_new_training':True,'neural_calls':0,'gradient_queries':0,'model_construction':False,'optimizer_construction':False,'aggregates':aggregates,'cases':rows,'cycle_receipt_pins':pins,'max_CE_absolute_roundoff':max(r['uniform_CE_absolute_roundoff'] for r in rows)}
path=HERE/'FIT_loss_geometric_decomposition.json'
with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'output':pin(path),'all':aggregates['all'],'max_CE_roundoff':result['max_CE_absolute_roundoff']},indent=2))
