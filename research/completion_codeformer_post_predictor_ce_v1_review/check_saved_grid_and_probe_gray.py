"""Independent plain-array EVAL grid QA and retained four-arm gray diagnostic."""
import hashlib,json,os
from pathlib import Path
import numpy as np
from PIL import Image
B=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION');H=B/'research/completion_codeformer_post_predictor_ce_v1_review';R=B/'outputs/completion_codeformer_predictor_ce_v1_return';P=R/'root_saved_code_geometry_diagnosis.json'
def xp(p):return '\\\\?\\'+str(Path(p).resolve())
def sha(p):
    h=hashlib.sha256()
    with open(xp(p),'rb') as f:
        for v in iter(lambda:f.read(1024**2),b''):h.update(v)
    return h.hexdigest()
def pin(p):return {'path':str(p),'bytes':os.stat(xp(p)).st_size,'sha256':sha(p)}
def read(p):
    with open(xp(p),'r',encoding='utf-8-sig') as f:return json.load(f)
checks=0
def need(ok,label):
    global checks
    if not ok:raise ValueError(label)
    checks+=1
root=read(P);need(sha(P)=='647bfa63a9f7aaa62d18cc7e494a5bba5ccf5468d224e3293d5370a469a7fca8','Root receipt exact');rows=root['records'];base={};recomputed=[]
for row in rows:
    path=Path(row['predicted_codes_file']['path']);need(pin(path)==row['predicted_codes_file'],'Code array pin')
    with np.load(xp(path),allow_pickle=False,max_header_size=512) as a:code=a['predicted_codes'].reshape(256).copy();mask=a['mask256'][0,0].copy()
    labelpath=Path(row['fixed_label_file']['path']);need(pin(labelpath)==row['fixed_label_file'],'Teacher pin')
    with np.load(xp(labelpath),allow_pickle=False,max_header_size=512) as a:teacher=a['nearest_codes'].reshape(256).copy()
    need(np.isin(mask,[0,1]).all(),'Binary mask');occupancy=mask.astype(bool).reshape(16,16,16,16).sum((1,3)).reshape(256);regions={'all':np.ones(256,bool),'touched':occupancy>0,'majority':occupancy>=128,'fully_hidden':occupancy==256,'visible':occupancy==0,'crossing':(occupancy>0)&(occupancy<256)}
    key=(row['cohort'],row['id'])
    if row['cycle']==0:base[key]=code.copy()
    need(key in base,'Initial samecase before snapshots');out={}
    for region,s in regions.items():
        n=int(s.sum());correct=int((code[s]==teacher[s]).sum());changed=int((code[s]!=base[key][s]).sum());actual={'tokens':n,'correct':correct,'agreement':correct/n if n else None,'changed_from_R1_tokens':changed,'changed_from_R1_fraction':changed/n if n else None};need(actual==row['regions'][region],'Exact all EVAL geometric counts/arithmetic');out[region]=actual
    recomputed.append({**{k:row[k] for k in ('id','cohort','cycle','mask_kind')},'regions':out})
need(len(rows)==192 and len(base)==48,'All192actual48fixedcases')
def group(items):
    values={}
    for region in items[0]['regions']:
        n=sum(r['regions'][region]['tokens'] for r in items);correct=sum(r['regions'][region]['correct'] for r in items);change=sum(r['regions'][region]['changed_from_R1_tokens'] for r in items)
        values[region]={'cases':len(items),'tokens':n,'mean_case_agreement':float(np.mean([r['regions'][region]['agreement'] for r in items])),'pooled_token_agreement':correct/n,'mean_case_changed_fraction_from_R1':float(np.mean([r['regions'][region]['changed_from_R1_fraction'] for r in items])),'pooled_changed_tokens':change}
    return values
for cohort in ('monitor','challenge'):
    for cycle in (0,8,16,32):
        items=[r for r in recomputed if r['cohort']==cohort and r['cycle']==cycle]
        for name,records in [('macro',items)]+[(kind,[r for r in items if r['mask_kind']==kind]) for kind in sorted({r['mask_kind'] for r in items})]:
            expected=root['aggregates'][cohort][str(cycle)]['macro'] if name=='macro' else root['aggregates'][cohort][str(cycle)]['geometry'][name]
            actual=group(records)
            for region in actual:
                for k,v in actual[region].items():need(abs(v-expected[region][k])<=1e-15,'All macro/geometry pooled andmacro arithmetic')
probe=B/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results';gray=[]
for dirname in ('ACTUAL_A_R1_w1','CONTROL_B_predicted_codes_w0','ORACLE_C_unavailable_clean_information_w1','ORACLE_D_unavailable_clean_information_w0'):
    report=read(probe/'outputs'/dirname/'challenge/results.json')
    for row in report['records']:
        if row['style']!='GRAY':continue
        png=probe/'outputs'/row['PNG'];need(sha(png)==row['PNG_sha256'],'Retained probe PNG pin');path=png.with_suffix('.npz')
        with np.load(xp(path),allow_pickle=False,max_header_size=512) as a:
            value=a['result256'][0].transpose(1,2,0);support=a['mask256'][0,0].astype(bool);need(np.isfinite(value).all(),'Retained gray arrays finite')
            oracle=bool(a['oracle_unavailable_clean_information']);arm=chr(int(a['arm_code'])) if int(a['arm_code'])>3 else 'ABCD'[int(a['arm_code'])]
        with Image.open(xp(png)) as im:pixels=np.array(im)
        need(np.array_equal(np.rint(value.clip(0,1)*255).astype(np.uint8),pixels),'Retained gray float/PNG exact')
        gray.append({'arm':row['arm'],'id':row['id'],'mask_kind':row['mask_kind'],'oracle_unavailable_clean_information':row['oracle_unavailable_clean_information'],'not_deployable':row.get('not_deployable',False),'raw_support_mean_channel_spread':float(np.ptp(value,axis=-1)[support].mean()),'PNG_support_mean_channel_spread':float(np.ptp(pixels.astype(np.int16),axis=-1)[support].mean()),'PNG_hole_L1_previously_audited':row['PNG_hole_L1'],'NPZ_pin':pin(path),'PNG_pin':pin(png)})
need(len(gray)==32,'All8graycases times4retainedarms')
summary={}
for arm in 'ABCD':
    items=[r for r in gray if r['arm']==arm];need(len(items)==8,'Eightgrayeacharm');summary[arm]={'gray_cases':8,'macro_raw_channel_spread':float(np.mean([r['raw_support_mean_channel_spread'] for r in items])),'macro_PNG_channel_spread':float(np.mean([r['PNG_support_mean_channel_spread'] for r in items])),'macro_PNG_hole_L1':float(np.mean([r['PNG_hole_L1_previously_audited'] for r in items]))}
result={'format':'post-predictor-ce-v1-independent-grid-QA-retained-gray-v1','complete':True,'root_geometric_diagnosis_pin':pin(P),'all192_geometric_token_counts_and_aggregates_match':True,'check_count':checks,'retained_probe_four_arm_gray':summary,'retained_probe_gray_cases':gray,'limits':'All token regions are area proxies, not decoder receptivefields or causalpaths. OracleC/D use unavailablecleancontext. Channelspread measures grayscale color artifact, not actualhiddenidentity or perceptual utility. Savedimages only; noNN/autograd/optimizer/VM. Existing differentFIT observations do not provide controlled learningcurve.','neural_calls':0,'gradient_queries':0,'model_construction':False,'optimizer_construction':False}
path=H/'independent_grid_QA_and_retained_gray.json'
with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
print(json.dumps({'pin':pin(path),'checks':checks,'gray':summary},indent=2))
