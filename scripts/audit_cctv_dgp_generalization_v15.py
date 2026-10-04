"""Independent serialized-output arithmetic audit; no neural imports/forwards."""
import argparse
from pathlib import Path
import sys
import time


def audit(root, parent, capacity, expected_sha, out, receipt):
    sys.path.insert(0,str(root))
    import cctv_dgp_generalization_v15 as v
    import numpy as np
    from PIL import Image
    start=time.monotonic();p=v.verify(root,parent,expected_sha)
    r=v.read(out/'results.json');n=v.read(out/'neural_execution_receipt.json')
    v.require(r['complete'] and r['protocol_sha256']==expected_sha and r['validation_used'] and
              all(r[k] is False for k in ['training','native_used','native_reserved_used','production_promoted','checkpoint_selected']) and
              r['optimizer_updates']==r['backward_calls']==0,'Wrong returned scope')
    for name,pin in r['artifacts_sha256'].items():v.require(v.sha(v.safe(out,name))==pin,'Returned artifact differs:'+name)
    expected={'dgp':570,'prior_encoder':570,'prior_transformer_head':570,'direct_conditioner':570,
              'prior_generator':1090,'unused_v11':0,'recognizer':2184}
    v.require(n['complete'] and n['protocol_sha256']==expected_sha and n['counts']==n['expected_counts']==expected and
              n['states_before']==n['states_after'] and n['states_after']['conditioner']==p['trained_head_state_hash'] and
              n['optimizer_updates']==n['backward_calls']==0 and n['peak_vram_bytes']<=v.DESIGN['peak_vram_cap_bytes'],
              'Recorded frozen execution differs')
    v.require(r['seconds']<=1200,'Finite runner cap failed')
    refs={x['id']:x for x in p['references']};cases={x['id']:x for x in p['cases']}
    v.require(len(r['rows'])==520 and {x['id'] for x in r['rows']}==set(cases),'Returned cases missing/repeated')
    rebuilt={a:[] for a in v.ARMS+['input']};images=raws=cosines=0
    def vector(name):
        x=np.load(v.safe(out,name),allow_pickle=False)
        v.require(x.dtype==np.float32 and x.shape==(512,) and np.isfinite(x).all() and
                  abs(float(np.linalg.norm(x))-1)<1e-5,'Invalid embedding')
        return x
    def near(actual,expected):
        v.require(set(actual)==set(expected),'Metric keys differ')
        for k,value in expected.items():
            if isinstance(value,dict):near(actual[k],value)
            elif value is None or isinstance(value,(bool,int,str,list)):v.require(actual[k]==value,'Value differs:'+k)
            else:v.require(np.isfinite(actual[k]) and abs(actual[k]-value)<=1e-6,'Arithmetic differs:'+k)
    for row in r['rows']:
        c=cases[row['id']];ref=refs[c['reference_id']]
        v.require(all(row[k]==c[k] for k in ['reference_id','source','profile']) and set(row['arms'])==set(v.ARMS),'Row metadata differs')
        target=v.rgb(root/ref['target']);camera=v.rgb(root/c['input'])
        support=np.asarray(Image.open(root/ref['observed']))>0
        truth=vector('target_embeddings/'+ref['id']+'.npy')
        for arm,item in [('input',row['input_metrics'])]+list(row['arms'].items()):
            image=camera if arm=='input' else v.rgb(out/item['prediction'])
            v.require(np.array_equal(image[~support],camera[~support]),'Unsupported pixels changed')
            vec=vector(item['embedding']);cosine=float(np.clip(vec@truth,-1,1));cosines+=1
            measured={**v.metrics(image,target,support),'ArcFace_observed_fixed':cosine}
            near({k:item[k] for k in measured},measured)
            rebuilt[arm].append({**{k:c[k] for k in ['id','source','profile']},**measured})
            if arm!='input':images+=1
            preview=ref['id'] in p['preview_reference_ids']
            v.require(('raw' in item)==(arm!='input' and preview),'Raw preview coverage differs')
            if 'raw' in item:
                raw=np.load(out/item['raw'],allow_pickle=False)
                v.require(np.array_equal(v.png(raw,camera,support),image),'Raw/PNG conversion differs');raws+=1
    parity=v.read(out/'fresh_image_parity.json');lookup={x['id']:x for x in p['parity_cases']}
    pref={x['id']:x for x in p['parity_references']}
    v.require(parity['complete'] and len(parity['cases'])==50 and {x['id'] for x in parity['cases']}==set(lookup),'Missing parity cases')
    v.require(v.sha(capacity/'results.json')==v.CAPACITY_PIN,'Capacity results differ')
    for item in parity['cases']:
        c=lookup[item['id']];ref=pref[c['reference_id']]
        camera=v.rgb(parent/c['input']);support=np.asarray(Image.open(parent/ref['observed']))>0
        raw=np.load(out/item['raw'],allow_pickle=False);image=v.rgb(out/item['prediction'])
        old='update1000/none/raw/'+c['id']+'.npy';oldpng='update1000/none/images/'+c['id']+'.png'
        for name in [old,oldpng]:v.require(v.sha(capacity/name)==p['capacity_artifacts_sha256'][name],'Old capacity evidence differs')
        delta=float(np.max(np.abs(raw-np.load(capacity/old,allow_pickle=False))))
        v.require(delta<=2e-6 and abs(delta-item['maximum_float_difference'])<=1e-12 and item['png_equal'] and
                  np.array_equal(v.png(raw,camera,support),image) and np.array_equal(image,v.rgb(capacity/oldpng)), 'Fresh-image parity differs')
    summaries={a:v.aggregate(items) for a,items in rebuilt.items()};near(r['summaries'],summaries)
    v.require(r['guard_report']==v.guard_report(summaries),'Diagnostic guard differs')
    cells=0;rowmap={x['id']:x for x in r['rows']}
    for profile,name in zip(v.PROFILES,r['grids']):
        sheet=np.asarray(Image.open(out/name).convert('RGB'));v.require(sheet.shape==(2904,1300,3),'Grid geometry differs')
        for i,rid in enumerate(p['preview_reference_ids']):
            c=next(x for x in p['cases'] if x['reference_id']==rid and x['profile']==profile)
            imgs=[v.rgb(root/c['input'])]+[v.rgb(out/rowmap[c['id']]['arms'][a]['prediction']) for a in v.ARMS]+[v.rgb(root/refs[rid]['target'])]
            for j,img in enumerate(imgs):
                y=24+i*288+28;x=j*260+2
                v.require(np.array_equal(sheet[y:y+256,x:x+256],img),'Grid cell differs');cells+=1
    v.require(len(r['grids'])==5 and images==1560 and raws==150 and cosines==2080 and cells==250,'Coverage count differs')
    value={'complete':True,'protocol_sha256':expected_sha,'results_sha256':v.sha(out/'results.json'),
        'auditor_sha256':v.sha(Path(__file__)),'pngs_checked':images,'raw_previews_checked':raws,
        'fresh_capacity_parity_checked':50,'embedding_cosines_rebuilt':cosines,'grid_cells_checked':cells,
        'summaries':summaries,'guard_report':v.guard_report(summaries),'seconds':time.monotonic()-start,
        'neural_forwards':0,'backward_calls':0,'optimizer_updates':0,'production_promoted':False,
        'native_reserved_used':False,'limitation':'Checks serialized pixels/embeddings, finite frozen-state/count receipts and fresh/cached parity; no CUDA/model/recognizer replay or independent final visual review.'}
    v.write(receipt,value);print({k:value[k] for k in ['complete','pngs_checked','raw_previews_checked','fresh_capacity_parity_checked','seconds']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','parent','capacity','results','receipt']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--expected-sha',required=True);a=p.parse_args()
    audit(a.root.resolve(),a.parent.resolve(),a.capacity.resolve(),a.expected_sha,a.results.resolve(),a.receipt.resolve())
