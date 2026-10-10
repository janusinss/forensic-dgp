"""Inspect the first failed group check from saved arrays only; no models."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_group_diagnostic_v1'
RETURN=ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_return'
MIXED=ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text())


def rgb(path):
    with Image.open(path) as im:return np.array(im.convert('RGB'))


def main():
    assert not OUT.exists();started=time.monotonic()
    path=ROOT/'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py'
    spec=importlib.util.spec_from_file_location('metric_readback',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    p=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_vm/protocol.json')
    cohort=p['cohorts'][0];folder=RETURN/('outputs/state0_'+cohort['name']+'/before');r=read(folder/'receipt.json')
    actual_rows=[];worst_case=0.
    for c,row in zip(cohort['cases'],r['rows']):
        cid=c['id'];camera,target=rgb(MIXED/c['input']),rgb(MIXED/c['target'])
        with Image.open(MIXED/c['observed']) as im:mask=np.array(im)>0
        feature=np.zeros((256,256),bool)
        for point in c['landmarks5_canvas_xy']:
            x,y=np.floor(point).astype(int);feature[max(0,y-12):min(256,y+12),max(0,x-12):min(256,x+12)]=True
        feature &= m.erode(mask,6)
        raw=np.load(folder/(cid+'.npy'),allow_pickle=False)
        vector,rv,truth=[np.load(folder/(cid+n),allow_pickle=False) for n in ['_embedding.npy','_raw_embedding.npy','_target_embedding.npy']]
        value=m.case_metrics(raw,rgb(folder/(cid+'.png')),target,camera,mask,feature,raw,vector,rv,truth)
        for key,v in value.items():
            assert np.allclose(v,row['metrics'][key],rtol=2e-10,atol=1e-11)
            worst_case=max(worst_case,float(np.abs(np.array(v)-np.array(row['metrics'][key])).max()))
        actual_rows.append({'id':cid,'source':c['source'],'profile':c['profile'],'metrics':value})
        assert time.monotonic()-started<120
    computed=m.groups(actual_rows);assert set(computed)==set(r['groups'])
    differences=[]
    for key,group in computed.items():
        assert group['cases']==r['groups'][key]['cases']
        for metric in m.METRICS:
            saved=r['groups'][key][metric];actual=group[metric]
            if actual!=saved:differences.append({'group':key,'metric':metric,'saved':saved,'recomputed':actual,'absolute_error':abs(saved-actual)})
    stored_rows=[{'id':c['id'],'source':c['source'],'profile':c['profile'],'metrics':row['metrics']} for c,row in zip(cohort['cases'],r['rows'])]
    record={'complete':True,'checker_sha256':sha(Path(__file__)),'metric_source_sha256':sha(path),
        'receipt_sha256':sha(folder/'receipt.json'),'first_state':0,'cohort':cohort['name'],'variant':'before',
        'cases':50,'groups':17,'differences':differences,'maximum_group_absolute_error':max((x['absolute_error'] for x in differences),default=0.),
        'maximum_case_metric_absolute_error':worst_case,'groups_from_stored_rows_exactly_match_saved':m.groups(stored_rows)==r['groups'],
        'per_case_original_tolerances_pass':True,'seconds':time.monotonic()-started,'cap_seconds':120,
        'model_forwards':0,'optimizer_updates':0,'quality_gates_changed':False,'new_audit_pass_not_implied':True}
    OUT.mkdir()
    with (OUT/'diagnostic.json').open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
