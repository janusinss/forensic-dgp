"""One explicit transport-only recovery, with separate evidence and no retries."""
import argparse
import hashlib
import http.client
import importlib.util
import json
from pathlib import Path
import time
from urllib.parse import urlsplit

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'outputs/codeformer_u96_context_study_v1'
OUT=BASE/'recovery_r1'
PARENT=ROOT/'scripts/run_u96_context_study_v1.py'
PLAN=BASE/'comparison_plan_v1.json'
PLAN_SHA='ab3daf6800f3cbfa7edc7923bd448e5f0be94bf59207d836fca512e59198590d'

def pin(p):
    b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def write(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.write('\n')
def request(url,*,body=None,headers=None,folder=None,deadline):
    target=urlsplit(url)
    assert target.scheme=='http' and target.hostname=='127.0.0.1' and target.port!=61815
    remaining=deadline-time.monotonic();assert remaining>0
    c=http.client.HTTPConnection(target.hostname,target.port,timeout=min(150,remaining));sink=None
    try:
        c.request('POST' if body is not None else 'GET',target.path,body=body,
            headers={**(headers or {}),'Connection':'keep-alive'})
        response=c.getresponse();lengths=response.headers.get_all('Content-Length')
        assert lengths and len(lengths)==1
        length=int(lengths[0]);assert 0<length<=10*1024**2
        if folder:
            write(folder/'response-headers.json',dict(response.headers))
            sink=(folder/'response.body').open('xb')
        chunks=[];count=0
        while count<length:
            remaining=deadline-time.monotonic();assert remaining>0
            if c.sock:c.sock.settimeout(min(150,remaining))
            part=response.read1(min(65536,length-count))
            if not part:break
            chunks.append(part);count+=len(part)
            if sink:sink.write(part);sink.flush()
        assert count==length,'Incomplete response; no automatic retry'
        data=b''.join(chunks)
        assert response.status==200,'HTTP '+str(response.status)+': '+data[:1000].decode(errors='replace')
        return json.loads(data)
    finally:
        if sink:sink.close()
        c.close()

def prepare():
    assert pin(PLAN)['sha256']==PLAN_SHA
    plan=json.loads(PLAN.read_bytes());assert plan['worker']==pin(PARENT)
    failed=json.loads((BASE/'comparison.json').read_bytes())
    assert failed['complete'] is False and failed['new_forwards']==0
    assert failed['submitted_HTTP_calls']==1 and failed['failure']['type']=='ConnectionResetError'
    assert failed['active_call']['key']=='original:original'
    OUT.mkdir(exist_ok=False)
    record={'complete':True,'status':'TRANSPORT_ONLY_RECOVERY_PREPARED_NOT_RUN',
        'scientific_plan':pin(PLAN),'sealed_original_worker':pin(PARENT),'driver':pin(Path(__file__)),
        'parent_stopped_result':pin(BASE/'comparison.json'),'failure_retained':True,
        'changed_transport':'Connection keep-alive through exact Content-Length; close only after full body',
        'unchanged_inputs_masks_models_settings_and_quality_requirements':True,
        'new_result_directory':OUT.relative_to(ROOT).as_posix(),
        'prior_actual_completed_original_forwards':1,'prior_verified_outputs':0,
        'fresh_run_forward_cap':96,'combined_all_attempt_forward_cap':97,
        'fresh_server_zero_counters_required':True,'automatic_retry_or_resume':False,
        'neural_training_optimizer_VM_calls_during_preparation':0}
    write(OUT/'recovery_plan_v1.json',record)
    print(json.dumps({'prepared':True,'recovery_plan':pin(OUT/'recovery_plan_v1.json')}))

def run(receipt):
    recovery=json.loads((OUT/'recovery_plan_v1.json').read_bytes())
    assert recovery['driver']==pin(Path(__file__)) and recovery['scientific_plan']==pin(PLAN)
    spec=importlib.util.spec_from_file_location('sealed_context_worker',PARENT)
    parent=importlib.util.module_from_spec(spec);spec.loader.exec_module(parent)
    assert parent.pin(PARENT)==recovery['sealed_original_worker']
    assert recovery['parent_stopped_result']==pin(BASE/'comparison.json')
    parent.OUT=OUT;parent.PLAN=PLAN;parent.request=request
    try:parent.run(receipt,PLAN_SHA)
    finally:
        if (OUT/'comparison.json').exists():
            value=json.loads((OUT/'comparison.json').read_bytes())
            value['transport_recovery']={'driver':pin(Path(__file__)),
                'recovery_plan':pin(OUT/'recovery_plan_v1.json'),'one_prior_forward_not_verified_as_output':True,
                'combined_completed_forward_count':1+value['new_forwards']}
            parent.dump(OUT/'comparison.json',value)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--run',action='store_true')
    p.add_argument('--server-receipt');a=p.parse_args()
    prepare() if a.prepare else run(a.server_receipt)
