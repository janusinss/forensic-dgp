"""Admit only47 durable saved DEV outputs via independent NumPy pixel readback.

No model/forward/state decode. Original partial stop and pending48th reservation
remain intact. This is carried-evidence integrity, never completeDEV acceptance.
"""
import argparse
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys
import time

import numpy as np
import audit_u96_smoothanchors_lr_fixed_DEV_saved_v1 as A

HELPER_SHA = '48c40199939fc14a4520133236d5a80e1aad8bf558e0d11d285ac13fca084465'
AUDIT = A.OUT
OLD_PARTIAL_SHA = '97d7c2c8d51b3d5b9f3cd7b510dddad7c30951af6aa8151365639b20fbfdba4c'


def exact_prefix(plan, paths):
    expected = [(case['id'],arm) for case in plan['cases'] for arm in A.ARMS]
    actual = {(Path(path).parent.parent.name,Path(path).parent.name) for path in paths}
    A.need(len(paths)==47 and len(actual)==47 and actual==set(expected[:47]),
           'Exactly first47 planned arms,23pairs plus case24 incumbent required')
    return expected[:47]


def audit(review_path):
    start=time.monotonic();review=A.read(review_path)
    A.need(review['complete']and review['source_review_pass']and A.pin(Path(__file__))in review['sources']
           and A.pin(Path(A.__file__))in review['sources'],'Fresh carried47/helper source review required')
    for row in review['sources']:A.bound(row)
    A.need(A.pin(Path(A.__file__))['sha256']==HELPER_SHA,'Held independent numerical helper unchanged')
    A.need(A.pin(A.PLAN)['sha256']==A.PLAN_SHA,'Exact original fixed68 plan');plan=A.read(A.PLAN)
    partial=AUDIT/'saved_pixel_audit_v1.json';A.need(A.pin(partial)['sha256']==OLD_PARTIAL_SHA,'Original partial audit remains exact')
    inventory_path=AUDIT/'saved_return_inventory_v1.json';inventory=A.read(inventory_path)
    destination=AUDIT/'carried47_pixel_admission_v1.json';A.need(not destination.exists(),'Immutable new carry admission only')
    checks=[]
    def ck(name,ok):
        checks.append({'name':name,'pass':bool(ok)});A.need(ok,name)
    current={}
    for path in sorted(A.RUN.rglob('*')):
        if path.is_file():current[path.relative_to(A.ROOT).as_posix()]=A.pin(path)
    ck('all454 original files exact before readback',current==inventory['files']and len(current)==454)
    status=A.read(A.RUN/'status.json');failure=A.read(A.RUN/'failure.json');pending=A.read(A.RUN/'status.pending')
    supervisor_path=A.PREP/'fixed_DEV_supervisor_v1.json';supervisor=A.read(supervisor_path)
    ck('original infrastructure stop unknown retained',supervisor['complete']and supervisor['worker_exit']==1 and not supervisor['timeout']and supervisor['forward_outcome_unknown']and failure['error']=='PermissionError'and failure['phase']=='forward_reserved'and failure['counters']==pending['counters']and pending['counters']['full_forward_attempts']==48 and pending['counters']['full_forward_returns']==pending['counters']['successful_saved_outputs']==47 and status['counters']['full_forward_attempts']==status['counters']['full_forward_returns']==47)
    paths=list(A.RUN.glob('cases/*/*/record.json'));prefix=exact_prefix(plan,paths);byid={row['id']:row for row in plan['cases']}
    ck('23complete cases nofabricatedcase24receipt',len(list(A.RUN.glob('cases/*/case.json')))==23 and not(A.RUN/'cases'/plan['cases'][23]['id']/'case.json').exists()and not(A.RUN/'comparison.json').exists())
    for row in [*plan['sources'],*plan['prerequisites'].values(),plan['protocol'],plan['head_file'],plan['cohort'],plan['data_reuse'],*plan['model'].values()]:A.bound(row)
    carried=[];common_cache={};metrics_bycase={};expected_files=set(current)
    for number,(cid,arm)in enumerate(prefix,1):
        A.need(time.monotonic()-start<180,'Finite carried47 audit')
        row=byid[cid];folder=A.RUN/'cases'/cid;armfolder=folder/arm;recordpath=armfolder/'record.json';record=A.read(recordpath)
        ck('exact durable arm record '+cid+arm,record['complete']and record['case_id']==cid)
        counters=record['counters'];ck('original ordered durable counter position '+cid+arm,counters['full_forward_attempts']==counters['full_forward_returns']==number and counters['successful_saved_outputs']==number-1 and counters['completed_cases']==(number-1)//2 and all(counters[k]==0 for k in['gradient_queries','backward_calls','optimizer_constructions','optimizer_updates']))
        if cid not in common_cache:
            source=A.image(A.bound(row['upload_source']),size=None);canvas,observed,geometry=A.prepare(source)
            for key in['input','support','target','source1024','pair_record']:
                if row.get(key):A.bound(row[key])
            ck('originalprepared exact geometry '+cid,geometry==row['geometry']and A.digest(canvas.tobytes())==row['input_pixels']['RGB8_sha256']and np.array_equal(source,A.image(folder/'original.png',size=None))and np.array_equal(canvas,A.image(A.bound(row['input'])))and np.array_equal(observed,A.image(A.bound(row['support']),'L')>0)and np.array_equal(canvas,A.image(folder/'prepared.png'))and np.array_equal(observed,A.image(folder/'observed_support.png','L')>0))
            A.need(row['target']is not None and observed.all(),'Current carried47 allpaired full observed; no native targets invented')
            target=A.image(A.bound(row['target']));ck('pairedtarget exact '+cid,np.array_equal(target,A.image(folder/'target.png'))and A.pin(folder/'target.png')['sha256']==row['target']['sha256'])
            common={key:A.pin(folder/name)for key,name in[('original','original.png'),('prepared','prepared.png'),('support','observed_support.png'),('target','target.png')]}
            common_cache[cid]=(canvas,observed,geometry,target,common)
        canvas,observed,geometry,target,common=common_cache[cid]
        casepath=folder/'case.json'
        if casepath.exists():
            case=A.read(casepath)
            ck('completecase receipts bind exact output/input '+cid+arm,case['complete']and case['id']==cid and case['geometry']==geometry and case['inputs']==common and case['paired_target']==row['target']and all(case['outputs'][arm][key]==record[key]for key in case['outputs'][arm]))
        shapes={'native512':(1,3,512,512),'logits':(1,256,1024),'LQ':(1,256,16,16),'unclipped':(256,256,3),'raw':(256,256,3)}
        arrays={};schemas={}
        for key,shape in shapes.items():
            path=A.bound(record[key]);name={'native512':'native512.npy','logits':'logits.npy','LQ':'LQ.npy','unclipped':'unclipped256.npy','raw':'raw.npy'}[key]
            A.need(path==armfolder/name,'Exact keyed savedarray namespace');arrays[key]=A.array(path,shape)
            schemas[key]={'file':record[key],'shape':list(shape),'dtype':'float32','minimum':float(arrays[key].min()),'maximum':float(arrays[key].max()),'tensor_bytes_sha256':A.digest(arrays[key].tobytes())}
        un,raw,png=A.delivery(arrays['native512'],canvas,observed);pngpath=A.bound(record['png']);A.need(pngpath==armfolder/'output.png','Exact PNG namespace')
        ck('native512->unclipped/raw/floorPNG exact '+cid+arm,np.array_equal(un,arrays['unclipped'])and np.array_equal(raw,arrays['raw'])and np.array_equal(png,A.image(pngpath)))
        ck('outside support exact '+cid+arm,np.array_equal(raw[~observed],canvas.astype(np.float32)[~observed]/np.float32(255))and np.array_equal(png[~observed],canvas[~observed]))
        candidate=arm==A.ARMS[1]
        provenance={'arm':arm,'R2_file_sha256':plan['model']['r2']['sha256'],'U96_file_sha256':plan['model']['u96']['sha256'],'head_file_sha256':plan['head_file']['sha256']if candidate else None,'head_canonical_sha256':plan['candidate_head_canonical_sha256']if candidate else None,'w':1.,'detach16':True,'AdaIN':True,'CPU_FP32':True,'display_processing':'none','source_candidate_training_updates':64 if candidate else None,'model_qualified':False,'same_observed_geometry':True,'outside_support_exact_observed_pixels':True,'forward_sites':['32','64','128','256']}
        ck('exact modelandpolicy provenance '+cid+arm,record['provenance']==provenance)
        measured={view:A.paired(value,target.astype(np.float64)/255,observed)for view,value in[('raw',raw),('PNG',png.astype(np.float64)/255)]}
        metrics_bycase.setdefault(cid,{})[arm]=measured
        carried.append({'case_id':cid,'arm':arm,'planned_case_index':plan['case_order'].index(cid),'record':A.pin(recordpath),'complete_case_receipt':A.pin(casepath)if casepath.exists()else None,'inputs':common,'outputs':{key:record[key]for key in['native512','logits','LQ','unclipped','raw','png']},'provenance':record['provenance'],'geometry':geometry,'profile':row['profile'],'reference_id':row['reference_id'],'paired_metrics':measured,'saved_array_schema':schemas,'all_saved_delivery_pixels_exact':True})
    for row in current.values():ck('originalfile unchanged afterpixelreadback '+row['path'],A.pin(A.ROOT/row['path'])==row)
    ck('carried47 order and23matchedpairs exact',len(carried)==47 and [(x['case_id'],x['arm'])for x in carried]==prefix and sum(len(x)==2 for x in metrics_bycase.values())==23)
    need_no_torch='torch'not in sys.modules;ck('no Torch/model import',need_no_torch)
    report={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'carried47_pixel_admission_pass':True,'checks':checks,'failures':[],
        'carried47_records_count':47,'completed_pair_cases':23,'partial_case24_incumbent_only':{'case_id':plan['cases'][23]['id'],'arm':A.ARMS[0]},'carried_outputs':carried,
        'original_plan':A.pin(A.PLAN),'source_review':A.pin(review_path),'script':A.pin(Path(__file__)),'numerical_helper':A.pin(Path(A.__file__)),
        'old_partial_audit':A.pin(partial),'old_inventory':A.pin(inventory_path),'original_file_count':454,'original_file_bytes':inventory['bytes'],
        'old_stop_evidence':[A.pin(path)for path in[A.RUN/'status.json',A.RUN/'status.pending',A.RUN/'failure.json',supervisor_path]],
        'durable_status_attempts_returns':[47,47],'retained_failure_pending_attempts_returns':[48,47],'original_unknown_reservation_retained':True,
        'all47_native_to_RAW_floorPNG_and_geometry_exact':True,'all47_logit_LQ_native_schema_finite':True,'all_model_source_input_provenance_pins_current':True,
        'scope':'INTEGRITY ADMISSION OF47CARRIED SAVED OUTPUTS ONLY. Paired23completecase metrics plus orphanincumbent diagnostic; no complete68DEV result, statistical selection, acceptance, modelquality or permission to rerun old47 forwards.',
        'prospective_tail_outputs_allowed_by_this_receipt':False,'automatic_retry_resume_performed':False,'full136_pixel_audit_complete':False,'no_visual_or_quality_acceptance_claim':True,
        'model_qualified':False,'app_default_changed':False,'Torch_neural_optimizer_training_VM_calls':0,'seconds':time.monotonic()-start}
    result=A.write(destination,report)
    print(__import__('json').dumps({'complete':True,'carried47_integrity_pass':True,'outputs':47,'pairs':23,'receipt':result}));return 0


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--source-review',type=Path,required=True)
    args=parser.parse_args();return audit(args.source_review)


if __name__=='__main__':raise SystemExit(main())
