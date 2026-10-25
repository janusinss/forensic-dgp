"""Build a fresh self-contained manual-L4 packet; never imports Torch."""
import argparse
import ast
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from research.codeformer_spatial_objectives_v1 import packet as p

STAGE=ROOT/'outputs'/p.VM_NAME
RETURN=ROOT/'outputs/codeformer_skip_feature_adapter_v1_return'
DESIGN=ROOT/'outputs/codeformer_skip_spatial_objective_design_v1'
REVIEW=ROOT/'outputs/codeformer_spatial_objectives_packet_review_v1'


def stage_packet():
    p.require(not STAGE.exists(),'New stage already exists; preserve it and do not rebuild over it')
    STAGE.mkdir(); REVIEW.mkdir(exist_ok=True)
    copied={}; source_map={}
    def copy(source,name):
        source=p.regular(source);p.canonical(name);destination=STAGE/name
        digest=p.sha256(source);size=source.stat().st_size
        if name in copied:
            p.require(copied[name]=={'bytes':size,'sha256':digest},'Content path collision')
            return {'path':name,**copied[name]}
        destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as a,destination.open('xb') as b:shutil.copyfileobj(a,b,1024**2)
        p.require(destination.stat().st_size==size and p.sha256(destination)==digest,'Stage copy differs')
        copied[name]={'bytes':size,'sha256':digest}
        source_map[name]={'local_source':str(source.relative_to(ROOT)),'bytes':size,'sha256':digest}
        return {'path':name,**copied[name]}
    def data(binding,base=RETURN):
        source=ROOT/binding['path'] if binding['path'].startswith('outputs/') and (ROOT/binding['path']).is_file() else base/binding['path']
        p.require(source.stat().st_size==binding['bytes'] and p.sha256(source)==binding['sha256'],'Retained asset pin differs')
        suffix=source.suffix.lower();name='assets/'+binding['sha256']+suffix
        return copy(source,name)
    def write(name,value):
        target=STAGE/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as f:f.write(p.json_bytes(value))
        copied[name]={'bytes':target.stat().st_size,'sha256':p.sha256(target)}
    d=p.read_json(DESIGN/'prospective_protocol_design_v1.json')
    admission=[];rows=[]
    from PIL import Image
    import numpy as np
    def rgbsha(pin):
        with Image.open(STAGE/pin['path']) as image:
            p.require(image.mode=='RGB' and image.size==(256,256),'RGB256 static input/target required')
            return hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
    for row in d['dataset']['observations']:
        item={k:row[k] for k in ('id','reference_id','profile','split','input_only_decision')}
        item['input']=data(row['input']);item['input_RGB8_sha256']=rgbsha(item['input'])
        p.require(item['input_RGB8_sha256']==row['input_RGB8_sha256'],'TRAIN pixels changed')
        item['target']=data(row['paired_photo_target']);item['target_RGB8_sha256']=rgbsha(item['target'])
        rows.append(item)
        admission.append({**item,'admitted_for_this_finite_TRAIN_screen':True,
            'new_role_record':'Four readable strong inputs move from historical nonoptimizing TRAIN probes to this explicit prospective TRAIN fit; old flags remain intact.',
            'no_DEV_FINAL_native_optimization':True})
    old_guards=p.read_json(RETURN/'outputs/skip_feature_adapter_v1/guard_0000/bank.json')['cases']
    old_by_id={r['id']:r for r in old_guards}
    teacher=p.read_json(RETURN/'outputs/skip_feature_adapter_v1/teacher/manifest.json')
    guards=[]
    for old in old_guards:
        item={k:old[k] for k in ('id','reference_id','profile','role','split','metrics','upstream')}
        for field in ('input','target'):
            item[field]=data(old[field]) if old[field] is not None else None
            item[field+'_RGB8_sha256']=rgbsha(item[field]) if item[field] is not None else None
        item['outputs']={k:data(v) for k,v in old['outputs'].items()}
        if old['role'] in ('clear','dark'):
            item['behavior_distance']=old['behavior_distance']
            behavior=old['behavior_teacher']
            item['behavior_teacher']={'raw':data(behavior['RAW']),'png':data(behavior['PNG']),'kind':behavior['kind']}
        guards.append(item)
    p.require([r['id'] for r in guards]==[r['id'] for r in d['output_guards']['cases']],'Exact guard order mismatch')
    strong=[]
    for row in d['strong_output_probes']['cases']:
        item=next(dict(r) for r in rows if r['id']==row['id'])
        item['outputs']={k:data(v) for k,v in row['initial_R2_cached_outputs'].items()}
        item['metrics']=row['initial_R2_cached_metrics'];item['upstream']=row['initial_R2_upstream']
        strong.append(item)
    clean={}
    for row in d['model']['clean_features']:
        rid=row['reference_id'];clean[rid]={'features':{s:data(b) for s,b in row['features'].items()},'scales':row['scales']}
    weight=copy(RETURN/'weights/codeformer_r2.pth','weights/codeformer_r2.pth')
    # Local active sources are new, not edits to any historical packet.
    for source in sorted((ROOT/'research/codeformer_spatial_objectives_v1').iterdir()):
        if source.suffix in ('.py','.md'):copy(source,'research/codeformer_spatial_objectives_v1/'+source.name)
    copy(ROOT/'research/__init__.py','research/__init__.py')
    for name in ('__init__.py','codeformer/__init__.py','codeformer/codeformer_arch.py',
                 'codeformer/vqgan_arch.py','codeformer/registry.py','codeformer/LICENSE','codeformer/NOTICE.md'):
        copy(RETURN/'third_party'/name,'third_party/'+name)
    copy(RETURN/'research/codeformer_skip_feature_adapter_v1/metrics.py','provenance/metrics.py')
    copy(ROOT/'scripts/prepare_codeformer_spatial_objectives_v1.py','scripts/prepare_codeformer_spatial_objectives_v1.py')
    copy(ROOT/'scripts/run_codeformer_spatial_objectives_v1.py','scripts/run_codeformer_spatial_objectives_v1.py')
    copy(ROOT/'scripts/supervise_codeformer_spatial_objectives_v1.py','scripts/supervise_codeformer_spatial_objectives_v1.py')
    copy(ROOT/'scripts/run_spatial_objectives.sh','run_spatial_objectives.sh')
    provenance=[]
    for source in sorted(DESIGN.iterdir()):
        if source.suffix in ('.json','.md'):
            provenance.append(copy(source,'provenance/design/'+source.name))
    for name in ('AGENTS.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
                 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md','CCTV_CODEFORMER_SKIP_ADAPTER_ROLE_DIAGNOSTIC_V1_REVIEW.md'):
        provenance.append(copy(ROOT/name,'provenance/instructions/'+name))
    for name in ('data_bundle.json','data_admission.json','auxiliary_bundle.json','provenance/auxiliary_root_input_review_v1_r1.json',
                 'provenance/retained_filter_fusion/reference_manifest.json',
                 'provenance/retained_filter_fusion/upstream_CodeFormer_LICENSE',
                 'provenance/retained_filter_fusion/data_independent_scope_pass_v2.json'):
        provenance.append(copy(RETURN/name,'provenance/retained/'+Path(name).name))
    sampler=[{'id':r['observation_id'],'update':r['update'],'slot':r['slot']} for r in p.read_json(DESIGN/'sampler72_v1.json')['rows']]
    write('admission.json',{'format':'spatial-objectives-explicit-TRAIN-admission-v1','complete':True,'rows':admission,
        'preserves_historical_flags':True,'unknown_wild_images_used':False,'borderline_strong_and_mosaics_excluded':True})
    write('source_map.json',source_map)
    vm_only=[weight['path']]+[b['path'] for r in clean.values() for b in r['features'].values()]
    spec={'format':'codeformer-spatial-objectives-manual-L4-protocol-v1','vm_name':p.VM_NAME,
        'design_sha256':p.sha256(DESIGN/'prospective_protocol_design_v1.json'),
        'train_rows':rows,'clean_features':clean,'guards':guards,'strong':strong,'sampler':sampler,
        'r2_asset':weight,'parity_ids':[r['id'] for r in d['output_guards']['fresh_GPU_parity_witnesses']],
        'initial_adapter_sha256':d['model']['initializer_adapter24_canonical_sha256'],
        'base515_sha256':d['model']['base515_canonical_sha256'],'max_calls':p.CAPS,'limits':p.LIMITS,
        'files':copied,'vm_only_static':sorted(vm_only),'return_static':sorted(set(copied)-set(vm_only)),
        'provenance':provenance,'max_updates_per_arm':24,'arms':['U','S'],'snapshots':[0,6,12,24],
        'worker_seconds':2700,'manual_supervised_maximum_seconds_including_export':3120,
        'earlier45min_design_cap_provisional_refinement':'Worker45min; external supervision46min; export5min;52min entire supervised ceiling. No unchanged extension of a failed training recipe.',
        'automatic_launch':False,'automatic_resume':False,'model_qualified':False,
        'full_development_and_FINAL_still_required':True,'return_omits_only_VM_static_and_observed_cache':True}
    spec['allowed_outputs']=p.output_names(spec)
    path=STAGE/'protocol.json'
    with path.open('xb') as f:f.write(p.json_bytes(spec))
    verified=p.verify(STAGE,p.sha256(path))
    from research.codeformer_spatial_objectives_v1 import audit
    checked=audit.audit_static(STAGE,verified)
    for name in verified['files']:
        if name.endswith('.py'):ast.parse((STAGE/name).read_text(encoding='utf-8'),feature_version=(3,10))
    report={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'stage':str(STAGE),
        'protocol_sha256':p.sha256(path),'static_files':len(copied),'static_bytes':sum(v['bytes'] for v in copied.values()),
        'vm_only_static_bytes':sum(copied[n]['bytes'] for n in vm_only),
        'return_static_bytes':sum(copied[n]['bytes'] for n in spec['return_static'])+path.stat().st_size,
        'static_audit':checked,'Torch_neural_optimizer_calls':0,'training_unrun':True,'model_qualified':False}
    reportpath=REVIEW/'stage_preparation_v1.json'
    with reportpath.open('xb') as f:f.write(p.json_bytes(report))
    print(json.dumps({k:v for k,v in report.items() if k!='static_audit'}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',action='store_true',required=True)
    parser.parse_args();stage_packet()
