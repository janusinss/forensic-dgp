"""CPU-only data/source staging; never archive, upload, infer or launch."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.codeformer_fixed_state_response_v1 import contract as c

PRIOR_STAGE=ROOT/'outputs/codeformer_tone_control_training_v1_preparation/stage_v1.json'
PRIOR_STAGE_SHA='9750ce0308bb2d677a96e4eb0b91544f5da7831aef512a4e36d17b8f7741d7cb'
TONE_IMPORT=ROOT/'outputs/codeformer_tone_control_training_v1_return_review/return_import_v1.json'
TONE_IMPORT_SHA='5a8128b239732f29f2f0195df6d3d4b42bd8c90ae895a04d1329b7d3f2afdd05'
STATE_AUDIT=ROOT/'outputs/codeformer_tone_control_training_v1_return_review/state_v1/state_audit_v1.json'
STATE_AUDIT_SHA='7e918ccaa79304a846ff9fd02f5812c0b35255b0626948bee4dca22883fbf5f8'


def encoded(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')


def fresh(path,pin):
    path=c.no_links(path);before=path.stat()
    c.require(path.resolve(strict=True).is_relative_to(ROOT)and path.is_file()and before.st_nlink==1 and
        before.st_size==pin['bytes']and c.sha256(path)==pin['sha256'],'Exact local parent binding differs: '+str(path))
    after=path.stat();c.require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
        'Parent bytes changed while hashing')
    return str(path)


def dependency_bindings():
    c.require(c.sha256(PRIOR_STAGE)==PRIOR_STAGE_SHA and c.sha256(TONE_IMPORT)==TONE_IMPORT_SHA and
        c.sha256(STATE_AUDIT)==STATE_AUDIT_SHA,'Closed parent import/resolver/state review changed')
    stage=c.read_json(PRIOR_STAGE);imported=c.read_json(TONE_IMPORT);state=c.read_json(STATE_AUDIT)
    c.require(imported['complete']is True and imported['protocol_sha256']==c.PARENT_PROTOCOL_SHA256['tone_training']and
        state['evidence_consistency_pass']is True,'Verified exact tone parent required')
    tone_root=Path(imported['import_root']);deps={'tone_training':{},'adain':{}};local={'tone_training':{},'adain':{}}
    def add(alias,name):
        if name in deps[alias]:return Path(local[alias][name])
        if alias=='adain':pin=stage['dependencies'][name];path=Path(stage['local_dependency_resolution'][name])
        else:pin={'sha256':imported['files_sha256'][name],'bytes':imported['files_bytes'][name]};path=c.safe_path(tone_root,name)
        deps[alias][name]=deepcopy(pin);local[alias][name]=fresh(path,pin);return path
    for alias in c.ROOT_ALIAS:add(alias,'protocol.json')
    for name in (*c.HISTORICAL_RUNTIME_FILES,*c.CAMERA_FILES,'references.json','training_parity_v1.json','comparison_cases.json'):
        add('adain',name)
    for asset in c.ASSETS.values():
        add(asset['root'],asset['path']);c.require(deps[asset['root']][asset['path']]=={k:asset[k]for k in ('sha256','bytes')},'Exact asset changed')
    for name in ('research/codeformer_tone_control_training_v1/model.py','research/codeformer_tone_control_training_v1/data.py',
            'research/codeformer_tone_control_training_v1/design.json','outputs/tone_control_training_v1/runtime.json'):
        add('tone_training',name)
    references=c.read_json(add('adain','references.json'))
    parity=c.read_json(add('adain','training_parity_v1.json'))
    fitted=[]
    for p in parity['cases']:
        identifier='train_u%04d_slot%d'%(p['update'],p['slot']);stem='outputs/tone_control_training_v1/TRAIN/'+identifier+'/'
        for name in (p['pair_record_path'],):add('adain',name)
        paths={key:stem+filename for key,filename in [('input_path','input.png'),('cache_path','cache.json'),
            ('base_raw_path','base_raw.npy'),('original_raw_path','original_raw.npy')]}
        for name in paths.values():add('tone_training',name)
        fitted.append({'id':identifier,'cohort':'fitted','reference_id':p['reference_id'],'source_sha256':p['source_sha256'],
            'role':'normal'if p['slot']in (1,3)else'dark','update':p['update'],'slot':p['slot'],**paths})
    return deps,local,references,parity,{'format':'codeformer-fixed-state-fitted-cases-v1','cases':fitted},add


def exclusion_metadata(references,comparison):
    ids={r['id']for r in references['references']if r['historical_role']!='train'}
    hashes={r['source_sha256']for r in references['references']if r['historical_role']!='train'}
    metadata={}
    sources=[('comparison_metadata',comparison)]
    for name in ('outputs/cctv_native_development_v2/frozen_subset.json','outputs/cctv_chokepoint_native_development_v1/frozen_subset.json'):
        path=c.safe_path(ROOT,name);raw=path.read_bytes();doc=c.read_json(path)
        metadata[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'metadata_only':True}
        sources.append((name,doc['cases']))
    for source,rows in sources:
        for row in rows:
            for key in ('id','reference_id','identity','global_person_id','source_person_id'):
                if row.get(key)is not None:ids.add(str(row[key]))
            for key,value in row.items():
                if key.endswith('sha256')and isinstance(value,str)and c.HEX.fullmatch(value):hashes.add(value)
    return {'format':'codeformer-fixed-state-exclusions-v1','reference_ids':sorted(ids),'source_sha256':sorted(hashes),
        'metadata_sources':metadata,'scope':'Exact file/photo and native split metadata exclusion; numeric/source IDs do not prove human identity disjointness',
        'DEV_FINAL_native_pixels_opened':0,'FINAL_targets_or_metrics_used':False}


def prepare(stage,receipt):
    from research.codeformer_fixed_state_response_v1 import data,budget
    stage=c.no_links(stage);receipt=c.no_links(receipt)
    c.require(stage.parent.resolve()==(ROOT/'outputs').resolve()and stage.name==c.VM_NAME and
        not stage.exists()and not receipt.exists(),'Exact new unused stage and receipt required')
    deps,local,references,parity,fitted,add=dependency_bindings()
    refs=data.reference_index(references)
    comparison=c.read_json(Path(local['adain']['comparison_cases.json']))['cases']
    exclusions=exclusion_metadata(references,comparison)
    selected=data.select_heldout(refs,parity,exclusions)
    for row in selected:add('adain',row['source_path'])
    resolver=lambda alias,name:Path(local[alias][name])
    selection={'format':'codeformer-fixed-state-selection-v1','selected':selected,
        'selection_rule':c.DESIGN['data']['selection'],'camera_rule':c.DESIGN['data']['heldout_camera'],
        'source_ids_are_not_proven_person_ids':True,'fitted_reference_ids':sorted({r['reference_id']for r in parity['cases']}),
        'exclusions_sha256':hashlib.sha256(encoded(exclusions)).hexdigest()}
    heldout,generated=data.render_heldout(selected,deps['adain']['references.json']['sha256'],resolver,deps['adain'])
    heldout['selection_sha256']=hashlib.sha256(encoded(selection)).hexdigest()
    payloads=dict(generated,**{'fitted_cases.json':encoded(fitted),'heldout_pairs.json':encoded(heldout),
        'selection.json':encoded(selection),'exclusions.json':encoded(exclusions)})
    source_names=sorted(c.REQUIRED_SOURCE_FILES-{'research/__init__.py'})
    source_names+=['tests/test_codeformer_fixed_state_response_'+kind+'_v1.py'for kind in ('data','model','transport','worker')]
    current={}
    for name in source_names:payloads[name]=c.safe_path(ROOT,name).read_bytes();current[name]=name
    instructions=('AGENTS.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','PROJECT_HANDOFF.md',
        'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md','CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN_REVIEW.md')
    for name in instructions:payloads['instructions/'+name]=c.safe_path(ROOT,name).read_bytes();current['instructions/'+name]=name
    for name in generated:current[name]=False
    for name in c.REQUIRED_DATA_FILES:current[name]=False
    marker=c.safe_path(ROOT,'research/__init__.py').read_bytes();payloads['research/__init__.py']=b'';current['research/__init__.py']=False
    synthetic={'research/__init__.py':{'policy':'intentional empty transport namespace; local initializer not copied',
        'packet_sha256':hashlib.sha256(b'').hexdigest(),'packet_bytes':0,'local_reference_path':'research/__init__.py',
        'local_reference_sha256':hashlib.sha256(marker).hexdigest(),'local_reference_bytes':len(marker),'copied_or_executed':False}}
    hashes={n:hashlib.sha256(raw).hexdigest()for n,raw in payloads.items()};sizes={n:len(raw)for n,raw in payloads.items()}
    p={'format':c.FORMAT,'module':c.MODULE,'vm_name':c.VM_NAME,'outputs':c.OUTPUTS,'export_prefix':c.PREFIX,
        'parent_roots':deepcopy(c.PARENT_ROOTS),'parent_protocol_sha256':deepcopy(c.PARENT_PROTOCOL_SHA256),
        'assets':deepcopy(c.ASSETS),'model_digests':deepcopy(c.MODEL_DIGESTS),'design':deepcopy(c.DESIGN),
        'design_sha256':c.DESIGN_SHA256,'call_caps':deepcopy(c.COUNTS),'limits':deepcopy(c.LIMITS),
        'fitted_index_file':'fitted_cases.json','new_pairs_file':'heldout_pairs.json','selection_file':'selection.json','exclusions_file':'exclusions.json',
        'dependencies':deps,'files_sha256':hashes,'files_bytes':sizes,'status':'PREPARED_UNRUN','manual_tmux_required':True,'all_states_frozen':True,
        **{k:False for k in ('automatic_VM_launch','automatic_training','automatic_resume','new_training_recipe','model_qualified',
            'automatic_app_promotion','quality_gates_changed','FINAL_used','optimizer_loaded','cached_gradient_reuse')}}
    c.verify_protocol(p)
    protocol_bytes=encoded(p)
    c.require(sum(sizes.values())+len(protocol_bytes)<=c.LIMITS['source_packet_bytes'],'Whole static packet including protocol exceeds32MiB')
    ledger=budget.resource_ledger(len(payloads)+1,sum(sizes.values())+len(protocol_bytes))
    c.require(not any(x in sys.modules for x in ('torch','tensorflow','jax')),'Framework imported during preparation')
    # All image/model dependencies are read-only. Source selection, rendering,
    # hashes and resource closure precede staging. Full decode/readback follows;
    # a failed partial stage is preserved and receives no complete receipt.
    stage.mkdir()
    for name,raw in payloads.items():
        path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb')as stream:stream.write(raw)
    with (stage/'protocol.json').open('xb')as stream:stream.write(protocol_bytes)
    index=data.load_index(stage,p,{},resolver=resolver)
    c.write_new_json(receipt,{'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'stage':str(stage),
        'protocol_sha256':c.sha256(stage/'protocol.json'),'source_files':hashes,'source_bytes':sizes,'current_sources':current,
        'synthetic_sources':synthetic,'generated_data_files':sorted(set(generated)|c.REQUIRED_DATA_FILES),
        'dependencies':deps,'local_dependency_resolution':local,'dependency_counts':{k:len(v)for k,v in deps.items()},
        'parent_import_sha256':TONE_IMPORT_SHA,'parent_state_audit_sha256':STATE_AUDIT_SHA,
        'source_sha256':c.sha256(__file__),'resource_ledger':ledger,'data_binding_receipt':index['binding_receipt'],
        'source_stage_not_sealed':True,'actual_neural_autograd_optimizer_VM_calls':0,'model_qualified':False})
    return {'complete':True,'stage_only_not_sealed':True,'protocol_sha256':c.sha256(stage/'protocol.json'),
        'static_files':len(payloads)+1,'heldout_PNGs':64,'fitted_rows':200,'heldout_rows':64,
        'dependencies':{k:len(v)for k,v in deps.items()},'neural_calls':0}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,default=ROOT/'outputs'/c.VM_NAME)
    p.add_argument('--receipt',type=Path,default=ROOT/'outputs/codeformer_fixed_state_response_v1_preparation/stage_v1.json')
    a=p.parse_args(argv);print(json.dumps(prepare(a.stage,a.receipt)))


if __name__=='__main__':main()
