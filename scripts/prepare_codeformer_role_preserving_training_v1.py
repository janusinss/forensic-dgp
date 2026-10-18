"""CPU-only minimal read-only dependency/data stage; no seal, NN or VM launch."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.codeformer_role_preserving_training_v1 import contract as c

PRIOR_STAGE=ROOT/'outputs/codeformer_normal_tone_training_v1_preparation/stage_v1.json'
PRIOR_STAGE_SHA='869ad664a9b30eead9fe68c4f33e7afcc312751bbffb8d856a1b39f2c5c40bc7'
OLD_RETURN=ROOT/'outputs/codeformer_adain_gradient_isolation_v1_return'
OLD_IMPORT=ROOT/'outputs/codeformer_adain_gradient_isolation_v1_return_review/transport_v1.json'
OLD_IMPORT_SHA='4be3058a4ee9782cc314510884c303749b86ae5b8540a2200fbfa429dc6ad70a'


def encoded(value):
    import json
    return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode('utf-8')


def compact(value):
    import json
    return (json.dumps(value,separators=(',',':'),allow_nan=False)+'\n').encode('utf-8')


def fresh(path,pin):
    path=c.no_links(path);before=path.stat()
    c.require(path.resolve(strict=True).is_relative_to(ROOT) and path.is_file() and before.st_nlink==1 and
        before.st_size==pin['bytes'] and c.sha256(path)==pin['sha256'],'Fresh exact dependency differs: '+str(path))
    after=path.stat()
    c.require((before.st_size,before.st_mtime_ns,before.st_ino)==(after.st_size,after.st_mtime_ns,after.st_ino),
        'Dependency changed while hashing')
    return str(path)


def dependency_bindings():
    """Close only selected physical inputs, retaining full historical metadata."""
    c.require(c.sha256(PRIOR_STAGE)==PRIOR_STAGE_SHA and c.sha256(OLD_IMPORT)==OLD_IMPORT_SHA,
        'Pinned historical resolver/import receipt differs')
    prior=c.read_json(PRIOR_STAGE);imported=c.read_json(OLD_IMPORT)
    catalog=deepcopy(prior['dependencies'])
    c.require(len(catalog)==2011 and c.canonical_sha256(catalog)==c.DEPENDENCIES_SHA256,
        'Historical metadata catalog differs')
    c.require(imported['complete']is True and imported['protocol_sha256']==c.PARENT_PROTOCOL_SHA and
        imported['whole_archive_and_all_member_hashes_verified_before_import']is True and
        imported['exact_namespace_verified']is True and Path(imported['import_directory']).resolve()==OLD_RETURN.resolve(),
        'Verified historical inert return required')
    deps={};retained={};extra={};local={}
    def add(name,returned=False):
        group=extra if returned else deps
        if name in group:return Path(local[name])
        if returned:
            pin={'sha256':imported['files_sha256'][name],'bytes':imported['files_bytes'][name]}
            path=c.safe_path(OLD_RETURN,name)
        else:
            pin=catalog[name];path=Path(prior['local_dependency_resolution'][name])
        group[name]=deepcopy(pin);local[name]=fresh(path,pin)
        return path
    for name in ('protocol.json','references.json','comparison_cases.json','training_parity_v1.json',
            'hq_cache_parity_v1.json','validation_saved_v1.json','perceptual/manifest.json',
            'perceptual/LICENSE_LPIPS.txt','perceptual/LICENSE_TORCHVISION.txt','perceptual/NOTICE.md',
            'third_party/codeformer/LICENSE','third_party/codeformer/NOTICE.md',*c.HISTORICAL_RUNTIME_FILES):add(name)
    for pin in c.ASSETS.values():
        add(pin['path']);c.require(deps[pin['path']]=={k:pin[k]for k in ('sha256','bytes')},'Named asset differs')
    old=c.read_json(Path(local['protocol.json']))
    c.require(c.sha256(Path(local['protocol.json']))==c.PARENT_PROTOCOL_SHA and
        c.canonical_sha256(old)==c.PARENT_PROTOCOL_CANONICAL_SHA,'Exact parent protocol differs')
    manifest_path=add(c.RETURN_MANIFEST['path'],True)
    c.require(extra[c.RETURN_MANIFEST['path']]=={k:c.RETURN_MANIFEST[k]for k in ('sha256','bytes')},'Parent manifest differs')
    manifest=c.read_json(manifest_path)
    c.require(manifest['complete']is True and manifest['protocol_sha256']==c.PARENT_PROTOCOL_SHA,
        'Historical output provenance differs')
    for name,pin in c.FIXED_RETURNED_DEPENDENCIES.items():
        add(name,True);c.require(extra[name]==pin,'Fixed returned metadata differs')
    for name,digest in sorted(manifest['files_sha256'].items()):
        if any(name.startswith('outputs/adain_gradient_isolation_v1/snapshot_'+s+'/')for s in ('0000','0050')) and name.endswith(('/raw.npy','/native.png')):
            pin={'sha256':digest,'bytes':imported['files_bytes'][name]}
            c.require(imported['files_sha256'][name]==digest,'Historical import/manifest disagrees')
            retained[name]=pin;local[name]=fresh(c.safe_path(OLD_RETURN,name),pin)
    c.require(len(retained)==1624 and c.canonical_sha256(retained)==c.RETAINED_OUTPUTS_SHA256,
        'Exact original/base406 RAW/PNG pairs required')
    parity=c.read_json(Path(local['training_parity_v1.json']))
    references={r['id']:r for r in c.read_json(Path(local['references.json']))['references']}
    c.require([(r['update'],r['slot'])for r in parity['cases']]==[(u,s)for u in range(1,51)for s in range(1,5)],
        'Exact chronological200TRAIN observations required')
    training=[]
    for row in parity['cases']:
        for name in (row['input_path'],row['pair_record_path'],row['clean_source_path']):add(name)
        role='normal'if row['slot']in (1,3)else'dark';ref=references[row['reference_id']]
        c.require(ref['historical_role']=='train' and ref['source_sha256']==row['source_sha256'],
            'TRAIN role/source binding differs')
        training.append({**deepcopy(row),'id':'train_u%04d_slot%d'%(row['update'],row['slot']),
            'role':role,'sourcebindings':deepcopy(row)})
    selected=sorted({r['reference_id']for r in training if r['role']=='dark'})
    c.require(len(selected)==100 and set(selected)=={r['reference_id']for r in training},'Exactly100 paired TRAIN sources required')
    hq_parity={r['reference_id']:r for r in c.read_json(Path(local['hq_cache_parity_v1.json']))['cases']}
    hq=[]
    for rid in selected:
        oldrow=hq_parity[rid];source=next(r for r in training if r['reference_id']==rid)
        stem='outputs/adain_gradient_isolation_v1/HQ_cache/'+rid+'/'
        for name in ('indices.npy','features.npy','record.json'):add(stem+name,True)
        record=c.read_json(Path(local[stem+'record.json']))
        c.require(record['reference_id']==rid and record['exact_retained_FP32_target_parity']['pass']is True and
            all(record['exact_retained_FP32_target_parity'][k]==oldrow[k]for k in oldrow),
            'Old HQ array/parity provenance differs')
        hq.append({**deepcopy(oldrow),'clean_source_path':oldrow['source_path'],
            'clean512_rgb8_sha256':source['clean512_rgb8_sha256'],'indices_path':stem+'indices.npy',
            'features_path':stem+'features.npy','record_path':stem+'record.json','sourcebindings':record})
    saved={r['id']:r for r in c.read_json(Path(local['validation_saved_v1.json']))['items']}
    comparisons={'comparison_'+r['id']:r for r in c.read_json(Path(local['comparison_cases.json']))['cases']}
    metrics={s:c.read_json(Path(local['outputs/adain_gradient_isolation_v1/snapshot_'+s+'/metrics.json']))for s in ('0000','0050')}
    ids=[r['id']for r in metrics['0000']['rows']]
    c.require(len(ids)==len(set(ids))==406 and set(ids)==set(saved)|set(comparisons) and
        set(ids)=={r['id']for r in metrics['0050']['rows']},'Exact406 development metadata namespace required')
    development=[]
    for row in metrics['0000']['rows']:
        rid=row['id']
        if rid in saved:
            origin=saved[rid];inp,mask,target=origin['input_path'],None,origin['target_path']
            for name in (inp,target,origin['pair_record_path']):add(name)
            sourcebindings={'validation_saved':deepcopy(origin),'pair_record':c.read_json(Path(local[origin['pair_record_path']]))}
        else:
            origin=comparisons[rid];inp,mask,target=origin['prepared_path'],origin['observed_path'],origin.get('target_path')
            for name in (inp,mask,target):
                if name is not None:add(name)
            sourcebindings=deepcopy(origin)
        paths={'original_raw_path':'outputs/adain_gradient_isolation_v1/snapshot_0000/'+rid+'/raw.npy',
            'original_png_path':'outputs/adain_gradient_isolation_v1/snapshot_0000/'+rid+'/native.png',
            'parent_raw_path':'outputs/adain_gradient_isolation_v1/snapshot_0050/'+rid+'/raw.npy',
            'parent_png_path':'outputs/adain_gradient_isolation_v1/snapshot_0050/'+rid+'/native.png'}
        c.require(all(name in retained for name in paths.values()),'Historical DEV output pair missing')
        development.append({'id':rid,'source':row['source'],'profile':row['profile'],'evidence':row['evidence'],
            'input_path':inp,'observed_path':mask,'target_path':target,**paths,'sourcebindings':sourcebindings})
    c.require(set(extra)==c.expected_extra_returned_paths(selected),'Exact100HQ/fixedmetadata physical closure required')
    bundle={'format':'role-preserving-data-bundle-v1','training':training,'hq':hq,'development':development}
    return catalog,deps,retained,extra,local,old,bundle,selected


def prepare(stage,receipt,provenance=()):
    from research.codeformer_role_preserving_training_v1 import transport
    stage,receipt=c.no_links(stage),c.no_links(receipt)
    c.require(stage.parent.resolve()==(ROOT/'outputs').resolve() and stage.name==c.VM_NAME and
        not stage.exists() and not receipt.exists() and receipt.resolve(strict=False).is_relative_to(ROOT/'outputs'),
        'Exact unused source-stage/receipt required; preserve prior attempts')
    catalog,deps,retained,extra,local,old,bundle,selected=dependency_bindings()
    sources=sorted(c.REQUIRED_SOURCE_FILES-{'research/__init__.py'})
    sources+=['scripts/prepare_codeformer_role_preserving_training_v1.py']
    sources+=['tests/test_codeformer_role_preserving_'+kind+'_v1.py'for kind in ('contract','model','worker')]
    sources+=['tests/test_codeformer_role_preserving_training_'+kind+'_v1.py'for kind in ('data','transport')]
    payloads={name:c.safe_path(ROOT,name).read_bytes()for name in sources};current={name:name for name in sources}
    for name in ('AGENTS.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','PROJECT_HANDOFF.md','CCTV_CODEFORMER_FINETUNING_PLAN_V1.md'):
        target='instructions/'+name;payloads[target]=c.safe_path(ROOT,name).read_bytes();current[target]=name
    for name in provenance:
        path=c.safe_path(ROOT,name);c.require(path.suffix in ('.md','.json'),'Bounded source provenance only')
        target='provenance/'+name;c.require(target not in payloads,'Provenance source-path collision')
        payloads[target]=path.read_bytes();current[target]=name
    payloads['data_bundle.json']=compact(bundle);current['data_bundle.json']=False
    c.require(len(payloads['data_bundle.json'])<=c.LIMITS['one_JSON_bytes'],'All-field compactdata bundle exceeds2MiB')
    payloads['research/__init__.py']=b'';current['research/__init__.py']=False
    marker=c.safe_path(ROOT,'research/__init__.py').read_bytes()
    synthetic={'research/__init__.py':{'policy':'intentional empty transport namespace; localinitializer notcopied',
        'packet_sha256':hashlib.sha256(b'').hexdigest(),'packet_bytes':0,'local_reference_path':'research/__init__.py',
        'local_reference_sha256':hashlib.sha256(marker).hexdigest(),'local_reference_bytes':len(marker),'copied_or_executed':False}}
    hashes={n:hashlib.sha256(v).hexdigest()for n,v in payloads.items()};sizes={n:len(v)for n,v in payloads.items()}
    protocol={'format':c.FORMAT,'module':c.MODULE,'vm_name':c.VM_NAME,'parent_dir':c.PARENT_NAME,
        'parent_protocol_sha256':c.PARENT_PROTOCOL_SHA,'outputs':c.OUTPUTS,'export_prefix':c.PREFIX,
        'design':deepcopy(c.DESIGN),'design_sha256':c.DESIGN_SHA256,'state_policy':deepcopy(c.STATE_POLICY),
        **{k:deepcopy(getattr(c,k.upper()))for k in ('training','objective','evaluation','gradient_route','initializer_parity','perceptual_prerequisite','assets','model_digests')},
        'call_caps':deepcopy(c.COUNTS),'limits':deepcopy(c.LIMITS),**deepcopy(c.PARENT_CONTROLS),
        'data_manifest':'references.json','comparison_cases':'comparison_cases.json','data_bundle_file':'data_bundle.json',
        'historical_parent_protocol':old,'historical_dependency_bindings':catalog,'dependencies':deps,
        'retained_output_dependencies':retained,'extra_returned_dependencies':extra,'selected_train_reference_ids':selected,
        'files_sha256':hashes,'files_bytes':sizes,'design_document':{'path':c.DESIGN_DOCUMENT,'sha256':hashes[c.DESIGN_DOCUMENT]},
        'status':'PREPARED_UNRUN','manual_tmux_required':True,'new_training_recipe':True,'actual_candidate_checkpoints_required':True,
        **{k:False for k in ('automatic_training','automatic_VM_launch','automatic_resume','model_qualified','automatic_app_promotion',
            'quality_gates_changed','FINAL_used','old_optimizer_loaded','historical_RNG_loaded','cached_gradient_reuse','tone_head_used')}}
    c.verify_protocol(protocol)
    protocol_payload=compact(protocol)
    c.require(len(protocol_payload)<=c.LIMITS['one_JSON_bytes'],'All-field compactprotocol exceeds2MiB')
    c.require(len(payloads)+1<=c.LIMITS['source_packet_files'] and sum(sizes.values())+len(protocol_payload)<=c.LIMITS['source_packet_bytes'],
        'Protocol-inclusive source packet cap exceeded')
    # Verify metadata/source joins before creating a stage; resolver never copies parent data.
    metadata_receipt=transport.verify_data_bundle(None,protocol,None,resolver=lambda name:Path(local[name]),bundle=bundle)
    # Final numerical image/HQ joins run once before any source-stage mutation.
    from research.codeformer_role_preserving_training_v1 import data
    loaded=data.load_index(None,protocol,bundle,resolver=lambda name:Path(local[name]))
    data_receipt=loaded['binding_receipt']
    del loaded
    import gc
    gc.collect()
    ledger=transport.resource_ledger(len(payloads)+1,sum(sizes.values())+len(protocol_payload))
    c.require(not any(n in sys.modules for n in ('torch','tensorflow','jax')),'Framework imported during stage preparation')
    stage.mkdir()
    for name,value in payloads.items():
        path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb')as stream:stream.write(value)
    with (stage/'protocol.json').open('xb')as stream:stream.write(protocol_payload)
    c.write_new_json(receipt,{'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'stage':str(stage),
        'protocol_sha256':c.sha256(stage/'protocol.json'),'source_files':hashes,'source_bytes':sizes,'current_sources':current,
        'synthetic_sources':synthetic,'generated_data_files':['data_bundle.json'],'historical_dependency_bindings':catalog,
        'dependencies':deps,'retained_output_dependencies':retained,'extra_returned_dependencies':extra,
        'local_dependency_resolution':local,'dependency_count':len(deps),'retained_output_dependency_count':len(retained),
        'extra_returned_dependency_count':len(extra),'total_readonly_bindings':len(local),'resource_ledger':ledger,
        'data_metadata_receipt':metadata_receipt,'data_binding_receipt':data_receipt,
        'selected_train_reference_ids':selected,'source_sha256':c.sha256(__file__),
        'source_stage_not_sealed':True,'local_neural_gradient_optimizer_VM_calls':0,'model_qualified':False})
    return {'complete':True,'stage_only_not_sealed':True,'static_files':len(payloads)+1,'protocol_sha256':c.sha256(stage/'protocol.json'),
        'minimal_dependencies':len(deps),'retained_output_dependencies':len(retained),'extra_returned_dependencies':len(extra),'local_neural_calls':0}


def plan_only(directory):
    """Write reviewed data/physical-resolution metadata without a source stage."""
    from research.codeformer_role_preserving_training_v1 import transport
    directory=c.no_links(directory)
    c.require(directory.resolve(strict=False).is_relative_to(ROOT/'outputs') and not directory.exists(),
        'Exclusive in-workspace data-plan directory required')
    catalog,deps,retained,extra,local,old,bundle,selected=dependency_bindings()
    p={'dependencies':deps,'retained_output_dependencies':retained,'extra_returned_dependencies':extra,
        'selected_train_reference_ids':selected}
    data_receipt=transport.verify_data_bundle(None,p,None,resolver=lambda n:Path(local[n]),bundle=bundle)
    directory.mkdir(parents=True)
    payload=compact(bundle);c.require(len(payload)<=c.LIMITS['one_JSON_bytes'],'Compactdata plan exceeds2MiB')
    with (directory/'data_bundle.json').open('xb')as stream:stream.write(payload)
    c.write_new_json(directory/'receipt.json',{'complete':True,'source_stage_created':False,'sealed':False,
        'data_bundle_sha256':c.sha256(directory/'data_bundle.json'),'data_bundle_bytes':(directory/'data_bundle.json').stat().st_size,
        'historical_dependency_bindings':catalog,'dependencies':deps,'retained_output_dependencies':retained,
        'extra_returned_dependencies':extra,'local_dependency_resolution':local,'selected_train_reference_ids':selected,
        'data_binding_receipt':data_receipt,'source_sha256':c.sha256(__file__),'neural_gradient_optimizer_VM_calls':0})
    return {'complete':True,'data_only_plan':str(directory),'minimal_dependencies':len(deps),
        'retained_outputs':len(retained),'extra_returned':len(extra),'data_bundle_sha256':c.sha256(directory/'data_bundle.json')}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',type=Path,default=ROOT/'outputs'/c.VM_NAME)
    p.add_argument('--receipt',type=Path,default=ROOT/'outputs/codeformer_role_preserving_training_v1_preparation/stage_v1.json')
    p.add_argument('--provenance',action='append',default=[])
    p.add_argument('--plan-only',type=Path)
    a=p.parse_args(argv);print(plan_only(a.plan_only)if a.plan_only is not None else prepare(a.stage,a.receipt,a.provenance))


if __name__=='__main__':main()
