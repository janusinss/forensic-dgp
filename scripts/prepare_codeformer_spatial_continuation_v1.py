"""Stage a new, self-contained reviewed manual-L4 continuation. Never seals a tar.

Imports and checks are non-neural. Existing stages, returns and checkpoints are
read-only sources. A hash-bound independent source review is required to stage.
"""
import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.codeformer_spatial_continuation_v1 import packet as p

PARENT=ROOT/'outputs/cctv_codeformer_spatial_objectives_v1_vm'
RETURN=ROOT/'outputs/cctv_codeformer_spatial_objectives_v1_return_recovery_r1'
REVIEW=ROOT/'outputs/codeformer_spatial_objectives_return_review_v1'
STAGE=ROOT/'outputs'/p.VM_NAME
REPORT=ROOT/'outputs/codeformer_spatial_continuation_packet_review_v1'
PARENT_SHA='3fe24e684a835268ac805ca56bfafefff429ef5f1aa13a90ec11faa644380dd9'
EVIDENCE={
 'parent_result':(RETURN/'outputs/spatial_objectives_v1/result.json','4dc5a10bb73eea23b2f495c0de07432b7bc17671ae501736979ea7065e39b5dc'),
 'pure_return_audit':(REVIEW/'pure_return_audit_v1_r2.json','94344dc1494c2da4818222bc8c28d9b8986179b9ccf0ee14b2587e84bca52f4a'),
 'checkpoint_content_audit':(REVIEW/'independent_checkpoint_contents_v1.json','a3d3f8b5533b2d98a8445bbfc010fe8cf83e046296cd9eb0711894e0e3fb9704'),
 'export_manifest':(RETURN/'export.json','9da40ef8b7bd237050596b4e97de6fa3be4d8316d65a03587df211bd39d82929')}
SOURCES=[
 'research/codeformer_spatial_continuation_v1/'+n for n in ('packet.py','model.py','worker.py','audit.py')]
SOURCES += ['scripts/'+n for n in ('prepare_codeformer_spatial_continuation_v1.py',
 'run_codeformer_spatial_continuation_v1.py','supervise_codeformer_spatial_continuation_v1.py','run_spatial_continuation.sh')]


def output_cap(name):
    """Closed complete-run payload bounds, exclusive of fresh cache/static."""
    if name.endswith('.partial') or name.startswith('cache/'):return 0
    if name.endswith('.npy'):return 786560
    if name.endswith('.png'):return 200*1024
    if name.endswith('.npz'):return 1274842
    if name.endswith('.pt'):return p.MiB
    if name.endswith('.jsonl'):return 16*p.MiB
    if name=='trainer.log':return 2*p.MiB
    if name.endswith('bank.json'):return 256*1024
    if name.endswith('gates.json'):return 128*1024
    if name.startswith(p.O+'stats/'):return 32*1024
    if name.startswith(p.O+'branches/') or name=='launch.json':return p.MiB
    if name in (p.O+'result.json','export.json'):return 2*p.MiB
    if name.startswith((p.O+'parity/',p.O+'guards/',p.O+'strong/')):return 16*1024
    return 512*1024


def stage_packet(peer_path, peer_sha):
    p.require(not STAGE.exists(),'Preserve previous or partial stage; no rebuild-over')
    peer_path=p.regular(peer_path)
    p.require(p.sha256(peer_path)==peer_sha,'Independent peer receipt SHA differs')
    peer=p.read_json(peer_path);p.require(peer.get('pass') is True,'Independent source peer must pass')
    pins={r['path']:r for r in peer['source_pins']}
    for name in SOURCES:
        actual=p.pin(ROOT,ROOT/name)
        p.require(pins.get(name)==actual,'Peer receipt does not cover current source: '+name)
    from research.codeformer_spatial_continuation_v1 import audit
    fixture=audit.objective_fixtures()
    p.require(fixture.get('audit_pass') is True or fixture.get('pass') is True,'Independent objective fixtures must pass')
    p.require('torch' not in sys.modules,'Staging must never import Torch')
    p.require(p.sha256(PARENT/'protocol.json')==p.sha256(RETURN/'protocol.json')==PARENT_SHA,
              'Original sealed/returned protocol differs')
    parent=p.read_json(PARENT/'protocol.json')
    # Full original static closure, without importing or executing old runtime.
    for name,binding in parent['files'].items():
        source=p.regular(PARENT/name)
        p.require(source.stat().st_size==binding['bytes'] and p.sha256(source)==binding['sha256'],
                  'Original static source/asset changed: '+name)
    for source,digest in EVIDENCE.values():p.require(p.sha256(source)==digest,'Audited source evidence changed')
    result=p.read_json(EVIDENCE['parent_result'][0])
    p.require(result['complete'] and not result['pending'] and result['counts']==result['returns'] and
        all(b['terminal_status']=='complete' and b['completed_updates']==24 for b in result['branches']),
        'Only complete audited own24 states may continue')
    source_bytes=sum(b['bytes'] for b in parent['files'].values())
    p.require(shutil.disk_usage(ROOT).free>=source_bytes+2*p.GiB,'Insufficient local space for fresh verified stage')
    STAGE.mkdir();REPORT.mkdir(exist_ok=True)
    copied={};source_map={}
    def copy(source,name):
        source=p.regular(source);p.canonical(name);binding={'bytes':source.stat().st_size,'sha256':p.sha256(source)}
        if name in copied:
            p.require(copied[name]==binding,'Content-path collision');return {'path':name,**binding}
        destination=STAGE/name;destination.parent.mkdir(parents=True,exist_ok=True)
        with source.open('rb') as a,destination.open('xb') as b:shutil.copyfileobj(a,b,p.MiB)
        p.require(destination.stat().st_size==binding['bytes'] and p.sha256(destination)==binding['sha256'],'Copied bytes differ')
        copied[name]=binding;source_map[name]={'local_source':str(source.relative_to(ROOT)),**binding}
        return {'path':name,**binding}
    def write(name,value):
        destination=STAGE/name;destination.parent.mkdir(parents=True,exist_ok=True)
        with destination.open('xb') as f:f.write(p.json_bytes(value))
        copied[name]={'bytes':destination.stat().st_size,'sha256':p.sha256(destination)}
    def returned_asset(item):
        source=p.regular(RETURN/item['path'])
        p.require(p.pin(RETURN,source)==item,'Actual returned asset pin differs')
        return copy(source,'assets/'+item['sha256']+source.suffix.lower())
    # Preserve original assets, licenses, terms and split evidence. Runtime code
    # is inert provenance in the new packet, never the active import path.
    for name in parent['files']:
        if name.startswith(('assets/','weights/','third_party/','provenance/')):
            copy(PARENT/name,name)
        elif name=='research/__init__.py':copy(PARENT/name,name)
        else:copy(PARENT/name,'provenance/parent_runtime/'+name)
    for name in SOURCES:copy(ROOT/name,name if name!='scripts/run_spatial_continuation.sh' else 'run_spatial_continuation.sh')
    copy(ROOT/'research/codeformer_spatial_continuation_v1/__init__.py','research/codeformer_spatial_continuation_v1/__init__.py')
    parent_pin=copy(RETURN/'protocol.json','provenance/parent_evidence/protocol.json')
    evidence={key:copy(source,'provenance/parent_evidence/'+key+'.json') for key,(source,_) in EVIDENCE.items()}
    peer_pin=copy(peer_path,'provenance/peer_source_review.json')
    design=REVIEW/'method_review_v1/CONTINUATION_96_DESIGN_REVIEW.json'
    design_pin=copy(design,'provenance/continuation_design_review.json')
    copy(design.with_suffix('.md'),'provenance/continuation_design_review.md')
    resume={};source_guards={r['id']:r for r in parent['guards']}
    for branch in result['branches']:
        arm=branch['arm'];state=branch['last_checkpoint']
        source=p.regular(RETURN/state['path']);p.require(p.pin(RETURN,source)==state,'Own24 checkpoint source differs')
        cp=copy(source,'resume/'+arm+'/u0024.pt')
        stats=next(v['rows'] for v in branch['feature_statistics'] if v['update']==24)
        features=[]
        for row in stats:
            item=deepcopy(row);item['arrays']=returned_asset(row['arrays']);features.append(item)
        parity=[]
        for identifier in parent['parity_ids']:
            source_row=p.read_json(RETURN/f'outputs/spatial_objectives_v1/guards/{arm}/u0024/{identifier}.json')
            # These are actual own24 results, including arm/update and the
            # measured behavior distance. Guard0 remains separately unchanged.
            item=deepcopy(source_row)
            baseline=source_guards[identifier]
            p.require(item['input']==baseline['input'] and item['target']==baseline['target'],
                      'Own24 parity input/target differs from its retained guard input')
            item['input_RGB8_sha256']=baseline['input_RGB8_sha256']
            item['target_RGB8_sha256']=baseline['target_RGB8_sha256']
            item['outputs']={k:returned_asset(v) for k,v in source_row['outputs'].items()}
            parity.append(item)
        digests={row['adapter_state_sha256'] for row in stats};p.require(len(digests)==1,'Own24 state stats digest ambiguity')
        resume[arm]={'checkpoint':cp,'completed_updates':24,'adapter_sha256':next(iter(digests)),
            'parity_rows':parity,'feature_rows':features}
    sampler=[dict(row,update=row['update']+24+24*repeat)
             for repeat in range(3) for row in parent['sampler']]
    spec={k:deepcopy(parent[k]) for k in ('train_rows','clean_features','guards','strong','r2_asset','parity_ids',
        'initial_adapter_sha256','base515_sha256')}
    spec.update(format='codeformer-spatial-continuation-manual-L4-protocol-v1',vm_name=p.VM_NAME,
        parent_protocol=parent_pin,parent_protocol_sha256=PARENT_SHA,parent_evidence=evidence,resume=resume,
        design_sha256=design_pin['sha256'],sampler_original72=parent['sampler'],sampler=sampler,
        start_update=24,max_updates_per_arm=96,additional_updates_per_arm=72,arms=['U','S'],
        guard_updates=[30,36,48,72,96],checkpoint_updates=[24,30,36,48,72,96],
        stats_updates=[24,48,72,96],strong_updates=[48,72,96],max_calls=p.CAPS,limits=p.LIMITS,
        worker_seconds=2700,manual_supervised_maximum_seconds_including_export=3120,
        automatic_launch=False,automatic_resume=False,model_qualified=False,
        full_development_and_FINAL_still_required=True,provenance=[parent_pin,*evidence.values(),peer_pin,design_pin],
        return_omits_only_VM_static_and_observed_cache=True)
    vm_only=parent['vm_only_static']
    spec['allowed_outputs']=p.output_names(spec)
    runtime_cap=sum(output_cap(n) for n in spec['allowed_outputs'])
    final_count=sum(not n.endswith('.partial') for n in spec['allowed_outputs'])
    # Atomic final/partial names share one artifact allowance; no run retry.
    # Up to four retained exceptional siblings are conservatively reserved for
    # worker-failure, failure receipt, supervisor and export metadata.
    payload_proof={'complete':True,'runtime_final_slot_upper_bytes':runtime_cap,
        'additional_retained_partial_upper_bytes':4*2*p.MiB,'protocol_upper_bytes':p.MiB,
        'all_final_runtime_slots_including_cache':final_count,'retained_partial_slots':4,
        'allowed_names_include_mutually_exclusive_partials':len(spec['allowed_outputs']),
        'static_return_bytes_before_proof':sum(v['bytes'] for n,v in copied.items() if n not in vm_only),
        'static_files_before_proof':len(copied),'proof_and_source_map_upper_bytes':p.MiB,
        'partial_reason':'One atomic write at a time. Infrastructure failure stops worker/no resume; only failure/supervisor/export metadata remain. Final and partial for the same slot are not double payload.'}
    payload_proof['payload_upper_bytes']=runtime_cap+payload_proof['static_return_bytes_before_proof']+10*p.MiB
    payload_proof['physical_file_upper']=len(copied)+final_count+4+3
    p.require(payload_proof['payload_upper_bytes']<=p.LIMITS['return_bytes'],'Closed namespace cannot fit512MiB; stop without shrinking data')
    p.require(payload_proof['physical_file_upper']<=p.LIMITS['files'],'Closed physical namespace exceeds2000 files')
    write('payload_namespace_bounds.json',payload_proof)
    write('source_map.json',source_map)
    spec.update(files=copied,vm_only_static=sorted(vm_only),return_static=sorted(set(copied)-set(vm_only)))
    protocol_path=STAGE/'protocol.json'
    with protocol_path.open('xb') as f:f.write(p.json_bytes(spec))
    p.require(protocol_path.stat().st_size<=p.MiB and (STAGE/'source_map.json').stat().st_size+
        (STAGE/'payload_namespace_bounds.json').stat().st_size<=p.MiB,'Metadata proof reservation exceeded')
    verified=p.verify(STAGE,p.sha256(protocol_path));checked=audit.audit_static(STAGE,verified)
    for name in copied:
        if name.endswith('.py'):ast.parse((STAGE/name).read_text(encoding='utf-8'),feature_version=(3,10))
    report={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'stage':str(STAGE),
        'protocol_sha256':p.sha256(protocol_path),'files':len(copied),'payload_proof':payload_proof,
        'static_audit':checked,'objective_fixtures':fixture,'peer_source_receipt':peer_pin,
        'training_unrun':True,'Torch_neural_optimizer_calls':0,'archive_unbuilt':True,'model_qualified':False}
    path=REPORT/'stage_preparation_v1.json'
    with path.open('xb') as f:f.write(p.json_bytes(report))
    print(json.dumps({k:v for k,v in report.items() if k not in ('static_audit','objective_fixtures')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',action='store_true',required=True)
    parser.add_argument('--peer-review',type=Path,required=True);parser.add_argument('--peer-review-sha',required=True)
    args=parser.parse_args();stage_packet(args.peer_review,args.peer_review_sha)
