"""CPU-only new source stage; never connect to a VM, launch training, or seal."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.codeformer_normal_tone_training_v1 import contract as c


def prepare(stage,receipt):
    stage=c.no_links(stage);receipt=c.no_links(receipt)
    c.require(stage.parent.resolve()==(ROOT/'outputs').resolve() and stage.name==c.VM_NAME,
              'Exact new canonical local stage required')
    c.require(not stage.exists() and not receipt.exists(),'Preserve prior/partial stage and receipt')
    old=ROOT/'outputs/codeformer_loss_response_v1_return'
    c.require(c.sha256(old/'protocol.json')=='23c19c394e34261861b8853cfd60aa60d1cfaefc91b12e92a2f8b8dce85112ab',
              'Pinned prior dependency manifest source required')
    prior=c.read_json(old/'protocol.json')
    resolved=c.read_json(ROOT/'outputs/codeformer_loss_response_design_v1/stage_v1_r1.json')
    dependencies=deepcopy(prior['dependencies'])
    c.require(dependencies==resolved['dependencies'],'Prior dependency namespace changed')
    local={}
    for name,pin in sorted(dependencies.items()):
        path=c.no_links(Path(resolved['local_dependency_resolution'][name]))
        c.require(path.resolve(strict=True).is_relative_to(ROOT),'Dependency resolution escapes workspace')
        before=path.stat()
        c.require(path.is_file() and before.st_size==pin['bytes'] and c.sha256(path)==pin['sha256'],
                  'Fresh parent dependency differs: '+name)
        after=path.stat()
        c.require((before.st_size,before.st_mtime_ns,before.st_ino)==
                  (after.st_size,after.st_mtime_ns,after.st_ino),'Dependency changed while hashing')
        local[name]=str(path)
    parent=c.read_json(local['protocol.json'])
    c.require(c.sha256(local['protocol.json'])==c.PARENT_PROTOCOL_SHA and
              all(dependencies[n]['sha256']==v for n,v in parent['files_sha256'].items()),
              'Entire1939immutable parent namespace required')
    sources=['research/codeformer_normal_tone_training_v1/'+n for n in
             ('__init__.py','contract.py','transport.py','worker.py','model.py')]
    sources+=['scripts/supervise_codeformer_normal_tone_training_v1.py']
    instructions=['AGENTS.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
                  'PROJECT_HANDOFF.md','CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
                  'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md']
    payloads={n:c.safe_path(ROOT,n).read_bytes() for n in sources}
    current_sources={n:n for n in sources}
    payloads.update({'instructions/'+n:c.safe_path(ROOT,n).read_bytes() for n in instructions})
    current_sources.update({'instructions/'+n:n for n in instructions})
    # Independent evidence is immutable source provenance, never an executing cache.
    provenance=['outputs/codeformer_normal_tone_v1_return_review/independent_transport_v1.md',
                'outputs/codeformer_normal_tone_v1_return_review/independent_transport_v1.json',
                'outputs/codeformer_normal_tone_v1_return_review/source_root_cause_v1.md',
                'outputs/codeformer_normal_tone_v1_return_review/source_root_cause_v1.json',
                'outputs/codeformer_normal_tone_training_v1_design/independent_math_review_v1.md',
                'outputs/codeformer_normal_tone_training_v1_design/independent_math_review_v1.json',
                'outputs/codeformer_normal_tone_training_v1_preparation/independent_source_review_v1.md',
                'outputs/codeformer_normal_tone_training_v1_preparation/independent_source_review_v1.json']
    for name in provenance:
        target='provenance/'+Path(name).name
        payloads[target]=c.safe_path(ROOT,name).read_bytes();current_sources[target]=name
    tests=['test_codeformer_normal_tone_training_transport_v1.py',
           'test_codeformer_normal_tone_training_math_v1.py',
           'test_codeformer_normal_tone_training_worker_v1.py','test_codeformer_normal_tone_math_v1.py']
    for name in tests:
        payloads['tests/'+name]=c.safe_path(ROOT,'tests/'+name).read_bytes()
        current_sources['tests/'+name]='tests/'+name
    payloads['research/__init__.py']=b''
    own={name:__import__('hashlib').sha256(data).hexdigest() for name,data in payloads.items()}
    sizes={name:len(data) for name,data in payloads.items()}
    design='instructions/CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_DESIGN.md'
    protocol={'format':c.FORMAT,'seed':20261015,'data_manifest':'references.json',
        'comparison_cases':'comparison_cases.json','outputs':c.OUTPUTS,'export_prefix':c.PREFIX,
        'parent_dir':c.PARENT_NAME,'parent_protocol_sha256':c.PARENT_PROTOCOL_SHA,
        'training':deepcopy(c.TRAINING),'evaluation':deepcopy(c.EVALUATION),'limits':deepcopy(c.LIMITS),
        'gradient_route':deepcopy(c.GRADIENT_ROUTE),'objective':deepcopy(c.OBJECTIVE),
        'normal_formula_prerequisite':deepcopy(c.NORMAL_FORMULA_PREREQUISITE),
        'assets':deepcopy(c.ASSETS),**deepcopy(c.PARENT_CONTROLS),
        'dependencies':dependencies,'files_sha256':own,'files_bytes':sizes,
        'design':{'path':design,'sha256':own[design]},'status':'PREPARED_UNRUN',
        'manual_tmux_required':True,'new_training_recipe':True,'actual_candidate_checkpoints_required':True,
        'model_qualified':False,'automatic_app_promotion':False,'automatic_training':False,
        'quality_gates_changed':False,'resume_authorized':False,'FINAL_used':False,'cached_gradient_reuse':False}
    c.verify_protocol(protocol)
    # No destination or packet exists before every required input has been read.
    stage.mkdir()
    for name,data in payloads.items():
        path=stage/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
    c.write_new_json(stage/'protocol.json',protocol)
    c.write_new_json(receipt,{'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),
        'stage':str(stage),'protocol_sha256':c.sha256(stage/'protocol.json'),
        'source_files':own,'source_bytes':sizes,'dependencies':dependencies,
        'current_sources':current_sources,
        'local_dependency_resolution':local,'dependency_count':len(dependencies),
        'dependency_bytes':sum(v['bytes'] for v in dependencies.values()),
        'source_sha256':c.sha256(__file__),'source_stage_not_sealed':True,
        'local_neural_gradient_optimizer_VM_calls':0,'model_qualified':False})
    return {'complete':True,'protocol_sha256':c.sha256(stage/'protocol.json'),'static_files':len(own)+1,
            'dependency_count':len(dependencies),'stage_only_not_sealed':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,default=ROOT/'outputs'/c.VM_NAME)
    parser.add_argument('--receipt',type=Path,
        default=ROOT/'outputs/codeformer_normal_tone_training_v1_preparation/stage_v1.json')
    args=parser.parse_args();print(prepare(args.stage,args.receipt))


if __name__=='__main__':main()
