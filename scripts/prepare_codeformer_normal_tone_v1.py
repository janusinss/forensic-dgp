"""Prepare a source-only stage after review; never connect to/run a VM or seal."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.codeformer_normal_tone_v1 import contract as c


def prepare(stage, review):
    stage=c.no_links(stage);review=c.no_links(review)
    c.require(stage.parent.resolve()==(ROOT/'outputs').resolve() and stage.name==c.VM_NAME,
              'Stage must use exact new canonical name in workspace outputs')
    c.require(not stage.exists(),'Preserve existing stage; use a separately reviewed revision')
    c.require(not review.exists(),'Preserve prior staging receipt')
    old_return=ROOT/'outputs/codeformer_loss_response_v1_return'
    imported=c.read_json(ROOT/'outputs/codeformer_loss_response_v1_return_review/return_import_v1.json')
    c.require(imported['complete'] is True and imported['protocol_sha256']==c.ROOTS['loss_response']['protocol_sha256'],
              'Verified prior loss-response import required')
    old_protocol=c.read_json(old_return/'protocol.json')
    old_manifest=c.read_json(old_return/'outputs/loss_response_v1/export_manifest.json')
    c.require(c.sha256(old_return/'outputs/loss_response_v1/export_manifest.json')==c.LOSS_RESPONSE_RETURN_MANIFEST_SHA,
              'Exact completed prior loss-response return manifest required')
    previous=c.read_json(ROOT/'outputs/codeformer_loss_response_design_v1/stage_v1_r1.json')
    c.require(c.sha256(old_return/'protocol.json')==c.ROOTS['loss_response']['protocol_sha256'] and
              old_protocol['dependencies']==previous['dependencies'],'Prior frozen parent identity differs')
    dependencies={'adain':{},'loss_response':{}}; local={'adain':{},'loss_response':{}}
    def bind(alias,name,path,expected):
        c.safe_relative(name);path=c.no_links(path);before=path.stat()
        c.require(path.is_file() and before.st_size==expected['bytes'] and c.sha256(path)==expected['sha256'],
                  'Fresh dependency differs: '+alias+'/'+name)
        after=path.stat()
        c.require((before.st_size,before.st_mtime_ns,before.st_ino)==
                  (after.st_size,after.st_mtime_ns,after.st_ino),'Dependency changed while hashing')
        dependencies[alias][name]=expected;local[alias][name]=str(path)
    for name,pin in old_protocol['dependencies'].items():
        bind('adain',name,Path(previous['local_dependency_resolution'][name]),pin)
    for name in sorted(c.required_loss_response_paths(old_protocol,old_manifest)):
        bind('loss_response',name,old_return/name,
             {'sha256':imported['files_sha256'][name],'bytes':imported['files_bytes'][name]})
    cases=c.read_json(old_return/'cases.json');c.verify_cases(cases)
    own_sources=['research/codeformer_normal_tone_v1/'+name for name in
                 ('__init__.py','contract.py','transport.py','worker.py')]
    own_sources+=['scripts/supervise_codeformer_normal_tone_v1.py']
    instructions=['AGENTS.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
                  'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md','PROJECT_HANDOFF.md',
                  'CCTV_CODEFORMER_NORMAL_TONE_V1_DESIGN.md']
    # Read every source before creating the stage. This builder never seals it.
    payloads={name:c.safe_path(ROOT,name).read_bytes() for name in own_sources}
    payloads.update({'instructions/'+name:c.safe_path(ROOT,name).read_bytes() for name in instructions})
    for relative in ('outputs/codeformer_normal_tone_v1_design/source_design_review_v1.md',
                     'outputs/codeformer_normal_tone_v1_design/source_design_review_v1.json',
                     'outputs/codeformer_loss_response_v1_return_review/next_design_independent_review_v1.md',
                     'outputs/codeformer_loss_response_v1_return_review/next_design_independent_review_v1.json'):
        payloads['provenance/'+Path(relative).name]=c.safe_path(ROOT,relative).read_bytes()
    for name in ('test_codeformer_normal_tone_math_v1.py','test_codeformer_normal_tone_transport_v1.py'):
        payloads['tests/'+name]=c.safe_path(ROOT,'tests/'+name).read_bytes()
    payloads['research/__init__.py']=b''
    payloads['cases.json']=c.safe_path(old_return,'cases.json').read_bytes()
    stage.mkdir()
    for name,payload in payloads.items():
        target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('xb') as stream:stream.write(payload)
    own={name:c.sha256(stage/name) for name in payloads};sizes={name:len(p) for name,p in payloads.items()}
    assets={name:{'root':'adain',**value} for name,value in old_protocol['assets'].items()}
    p={'format':c.FORMAT,'seed':20261015,'outputs':'outputs/normal_tone_v1','parent_roots':c.ROOTS,
       'source_updates':c.SOURCE_UPDATES,'cases_file':'cases.json','snapshots':['original0','stopped50'],
       'branches':c.BRANCHES,'loss_weights':c.WEIGHTS,'optimizer':c.OPTIMIZER,'numerical':c.NUMERICAL,
       'objective':c.OBJECTIVE,
       'limits':c.LIMITS,'call_caps':c.COUNTS,'total_neural_forwards':224,'total_gradient_queries':48,
       'files_sha256':own,'files_bytes':sizes,'dependencies':dependencies,'assets':assets,
       'cached_basis':c.cached_basis_paths(),'prior_controls':c.prior_controls(),
       'automatic_app_promotion':False,'model_qualified':False,'new_training_recipe':False,
       'export_candidate_checkpoint':False,'manual_tmux_required':True,'all_branches_disposable':True,
       'quality_gates_changed':False,'status':'PREPARED_UNRUN'}
    c.verify_protocol(p);c.write_new_json(stage/'protocol.json',p)
    c.write_new_json(review,{'complete':True,'stage':str(stage),'protocol_sha256':c.sha256(stage/'protocol.json'),
       'source_files':own,'source_bytes':sizes,'dependencies':dependencies,'local_dependency_resolution':local,
       'parent_roots':c.ROOTS,'dependency_counts':{k:len(v) for k,v in dependencies.items()},
       'dependency_bytes':{k:sum(x['bytes'] for x in v.values()) for k,v in dependencies.items()},
       'source_sha256':c.sha256(__file__),'UTC':datetime.now(timezone.utc).isoformat(),
       'source_cases':16,'actual_neural_gradient_optimizer_VM_calls':0,
       'execution_archive_sealed':False,'model_qualified':False})
    return {'complete':True,'protocol_sha256':c.sha256(stage/'protocol.json'),
            'static_files':len(own)+1,'dependency_counts':{k:len(v) for k,v in dependencies.items()},
            'stage_only_not_sealed':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',type=Path,default=ROOT/'outputs'/c.VM_NAME)
    parser.add_argument('--receipt',type=Path,default=ROOT/'outputs/codeformer_normal_tone_v1_preparation/stage_v1.json')
    args=parser.parse_args()
    c.require(args.stage.parent.resolve()==(ROOT/'outputs').resolve() and args.stage.name==c.VM_NAME,
              'Stage must use exact new canonical name in workspace outputs')
    c.require(not args.receipt.exists(),'Preserve prior staging receipt')
    print(prepare(args.stage,args.receipt))


if __name__=='__main__':main()
