"""Build a source-only manual packet using exact read-only AdaIN parent bindings."""
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_loss_response_v1 import contract as c

REVIEW = ROOT / 'outputs/codeformer_loss_response_design_v1'
PARENT_LOCAL = ROOT / 'outputs/cctv_codeformer_adain_gradient_isolation_v1_vm'
IMPORTED = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return'
AUDIT = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review'

def write_new(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', default='', help='Preserve an earlier stage; use a new rN local preparation revision')
    args = parser.parse_args()
    c.require(args.revision == '' or re.fullmatch(r'r[1-9][0-9]*', args.revision), 'Revision must be rN')
    suffix = '_' + args.revision if args.revision else ''
    STAGE = ROOT / 'outputs' / (c.VM_NAME + suffix)
    receipt = REVIEW / ('stage_v1' + suffix + '.json')
    c.require(not STAGE.exists(), 'Preserve previous stage; do not replace')
    c.require(not receipt.exists(), 'Preserve previous stage receipt; do not replace')
    parent = c.read_json(PARENT_LOCAL / 'protocol.json')
    c.require(c.sha256(PARENT_LOCAL / 'protocol.json') == c.PARENT_PROTOCOL_SHA, 'Frozen parent protocol differs')
    transport = c.read_json(AUDIT / 'transport_v1.json')
    c.require(transport['complete'] and transport['protocol_sha256'] == c.PARENT_PROTOCOL_SHA,
              'Closed parent transport required')
    resolved = transport['immutable_sources_and_assets_freshly_verified']
    dependencies = {}
    local = {}
    def add(name, path, digest, size=None):
        path = c.no_links(path)
        c.safe_relative(name)
        c.require(path.is_file() and c.sha256(path) == digest, 'Fresh dependency bytes differ: ' + name)
        size = path.stat().st_size if size is None else size
        c.require(path.stat().st_size == size, 'Dependency size differs: ' + name)
        pin = {'sha256': digest, 'bytes': size}
        c.require(name not in dependencies or dependencies[name] == pin, 'Conflicting dependency identity')
        dependencies[name], local[name] = pin, str(path)
        return pin
    for name, digest in parent['files_sha256'].items():
        row = resolved[name]
        c.require(row['sha256'] == digest, 'Frozen resolution map differs')
        add(name, Path(row['resolved_path']), digest, row['bytes'])
    add('protocol.json', PARENT_LOCAL / 'protocol.json', c.PARENT_PROTOCOL_SHA)
    run = 'outputs/adain_gradient_isolation_v1/'
    def add_return(name):
        c.require(name in transport['files_sha256'], 'Unbound parent result')
        return add(name, IMPORTED / name, transport['files_sha256'][name], transport['files_bytes'][name])
    history_name = run + 'training.jsonl'
    add_return(history_name)
    history = [json.loads(line) for line in (IMPORTED / history_name).read_text(encoding='utf-8').splitlines()]
    c.require([r['update'] for r in history] == list(range(1,51)), 'Exact50-update parent history required')
    rows = []
    for update in c.SOURCE_UPDATES:
        old = history[update-1]
        c.require(old['optimizer_committed'] is True, 'Historical update incomplete')
        for item in old['microbatches']:
            record = item['pair_record']
            row = {'id': 'u%04d_s%d' % (update,item['slot']), 'update':update, 'slot':item['slot'],
                   'reference_id':item['id'], 'profile':item['profile'], 'parameters':record['parameters'],
                   'record':record, 'input_path':record['saved_input_path'],
                   'pair_record_path':run+'training_inputs/update_%04d/slot_%d/pair_record.json' % (update,item['slot']),
                   'source_path':record['source_path']}
            for key in ('input_path','pair_record_path'):
                add_return(row[key])
            c.require(c.read_json(IMPORTED / row['pair_record_path']) == record, 'Case record/history differs')
            c.require(dependencies[row['input_path']]['sha256'] == record['saved_input_sha256'] and
                      dependencies[row['source_path']]['sha256'] == record['source_sha256'], 'Case pixel source differs')
            hq = run+'HQ_cache/'+item['id']+'/'
            for key, name in [('hq_indices_path','indices.npy'), ('hq_features_path','features.npy'), ('hq_record_path','record.json')]:
                row[key] = hq+name; add_return(row[key])
            if item['slot'] == 3:
                anchor = run+'behavior_anchors/update_%04d/' % update
                for key,name in [('behavior_delivered_path','delivered256.npy'), ('behavior_logits_path','logits.npy'), ('behavior_record_path','record.json')]:
                    row[key] = anchor+name; add_return(row[key])
            rows.append(row)
    cases = {'format':c.CASES_FORMAT,'source_updates':c.SOURCE_UPDATES,'cases':rows,
             'source_and_probe_are_identical_TRAIN_cases':True,'new_input_policy':False,
             'DEV_or_FINAL_cases':0,'model_qualified':False}
    c.verify_cases(cases)
    assets = dict(parent['assets'])
    for name,path,digest in [
        ('original_checkpoint',run+'snapshot_0000/checkpoint.pt','657fc55862143c66f7e6d1ba5d1875e87d423b78e53bee8f1a4376baa345c535'),
        ('stopped_checkpoint',run+'stopped_checkpoint.pt','43c0f313365e13849ddcd852f401511e4c7ae52cbac55fdd98a6b032f06ef37b')]:
        pin = add_return(path)
        c.require(pin['sha256'] == digest, 'Original/stopped checkpoint fingerprint differs')
        assets[name] = {'path':path,**pin}
    STAGE.mkdir()
    c.write_new_json(STAGE / 'cases.json',cases)
    source_names = ['research/codeformer_loss_response_v1/'+name for name in
                    ('__init__.py','contract.py','worker.py','transport.py')]
    source_names += ['scripts/supervise_codeformer_loss_response_v1.py']
    for name in source_names:
        target = STAGE / name
        write_new(target,(ROOT/name).read_bytes())
    write_new(STAGE/'research/__init__.py',b'')
    for name in ('AGENTS.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
                 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md','PROJECT_HANDOFF.md',
                 'CCTV_CODEFORMER_LOSS_RESPONSE_V1_DESIGN.md'):
        write_new(STAGE/'instructions'/name,(ROOT/name).read_bytes())
    for name in ('loss_contract_review.md','train_pixels_findings_v1.md',
                 'train_pixels_finite_design_review_v1.md'):
        write_new(STAGE/'provenance'/name,(REVIEW/name).read_bytes())
    files = sorted(p for p in STAGE.rglob('*') if p.is_file())
    own = {p.relative_to(STAGE).as_posix():c.sha256(p) for p in files}
    sizes = {p.relative_to(STAGE).as_posix():p.stat().st_size for p in files}
    p = {'format':c.FORMAT,'seed':20261015,'outputs':'outputs/loss_response_v1',
         'parent_directory':c.PARENT_NAME,'parent_protocol_sha256':c.PARENT_PROTOCOL_SHA,
         'source_updates':c.SOURCE_UPDATES,'cases_file':'cases.json',
         'snapshots':['original0','stopped50'],'branches':c.BRANCHES,'loss_weights':c.WEIGHTS,
         'optimizer':c.OPTIMIZER,'numerical':c.NUMERICAL,'limits':c.LIMITS,'call_caps':c.COUNTS,
         'total_neural_forwards':425,'total_gradient_queries':193,
         'files_sha256':own,'files_bytes':sizes,'dependencies':dependencies,'assets':assets,
         'automatic_app_promotion':False,'model_qualified':False,'new_training_recipe':False,
         'export_candidate_checkpoint':False,'manual_tmux_required':True,
         'status':'PREPARED_UNRUN','all_branches_disposable':True,'quality_gates_changed':False}
    c.verify_protocol(p)
    c.write_new_json(STAGE/'protocol.json',p)
    c.write_new_json(receipt,{'complete':True,'stage':str(STAGE),'protocol_sha256':c.sha256(STAGE/'protocol.json'),
        'source_files':own,'source_bytes':sizes,'dependencies':dependencies,'local_dependency_resolution':local,
        'dependencies_freshly_hashed':len(dependencies),'source_cases':16,'image_eligible':12,
        'source_sha256':c.sha256(__file__),'UTC':datetime.now(timezone.utc).isoformat(),
        'actual_neural_gradient_optimizer_VM_calls':0,'execution_archive_sealed':False,'model_qualified':False,
        'local_preparation_revision':args.revision or 'original', 'canonical_VM_directory':c.VM_NAME})
    print(json.dumps({'complete':True,'static_files':len(own)+1,'dependencies':len(dependencies),
                     'protocol_sha256':c.sha256(STAGE/'protocol.json'),'neural_or_VM_calls':0}))

if __name__ == '__main__':
    main()
