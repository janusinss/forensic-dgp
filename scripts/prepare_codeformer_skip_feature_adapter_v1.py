"""Build a self-contained, inert, new manual L4 packet; never launch training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_skip_feature_adapter_v1 import data as d

VM_NAME = 'cctv_codeformer_skip_feature_adapter_v1_vm'
STAGE = ROOT / 'outputs' / VM_NAME
PREP = ROOT / 'outputs/codeformer_skip_feature_adapter_v1_preparation'
OLD = ROOT / 'outputs/codeformer_filter_fusion_training_v1_return'
ORACLE = ROOT / 'outputs/codeformer_hq_code_oracle_inputs_20261019_v1'
PROPOSAL = ROOT / 'outputs/codeformer_skip_adapter_pilot_design_20261019_v1/dependency_and_finite_proposal_v1.json'
SOURCE_NAMES = ['research/__init__.py'] + [
    'research/codeformer_skip_feature_adapter_v1/' + n + '.py'
    for n in ('__init__', 'data', 'model', 'metrics', 'transport', 'worker')]
SOURCE_NAMES += ['scripts/supervise_codeformer_skip_feature_adapter_v1_vm.py',
    'scripts/prepare_codeformer_skip_feature_adapter_v1.py']
ARCH = ['third_party/__init__.py', 'third_party/codeformer/__init__.py',
    'third_party/codeformer/codeformer_arch.py', 'third_party/codeformer/vqgan_arch.py',
    'third_party/codeformer/registry.py', 'third_party/codeformer/LICENSE', 'third_party/codeformer/NOTICE.md']
WEIGHTS = {
    'weights/codeformer_original.pth': ROOT / 'outputs/codeformer_restoration_pretrained_v1/codeformer.pth',
    'weights/codeformer_r2.pth': ROOT / 'outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth',
    'weights/codeformer_parent.pth': OLD / 'weights/codeformer_parent.pth'}


def require(value, message):
    if not value:
        raise ValueError(message)


def payload(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'


def pin(path, allow_original=False):
    path = Path(path).absolute()
    require(path.resolve(strict=True).is_relative_to(ROOT), 'Source must remain in workspace')
    for parent in (path, *path.parents):
        st = parent.lstat()
        require(not stat.S_ISLNK(st.st_mode) and not getattr(st, 'st_file_attributes', 0) & 1024,
                'Linked/reparse source refused')
    before = path.stat()
    require(stat.S_ISREG(before.st_mode) and (before.st_nlink == 1 or
            (allow_original and path == WEIGHTS['weights/codeformer_original.pth'].absolute() and before.st_nlink == 3)),
            'Unique ordinary source or exact retained original hardlink required')
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    after = path.stat()
    require((before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during hash')
    return {'bytes': before.st_size, 'sha256': digest}


def read(path):
    return d.read_json(path)


def plan():
    from research.codeformer_skip_feature_adapter_v1 import transport as t, worker as w, model as m
    proposal = read(PROPOSAL)
    require(proposal['complete'] is True, 'Completed predecessor proposal required')
    for key in ('data_bundle.json', 'evaluation_bundle.json', 'data_schedule_v2.json', 'data_admission_v2.json'):
        rel = (OLD / key).relative_to(ROOT).as_posix()
        require(pin(OLD / key) == proposal['source_pins'][rel], 'Frozen predecessor metadata differs')
    old_bundle = read(OLD / 'data_bundle.json'); evaluation = read(OLD / 'evaluation_bundle.json')
    old_schedule = read(OLD / 'data_schedule_v2.json'); old_admission = read(OLD / 'data_admission_v2.json')
    order = old_schedule['anchor_source_order']; schedule = d.schedule_for(order)
    pins, copies, generated = {}, {}, {}

    def add(source, name, expected=None, original=False):
        d.canonical(name); p = pin(source, original)
        require(expected is None or p == expected, 'Physical source pin differs: ' + name)
        if name in pins:
            require(pins[name] == p, 'Conflicting content-addressed path'); return {'path': name, **p}
        pins[name] = p; copies[name] = str(Path(source).relative_to(ROOT)).replace('\\', '/')
        return {'path': name, **p}

    def new(name, value):
        d.canonical(name); require(name not in pins, 'Duplicate new packet file')
        b = value if type(value) is bytes else payload(value)
        pins[name] = {'bytes': len(b), 'sha256': hashlib.sha256(b).hexdigest()}; generated[name] = b

    for name in SOURCE_NAMES + ARCH:
        add(ROOT / name, name)
    for name, source in WEIGHTS.items():
        add(source, name, {k: m.ASSETS[{'weights/codeformer_original.pth':'original',
            'weights/codeformer_r2.pth':'r2', 'weights/codeformer_parent.pth':'parent'}[name]][k]
            for k in ('bytes', 'sha256')}, original=name.endswith('original.pth'))

    def operational(row, source_root):
        out = copy.deepcopy(row)
        for key in ('input', 'target', 'target256', 'support', 'corruption_footprint'):
            if out.get(key) is not None:
                p = out[key]
                out[key] = add(source_root / p['path'], 'data/pixels/' + p['sha256'] + '.png',
                               {k:p[k] for k in ('bytes', 'sha256')})
        return out

    role_alias = {'filter_a':'filter', 'filter_b':'filter', 'clear':'clear', 'dark':'dark'}
    require(all(r['role'] in role_alias for r in old_admission['admitted_role_cases']),
            'Known historical two-filter-slot admission roles required')
    prior_allowed = {(role_alias[r['role']], r['id']) for r in old_admission['admitted_role_cases']}
    cases, admitted = {}, []
    for rid in order:
        for role, profile in d.PROFILES.items():
            cid = rid + '__' + profile
            require((role, cid) in prior_allowed, 'New role lacks retained input-only admission')
            row = old_bundle['case_by_id'][cid]
            require(row['split'] == 'TRAIN' and row['reference_id'] == rid and
                    rid not in d.HELD, 'TRAIN/source exclusion differs')
            out = operational(row, OLD); out['active_role'] = role
            out['new_pilot_explicit_admission'] = 'data_admission.json'
            out['historical_optimization_flags_unmodified'] = True
            cases[cid] = out; admitted.append({'id':cid, 'reference_id':rid, 'role':role})
    new('data_bundle.json', {'format':'codeformer-skip-feature-training-bundle-v1', 'complete':True,
        'case_by_id':cases, 'source_order':order, 'sources_not_proven_unique_people':True,
        'historical411_roles_in_provenance_only':True, 'old_flags_unmodified':True})
    new('data_schedule.json', schedule)
    new('data_admission.json', {'format':'codeformer-skip-feature-explicit-admission-v1',
        'complete':True, 'root_approved':True, 'admitted_role_cases':admitted,
        'retained_admission':{'path':'provenance/previous_data_admission_v2.json', **pin(OLD/'data_admission_v2.json')},
        'old_flags_unmodified':True, 'held_sources':list(d.HELD), 'automatic_launch_authorized':False,
        'historical_role_aliases_for_subset_join':role_alias,
        'no_new_input_rendering_or_optimizing_auxiliary_strong_cases':True})
    eval_copy = copy.deepcopy(evaluation)
    eval_copy['cases'] = [operational(r, OLD) for r in evaluation['cases']]
    eval_copy['refused_controls'] = [operational(r, OLD) for r in evaluation['refused_controls']]
    new('evaluation_bundle.json', eval_copy)
    manifest = read(ORACLE/'manifest_v1.json'); review = read(ORACLE/'root_input_review_v1_r1.json')
    require(pin(ORACLE/'manifest_v1.json')['sha256'] == 'cdb70da025284909c96144fd1884464cf89fea6c7f4167e9c5d99bd834dbc2ed',
            'Frozen auxiliary manifest differs')
    labels = {r['id']:r['input_only_label'] for r in review['cases']}; aux = []
    for row in manifest['cases']:
        out = {'id':row['id'], 'reference_id':row['reference_id'], 'profile':row['profile'],
            'input':row['input'], 'input_RGB8_sha256':row['input_RGB8_sha256'],
            'target256':row['metrics_target256'],
            'target256_pixels':{'RGB8_sha256':row['metrics_target256']['RGB8_sha256']},
            'evidence':'paired_synthetic_TRAIN', 'split':'TRAIN', 'source':'six historical TRAIN photographs',
            'cohort':'auxiliary18_TRAIN_diagnostic_not_independent_holdout',
            'input_only_decision':labels[row['id']], 'optimization_permitted_now':False}
        aux.append(operational(out, ROOT))
    new('auxiliary_bundle.json', {'complete':True, 'cases':aux, 'not_independent_holdout':True,
        'optimization_permitted_now':False, 'no_FINAL_or_native_inputs':True})
    guards = [{'role':role, 'reference_id':rid, 'id':rid+'__'+d.PROFILES[role]}
        for role in d.PROFILES for rid in order[:8]]
    witnesses = {role:order[0]+'__'+profile for role,profile in d.PROFILES.items()}
    for name in ('data_bundle.json','evaluation_bundle.json','data_schedule_v2.json','data_admission_v2.json',
                 'root_dark_linear_input_review_v2.json'):
        add(OLD/name, 'provenance/previous_'+name)
    for source in sorted((OLD/'provenance').rglob('*')):
        if source.is_file():
            add(source, 'provenance/retained_filter_fusion/'+source.relative_to(OLD/'provenance').as_posix())
    for name in ('manifest_v1.json','root_input_review_v1_r1.json'):
        add(ORACLE/name, 'provenance/auxiliary_'+name)
    for name in ('PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
                 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md','AGENTS.md','CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md'):
        add(ROOT/name, 'provenance/current_instructions/'+name)
    add(PROPOSAL, 'provenance/earlier_proposal_only.json')
    new('run_skip_adapter.sh', b'#!/usr/bin/env bash\nset -uo pipefail\ncd "$(dirname "$0")" || exit 1\nexec python -B -u scripts/supervise_codeformer_skip_feature_adapter_v1_vm.py --root . --protocol-sha "$1"\n')
    new('provenance/NEW_DESIGN_DECISIONS.md', b'''# New finite design decisions\n\nThis packet follows the completed clean-SFT oracle, not a repeated fusion-only recipe.\nAll515 original R2 tensors are frozen; four new width32 residual translators add\n24 tensors/51,328 parameters after the stock detach. Initialization is identity.\nUse all55 admitted TRAIN photos:55 Gaussian2.4/55 clear/55 linear-sRGB dark controls.\n110 Adam updates/440 microbatches; blur each4 times, clear/dark each2 times.\nFeature loss0.1 times mean of four site L1 errors normalized by each source's fixed\nclear feature RMS clamped1e-6. Plus paired RGB L1 and0.25 adjacent-residual L1.\nClear original-CodeFormer behavior2x and R2 dark behavior2x; divide each slot by4.\nAdam LR1e-4 linear10-step warmup, betas(.9,.99),eps1e-8,weightdecay0,clipnorm1,seed20261019.\nThis changes the earlier PROPOSAL's three queries to FOUR BEFORE any training:\nblur total + isolated image-only on one graph, clear total, dark total.\nThe isolated image query must prove connected finite nonzero last-projection\ngradients at every site. Feature loss alone cannot prove decoder connectivity.\n110 is finite updates, not epochs. All478 comparison slots remain at0/55/110.\nAt55 report unchanged qualification gates; enforce additional no-new-regression\ninterim check plus both dark requirements. At110 require original full gates.\nInitial inherited failures are explicit and never count as qualification.\nAuxiliary18 remain repeated TRAIN diagnostic only, with no optimization of strong\ninputs, no identity claim and no independent holdout claim. Four readable strong\ncases remain separate from two borderline stress observations.\nNo native clean truth, new ethnic/Zamboanga claim, automatic training or resume.\n''')
    protocol = {'format':'codeformer-skip-feature-adapter-finite-protocol-v1', 'vm_name':VM_NAME,
        'outputs':'outputs/skip_feature_adapter_v1', 'files':pins, 'max_updates':110,
        'snapshots':[0,55,110], 'guard_snapshots':[0,20,55,110], 'call_caps':w.call_caps(),
        'data':{'bundle_path':'data_bundle.json','schedule_path':'data_schedule.json','admission_path':'data_admission.json'},
        'evaluation_path':'evaluation_bundle.json','auxiliary_path':'auxiliary_bundle.json',
        'guard_cases':guards, 'witness_cases':witnesses, 'source_order':order,
        'weight_assets':{row['path']:{k:row[k] for k in ('bytes','sha256')} for row in m.ASSETS.values()},
        'automatic_resume':False, 'automatic_training_launch':False, 'dependencies':{},
        'gradient_queries':['filter_total','filter_image_only','clear_total','dark_total'],
        'limits':dict(t.LIMITS), 'stages':dict(w.STAGES)}
    supervision = ['outputs/supervision/'+n for n in ('preflight.log','worker.log',
        'run_receipt.json','export_invocation.json','export.log','finished.json')]
    outputs = w.planned_output_names(protocol, eval_copy['cases'], cases, aux)
    protocol['allowed_output_names'] = sorted(set(outputs+supervision))
    optional = {'outputs/skip_feature_adapter_v1/'+n for n in
        ('failure.json','stopped_checkpoint.pt','export_manifest.json')}
    protocol['required_success_output_names'] = sorted(set(outputs)-optional)
    w.validate_plan(protocol)
    protocol_bytes = payload(protocol)
    return {'pins':pins, 'copies':copies, 'generated':generated, 'protocol':protocol,
            'protocol_bytes':protocol_bytes, 'sha256':hashlib.sha256(protocol_bytes).hexdigest()}


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--stage',action='store_true')
    args=parser.parse_args(); p=plan()
    summary={'complete':True,'prospective':True,'files':len(p['pins'])+1,
        'bytes':sum(v['bytes'] for v in p['pins'].values())+len(p['protocol_bytes']),
        'protocol_sha256':p['sha256'],'actual_neural_gradient_optimizer_VM_calls':0,'stage_created':False}
    if not args.stage:
        print(json.dumps(summary));return
    require(not STAGE.exists() and not (PREP/'packet_stage_v1.json').exists(), 'Preserve previous/partial stage')
    require(shutil.disk_usage(ROOT).free >= summary['bytes']+4*1024**3, 'FourGiB local reserve required')
    STAGE.mkdir()
    for name, expected in sorted(p['pins'].items()):
        dest=STAGE.joinpath(*d.canonical(name).parts);dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as out:
            if name in p['generated']:out.write(p['generated'][name])
            else:
                source=ROOT/p['copies'][name]; before=source.stat()
                with source.open('rb') as inp:shutil.copyfileobj(inp,out,4*1024**2)
                after=source.stat()
                require((before.st_ino,before.st_size,before.st_mtime_ns)==
                        (after.st_ino,after.st_size,after.st_mtime_ns),'Source changed during copy')
            out.flush();os.fsync(out.fileno())
        require(pin(dest)==expected,'Physical staged file differs')
    with (STAGE/'protocol.json').open('xb') as out:out.write(p['protocol_bytes']);out.flush();os.fsync(out.fileno())
    from research.codeformer_skip_feature_adapter_v1 import transport as t, worker as w
    verified=t.verify_transfer(STAGE,p['sha256']);w.validate_plan(verified['protocol']);d._local_metadata(STAGE,p['protocol'])
    summary.update(stage_created=True,stage=str(STAGE),static_transfer_verified=True,
        actual_stage_pin_map={**p['pins'],'protocol.json':pin(STAGE/'protocol.json')},local_source_map=p['copies'])
    PREP.mkdir(parents=True,exist_ok=True)
    with (PREP/'packet_stage_v1.json').open('xb') as out:out.write(payload(summary))
    print(json.dumps({k:v for k,v in summary.items() if k not in ('actual_stage_pin_map','local_source_map')}))


if __name__=='__main__':main()
