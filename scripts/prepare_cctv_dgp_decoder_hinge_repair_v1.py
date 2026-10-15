"""Stage and seal a human-manual hinge correction; no model or VM execution."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile

from cctv_dgp_decoder_hinge_repair_v1_contract import (
    NAME, STEM, SOURCE_PIN, PARENT_PIN, REPLAY_PIN, CHECKPOINT,
    LEGACY_HELPER_SHA, CORRECTED_HELPER_SHA, BUDGETS, OPTIMIZER, UPDATES, PROBES,
    read, write, sha, verify_transfer,
)

ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT/'outputs/cctv_dgp_decoder_hinge_repair_v1_design'
PACKET = ROOT/'outputs'/NAME
SOURCE = ROOT/'outputs/cctv_dgp_spatial_readout_v1_return'
PARENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
REPLAY = ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_return'
ANALYSIS = ROOT/'outputs/cctv_dgp_spatial_path_replay_v1_analysis'


def calibrate():
    """Exercise the prospective saved-output scorer on real existing controls."""
    import numpy as np
    from PIL import Image
    from audit_cctv_dgp_decoder_hinge_repair_v1_return import _numeric_scalar_checks
    from audit_cctv_dgp_spatial_path_replay_v1_return import recompute_metrics
    sys.path.insert(1, str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support
    p, data = read(REPLAY/'protocol.json'), read(PARENT/'protocol.json')
    cases, refs = {c['id']:c for c in data['cases']}, {r['id']:r for r in data['references']}
    def pixels(path, mode='RGB'):
        with Image.open(path) as image:
            assert image.size == (256,256)
            return np.asarray(image.convert(mode)).copy()
    checked = 0
    for label in ['baseline', 'A_u001']:
        saved_root = REPLAY/'outputs/spatial_path_replay_v1'/label
        rows = {r['case']:r for r in read(saved_root/'metrics.json')['rows']}
        for index, block in enumerate(p['reference_blocks']):
            ref = refs[block['reference']]
            target, mask = pixels(PARENT/ref['target']), pixels(PARENT/ref['observed'],'L')>0
            retained, ev = unpack(PARENT/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz",block['case_ids'])
            with np.load(saved_root/'packs'/f'r{index:02d}.npz', allow_pickle=False) as pack:
                assert pack['ids'].tolist() == block['case_ids']
                raw, er, ep = [pack[key].copy() for key in ['raw','embeddings_raw','embeddings_png']]
            for slot, cid in enumerate(block['case_ids']):
                c = cases[cid]
                camera = pixels(PARENT/c['input']); png = pixels(saved_root/'png'/(cid+'.png'))
                support = feature_support(mask,c['landmarks5_canvas_xy'])
                rm, pm = recompute_metrics(raw[slot],png,camera,target,mask,support,retained[slot],ev,er,ep,slot,label=='baseline')
                _numeric_scalar_checks(rows[cid]['raw'],rm)
                _numeric_scalar_checks(rows[cid]['png'],pm)
                checked += 1
    write(DESIGN/'scorer_calibration.json', dict(complete=True, compositions=checked,
        raw_compositions=checked, png_compositions=checked, source_count=2,
        exact_control_pixels_checked=True, saved_metrics_match_prospective_checker=True,
        neural_calls=0, gradient_queries=0, optimizer_updates=0,
        checker_sha256=sha(ROOT/'scripts/audit_cctv_dgp_decoder_hinge_repair_v1_return.py')))


def stage():
    assert not DESIGN.exists() and not PACKET.exists(), 'Preserve existing design/packet'
    assert all(read(ANALYSIS/name)['complete'] for name in ['independent_audit.json','visual_binding_audit.json','hinge_import_regression_audit.json'])
    old, data, replay = read(SOURCE/'protocol.json'), read(PARENT/'protocol.json'), read(REPLAY/'protocol.json')
    assert (sha(SOURCE/'protocol.json'),sha(PARENT/'protocol.json'),sha(REPLAY/'protocol.json')) == (SOURCE_PIN,PARENT_PIN,REPLAY_PIN)
    assert sha(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == CHECKPOINT
    DESIGN.mkdir()
    protected_names = ['SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','PROJECT_HANDOFF.md',
        'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_NEXT_PILOT.md',
        'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth']
    write(DESIGN/'protected_before_preparation.json',dict(complete=True,files={name:sha(ROOT/name) for name in protected_names}))
    test = ROOT/'scripts/test_cctv_dgp_decoder_hinge_repair_v1.py'
    result = subprocess.run([sys.executable,'-B',str(test)],cwd=ROOT,capture_output=True,text=True,timeout=120)
    text = result.stdout+result.stderr
    with (DESIGN/'cpu_tests.txt').open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    match = re.search(r'Ran (\d+) tests',text)
    assert result.returncode == 0 and match and int(match.group(1)) >= 12, text
    write(DESIGN/'cpu_tests.json',dict(complete=True,exit_code=result.returncode,tests_run=int(match.group(1)),
        test_sha256=sha(test),checker_sha256=sha(ROOT/'scripts/audit_cctv_dgp_decoder_hinge_repair_v1_return.py'),
        worker_sha256=sha(ROOT/'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py'),
        local_neural_calls=0,local_gradient_queries=0,local_optimizer_updates=0))
    calibrate()
    manifest = read(REPLAY/'result_manifest.json')['files']
    replay_names = set(replay['assets_sha256']) | {'protocol.json',
        'outputs/spatial_path_replay_v1/run_results.json',
        'outputs/spatial_path_replay_v1/A_u001_vector.npy',
        'outputs/spatial_path_replay_v1/A_u001_state.json',
        'outputs/spatial_path_replay_v1/A_u001_outcome.json'}
    replay_names |= {name for name in manifest if name.startswith('outputs/spatial_path_replay_v1/A_u001/')}
    replay_required = {name:sha(REPLAY/name) for name in sorted(replay_names)}
    assert all(replay_required[name] == manifest[name]['sha256'] for name in replay_required)
    d = dict(name=NAME,status='DESIGN_FROZEN_NOT_YET_SEALED',date_local='2026-10-15',
        source_protocol_sha256=SOURCE_PIN,parent_protocol_sha256=PARENT_PIN,replay_protocol_sha256=REPLAY_PIN,
        checkpoint_sha256=CHECKPOINT,route='A',optimizer=OPTIMIZER,budgets=BUDGETS,
        snapshot_updates=UPDATES,probe_amplitudes=dict(PROBES),common_radius_L2=0.04018470287178282,
        maximum_new_optimizer_updates=50,maximum_full_corpus_epochs=0,trainable_tensors=15,trainable_elements=609219,
        legacy_helper_sha256=LEGACY_HELPER_SHA,corrected_helper_sha256=CORRECTED_HELPER_SHA,
        source_required_sha256=replay['source_required_sha256'],parent_required_sha256=replay['parent_required_sha256'],
        replay_required_sha256=replay_required,case_ids=old['case_ids'],reference_blocks=old['reference_blocks'],
        schedule=old['schedule'],scientific_thresholds=old['scientific_thresholds'],
        trainable_parameter_names=old['trainable_parameter_names']['A'],automatic_launch=False,
        model_qualified=False,new_incumbent_selection=False,probe_admission_used=False,
        failed_recipe_resume_allowed=False,local_neural_construction_allowed=False,learning_rate_changed=False,
        derivative_boundary_is_only_intended_objective_change=True,
        method_origin='Fine-tuned DeblurGAN-v2-compatible current decoder; frozen external StyleGAN2 prior; neither random-weight training nor own-trained StyleGAN2',
        hypothesis='Correct the demonstrated inactive-hinge gradient import regression before testing a higher rate or additional epochs; correction alone is not a quality claim',
        intended_change='Load the exact previously audited ReLU guard; preserve scalar loss, weights, data, schedule, Adam, projection and acceptance thresholds',
        first_step_controls='Fresh original and A0 plus saved legacy A1 exact RAW/PNG; corrected first full state and 1/16,1/4 direction amplitudes, never probe selection',
        trajectory_stop='Only full corrected scheduled states with positive global and each-source structure and zero RAW/PNG preservation/brightness failures permit continuation; full qualification still requires >=1% in all sources and global',
        stop_updates=[1,10,20,30,40,50],native_pixels_used=0,DEV_pixels_used=0,final_identity_pixels=0,completion_forwards=0,
        new_actual_training_human_manual_only=True,agent_VM_execution=False,
        previous_replay_evidence_sha256={name:sha(ANALYSIS/name) for name in ['independent_audit.json','visual_observations.json','visual_binding_audit.json','curve_summary.json','hinge_import_regression_audit.json']})
    write(DESIGN/'design.json',d)
    PACKET.mkdir();(PACKET/'scripts').mkdir();(PACKET/'evidence').mkdir()
    copies = {
        'cctv_dgp_decoder_hinge_repair_v1_contract.py':ROOT/'scripts/cctv_dgp_decoder_hinge_repair_v1_contract.py',
        'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py':ROOT/'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py',
        'audit_cctv_dgp_decoder_hinge_repair_v1_return.py':ROOT/'scripts/audit_cctv_dgp_decoder_hinge_repair_v1_return.py',
        'cctv_dgp_bank_hinge_boundary_v1_loss.py':ROOT/'scripts/cctv_dgp_bank_hinge_boundary_v1_loss.py',
        'cctv_dgp_spatial_path_replay_v1_contract.py':ROOT/'scripts/cctv_dgp_spatial_path_replay_v1_contract.py',
        'audit_cctv_dgp_spatial_path_replay_v1_return.py':ROOT/'scripts/audit_cctv_dgp_spatial_path_replay_v1_return.py',
        'cctv_dgp_spatial_readout_v1_contract.py':SOURCE/'cctv_dgp_spatial_readout_v1_contract.py',
        'audit_cctv_dgp_spatial_readout_v1_return.py':SOURCE/'audit_cctv_dgp_spatial_readout_v1_return.py',
        'design.json':DESIGN/'design.json',
        'evidence/cpu_tests.json':DESIGN/'cpu_tests.json','evidence/cpu_tests.txt':DESIGN/'cpu_tests.txt',
        'evidence/test_source.py':test,'evidence/scorer_calibration.json':DESIGN/'scorer_calibration.json',
        'evidence/replay_results.md':ROOT/'CCTV_DGP_SPATIAL_PATH_REPLAY_V1_RESULTS.md',
        'evidence/post_replay_review.md':ROOT/'CCTV_DGP_POST_PATH_REPLAY_REVIEW_V1.md',
        'evidence/import_regression_audit.json':ANALYSIS/'hinge_import_regression_audit.json',
    }
    for name,path in copies.items():shutil.copyfile(path,PACKET/name)
    shell='''#!/usr/bin/env bash
set -euo pipefail
test -n "${TMUX:-}" || { echo "Run manually inside tmux" >&2; exit 1; }
test "$#" -eq 1 || { echo "Pass the exact protocol SHA256" >&2; exit 1; }
python -B -u scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py --root . --protocol-sha "$1" --supervise
'''
    with (PACKET/'scripts/run_decoder_hinge.sh').open('x',encoding='utf-8',newline='\n') as stream:stream.write(shell)
    for path in PACKET.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
    write(DESIGN/'staging.json',dict(complete=True,status='STAGED_NOT_SEALED',design_sha256=sha(DESIGN/'design.json'),
        copied_source_sha256={name:sha(path) for name,path in copies.items()},source_assets=len(d['source_required_sha256']),
        parent_assets=len(d['parent_required_sha256']),replay_assets=len(replay_required),neural_calls=0,optimizer_updates=0))
    print(dict(complete=True,status='STAGED_NOT_SEALED',CPU_tests=int(match.group(1)),scorer_compositions=200,
        design_sha256=sha(DESIGN/'design.json')))


def seal():
    assert read(DESIGN/'staging.json')['complete']
    assert not (PACKET/'protocol.json').exists() and not (DESIGN/'preparation.json').exists()
    staging=read(DESIGN/'staging.json')
    assert all(sha(PACKET/name)==digest for name,digest in staging['copied_source_sha256'].items())
    bash=read(DESIGN/'bash_syntax.json')
    assert bash['exit_code']==0 and bash['shell_sha256']==sha(PACKET/'scripts/run_decoder_hinge.sh')
    shutil.copyfile(DESIGN/'bash_syntax.json',PACKET/'evidence/bash_syntax.json')
    d=read(DESIGN/'design.json')
    p=d|dict(design_sha256=sha(DESIGN/'design.json'),assets_sha256={v.relative_to(PACKET).as_posix():sha(v) for v in sorted(PACKET.rglob('*')) if v.is_file()})
    write(PACKET/'protocol.json',p);pin=sha(PACKET/'protocol.json')
    verify_transfer(PACKET,SOURCE,PARENT,pin,replay=REPLAY)
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    assert not archive.exists() and not Path(str(archive)+'.sha256').exists()
    with archive.open('xb') as output:
        with tarfile.open(fileobj=output,mode='w:gz',compresslevel=9) as tar:
            for path in sorted(PACKET.rglob('*')):
                if path.is_file():tar.add(path,arcname=NAME+'/'+path.relative_to(PACKET).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='utf-8',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    write(DESIGN/'preparation.json',dict(complete=True,status='PREPARED_UNRUN_PENDING_INDEPENDENT_AUDIT',
        UTC=datetime.now(timezone.utc).isoformat(),protocol_sha256=pin,design_sha256=sha(DESIGN/'design.json'),
        archive_sha256=digest,archive_bytes=archive.stat().st_size,packet_assets=len(p['assets_sha256']),
        source_assets=len(p['source_required_sha256']),parent_assets=len(p['parent_required_sha256']),replay_assets=len(p['replay_required_sha256']),
        maximum_manual_optimizer_updates=50,actual_new_optimizer_updates=0,local_neural_calls=0,local_gradient_queries=0,
        actual_L4_repair_success_unverified=True,model_qualified=False,agent_VM_calls=0))
    print(read(DESIGN/'preparation.json'))


def refresh_unsealed():
    """Retain the first draft before applying pre-seal review corrections."""
    assert not (PACKET/'protocol.json').exists() and not (DESIGN/'preparation.json').exists()
    assert not (ROOT/'outputs'/(STEM+'-execution.tar.gz')).exists()
    revision=DESIGN/'preseal_revision_r0'
    assert not revision.exists(), 'Retain previous draft review'
    revision.mkdir();shutil.copytree(PACKET,revision/'packet')
    before={v.relative_to(PACKET).as_posix():sha(v) for v in PACKET.rglob('*') if v.is_file()}
    assert all(sha(revision/'packet'/name)==digest for name,digest in before.items())
    receipts={}
    for name in ['cpu_tests.json','cpu_tests.txt','scorer_calibration.json','staging.json']:
        path=DESIGN/name;receipts[name]=sha(path)
        assert path.resolve().is_relative_to(DESIGN.resolve()) and (revision/name).resolve().is_relative_to(DESIGN.resolve())
        path.rename(revision/name)
        assert sha(revision/name)==receipts[name]
    write(revision/'preservation.json',dict(complete=True,unsealed_draft_only=True,packet_sha256=before,receipt_sha256=receipts,
        sealed_protocol_or_archive_ever_existed=False,scientific_design_sha256=sha(DESIGN/'design.json')))
    test=ROOT/'scripts/test_cctv_dgp_decoder_hinge_repair_v1.py'
    result=subprocess.run([sys.executable,'-B',str(test)],cwd=ROOT,capture_output=True,text=True,timeout=120)
    text=result.stdout+result.stderr
    with (DESIGN/'cpu_tests.txt').open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    match=re.search(r'Ran (\d+) tests',text)
    assert result.returncode==0 and match and int(match.group(1))>=14,text
    write(DESIGN/'cpu_tests.json',dict(complete=True,exit_code=0,tests_run=int(match.group(1)),test_sha256=sha(test),
        checker_sha256=sha(ROOT/'scripts/audit_cctv_dgp_decoder_hinge_repair_v1_return.py'),
        worker_sha256=sha(ROOT/'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py'),
        local_neural_calls=0,local_gradient_queries=0,local_optimizer_updates=0))
    calibrate()
    changes={
        'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py':ROOT/'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py',
        'audit_cctv_dgp_decoder_hinge_repair_v1_return.py':ROOT/'scripts/audit_cctv_dgp_decoder_hinge_repair_v1_return.py',
        'evidence/test_source.py':test,'evidence/cpu_tests.json':DESIGN/'cpu_tests.json',
        'evidence/cpu_tests.txt':DESIGN/'cpu_tests.txt','evidence/scorer_calibration.json':DESIGN/'scorer_calibration.json'}
    for name,path in changes.items():shutil.copyfile(path,PACKET/name)
    old=read(revision/'staging.json');copied=old['copied_source_sha256']|{name:sha(path) for name,path in changes.items()}
    assert all(sha(PACKET/name)==digest for name,digest in copied.items())
    old.update(copied_source_sha256=copied,unsealed_draft_preserved_sha256=sha(revision/'preservation.json'),
        preseal_review_fixes='Combined 180s preflight, runtime environment versions, exact probe model/RNG restoration, saved-Adam calibration and checker bindings')
    write(DESIGN/'staging.json',old)
    print(dict(complete=True,status='STAGED_NOT_SEALED',CPU_tests=int(match.group(1)),scorer_compositions=200,
        design_sha256=sha(DESIGN/'design.json'),original_unsealed_draft_preserved=True))


if __name__=='__main__':
    parser=argparse.ArgumentParser();flags=parser.add_mutually_exclusive_group(required=True)
    flags.add_argument('--stage',action='store_true');flags.add_argument('--seal',action='store_true')
    flags.add_argument('--refresh-unsealed',action='store_true')
    args=parser.parse_args()
    if args.stage:stage()
    elif args.seal:seal()
    else:refresh_unsealed()
