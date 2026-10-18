"""Seal one finite human-manual experiment; no solve or learned execution."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile

from cctv_dgp_decoder_feasibility_v1_contract import (
    NAME,STEM,STAGE,ACTIVATION_PIN,CHECKPOINT,CORRECTED_HELPER_SHA,
    STATES,AMPLITUDES,RADIUS,SOLVER,BUDGETS,THRESHOLDS,
    read,write,sha,validate,verify_transfer,
)

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
PREP=OUT/'cctv_dgp_decoder_feasibility_v1_preparation'


def main():
    assert not (PREP/'preparation.json').exists()
    root=OUT/NAME
    assert not root.exists(),'Retain any prior complete or partial packet'
    activation=OUT/'cctv_dgp_decoder_activation_v1_return'
    repair=OUT/'cctv_dgp_decoder_hinge_repair_v1_return'
    source=OUT/'cctv_dgp_spatial_readout_v1_return'
    parent=OUT/'cctv_dgp_bank_photometric_guard_v1_return'
    replay=OUT/'cctv_dgp_spatial_path_replay_v1_return'
    analysis=OUT/'cctv_dgp_decoder_activation_v1_analysis'
    ap=read(activation/'protocol.json')
    assert sha(activation/'protocol.json')==ACTIVATION_PIN
    assert read(analysis/'independent_readback.json')['full_saved_numerical_audit_complete']
    assert read(analysis/'publication/independent_audit.json')['complete']
    tests=read(PREP/'cpu_tests.json')
    assert tests['complete'] and tests['exit_code']==0 and tests['tests']>0
    copies={
        'cctv_dgp_decoder_feasibility_v1_contract.py':ROOT/'scripts/cctv_dgp_decoder_feasibility_v1_contract.py',
        'cctv_dgp_decoder_feasibility_v1_math.py':ROOT/'scripts/cctv_dgp_decoder_feasibility_v1_math.py',
        'cctv_dgp_decoder_feasibility_v1_context.py':ROOT/'scripts/cctv_dgp_decoder_feasibility_v1_context.py',
        'scripts/cctv_dgp_decoder_feasibility_v1_vm.py':ROOT/'scripts/cctv_dgp_decoder_feasibility_v1_vm.py',
        'audit_cctv_dgp_decoder_feasibility_v1_return.py':ROOT/'scripts/audit_cctv_dgp_decoder_feasibility_v1_return.py',
        'cctv_dgp_decoder_activation_v1_math.py':ROOT/'scripts/cctv_dgp_decoder_activation_v1_math.py',
        'evidence/test_math.py.txt':ROOT/'scripts/test_cctv_dgp_decoder_feasibility_v1_math.py',
        'evidence/test_contract.py.txt':ROOT/'scripts/test_cctv_dgp_decoder_feasibility_v1_contract.py',
        'evidence/test_audit.py.txt':ROOT/'scripts/test_cctv_dgp_decoder_feasibility_v1_audit.py',
        'evidence/cpu_tests.json':PREP/'cpu_tests.json',
        'evidence/cpu_tests.txt':PREP/'cpu_tests.txt',
        'evidence/activation_independent_audit_r2.json':analysis/'independent_audit_r2.json',
        'evidence/activation_schema_r2.json':analysis/'independent_audit_r2_schema_r2.json',
        'evidence/activation_independent_readback.json':analysis/'independent_readback.json',
        'evidence/activation_independent_causal_review.json':analysis/'independent_causal_review.json',
        'evidence/activation_results.md':ROOT/'CCTV_DGP_DECODER_ACTIVATION_V1_RESULTS.md',
        'evidence/post_activation_review.md':ROOT/'CCTV_DGP_POST_ACTIVATION_V1_REVIEW.md',
    }
    assert all(path.is_file() for path in copies.values())
    bound={name:sha(path) for name,path in copies.items()}
    manifest=read(activation/'result_manifest.json')
    activation_bindings={name:item['sha256'] for name,item in manifest['files'].items()}
    activation_bindings['result_manifest.json']=sha(activation/'result_manifest.json')
    design=dict(name=NAME,date_local='2026-10-15',status='PREPARED_UNRUN_FINITE_FEASIBILITY_ONLY',
        activation_protocol_sha256=ACTIVATION_PIN,checkpoint_sha256=CHECKPOINT,route='A',
        hypothesis='A separately selected direction in the current decoder saved-gradient span may improve all sampled degraded structure terms while avoiding the demonstrated individual preservation and idealized brightness conflicts. Actual finite RAW/PNG output checks remain authoritative.',
        state_labels=STATES,placements=AMPLITUDES,common_radius_L2=RADIUS,solver=SOLVER,budgets=BUDGETS,
        scientific_thresholds=THRESHOLDS,corrected_helper_sha256=CORRECTED_HELPER_SHA,
        trainable_tensors=15,trainable_elements=609219,
        trainable_parameter_names=ap['trainable_parameter_names'],gradient_layout=ap['gradient_layout'],
        derivative_reference_indices=ap['derivative_reference_indices'],derivative_reference_ids=ap['derivative_reference_ids'],
        case_ids=ap['case_ids'],reference_blocks=ap['reference_blocks'],objective_replay_max_abs=1e-10,
        activation_required_sha256=activation_bindings,
        repair_required_sha256=ap['repair_required_sha256'],source_required_sha256=ap['source_required_sha256'],
        parent_required_sha256=ap['parent_required_sha256'],replay_required_sha256=ap['replay_required_sha256'],
        new_optimizer_updates=0,new_gradient_queries=0,new_full_corpus_epochs=0,
        diagnostic_only=True,one_direction_only=True,strict_homogeneous_sufficient_subset=True,
        fixed_placements_comparison_not_optimizer_continuation=True,exact_rounded_displacement_certificate_required=True,
        no_certificate_implies_whole_decoder_infeasibility=False,
        automatic_launch=False,failed_recipe_resume_allowed=False,model_qualified=False,new_incumbent_selection=False,
        local_neural_construction_allowed=False,local_parameter_proposals_allowed=False,
        native_pixels_used=False,DEV_pixels_used=False,final_identity_pixels=False,automatic_extra_epochs=False,
        training_success_not_implied=True,
        search_semantics='One homogeneous ray search with strictly non-increasing per-profile preservation slopes and positive structure slopes on two TRAIN references. The coefficient box normalizes a ray; it does not replace the parameter L2 radius. This is a stricter sufficient subset than the original nonzero finite slack constraints. No certifiable candidate is not a whole-span, original-slack or decoder-infeasibility proof.',
        certificate_semantics='Use exact rational idealized mean-clip coefficients and exact dyadic saved-gradient products with each actual rounded float32 displacement. Require exact unchanged radius, all30 preservation slopes nonpositive, all8 idealized brightness constraints and all8 structure slopes improving; full placement predicts at least1% for each sampled degraded structure term. This does not certify neural linearization or PNG behavior.',
        finite_selection_semantics='Evaluate all three prospectively fixed placements as one bounded comparison; do not select a fourth placement or continue optimizer training after a quality failure. No incumbent is selected by this experiment.',
        planned_visual_review=dict(sheets=20,rows='five fixed degradation profiles per TRAIN reference',
            columns=['input','target','initial','quarter','half','full'],
            required_features=['eyes','brows','nose','mouth','jaw','visible appearance'],
            automatic_review_pass=False,no_certified_direction_sheets=0),
        method_origin=ap['method_origin'],
        limitations=['Only two references have underlying gradients; all20fixedreferences are finite checks.',
            'A selected neural derivative/idealized clip certificate is not a literal float32/PNG derivative or useful-restoration result.',
            'Every tested placement must retain separateRAW/PNG preservation, global and each-source1%structure criteria, exterior pixels and whole-face visual review before further TRAIN/DEV selection.',
            'Final identities remain outside training/epoch selection. All five milestones and seven automatic/assisted covering families remain required.'])
    root.mkdir()
    for name,path in copies.items():
        destination=root/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,destination);assert sha(destination)==bound[name]
    write(root/'design.json',design)
    shell='''#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
test -n "${TMUX:-}"
test "$#" -eq 1
python -B -u scripts/cctv_dgp_decoder_feasibility_v1_vm.py --root . --protocol-sha "$1" --supervise
'''
    with (root/'scripts/run_feasibility.sh').open('x',encoding='utf-8',newline='\n') as stream:stream.write(shell)
    assets={path.relative_to(root).as_posix():sha(path) for path in sorted(root.rglob('*')) if path.is_file()}
    protocol=dict(design,design_sha256=sha(root/'design.json'),assets_sha256=assets)
    validate(protocol,design);write(root/'protocol.json',protocol);pin=sha(root/'protocol.json')
    verify_transfer(root,activation,repair,source,parent,replay,pin)
    archive=OUT/(STEM+'-execution.tar.gz');sidecar=Path(str(archive)+'.sha256')
    assert not archive.exists() and not sidecar.exists()
    with tarfile.open(archive,'x:gz',compresslevel=6) as stream:
        for path in sorted(root.rglob('*')):
            if path.is_file():stream.add(path,arcname=NAME+'/'+path.relative_to(root).as_posix(),recursive=False)
    with sidecar.open('x',encoding='utf-8',newline='\n') as stream:stream.write(sha(archive)+'  '+archive.name+'\n')
    write(PREP/'preparation.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=pin,design_sha256=sha(root/'design.json'),archive_sha256=sha(archive),
        archive_bytes=archive.stat().st_size,packet_assets=len(assets),copied_source_sha256=bound,
        required_bindings={key:len(protocol[key]) for key in ['activation_required_sha256','repair_required_sha256',
            'source_required_sha256','parent_required_sha256','replay_required_sha256']},
        status='PREPARED_UNRUN; independent preparation audit pending',automatic_launch=False,
        new_local_model_calls=0,new_local_solver_calls=0,new_local_parameter_proposals=0,
        new_local_gradient_queries=0,new_local_optimizer_updates=0,agent_VM_calls=0,model_qualified=False))
    print(dict(complete=True,status='PREPARED_UNRUN',protocol_sha256=pin,archive_bytes=archive.stat().st_size))


if __name__=='__main__':main()
