"""Frozen transfer/role contract for one manual decoder-direction experiment."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import sys

NAME = 'cctv_dgp_decoder_feasibility_v1_vm'
STEM = 'cctv-dgp-decoder-feasibility-v1'
STAGE = 'outputs/decoder_feasibility_v1'
ACTIVATION_NAME = 'cctv_dgp_decoder_activation_v1_vm'
REPAIR_NAME = 'cctv_dgp_decoder_hinge_repair_v1_vm'
SOURCE_NAME = 'cctv_dgp_spatial_readout_v1_vm'
PARENT_NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
REPLAY_NAME = 'cctv_dgp_spatial_path_replay_v1_vm'
ACTIVATION_PIN = 'b93c7a8ce3ab730f1d20f8ebc8786db18469e8652db26554cf0b116dc5785481'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
CORRECTED_HELPER_SHA = '8368f97c13b4703834c6ea249a85a6970322dce45416bfecd23d8a299f6706f8'
STATES = ['initial','quarter','half','full']
AMPLITUDES = [0.,.25,.5,1.]
RADIUS = 0.04018470287178282
THRESHOLDS = dict(structure_gain=.01, brightness_gain_fraction=.2,
                  MSE_slack=1e-12, SSIM_ArcFace_slack=1e-6)
SOLVER = dict(method='highs-ds',time_limit=60.,maxiter=10000,
    primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,
    presolve=True,interior_margin_fraction=1e-6,coefficient_bound=1.,
    full_minimum_predicted_structure_gain=.01,placements=[.25,.5,1.],
    radius_interior_fraction=.999)
BUDGETS = dict(worker_seconds=2400,external_seconds=2430,cache_seconds=240,
    basis_seconds=90,solver_seconds=60,certificate_seconds=600,snapshot_seconds=300,
    saved_audit_seconds=900,saved_audit_external_seconds=930,
    export_seconds=300,export_external_seconds=330,kill_grace_seconds=30,
    minimum_free_bytes=4*1024**3,disk_reserve_bytes=1024**3,
    available_RAM_before_import_bytes=8*1024**3,output_byte_cap=1280*1024**2,
    emergency_metadata_bytes=8*1024**2,peak_VRAM_bytes=20*1024**3,
    peak_RSS_bytes=12*1024**3,maximum_optimizer_updates=0,
    maximum_backwards=0,maximum_gradient_queries=0,original_batch_calls=20,
    prior_feature_batch_calls=20,decoder_batch_calls=80,recognizer_batch_calls=240,
    maximum_solver_calls=1,maximum_candidate_directions=1,
    maximum_placement_assignments=3)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def sha(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''): value.update(block)
    return value.hexdigest()


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verify_paths(folder,bindings):
    folder=Path(folder).resolve()
    assert type(bindings) is dict and bindings
    for name,digest in bindings.items():
        assert type(name) is str and name and ':' not in name and '\\' not in name
        relative=PurePosixPath(name)
        assert relative.parts and not relative.is_absolute() and '..' not in relative.parts and str(relative)==name
        path=folder/name
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        assert stat.S_ISREG(path.stat().st_mode) and sha(path)==digest,name


def validate(p,design):
    assert all(p.get(key)==value for key,value in design.items()),'Frozen design differs'
    assert p['name']==NAME and p['activation_protocol_sha256']==ACTIVATION_PIN
    assert p['checkpoint_sha256']==CHECKPOINT and p['route']=='A'
    assert p['state_labels']==STATES and p['placements']==AMPLITUDES
    assert p['common_radius_L2']==RADIUS and p['solver']==SOLVER and p['budgets']==BUDGETS
    assert p['scientific_thresholds']==THRESHOLDS
    assert p['trainable_tensors']==15 and p['trainable_elements']==609219
    assert p['derivative_reference_indices']==[12,4]
    assert p['derivative_reference_ids']==['tr_asian_00176','tr_ffhq_53945']
    assert len(p['trainable_parameter_names'])==15
    assert len(p['case_ids'])==len(set(p['case_ids']))==100 and len(p['reference_blocks'])==20
    assert p['objective_replay_max_abs']==1e-10
    assert p['new_optimizer_updates']==p['new_gradient_queries']==p['new_full_corpus_epochs']==0
    for key in ['automatic_launch','failed_recipe_resume_allowed','model_qualified','new_incumbent_selection',
        'local_neural_construction_allowed','local_parameter_proposals_allowed','native_pixels_used',
        'DEV_pixels_used','final_identity_pixels','automatic_extra_epochs']:
        assert p[key] is False,key
    assert p['diagnostic_only'] is p['one_direction_only'] is p['strict_homogeneous_sufficient_subset'] is True
    assert p['fixed_placements_comparison_not_optimizer_continuation'] is True
    assert p['exact_rounded_displacement_certificate_required'] is True
    assert p['no_certificate_implies_whole_decoder_infeasibility'] is False


def verify_transfer(root,activation,repair,source,parent,replay,pin):
    import numpy as np
    root,activation,repair,source,parent,replay=[Path(x).resolve() for x in [root,activation,repair,source,parent,replay]]
    p,design=read(root/'protocol.json'),read(root/'design.json')
    assert sha(root/'protocol.json')==pin and sha(root/'design.json')==p['design_sha256']
    validate(p,design)
    for folder,key in [(root,'assets_sha256'),(activation,'activation_required_sha256'),
        (repair,'repair_required_sha256'),(source,'source_required_sha256'),
        (parent,'parent_required_sha256'),(replay,'replay_required_sha256')]:
        verify_paths(folder,p[key])
    ap,rp,old,data=[read(folder/'protocol.json') for folder in [activation,repair,source,parent]]
    assert sha(activation/'protocol.json')==ACTIVATION_PIN
    assert sha(repair/'protocol.json')==ap['repair_protocol_sha256']
    assert sha(source/'protocol.json')==ap['source_protocol_sha256']
    assert sha(parent/'protocol.json')==ap['parent_protocol_sha256']
    assert sha(replay/'protocol.json')==ap['replay_protocol_sha256']
    assert sha(repair/'cctv_dgp_bank_hinge_boundary_v1_loss.py')==CORRECTED_HELPER_SHA
    result=read(activation/'outputs/decoder_activation_v1/run_results.json')
    assert result['complete'] and result['counts']['gradient_queries']==70
    assert result['counts']['optimizer_updates']==result['counts']['backwards']==0
    assert result['exact_retained_RAW_PNG_cases']==400 and result['model_qualified'] is False
    # The historical VM saved audit is expected to have stopped. Its explicit
    # independently reviewed local R2 evidence is bound in this new packet.
    audit=read(root/'evidence/activation_independent_audit_r2.json')
    closure=read(root/'evidence/activation_independent_readback.json')
    assert audit['complete'] and audit['protocol_sha256']==ACTIVATION_PIN
    assert audit['derived_R2_numerical_checks_complete'] and not audit['original_checker_passed_unmodified']
    assert closure['complete'] and closure['full_saved_numerical_audit_complete']
    assert p['case_ids']==ap['case_ids']==rp['case_ids']==old['case_ids']==data['preview_case_ids']
    assert p['reference_blocks']==ap['reference_blocks']
    assert p['trainable_parameter_names']==old['trainable_parameter_names']['A']
    assert p['gradient_layout']==ap['gradient_layout']
    initial=np.load(source/'outputs/spatial_readout_v1/A_initial_vector.npy',allow_pickle=False)
    assert initial.dtype==np.float32 and initial.shape==(609219,) and np.isfinite(initial).all()
    for index,reference in zip(p['derivative_reference_indices'],p['derivative_reference_ids']):
        folder=activation/'outputs/decoder_activation_v1/gradients'/f'r{index:02d}'
        measured=read(folder/'measurement.json')
        assert measured['reference']==reference and measured['scalar_labels']==p['gradient_layout']
        matrix=np.load(folder/'gradients.npy',allow_pickle=False,mmap_mode='r')
        assert matrix.dtype==np.float32 and matrix.shape==(35,609219)
    cases={case['id']:case for case in data['cases']}
    for block in p['reference_blocks']:
        assert len(block['case_ids'])==5
        assert all(cases[c]['role']=='train' and cases[c]['reference_id']==block['reference'] for c in block['case_ids'])
        assert [cases[c]['profile'] for c in block['case_ids']]==['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
    return p,ap,rp,old,data


def require_manual(root):
    assert platform.system()=='Linux','Human manual Linux L4 tmux only'
    assert platform.node().split('.')[0]=='forensic-dgp-thesis'
    assert Path(root).resolve()==(Path.home()/'forensic-dgp'/NAME).resolve()
    assert Path.cwd().resolve()==Path(root).resolve() and os.environ.get('TMUX')
    assert sys.prefix==str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')
