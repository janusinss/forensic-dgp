"""Frozen derivative-only decoder comparison. Verification is model-free."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import sys

NAME = 'cctv_dgp_decoder_hinge_repair_v1_vm'
STEM = 'cctv-dgp-decoder-hinge-repair-v1'
SOURCE_NAME = 'cctv_dgp_spatial_readout_v1_vm'
PARENT_NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
REPLAY_NAME = 'cctv_dgp_spatial_path_replay_v1_vm'
SOURCE_PIN = '3bfbaf4b4a519f2926f0503731cd9d22390a7fc0c09da20e2f022e6a1b93c97e'
PARENT_PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
REPLAY_PIN = '6f9a8757466578ef57457994ded7fe9b6b170657f2ee7dd3b8b23f0bd2cc1bc7'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
LEGACY_HELPER_SHA = '7474a7688872a130bf60ced51eff37e25017f295f73a459b06d082c23d4ca393'
CORRECTED_HELPER_SHA = '8368f97c13b4703834c6ea249a85a6970322dce45416bfecd23d8a299f6706f8'
STAGE = 'outputs/decoder_hinge_repair_v1'
UPDATES = [0, 1, 10, 20, 30, 40, 50]
PROBES = [('probe_1_16', 1/16), ('probe_1_4', 1/4)]
OPTIMIZER = dict(name='Adam', lr=.0001, betas=[.9,.999], eps=1e-8,
                 weight_decay=1e-5, foreach=False, fused=False, clip_norm=1.)
BUDGETS = dict(worker_seconds=2400, external_seconds=2430,
    saved_audit_seconds=600, saved_audit_external_seconds=630,
    export_seconds=600, export_external_seconds=630, kill_grace_seconds=30,
    cache_seconds=240, preflight_seconds=180, training_seconds=600,
    snapshot_seconds=300, minimum_free_bytes=8*1024**3,
    available_RAM_before_import_bytes=8*1024**3, disk_reserve_bytes=1024**3,
    output_byte_cap=3*1024**3, peak_VRAM_bytes=20*1024**3,
    peak_RSS_bytes=12*1024**3, maximum_gradient_queries=42,
    maximum_backwards=50, maximum_optimizer_updates=50,
    original_batch_calls=20, prior_feature_batch_calls=20,
    decoder_batch_calls=252, recognizer_batch_calls=492)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verify_paths(folder, bindings):
    folder=Path(folder).resolve()
    for name,digest in bindings.items():
        posix=PurePosixPath(name)
        assert isinstance(name,str) and ':' not in name and '\\' not in name
        assert not posix.is_absolute() and '..' not in posix.parts and str(posix)==name
        path=folder/name
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        assert stat.S_ISREG(path.stat().st_mode) and sha(path)==digest,name


def validate(p,d):
    # Design is copied exactly into the execution protocol. Added file hashes
    # are checked separately; descriptions cannot silently alter scientific fields.
    assert all(p.get(k)==v for k,v in d.items()), 'Frozen design changed'
    assert p['name']==NAME and p['source_protocol_sha256']==SOURCE_PIN
    assert p['parent_protocol_sha256']==PARENT_PIN and p['replay_protocol_sha256']==REPLAY_PIN
    assert p['checkpoint_sha256']==CHECKPOINT
    assert p['route']=='A' and p['optimizer']==OPTIMIZER and p['budgets']==BUDGETS
    assert p['snapshot_updates']==UPDATES and p['probe_amplitudes']==dict(PROBES)
    assert p['common_radius_L2']==0.04018470287178282
    assert p['maximum_new_optimizer_updates']==50 and p['maximum_full_corpus_epochs']==0
    assert p['trainable_tensors']==15 and p['trainable_elements']==609219
    assert p['legacy_helper_sha256']==LEGACY_HELPER_SHA and p['corrected_helper_sha256']==CORRECTED_HELPER_SHA
    assert len(p['case_ids'])==len(set(p['case_ids']))==100 and len(p['reference_blocks'])==20
    for key in ['automatic_launch','model_qualified','new_incumbent_selection','probe_admission_used',
                'failed_recipe_resume_allowed','local_neural_construction_allowed','learning_rate_changed']:
        assert p[key] is False,key
    assert p['derivative_boundary_is_only_intended_objective_change'] is True
    return p


def verify_transfer(root,source,parent,pin,replay=None):
    import numpy as np
    root,source,parent=map(lambda q:Path(q).resolve(),(root,source,parent))
    replay=Path(replay).resolve() if replay is not None else root.parent/REPLAY_NAME
    p=read(root/'protocol.json');d=read(root/'design.json')
    assert sha(root/'protocol.json')==pin and sha(root/'design.json')==p['design_sha256']
    validate(p,d)
    for folder,key in [(root,'assets_sha256'),(source,'source_required_sha256'),
                       (parent,'parent_required_sha256'),(replay,'replay_required_sha256')]:
        verify_paths(folder,p[key])
    assert sha(source/'protocol.json')==SOURCE_PIN and sha(parent/'protocol.json')==PARENT_PIN
    assert sha(replay/'protocol.json')==REPLAY_PIN
    assert sha(root/'cctv_dgp_bank_hinge_boundary_v1_loss.py')==CORRECTED_HELPER_SHA
    assert sha(parent/'cctv_dgp_bank_photometric_guard_v1_loss.py')==LEGACY_HELPER_SHA
    old,data=read(source/'protocol.json'),read(parent/'protocol.json')
    assert p['case_ids']==old['case_ids']==data['preview_case_ids']
    assert p['reference_blocks']==old['reference_blocks'] and p['schedule']==old['schedule'] and len(p['schedule'])==50
    assert p['scientific_thresholds']==old['scientific_thresholds']
    assert p['trainable_parameter_names']==old['trainable_parameter_names']['A']
    assert read(source/'outputs/spatial_readout_v1/run_results.json')['complete']
    assert read(replay/'outputs/spatial_path_replay_v1/run_results.json')['complete']
    cases={c['id']:c for c in data['cases']}
    for b in p['reference_blocks']:
        assert len(b['case_ids'])==5
        assert all(cases[c]['role']=='train' and cases[c]['reference_id']==b['reference'] for c in b['case_ids'])
        assert [cases[c]['profile'] for c in b['case_ids']]==['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
    initial=np.load(source/'outputs/spatial_readout_v1/A_initial_vector.npy',allow_pickle=False)
    with np.load(source/'outputs/spatial_readout_v1/trajectory_A/update_001.npz',allow_pickle=False) as a:
        actual=a['actual']
        assert actual.shape==initial.shape==(609219,) and actual.dtype==initial.dtype==np.float32
        assert np.isfinite(actual).all() and np.isfinite(initial).all()
    assert float(np.linalg.norm(actual.astype(np.float64)-initial.astype(np.float64)))<=p['common_radius_L2']
    return p,old,data


def require_manual(root):
    assert platform.system()=='Linux','Human manual Linux L4 tmux only; no local model or derivative calls'
    assert platform.node().split('.')[0]=='forensic-dgp-thesis'
    assert Path(root).resolve()==(Path.home()/'forensic-dgp'/NAME).resolve()
    assert Path.cwd().resolve()==Path(root).resolve() and os.environ.get('TMUX')
    assert sys.prefix==str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')


def continuation_eligible(gates):
    # A research-trajectory stop, separate from the unchanged >=1% quality gate.
    if set(gates)!={'raw','png'}:return False
    return all(not g['preservation_failures'] and math.isfinite(g['global_structure_gain']) and g['global_structure_gain']>0
        and len(g['source_structure_gains'])==2 and all(math.isfinite(v) and v>0 for v in g['source_structure_gains'].values())
        and math.isfinite(g['brightness_gain_fraction']) and 0<=g['brightness_gain_fraction']<=.2 for g in gates.values())
