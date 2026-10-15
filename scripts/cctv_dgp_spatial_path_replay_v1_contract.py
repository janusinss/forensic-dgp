"""Frozen saved-state replay contract; hash checks never construct a model."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import stat
import sys

NAME = 'cctv_dgp_spatial_path_replay_v1_vm'
STEM = 'cctv-dgp-spatial-path-replay-v1'
SOURCE_NAME = 'cctv_dgp_spatial_readout_v1_vm'
PARENT_NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
DESIGN_SHA = '8937227984234db395779600ccd948d6b64fab6490d817ce5d7398c1c1fef38a'
SOURCE_PIN = '3bfbaf4b4a519f2926f0503731cd9d22390a7fc0c09da20e2f022e6a1b93c97e'
PARENT_PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
CHECKPOINT = '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
UPDATES = [0, 1, 10, 20, 30, 40, 50]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def array_sha(value):
    import numpy as np
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verify_paths(folder, bindings):
    folder = Path(folder).resolve()
    for name, digest in bindings.items():
        posix = PurePosixPath(name)
        assert isinstance(name, str) and ':' not in name and '\\' not in name
        assert not posix.is_absolute() and '..' not in posix.parts and str(posix) == name
        path = folder/name
        assert path.resolve().is_relative_to(folder) and not path.is_symlink()
        assert stat.S_ISREG(path.stat().st_mode) and sha(path) == digest, name


def validate(p, d):
    assert p['name'] == NAME and p['design_sha256'] == DESIGN_SHA
    assert p['source_protocol_sha256'] == SOURCE_PIN and p['parent_protocol_sha256'] == PARENT_PIN
    assert p['checkpoint_sha256'] == CHECKPOINT
    fields = ['source_required_sha256', 'parent_required_sha256', 'states', 'reference_blocks', 'case_ids',
        'scientific_thresholds', 'original_distance_maximum_fraction', 'common_radius_L2',
        'paired_TRAIN_cases_per_snapshot', 'snapshot_compositions', 'additional_fresh_original_controls',
        'visualization', 'budgets']
    assert all(p[key] == d[key] for key in fields), 'Frozen diagnostic changed'
    assert p['no_training_or_gradients'] is True and p['no_historical_train_call'] is True
    assert p['source_cwd_for_guarded_inference_only'] is True
    assert p['model_qualified'] is False and p['automatic_launch'] is False and p['new_incumbent_selection'] is False
    assert p['full_corpus_epochs'] == 0 and p['maximum_new_optimizer_updates'] == 0
    assert [(s['route'], s['update']) for s in p['states']] == [(r,u) for r in ['A','B'] for u in UPDATES]
    assert len(p['case_ids']) == len(set(p['case_ids'])) == 100
    assert len(p['reference_blocks']) == 20 and len(p['states']) == 14
    assert p['local_neural_construction_allowed'] is False
    assert p['export_external_seconds'] == p['budgets']['export_external_seconds'] == 630
    assert p['saved_audit_external_seconds'] == p['budgets']['saved_audit_seconds']+30 == 630
    return p


def selected_vector(source, state, initial):
    import numpy as np
    path = Path(source)/state['source_file']
    assert sha(path) == state['file_sha256']
    if state['npz_key'] is None:
        value = np.load(path, allow_pickle=False)
    else:
        with np.load(path, allow_pickle=False) as saved:
            value = saved[state['npz_key']].copy()
    assert value.shape == initial.shape == (state['elements'],)
    assert value.dtype == initial.dtype == np.float32 and np.isfinite(value).all() and np.isfinite(initial).all()
    assert array_sha(value) == state['vector_sha256']
    distance = float(np.linalg.norm(value.astype(np.float64)-initial.astype(np.float64)))
    assert distance <= state['scientific_radius_L2']
    assert abs(distance-state['original_distance_L2']) < 1e-12
    return value


def verify_transfer(root, source, parent, pin):
    import numpy as np
    root, source, parent = map(lambda p: Path(p).resolve(), (root, source, parent))
    assert sha(root/'protocol.json') == pin and sha(root/'design.json') == DESIGN_SHA
    p = validate(read(root/'protocol.json'), read(root/'design.json'))
    verify_paths(root, p['assets_sha256'])
    verify_paths(source, p['source_required_sha256'])
    verify_paths(parent, p['parent_required_sha256'])
    assert sha(source/'protocol.json') == SOURCE_PIN and sha(parent/'protocol.json') == PARENT_PIN
    original_p, original_data = read(source/'protocol.json'), read(parent/'protocol.json')
    assert p['case_ids'] == original_p['case_ids'] == original_data['preview_case_ids']
    assert p['reference_blocks'] == original_p['reference_blocks']
    assert read(source/'outputs/spatial_readout_v1/run_results.json')['complete']
    cases = {c['id']:c for c in original_data['cases']}
    for block in p['reference_blocks']:
        assert all(cases[c]['role'] == 'train' and cases[c]['reference_id'] == block['reference'] for c in block['case_ids'])
        assert [cases[c]['profile'] for c in block['case_ids']] == p['visualization']['rows']
    for route in ['A','B']:
        initial = np.load(source/('outputs/spatial_readout_v1/'+route+'_initial_vector.npy'), allow_pickle=False)
        for state in p['states']:
            if state['route'] == route:
                selected_vector(source, state, initial)
    return p, original_p, original_data


def require_manual(root):
    assert platform.system() == 'Linux', 'Human manual Linux L4 tmux replay only; no local neural construction'
    assert platform.node().split('.')[0] == 'forensic-dgp-thesis'
    assert Path(root).resolve() == (Path.home()/'forensic-dgp'/NAME).resolve()
    assert Path.cwd().resolve() == Path(root).resolve() and os.environ.get('TMUX')
    assert sys.prefix == str(Path.home()/'forensic-dgp/cctv_dgp_vm_bundle/.venv')


def label(state):
    assert state['route'] in ['A','B'] and state['update'] in UPDATES
    return state['route']+'_u'+str(state['update']).zfill(3)


def expected_state_hash(frozen_state, names, layout, vector):
    """Stream expected tensor bytes; no tensor/model mutation or construction."""
    import numpy as np
    assert [r['name'] for r in layout] == names
    slices = {r['name']: r for r in layout}
    h = hashlib.sha256()
    for name, tensor in sorted(frozen_state.items()):
        if name in slices:
            r = slices[name]
            assert list(tensor.shape) == r['shape'] and str(tensor.dtype) == 'torch.float32'
            array = vector[r['begin']:r['end']].reshape(r['shape'])
        else:
            array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode()+b'\0'+str(tensor.dtype).encode()+b'\0')
        h.update(str(tuple(tensor.shape)).encode()+b'\0'+np.ascontiguousarray(array).tobytes())
    return h.hexdigest()
