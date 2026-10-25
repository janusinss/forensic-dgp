"""Read-only source/data verification and exclusive prospective-scope receipt.

No model framework, checkpoint decoding, neural, gradient, optimizer or VM work.
This prepares evidence, NOT an executable archive or launch instructions.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import stat
import sys

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_u96_spatial_feasibility_v1 import contract, model, objective

PARENT = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm'
INITIALIZERS = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm'
DESTINATION = ROOT / 'outputs/codeformer_u96_next_spatial_design_v1/prospective_scope_readback_v1.json'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def regular(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink() or getattr(info, 'st_file_attributes', 0) & 1024:
        raise ValueError('Regular nonlinked source required: ' + str(path))
    current = path.parent
    while current != ROOT:
        if current == current.parent or current.is_symlink() or getattr(current.lstat(), 'st_file_attributes', 0) & 1024:
            raise ValueError('Source ancestry escapes or links the workspace')
        current = current.parent


def pin(path):
    regular(path)
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def main():
    assert 'torch' not in sys.modules
    contract.need(not DESTINATION.exists(), 'Prior scope receipts are immutable')
    parent_bytes = (PARENT / 'protocol.json').read_bytes()
    selected = contract.select_parent_cases(parent_bytes)
    parent = json.loads(parent_bytes)
    calls = contract.operation_plan(selected)
    checks = 0
    bindings = []
    pixels = []
    seen = set()

    def verify(base, row):
        nonlocal checks
        contract.need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'Exact file binding schema')
        relative = PurePosixPath(row['path'])
        contract.need(not relative.is_absolute() and '..' not in relative.parts and '\\' not in row['path'],
                      'Normalized confined relative file name')
        path = base / row['path']
        actual = pin(path)
        contract.need(actual['bytes'] == row['bytes'] and actual['sha256'] == row['sha256'],
                      'Retained source bytes changed: ' + str(path))
        checks += 1
        key = actual['path']
        if key not in seen:
            seen.add(key)
            bindings.append(actual)
        return path

    # Every retained parent source/data/evidence binding, not just selected outputs.
    for name, row in parent['files'].items():
        if name.startswith(('assets/', 'evidence/', 'third_party/')):
            verify(PARENT, {'path': name, **row})
    for row in parent['model'].values():
        verify(INITIALIZERS, row)  # bytes only; do not deserialize checkpoints

    for row in selected['optimization'] + selected['monitors'] + selected['guards']:
        for kind in ('input', 'target'):
            binding = row[kind]
            if binding is None:
                contract.need(row in selected['guards'] and row['role'] == 'dark' and kind == 'target',
                              'Only inherited dark guards can lack a clean target')
                checks += 1
                continue
            path = verify(PARENT, binding)
            with Image.open(path) as source:
                contract.need(source.format == 'PNG' and source.size == (256, 256) and source.mode == 'RGB',
                              'Exact RGB256 PNG source required')
                digest = hashlib.sha256(source.tobytes()).hexdigest()
            contract.need(digest == row[kind + '_RGB8_sha256'], 'Decoded photographic pixel binding differs')
            pixels.append({'id': row['id'], 'kind': kind, 'RGB8_sha256': digest})
            checks += 1

    contract.need(tuple(objective.TASK_NAMES) == tuple(role + '/' + component for role in contract.ROLES
                                                    for component in ('rgb', 'adjacent')),
                  'Eight explicit component task order differs')
    checks += 1
    prototype = ROOT / 'research/codeformer_u96_spatial_feasibility_v1'
    sources = []
    for path in sorted(prototype.glob('*.py')):
        ast.parse(path.read_bytes(), feature_version=(3, 10))
        sources.append(pin(path))
        checks += 1
    result = {
        'complete': True, 'recorded_UTC': datetime.now(timezone.utc).isoformat(),
        'status': contract.STATUS, 'checks': checks,
        'parent_protocol': pin(PARENT / 'protocol.json'), 'data_bindings': bindings,
        'decoded_pixel_bindings': pixels, 'selected_cases': selected,
        'operation_plan': calls, 'prospective_contract': contract.prospective_contract(),
        'model_contract': model.architecture_contract(), 'sources': sources + [pin(Path(__file__))],
        'training_input_only_admission_changed': False,
        'photograph_ID_separation_is_person_identity_or_pretraining_overlap_proof': False,
        'monitor_is_independent_holdout': False,
        'new_pixels_prepared': 0, 'checkpoint_deserializations': 0,
        'Torch_model_neural_gradient_optimizer_training_VM_calls': 0,
        'executable_packet_or_guide_ready': False, 'model_qualified': False,
        'goal_complete': False,
    }
    contract.need('torch' not in sys.modules, 'No framework import in scope readback')
    with DESTINATION.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'complete': True, 'checks': checks, 'bindings': len(bindings),
                      'pixel_bindings': len(pixels), 'forward_slots': len(calls),
                      'receipt': DESTINATION.relative_to(ROOT).as_posix(),
                      'model_training_VM_calls': 0}))


if __name__ == '__main__':
    main()
