"""Pure supplied-pixel records for matched TRAIN banks; NumPy is explicit.

No decoder, neural framework, worker, filesystem payload or model is loaded.
Input/target file pins are validated and copied as caller-supplied bindings;
this arithmetic neither opens those files nor proves their pixel provenance.
The worker must resolve each exact case through the authenticated contract.
"""
import copy
from pathlib import PurePosixPath

from . import contract, objective, quality


VIEWS = quality.VIEWS
FIELDS = quality.FIELDS
TERM_TO_FIELD = (
    ('RGB_L1', 'mae'),
    ('adjacent_residual_L1', 'adjacent_rgb_error'),
    ('block16_RGB_mean_residual_L1', 'coarse_rgb_mae'),
)
INHERITED_ROLES = (
    'local_gaussian_1p2', 'global_gaussian_1p2', 'clear',
    'lowlight_linear_0p35', 'filter', 'dark',
)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return (type(value) is str and len(value) == 64
            and all(character in '0123456789abcdef' for character in value))


def _pin(value, label):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'},
         'Exact supplied ' + label + ' file pin required')
    path, size, digest = value['path'], value['bytes'], value['sha256']
    need(type(path) is str and bool(path) and '\\' not in path and ':' not in path,
         'Canonical relative ' + label + ' path required')
    parsed = PurePosixPath(path)
    need(path != '.' and not parsed.is_absolute() and parsed.as_posix() == path
         and all(part not in ('', '.', '..') for part in parsed.parts),
         'Canonical relative ' + label + ' path required')
    need(type(size) is int and size > 0 and _digest(digest),
         'Positive byte count and exact SHA256 ' + label + ' pin required')
    return copy.deepcopy(value)


def _scope(row, *, bank):
    need(type(row) is dict and row.get('bank') == bank
         and type(row.get('id')) is str and bool(row['id'])
         and type(row.get('reference_id')) is str and bool(row['reference_id'])
         and type(row.get('role')) is str and row.get('split') == 'TRAIN'
         and 'target' in row
         and row.get('case_key') == bank + ':' + row['id'],
         'Exact bank-qualified TRAIN row and case_key required')
    result = {key: row[key] for key in ('id', 'reference_id', 'role', 'split', 'bank', 'case_key')}
    result['input'] = _pin(row.get('input'), 'input')
    result['target'] = None if row.get('target') is None else _pin(row['target'], 'target')
    return result


def _fp32(value, *, np, label):
    need(isinstance(value, np.ndarray) and value.shape == (256, 256, 3)
         and value.dtype == np.dtype('float32') and np.isfinite(value).all()
         and value.min() >= 0 and value.max() <= 1,
         'Finite FP32 HWC RGB256 ' + label + ' in [0,1] required')
    return value


def _rgb8(value, *, np):
    need(isinstance(value, np.ndarray) and value.shape == (256, 256, 3)
         and value.dtype == np.dtype('uint8'), 'Exact decoded RGB8 HWC256 PNG pixels required')
    # Match the historical saved-PNG normalization before float64 metrics.
    return value.astype(np.float32) / np.float32(255)


def record_new(raw, png, target, row, *, np):
    """Measure both supplied views with the whole256 photographic objective.

    The exact unweighted terms become metric fields; no coefficient, role
    division, clipping, image resize or optimization is applied here. PNG
    pixels are measured independently rather than reconstructed from RAW.
    Caller row must contain the contract-resolved metadata plus the scheduled
    operation's explicit bank/case_key; old/new clear IDs remain distinct.
    """
    need(type(row) is dict and row.get('bank') in ('new_optimization', 'new_monitors'),
         'Only explicitly scoped new optimization or monitor rows are admitted')
    bank = row['bank']
    result = _scope(row, bank=bank)
    optimize = bank == 'new_optimization'
    references = contract.OPTIMIZATION_REFERENCES if optimize else contract.MONITOR_REFERENCES
    need(row['reference_id'] in references and row['role'] in objective.ROLES
         and row['id'] == row['reference_id'] + '__' + row['role']
         and row.get('cohort') == ('optimize' if optimize else 'monitor')
         and row.get('optimization_permitted') is optimize and result['target'] is not None
         and row.get('behavior_teacher') is None,
         'Exact new photographic TRAIN reference, role, cohort and target required')
    need(result['input']['path'] == 'data/inputs/' + row['id'] + '.png'
         and result['target']['path'] == 'data/targets/' + row['reference_id'] + '.png',
         'Exact contract-normalized new input and target paths required')
    observed = _fp32(raw, np=np, label='RAW')
    photograph = _fp32(target, np=np, label='photographic target')
    quantized = _rgb8(png, np=np)
    metrics = {}
    for view, pixels in (('raw', observed), ('native_png', quantized)):
        _, terms = objective.objective_numpy(pixels, photograph, row['role'], np=np)
        need(type(terms) is dict and set(terms) == {name for name, _ in TERM_TO_FIELD},
             'Exact three unweighted photographic NumPy terms required')
        metrics[view] = {field: quality._number(terms[name]) for name, field in TERM_TO_FIELD}
    result.update({'cohort': row['cohort'], 'optimization_permitted': optimize,
        'metrics': metrics, 'metric_region': 'whole256', 'supplied_pixel_metrics_only': True,
        'file_bytes_or_pixel_provenance_verified_here': False,
        'raw_png_delivery_consistency_verified_here': False,
        'inherited_complete_bank_gates_required': True, 'model_qualified': False})
    return result


def record_inherited(raw, png, target, row, *, np, frozen_U96_raw, frozen_U96_png,
                     teacher_raw=None, teacher_png=None):
    """Call the byte/origin-pinned old56 recorder without changing arithmetic.

    Original regional, null-target, teacher and frozen-U96 distance semantics
    come solely from the unchanged historical helper. This wrapper adds case
    scope and copied caller pins; it does not substitute the new three fields.
    The complete inherited bank evaluator still validates exact56 membership.
    """
    result = _scope(row, bank='old_quality')
    need(row['role'] in INHERITED_ROLES and row.get('cohort') in (None, 'monitor')
         and (row.get('optimization_permitted') is None or row.get('optimization_permitted') is False)
         and ((target is None) == (result['target'] is None))
         and (target is not None or row['role'] == 'dark'),
         'Only inherited evaluation-only TRAIN monitor/guard rows with exact target presence required')
    need(result['input']['path'].startswith('assets/')
         and (result['target'] is None or result['target']['path'].startswith('assets/')),
         'Exact inherited asset namespace required; new-bank paths refused')
    _fp32(raw, np=np, label='inherited RAW')
    if target is not None:
        _fp32(target, np=np, label='inherited photographic target')
    measured = quality.inherited_metrics().record_metrics(
        raw, png, target, row, np=np,
        frozen_U96_raw=frozen_U96_raw, frozen_U96_png=frozen_U96_png,
        teacher_raw=teacher_raw, teacher_png=teacher_png)
    result.update(copy.deepcopy(measured))
    result.update({'supplied_pixel_metrics_only': True,
        'file_bytes_or_pixel_provenance_verified_here': False,
        'raw_png_delivery_consistency_verified_here': False,
        'inherited_complete_bank_gates_required': True, 'model_qualified': False})
    return result
