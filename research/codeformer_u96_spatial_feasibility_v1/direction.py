"""Inert common-descent arithmetic; NumPy is supplied explicitly by the caller.

This is a finite feasibility calculation, not an optimizer or a runtime worker.
Every task is mandatory. A failed numerical certificate is NOT a proof that no
common-descent direction exists. Even a successful certificate concerns only
the supplied mean vectors at their common state: it promises no finite model,
RAW, PNG, preservation, unseen-image or identity outcome.
"""
import hashlib
import math

from .objective import TASK_NAMES


STATUS = 'INERT_PROTOTYPE_UNSEALED_UNRUN_NOT_A_TRAINED_MODEL'
PARAMETER_SCHEMA = (
    ('base.generator.blocks.22.conv1.bias', (64,)),
    ('base.generator.blocks.22.conv1.weight', (64, 64, 3, 3)),
    ('base.generator.blocks.22.conv2.bias', (64,)),
    ('base.generator.blocks.22.conv2.weight', (64, 64, 3, 3)),
    ('base.generator.blocks.22.norm1.bias', (64,)),
    ('base.generator.blocks.22.norm1.weight', (64,)),
    ('base.generator.blocks.22.norm2.bias', (64,)),
    ('base.generator.blocks.22.norm2.weight', (64,)),
    ('base.generator.blocks.24.bias', (3,)),
    ('base.generator.blocks.24.weight', (3, 64, 3, 3)),
)
PARAMETER_COUNT = 75843
READOUT_START = 74112
READOUT_COUNT = 1731
STEP_MAXABS = (1e-5, 5e-5)
MAX_SUPPORT_SOLVES = 255
FLOAT64_EPS = 2.0 ** -52
MIN_GRADIENT_MAXABS = 2.0 ** -126
SIMPLEX_TOLERANCE = 512 * FLOAT64_EPS
LSTSQ_RCOND = 64 * 9 * FLOAT64_EPS


def need(condition, message):
    if not condition:
        raise ValueError(message)


def _gamma(count):
    """Standard conservative sequential floating-point accumulation bound."""
    product = count * FLOAT64_EPS
    return product / (1 - product)


# Unit vectors have norm one and their pairwise absolute dot sums are <= one.
# 64 times gamma_n deliberately covers dot accumulation, stable normalization,
# construction of an eight-vector convex sum and repeated scalar reductions.
# The least-squares solve itself is not trusted: its candidate must pass direct
# simplex, full-vector derivative and global convex-hull KKT checks afterward.
# These arithmetic margins are NOT tolerances on any image-quality gate.
UNIT_DOT_ERROR_BOUND = 64 * (_gamma(PARAMETER_COUNT) + _gamma(9))
KKT_GAP_BOUND = 4 * UNIT_DOT_ERROR_BOUND


def parameter_schema():
    """Exact sorted names, C-order flattening, no model introspection."""
    need(tuple(name for name, _ in PARAMETER_SCHEMA) ==
         tuple(sorted(name for name, _ in PARAMETER_SCHEMA)) and
         sum(math.prod(shape) for _, shape in PARAMETER_SCHEMA) == PARAMETER_COUNT and
         sum(math.prod(shape) for name, shape in PARAMETER_SCHEMA
             if '.blocks.24.' in name) == READOUT_COUNT,
         'Internal parameter packing contract differs')
    return PARAMETER_SCHEMA


def readout_subspace_mask(*, np):
    """Fixed old two-tensor control in the identical expanded coordinates."""
    parameter_schema()
    mask = np.zeros(PARAMETER_COUNT, dtype=bool)
    mask[READOUT_START:] = True
    return mask


def _read_gradients(gradients, *, np):
    need(type(gradients) is dict and set(gradients) == set(TASK_NAMES),
         'Exact eight declared role/component mean gradients required')
    rows = []
    for name in TASK_NAMES:
        value = gradients[name]
        need(isinstance(value, np.ndarray) and value.shape == (PARAMETER_COUNT,) and
             value.dtype.kind == 'f', 'Each mean needs a floating75843-vector')
        rows.append(value.astype(np.float64, copy=True))
    return np.stack(rows, axis=0)


def _digest(rows):
    digest = hashlib.sha256()
    for name, row in zip(TASK_NAMES, rows):
        digest.update(name.encode('ascii') + b'\0')
        digest.update(row.astype('<f8', copy=False).tobytes(order='C'))
    return digest.hexdigest()


def _prepare(rows, subspace, *, np):
    need(subspace in ('expanded', 'readout'), 'Only expanded and fixed readout are declared')
    result = {'status': STATUS, 'task_names': list(TASK_NAMES),
              'parameter_count': PARAMETER_COUNT, 'subspace': subspace,
              'gradient_sha256': _digest(rows), 'certified': False,
              'failure_is_infeasibility_proof': False,
              'support_solves': 0, 'maximum_support_solves': MAX_SUPPORT_SOLVES,
              'normalization': 'Each projected role/component mean to unitL2; all eight mandatory.',
              'unit_dot_error_bound': UNIT_DOT_ERROR_BOUND,
              'kkt_gap_bound': KKT_GAP_BOUND,
              'minimum_gradient_maxabs': MIN_GRADIENT_MAXABS}
    if not np.isfinite(rows).all():
        result['reason'] = 'Nonfinite mandatory mean gradient; no task may be dropped'
        return result, None
    projected = rows.copy()
    if subspace == 'readout':
        projected[:, :READOUT_START] = 0
    maximum = np.max(np.abs(projected), axis=1)
    result['projected_maxabs'] = maximum.tolist()
    if np.any(maximum < MIN_GRADIENT_MAXABS):
        result['reason'] = 'Zero or subnormal-scale mandatory gradient; normalization uncertified'
        return result, None
    scaled = projected / maximum[:, None]
    scaled_norm = np.sqrt(np.sum(scaled * scaled, axis=1, dtype=np.float64))
    # This division is stable without squaring the original potentially tiny or
    # huge mean. Its magnitude remains evidence, never a task-selection rule.
    unit = scaled / scaled_norm[:, None]
    with np.errstate(over='ignore', invalid='ignore'):
        norms = maximum * scaled_norm
    if not np.isfinite(norms).all() or not np.isfinite(unit).all():
        result['reason'] = 'Mandatory norm cannot be represented finitely'
        return result, None
    result['projected_norms'] = norms.tolist()
    return result, unit


def _evaluate(unit, weights, *, np):
    combined = np.sum(unit * weights[:, None], axis=0, dtype=np.float64)
    derivative = unit @ combined
    norm_squared = float(combined @ combined)
    gap = norm_squared - float(np.min(derivative))
    finite = (np.isfinite(combined).all() and np.isfinite(derivative).all() and
              math.isfinite(norm_squared) and math.isfinite(gap))
    certified = bool(finite and norm_squared > UNIT_DOT_ERROR_BOUND and
                     np.all(derivative > UNIT_DOT_ERROR_BOUND) and
                     abs(gap) <= KKT_GAP_BOUND)
    return {'combined_vector': combined, 'unit_gradient_dot_candidate': derivative.tolist(),
            'candidate_norm_squared': norm_squared,
            'convex_hull_kkt_gap': gap, 'certified': certified}


def solve_common_descent(gradients, *, np, subspace='expanded'):
    """Enumerate <=255 supports of eight unit means; certify their common descent.

    For a support S, solve [G_SS,-1;1^T,0][w,nu]=[0,1] by least squares.
    Keep simplex-valid candidates and choose the smallest Gram quadratic value.
    A final full-vector KKT gap checks global convex-hull optimality directly.
    If c is certified, displacement -c strictly decreases all eight supplied
    component means to first order. No favorable task subset is selected.
    """
    rows = _read_gradients(gradients, np=np)
    result, unit = _prepare(rows, subspace, np=np)
    if unit is None:
        return result
    gram = unit @ unit.T
    result['gram'] = gram.tolist()
    if not np.isfinite(gram).all():
        result['reason'] = 'Nonfinite Gram matrix'
        return result
    best = None
    rejected = 0
    failures = 0
    for bits in range(1, MAX_SUPPORT_SOLVES + 1):
        support = [i for i in range(8) if bits & (1 << i)]
        size = len(support)
        matrix = np.zeros((size + 1, size + 1), dtype=np.float64)
        matrix[:size, :size] = gram[np.ix_(support, support)]
        matrix[:size, size] = -1
        matrix[size, :size] = 1
        rhs = np.zeros(size + 1, dtype=np.float64)
        rhs[-1] = 1
        result['support_solves'] += 1
        try:
            solution = np.linalg.lstsq(matrix, rhs, rcond=LSTSQ_RCOND)[0]
        except np.linalg.LinAlgError:
            failures += 1
            continue
        local = solution[:size]
        if not np.isfinite(local).all() or np.min(local) < -SIMPLEX_TOLERANCE:
            rejected += 1
            continue
        local = np.maximum(local, 0)
        total = float(local.sum(dtype=np.float64))
        if not math.isfinite(total) or total <= 0:
            rejected += 1
            continue
        weights = np.zeros(8, dtype=np.float64)
        weights[support] = local / total
        value = float(weights @ gram @ weights)
        if math.isfinite(value) and (best is None or value < best[0]):
            best = (value, weights, support)
    result['negative_weight_or_invalid_supports'] = rejected
    result['failed_lstsq_supports'] = failures
    if best is None:
        result['reason'] = 'No numerical simplex candidate; feasibility remains unknown'
        return result
    _, weights, support = best
    result.update(_evaluate(unit, weights, np=np))
    result['weights'] = weights.tolist()
    result['selected_support'] = support
    with np.errstate(over='ignore', invalid='ignore'):
        original = rows @ result['combined_vector']
    result['mean_gradient_dot_candidate'] = original.tolist()
    if not np.isfinite(original).all():
        result['certified'] = False
    result['reason'] = ('All eight unit derivatives positive beyond arithmetic bound; KKT gap certified'
                        if result['certified'] else
                        'Common descent not numerically certified; not proof of nonexistence')
    return result


def compare_subspaces(gradients, *, np):
    """Two predeclared analytical controls; no additional neural graphs."""
    return {'status': STATUS, 'task_names': list(TASK_NAMES),
            'expanded': solve_common_descent(gradients, np=np, subspace='expanded'),
            'readout': solve_common_descent(gradients, np=np, subspace='readout'),
            'selection_after_observing_results': False,
            'readout_is_analytical_control_not_a_training_arm': True}


def _checked_solution(rows, solution, *, np):
    need(type(solution) is dict and solution.get('subspace') == 'expanded' and
         solution.get('gradient_sha256') == _digest(rows) and
         solution.get('task_names') == list(TASK_NAMES),
         'Expanded solution must bind these exact eight common-state means')
    result, unit = _prepare(rows, 'expanded', np=np)
    if unit is None or not solution.get('certified', False):
        return None
    weights = np.asarray(solution.get('weights'), dtype=np.float64)
    need(weights.shape == (8,) and np.isfinite(weights).all() and
         np.min(weights) >= 0 and abs(float(weights.sum()) - 1) <= SIMPLEX_TOLERANCE,
         'Finite nonnegative simplex weights required')
    # Reconstruct; a caller cannot swap the stored vector or true certificate.
    # Reuse the exact solver weights. A second normalization can perturb an
    # already valid floating simplex by one ULP and change its bound vector.
    evaluated = _evaluate(unit, weights, np=np)
    claimed = solution.get('combined_vector')
    need(isinstance(claimed, np.ndarray) and claimed.shape == (PARAMETER_COUNT,) and
         np.array_equal(claimed, evaluated['combined_vector']),
         'Stored candidate differs from its bound convex combination')
    return (unit, evaluated['combined_vector']) if evaluated['certified'] else None


def build_fp32_proposals(initial_fp32, gradients, solution, *, np):
    """Return both fixed, independent, direction-preserving FP32 proposals.

    Rounding is toward the SAME initializer whenever nearest rounding would
    overshoot the intended coordinate displacement. No coordinate exceeds its
    fixed maximum. Recheck all eight original means on each ACTUAL realized
    FP32 displacement. If either loses strict descent, apply neither: no scale
    search or successful-arm selection is hidden in this arithmetic helper.
    """
    need(isinstance(initial_fp32, np.ndarray) and
         initial_fp32.shape == (PARAMETER_COUNT,) and initial_fp32.dtype == np.float32 and
         np.isfinite(initial_fp32).all(), 'Exact finite FP32 initializer vector required')
    rows = _read_gradients(gradients, np=np)
    checked = _checked_solution(rows, solution, np=np)
    result = {'status': STATUS, 'step_maxabs': list(STEP_MAXABS),
              'same_initializer': True, 'cumulative': False,
              'optimizer_steps': 0, 'proposals': [], 'certified': False,
              'stop_before_any_proposal': True,
              'failure_is_infeasibility_proof': False}
    if checked is None:
        result['reason'] = 'Expanded common-descent certificate missing or failed'
        return result
    unit, combined = checked
    maximum = float(np.max(np.abs(combined)))
    need(math.isfinite(maximum) and maximum > 0, 'Nonzero finite certified candidate required')
    initial64 = initial_fp32.astype(np.float64)
    for scale in STEP_MAXABS:
        intended = -combined / maximum * scale
        ideal = initial64 + intended
        with np.errstate(over='ignore', invalid='ignore'):
            parameters = ideal.astype(np.float32)
        need(np.isfinite(parameters).all(), 'Nonfinite FP32 proposal; stop before application')
        realized = parameters.astype(np.float64) - initial64
        overshoot = np.abs(realized) > np.abs(intended)
        parameters[overshoot] = np.nextafter(parameters[overshoot], initial_fp32[overshoot])
        realized = parameters.astype(np.float64) - initial64
        need(np.isfinite(realized).all() and np.all(np.abs(realized) <= np.abs(intended)) and
             float(np.max(np.abs(realized))) <= scale,
             'Toward-initializer FP32 displacement exceeded its fixed coordinate cap')
        with np.errstate(over='ignore', invalid='ignore'):
            original_derivatives = rows @ realized
            original_products = np.sum(np.abs(rows * realized[None, :]), axis=1, dtype=np.float64)
        unit_derivatives = unit @ realized
        unit_products = np.sum(np.abs(unit * realized[None, :]), axis=1, dtype=np.float64)
        original_bounds = 64 * _gamma(PARAMETER_COUNT) * original_products
        unit_bounds = 64 * _gamma(PARAMETER_COUNT) * unit_products
        finite = (np.isfinite(original_derivatives).all() and np.isfinite(original_bounds).all() and
                  np.isfinite(unit_derivatives).all() and np.isfinite(unit_bounds).all())
        certified = bool(finite and np.any(realized != 0) and
                         np.all(original_derivatives < -original_bounds) and
                         np.all(unit_derivatives < -unit_bounds))
        result['proposals'].append({
            'maxabs_scale': scale, 'parameters_fp32': parameters,
            'intended_displacement_float64': intended, 'realized_displacement_float64': realized,
            'maximum_actual_absolute_displacement': float(np.max(np.abs(realized))),
            'toward_initializer_coordinates': int(overshoot.sum()),
            'mean_gradient_dot_realized_displacement': original_derivatives.tolist(),
            'mean_dot_roundoff_bounds': original_bounds.tolist(),
            'unit_gradient_dot_realized_displacement': unit_derivatives.tolist(),
            'unit_dot_roundoff_bounds': unit_bounds.tolist(),
            'certified': certified,
            'reason': ('Actual FP32 displacement strictly descends all eight stored means'
                       if certified else 'FP32 displacement lost certified common descent; stop')})
    result['certified'] = all(row['certified'] for row in result['proposals'])
    result['stop_before_any_proposal'] = not result['certified']
    result['reason'] = ('Both predeclared independent FP32 displacements pass stored-vector checks'
                        if result['certified'] else
                        'At least one fixed FP32 displacement failed; apply neither and retain stop')
    return result
