"""One homogeneous saved-gradient ray search and exact rounded-state certificates.

No neural package is imported.  The public experiment entrypoint requires the
manual Linux/L4 contract before examining real gradients or constructing a ray.
Small pure-array helpers exist for synthetic fixtures and saved-data auditing.
The LP is a stricter sufficient subset, not a solve of heterogeneous finite
scientific slack.  A failed candidate is not an infeasibility certificate.
"""
from fractions import Fraction
import hashlib
import itertools
import math
import time

import numpy as np

LABELS = ('quarter', 'half', 'full')
AMPLITUDES = (.25, .5, 1.)
LAYOUT = [*[f'MSE_{i}' for i in range(5)], *[f'SSIM_loss_{i}' for i in range(5)],
    *[f'ArcFace_loss_{i}' for i in range(5)], *[f'landmark_structure_{i}' for i in range(1,5)],
    *[f'channel_mean_{i}_{c}' for i in range(5) for c in range(3)], 'objective']


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def rational_record(value):
    value = Fraction(value)
    return dict(numerator=str(value.numerator), denominator=str(value.denominator), approximation=float(value))


def exact_sum(values):
    """Sum finite binary64 inputs exactly using bounded int64 exponent bins."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if not np.isfinite(values).all(): raise ValueError('Exact sum requires finite numbers')
    values = values[values != 0.]
    if not values.size: return Fraction(0)
    mantissa, exponent = np.frexp(values)
    integers = np.ldexp(mantissa, 53).astype(np.int64)
    lowest = int(exponent.min())-53
    total = 0
    for exp in np.unique(exponent):
        selected = integers[exponent == exp]
        blocks = selected.size//256
        # Each block has at most256 signed integers of magnitude <=2**53;
        # its exact sum fits int64.  Only block totals enter Python integers.
        subtotal = sum(map(int, selected[:blocks*256].reshape(blocks,256).sum(1,dtype=np.int64)))
        subtotal += int(selected[blocks*256:].sum(dtype=np.int64))
        total += subtotal << (int(exp)-53-lowest)
    return Fraction(total << lowest, 1) if lowest >= 0 else Fraction(total, 1 << -lowest)


def exact_float32_dot(left, right):
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype != np.float32 or right.dtype != np.float32 or left.shape != right.shape:
        raise ValueError('Exact endpoint dot requires equal-shaped float32 arrays')
    # A product of two finite binary32 values has <=48 significant bits and
    # exponent >=-298; therefore it is represented exactly in binary64.
    return exact_sum(left.astype(np.float64)*right.astype(np.float64))


def exact_endpoint_dot(gradient, initial, placed):
    return exact_float32_dot(gradient, placed)-exact_float32_dot(gradient, initial)


def exact_radius_proof(delta64, radius):
    """Exact dyadic square sum for an explicitly provided binary64 vector."""
    delta64 = np.asarray(delta64)
    if delta64.dtype != np.float64 or delta64.ndim != 1 or not np.isfinite(delta64).all():
        raise ValueError('Radius proof requires a finite float64 vector')
    radius = float(radius)
    if not math.isfinite(radius) or radius <= 0: raise ValueError('Positive finite radius required')
    ratios = [float(v).as_integer_ratio() for v in delta64 if v != 0.]
    if not ratios: raise ValueError('Zero displacement is inadmissible')
    scale = max(d.bit_length()-1 for _,d in ratios)
    square = Fraction(sum((n << (scale-(d.bit_length()-1)))**2 for n,d in ratios),1 << (2*scale))
    allowed = Fraction.from_float(radius)**2
    if square > allowed: raise ValueError('Exact squared displacement exceeds unchanged radius')
    return dict(complete=True, exact_squared_norm=rational_record(square), exact_squared_radius=rational_record(allowed),
                exact_nonzero=True, exact_radius_pass=True, displacement_sha256=array_sha(delta64))


def exact_endpoint_radius(initial, placed, radius):
    square = (exact_float32_dot(placed,placed)+exact_float32_dot(initial,initial)
              -2*exact_float32_dot(initial,placed))
    allowed = Fraction.from_float(float(radius))**2
    return dict(complete=True, exact_squared_norm=rational_record(square), exact_squared_radius=rational_record(allowed),
        exact_nonzero=square > 0, exact_radius_pass=0 < square <= allowed,
        initial_sha256=array_sha(initial), placed_sha256=array_sha(placed))


def exact_photo_coefficients(baseline, target, observed):
    """Exact real-valued clipped-mean derivative coefficients per sign/channel.

    Returns5x3x2 Fractions, where last-axis0 applies to nonnegative mean shift,
    and1 applies to negative mean shift.  Baseline0 blocks negative movement;
    baseline1 blocks positive movement.  Coefficients contain exact prepared
    binary32 pixels and the exact3*observed_count divisor, not rounded proxies.
    """
    baseline, target, observed = map(np.asarray,(baseline,target,observed))
    if baseline.dtype != np.float32 or target.dtype != np.float32 or baseline.shape != target.shape:
        raise ValueError('Prepared baseline and target must have matching float32 shapes')
    if baseline.ndim != 4 or baseline.shape[:2] != (5,3) or observed.shape != (5,1,*baseline.shape[2:]):
        raise ValueError('Require five NCHW RGB profiles and N1HW observed mask')
    if not all(np.isfinite(x).all() for x in (baseline,target,observed)):
        raise ValueError('Nonfinite photometric input')
    if not ((baseline >= 0).all() and (baseline <= 1).all() and (target >= 0).all() and (target <= 1).all()
            and np.isin(observed,[0,1]).all()): raise ValueError('Photometric range or binary mask differs')
    result=[]
    for profile in range(5):
        mask=observed[profile,0].astype(bool); count=int(mask.sum())
        if count <= 0: raise ValueError('Each profile needs observed pixels')
        channels=[]
        for channel in range(3):
            base=baseline[profile,channel]; clean=target[profile,channel]
            pairs=[]
            for allowed in (mask & (base != 1.), mask & (base != 0.)):
                pairs.append(2*(exact_sum(base[allowed])-exact_sum(clean[allowed]))/(3*count))
            if pairs[0] > pairs[1]: raise ValueError('Clip concavity invariant failed')
            channels.append(pairs)
        result.append(channels)
    return result


def branch_coefficients(photo_coefficients):
    """All8 affine pieces of concave clipped-mean MSE slope per profile."""
    return [[tuple(photo_coefficients[i][c][signs[c]] for c in range(3))
             for signs in itertools.product((0,1),repeat=3)] for i in range(5)]


def build_lp(gradients, structure_baselines, photo_coefficients, interior_margin):
    """Build one bounded homogeneous-ray LP; never consumes finite gate slack."""
    gradients=np.asarray(gradients)
    if gradients.dtype != np.float32 or gradients.ndim != 3 or gradients.shape[:2] != (2,35):
        raise ValueError('Require two35-row float32 gradient matrices')
    if not np.isfinite(gradients).all(): raise ValueError('Nonfinite saved gradients')
    base=np.asarray(structure_baselines,dtype=np.float64)
    if base.shape != (2,4) or not np.isfinite(base).all() or (base <= 0).any():
        raise ValueError('Positive finite structure baselines required for all8 terms')
    margin=float(interior_margin)
    if not math.isfinite(margin) or margin <= 0: raise ValueError('Positive prospective interior margin required')
    flat=gradients.reshape(70,-1).astype(np.float64)
    norms=np.linalg.norm(flat,axis=1)
    if not np.isfinite(norms).all(): raise ValueError('Basis gradient norms must be finite')
    zero_basis=np.flatnonzero(norms == 0.).tolist()
    basis=flat/np.where(norms > 0.,norms,1.)[:,None]
    projections=(flat@basis.T).reshape(2,35,70)
    rows=[];names=[];scales=[];zero=[]
    def append(row,name,structure=False):
        row=np.asarray(row,dtype=np.float64)
        maximum=float(np.abs(row).max())
        if not math.isfinite(maximum): raise ValueError('Nonfinite projected constraint')
        scale=math.ldexp(1.,-math.frexp(maximum)[1]) if maximum else 1.
        # Power-of-two scaling is for the numerical candidate finder only.
        # Every admitted state is checked from unscaled exact endpoint products.
        tail=scale if structure else (margin if maximum else 0.)
        rows.append(np.r_[row*scale,tail]);names.append(name);scales.append(scale)
        if not maximum: zero.append(name)
    for ref in range(2):
        for scalar in range(15): append(projections[ref,scalar],f'r{ref}/{LAYOUT[scalar]}')
        for profile in range(1,5):
            for branch,coeffs in enumerate(branch_coefficients(photo_coefficients[ref])[profile]):
                photo=sum(float(coeffs[c])*projections[ref,19+3*profile+c] for c in range(3))
                append(.2*projections[ref,profile]-photo,f'r{ref}/brightness_{profile}/branch{branch}')
        for profile in range(4): append(projections[ref,15+profile]/base[ref,profile],f'r{ref}/structure_{profile+1}',True)
    matrix=np.asarray(rows)
    if matrix.shape != (102,71): raise ValueError('Expected30 preservation+64 brightness+8 structure constraints')
    return dict(A_ub=matrix,b_ub=np.zeros(102),c=np.r_[np.zeros(70),-1.],bounds=[(-1.,1.)]*70+[(0.,None)],basis=basis,
        receipt=dict(constraint_names=names,power2_row_scales=scales,zero_projected_rows=zero,
            basis_gradient_norms=norms.tolist(),basis_zero_indices=zero_basis,effective_basis_rank_claimed=False,
            basis_normalization='float64 component L2; zero rows retained with divisor1; candidate arithmetic only',
            minimum_nonzero_scaled_coefficient=float(np.abs(matrix[matrix != 0]).min()),
            coefficients_below_solver_default_small_matrix_value=int(((np.abs(matrix)<1e-9)&(matrix != 0)).sum()),
            conservative_subset='all30 initial profile slopes nonpositive; all8 clipped-mean brightness slopes nonpositive',
            finite_scientific_slack_consumed=False,coefficient_box_is_only_homogeneous_ray_normalizer=True,
            exact_L2_radius_replaced_by_box=False,full_decoder_infeasibility_certified=False))


def solve_lp(lp, options):
    """One call; solver success only nominates a candidate for exact checking."""
    from scipy.optimize import linprog
    allowed=['time_limit','maxiter','primal_feasibility_tolerance','dual_feasibility_tolerance','presolve']
    started=time.monotonic()
    result=linprog(lp['c'],A_ub=lp['A_ub'],b_ub=lp['b_ub'],bounds=lp['bounds'],
                   method='highs-ds',options={key:options[key] for key in allowed})
    receipt=dict(complete=True,solver_calls=1,method='highs-ds',status=int(result.status),message=str(result.message),
        success=bool(result.success),iterations=int(result.nit),seconds=time.monotonic()-started,
        objective=None if result.fun is None else float(result.fun),options={key:options[key] for key in allowed},
        solver_success_is_certificate=False,full_decoder_infeasibility_certified=False)
    candidate=None if not result.success or result.x is None or not np.isfinite(result.x).all() or result.x[-1] <= 0 else result.x
    if candidate is not None:
        receipt.update(coefficients=candidate[:-1].tolist(),common_normalized_structure_slope=float(candidate[-1]),
            approximate_max_scaled_primal_residual=float((lp['A_ub']@candidate-lp['b_ub']).max()))
    return candidate,receipt


def certify_placement(initial, placed, gradients, structure_baselines, photo_coefficients, radius, minimum_gain, clock=lambda:None):
    """Check actual float32 state against exact saved-gradient/clip surrogate."""
    initial,placed,gradients=map(np.asarray,(initial,placed,gradients))
    if initial.dtype != np.float32 or placed.dtype != np.float32 or initial.ndim != 1 or placed.shape != initial.shape:
        raise ValueError('Require actual float32 initial and placed vectors')
    if gradients.dtype != np.float32 or gradients.shape != (2,35,initial.size): raise ValueError('Gradient certificate shape differs')
    radius_proof=exact_endpoint_radius(initial,placed,radius)
    rows=[];passes=radius_proof['exact_radius_pass']
    for ref in range(2):
        slopes=[]
        for scalar in range(34):
            clock();slopes.append(exact_endpoint_dot(gradients[ref,scalar],initial,placed))
        preservation=[s <= 0 for s in slopes[:15]]
        gains=[-slopes[15+i]/Fraction.from_float(float(structure_baselines[ref][i])) for i in range(4)]
        structure=[gain >= Fraction.from_float(float(minimum_gain)) if minimum_gain else gain > 0 for gain in gains]
        brightness=[];mean_slopes=[]
        for profile in range(1,5):
            means=slopes[19+3*profile:19+3*profile+3]
            mean=sum(photo_coefficients[ref][profile][c][0 if means[c] >= 0 else 1]*means[c] for c in range(3))
            residual=max(Fraction(0),-mean)+Fraction.from_float(.2)*slopes[profile]
            mean_slopes.append(mean);brightness.append(residual)
        passed=all(preservation) and all(structure) and all(v <= 0 for v in brightness)
        passes=passes and passed
        rows.append(dict(reference_index=ref,pass_all=passed,preservation_pass=preservation,structure_pass=structure,
            exact_scalar_slopes=[rational_record(s) for s in slopes],normalized_structure_gains=[rational_record(g) for g in gains],
            mean_only_MSE_slopes=[rational_record(s) for s in mean_slopes],brightness_residuals=[rational_record(s) for s in brightness]))
    return dict(complete=True,admitted=bool(passes),radius=radius_proof,reference_certificates=rows,
        full_minimum_predicted_structure_gain=float(minimum_gain),exact_endpoint_arithmetic=True,
        exact_prepared_pixel_clip_coefficients=True,certificate_scope='Saved selected neural-branch linear surrogate with idealized real-valued clipped mean control',
        neural_branch_or_finite_loss_certified=False,literal_float32_derivative_certified=False,
        PNG_quality_certified=False,model_qualified=False,scientific_gates_changed=False)


def select_direction(initial, gradient_matrices, measurements, photo_inputs, p, clock, manual_root, *, on_proposal=None, on_solver=None):
    """Guarded one-ray experiment, invoked only by the new manual VM worker."""
    from cctv_dgp_decoder_feasibility_v1_contract import require_manual
    require_manual(manual_root)
    import scipy
    if scipy.__version__ != '1.15.3' or np.__version__ != '1.26.4': raise ValueError('Frozen VM NumPy/SciPy versions differ')
    options=p['solver'];initial=np.asarray(initial);gradients=np.asarray(gradient_matrices)
    if initial.dtype != np.float32 or initial.shape != (609219,) or gradients.shape != (2,35,609219):
        raise ValueError('Actual15-tensor decoder shape differs')
    if list(p['placements']) != [0.,*AMPLITUDES] or list(options['placements']) != list(AMPLITUDES):
        raise ValueError('Fixed initial and three placements differ')
    if [m['reference'] for m in measurements] != ['tr_asian_00176','tr_ffhq_53945']:
        raise ValueError('Both fixed derivative references are required in order')
    started=time.monotonic();clock()
    photo=[exact_photo_coefficients(v['baseline'],v['target'],v['observed']) for v in photo_inputs]
    structure=[m['scalar_values'][15:19] for m in measurements]
    lp=build_lp(gradients,structure,photo,options['interior_margin_fraction']);clock()
    if time.monotonic()-started > p['budgets']['basis_seconds']: raise TimeoutError('Basis construction stop')
    candidate,solver=solve_lp(lp,options);solver['construction']=lp['receipt']
    if on_solver is not None:on_solver(solver)
    if solver['seconds'] > p['budgets']['solver_seconds']:
        raise TimeoutError('Completed solver exceeded prospective wall-time stop; receipt retained')
    clock()
    result=dict(admitted=False,admitted_to_neural=False,direction64=None,vectors={},certificate=dict(complete=True,admitted=False,reason='No certifiable nonzero LP ray',model_qualified=False),solver_receipt=solver)
    if candidate is None:return result
    ray=candidate[:-1]@lp['basis'];norm=float(np.linalg.norm(ray))
    if not math.isfinite(norm) or norm <= 0:return result
    direction=ray*(float(p['common_radius_L2'])*options['radius_interior_fraction']/norm)
    if not np.isfinite(direction).all():return result
    vectors={label:(initial.astype(np.float64)+amplitude*direction).astype(np.float32) for label,amplitude in zip(LABELS,AMPLITUDES)}
    # Preserve the only proposed ray and all three rounded states before the
    # expensive exact certificate.  A timeout/kill must not erase failed states.
    if on_proposal is not None:on_proposal(direction,vectors,solver)
    certificates={};started=time.monotonic()
    for label in LABELS:
        clock()
        certificates[label]=certify_placement(initial,vectors[label],gradients,structure,photo,p['common_radius_L2'],
            options['full_minimum_predicted_structure_gain'] if label=='full' else 0.,clock)
        if time.monotonic()-started > p['budgets']['certificate_seconds']: raise TimeoutError('Exact certificate stop')
    admitted=all(c['admitted'] for c in certificates.values())
    result.update(admitted=admitted,admitted_to_neural=admitted,direction64=direction,vectors=vectors,
        certificate=dict(complete=True,admitted=admitted,reason='Exact surrogate certificates passed' if admitted else 'Rounded placement failed exact surrogate certificate; no retry',
            placements=certificates,ideal_direction_sha256=array_sha(direction),model_qualified=False,
            exact_photo_coefficients=[[[[rational_record(v) for v in pair] for pair in profile] for profile in ref] for ref in photo],
            actual_finite_RAW_PNG_checks_still_required=True,full_decoder_infeasibility_certified=False))
    return result
