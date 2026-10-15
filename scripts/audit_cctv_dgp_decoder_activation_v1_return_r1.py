"""Local schema and exact-radius adapter around the retained saved-data auditor.

Adds the four descriptive fields emitted by the hash-bound historical gate.
Replaces one platform-dependent approximate norm equality with an exact radius
proof; the saved approximate norm is explicitly excluded from certified metadata.
Only finite_arithmetic.objective reduction replay uses the already frozen
absolute 1e-10 scalar bound; all its other leaves retain exact comparison.
All numerical gates, original checker source, thresholds and data remain exact.
No learned model, gradient or optimizer is constructed or executed.
"""
import argparse
import ast
import copy
from decimal import Decimal, localcontext
from fractions import Fraction
import json
from pathlib import Path
import types

import audit_cctv_dgp_decoder_activation_v1_return as original
from cctv_dgp_decoder_activation_v1_contract import read, write, sha, verify_transfer

ORIGINAL_SHA = '262a18ed9f4b8b96ca40678c405bfc16df5b782fc9adf3894784c3fc5dadd4a4'
FROZEN_GATE_SHA = '9741ea8e6eacebc0148131fb6a72715b9b6350d432f558abe19f9f856bfcf331'
FROZEN_GATE_FILE = 'cctv_dgp_spatial_readout_v1_contract.py'
SCIENTIFIC_THRESHOLDS = dict(early_structure_gain=.01, final_structure_gain=.1,
    MSE_regression_tolerance=1e-12, SSIM_ArcFace_regression_tolerance=1e-6,
    brightness_fraction_maximum=.2)
ADDED_FIELDS = dict(scientific_thresholds=SCIENTIFIC_THRESHOLDS,
    numerical_sample_gate_only=True, model_qualified=False, automatic_extra_epochs=False)
NUMERICAL_FIELDS = {'pass','global_structure_gain','source_structure_gains',
                    'brightness_gain_fraction','preservation_failures'}
ORIGINAL_GATE = original.unchanged_gate
NORM_ASSERTION = "direction_record['norm_L2'] == float(np.linalg.norm(direction)) <= p['common_radius_L2']"
FINITE_CALL = "close_tree(finite_saved, row['finite_arithmetic'], atol=0, rtol=0)"
REDUCTION_ENUMERATION_SHA = '4bc869281c623c9e097003116579db0e42ecdf918efcbd64e4c2f418a3308207'


def exact_direction_radius(direction,record,p):
    """Exact dyadic radius proof; approximate saved norm is kept unverified."""
    import numpy as np
    assert direction.dtype == np.float64 and direction.ndim==1 and np.isfinite(direction).all()
    assert record['direction_sha256']==original.array_sha(direction)
    assert record['dtype']=='float64' and record['elements']==direction.size
    assert record['new_parameter_proposal'] is False
    # All finite binary64 values are dyadic. Accumulate exact integer squares
    # at a common denominator; do not use a platform reduction to certify radius.
    ratios = [float(v).as_integer_ratio() for v in direction if v != 0.]
    assert ratios, 'Retained direction must be nonzero'
    exponent = max(denominator.bit_length()-1 for _,denominator in ratios)
    square_sum = sum((numerator << (exponent-(denominator.bit_length()-1)))**2
                     for numerator,denominator in ratios)
    squared = Fraction(square_sum,1 << (2*exponent))
    radius = Fraction.from_float(float(p['common_radius_L2']))
    assert squared > 0 and squared <= radius*radius, 'Exact dyadic retained direction exceeds unchanged scientific radius'
    approximations = []
    for precision in [110,160]:
        with localcontext() as context:
            context.prec = precision
            value = (Decimal(squared.numerator)/Decimal(squared.denominator)).sqrt()
            approximations.append(dict(precision=precision,decimal=str(value),rounded_binary64=float(value)))
    assert approximations[0]['rounded_binary64']==approximations[1]['rounded_binary64']
    saved = record['norm_L2']
    assert type(saved) is float and math_is_finite_positive(saved)
    local = float(np.linalg.norm(direction))
    exact = approximations[-1]['rounded_binary64']
    import math
    lower = (Fraction.from_float(math.nextafter(exact,-math.inf))+Fraction.from_float(exact))/2
    upper = (Fraction.from_float(exact)+Fraction.from_float(math.nextafter(exact,math.inf)))/2
    assert lower*lower < squared < upper*upper, 'Correct rounding needs strict exact midpoint proof; ties remain unsupported'
    return dict(complete=True,direction_elements=int(direction.size),direction_array_sha256=original.array_sha(direction),
        nonzero_elements=len(ratios),common_dyadic_denominator_exponent=exponent,
        exact_squared_norm_numerator=str(squared.numerator),exact_squared_norm_denominator=str(squared.denominator),
        exact_radius_numerator=str(radius.numerator),exact_radius_denominator=str(radius.denominator),
        exact_squared_norm_le_unchanged_squared_radius=True,
        original_radius_L2=p['common_radius_L2'],saved_approximate_norm=saved,
        local_BLAS_approximate_norm=local,converged_correct_rounded_norm=exact,
        sqrt_convergence=approximations,metadata_bitwise_replay=saved==local,
        exact_binary64_rounding_midpoint_proof=dict(lower_midpoint_numerator=str(lower.numerator),lower_midpoint_denominator=str(lower.denominator),
            upper_midpoint_numerator=str(upper.numerator),upper_midpoint_denominator=str(upper.denominator),
            exact_squared_norm_strictly_between_squared_midpoints=True,tie_case=False),
        saved_norm_matches_correct_rounded=saved==exact,saved_norm_approximate_unverified=True,
        saved_metadata_rewritten=False,ULP_acceptance_tolerance_used=False,
        scientific_radius_or_tolerance_changed=False)


def math_is_finite_positive(value):
    import math
    return math.isfinite(value) and value>0


def compare_finite_objective_only(actual,saved,bound):
    """One named reduction scalar uses its pre-existing absolute replay bound."""
    import math
    assert bound==1e-10 and actual.keys()==saved.keys() and 'objective' in actual
    other_actual = {key:value for key,value in actual.items() if key!='objective'}
    other_saved = {key:value for key,value in saved.items() if key!='objective'}
    original.close_tree(other_actual,other_saved,atol=0,rtol=0)
    a,b = actual['objective'],saved['objective']
    assert type(a) is float and type(b) is float and math.isfinite(a) and math.isfinite(b)
    error = abs(a-b)
    assert error<=bound, 'Named objective scalar exceeds existing absolute replay bound'
    return dict(local_objective=a,saved_objective=b,absolute_difference=error,
        existing_absolute_bound=bound,relative_tolerance=0.,original_exact_objective_check_passed=a==b,
        all_other_leaves_exact=True,allowed_leaf_path='finite_arithmetic.objective',
        new_or_wider_bound=False,scientific_gate_changed=False)


def load_adapter_module(norm_proofs,objective_proofs=None):
    """Compile only the two explicitly declared source-bound local checks."""
    assert sha(original.__file__)==ORIGINAL_SHA
    tree = ast.parse(Path(original.__file__).read_text(encoding='utf-8'))
    matches = [node for node in ast.walk(tree) if isinstance(node,ast.Assert)
               and ast.unparse(node.test)==NORM_ASSERTION]
    assert len(matches)==1 and matches[0].lineno==315, 'Only the known norm metadata assertion may change'
    def verify(direction,record,p):
        norm_proofs.append(exact_direction_radius(direction,record,p))
        return True
    matches[0].test = ast.Call(func=ast.Name(id='_r1_verify_exact_direction_radius',ctx=ast.Load()),
        args=[ast.Name(id='direction',ctx=ast.Load()),ast.Name(id='direction_record',ctx=ast.Load()),ast.Name(id='p',ctx=ast.Load())],keywords=[])
    calls = [node for node in ast.walk(tree) if isinstance(node,ast.Call) and ast.unparse(node)==FINITE_CALL]
    assert len(calls)==1 and calls[0].lineno==381, 'Only the named objective reduction replay may change'
    if objective_proofs is None:
        objective_proofs=[]
    def verify_objective(actual,saved,p,row,label):
        proof = compare_finite_objective_only(actual,saved,p['objective_replay_max_abs'])
        objective_proofs.append(dict(state=label,reference=row['reference'],**proof))
        return 0.
    calls[0].func = ast.Name(id='_r1_compare_finite_objective_only',ctx=ast.Load())
    calls[0].args += [ast.Name(id='p',ctx=ast.Load()),ast.Name(id='row',ctx=ast.Load()),ast.Name(id='label',ctx=ast.Load())]
    calls[0].keywords=[]
    module = types.ModuleType('retained_activation_saved_auditor_local_schema_r1')
    module.__file__ = original.__file__
    module.__dict__['_r1_verify_exact_direction_radius'] = verify
    module.__dict__['_r1_compare_finite_objective_only'] = verify_objective
    exec(compile(ast.fix_missing_locations(tree),original.__file__,'exec'),module.__dict__)
    module.unchanged_gate = repaired_gate
    return module,ast.unparse(tree)


def load_frozen_gate(source):
    """Compile only its pinned scalar gate AST, never import its module."""
    import math
    path = Path(source)/FROZEN_GATE_FILE
    assert sha(path) == FROZEN_GATE_SHA, 'Retain exact original gate source'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    gates = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='stage_gate']
    assignments = [node for node in tree.body if isinstance(node,ast.Assign)
                   and any(isinstance(target,ast.Name) and target.id=='THRESHOLDS' for target in node.targets)]
    assert len(gates)==len(assignments)==1 and isinstance(assignments[0].value,ast.Call)
    call = assignments[0].value
    assert isinstance(call.func,ast.Name) and call.func.id=='dict' and not call.args
    thresholds = {keyword.arg:ast.literal_eval(keyword.value) for keyword in call.keywords}
    assert thresholds == SCIENTIFIC_THRESHOLDS
    returned = [node.value for node in ast.walk(gates[0]) if isinstance(node,ast.Return)]
    assert len(returned)==1 and isinstance(returned[0],ast.Call)
    fields = {keyword.arg:keyword.value for keyword in returned[0].keywords if keyword.arg is not None}
    assert isinstance(fields['scientific_thresholds'],ast.Name) and fields['scientific_thresholds'].id=='THRESHOLDS'
    for key,value in ADDED_FIELDS.items():
        if key != 'scientific_thresholds':
            assert ast.literal_eval(fields[key]) is value
    namespace = {'math':math,'THRESHOLDS':thresholds}
    code = compile(ast.fix_missing_locations(ast.Module(body=gates,type_ignores=[])),str(path),'exec')
    exec(code,namespace)
    return namespace['stage_gate']


def repaired_gate(before,after):
    result = ORIGINAL_GATE(before,after)
    assert result.keys() == NUMERICAL_FIELDS
    return result | copy.deepcopy(ADDED_FIELDS)


def compare_gate_record(actual,saved):
    assert actual.keys() == saved.keys() == NUMERICAL_FIELDS | ADDED_FIELDS.keys()
    for key,value in ADDED_FIELDS.items():
        assert actual[key] == saved[key] == value, 'Original gate metadata/thresholds differ'
    # Exact retained scalar gate computations and dictionaries, without widening
    # the scientific or arithmetic slack to accommodate a false green.
    assert actual == saved, 'Original numerical gate values differ'


def validate_retained_gate_schema(repair,source):
    frozen = load_frozen_gate(source)
    control = Path(repair)/'outputs/decoder_hinge_repair_v1'
    baseline = read(control/'baseline/metrics.json')['groups']
    rows = []
    for label in ['initial','probe_1_16','probe_1_4','fit_u001']:
        metrics = read(control/label/'metrics.json')['groups']
        retained = read(control/(label+'_outcome.json'))['gates'] if label!='initial' else None
        for mode in ['raw','png']:
            expected = frozen(baseline[mode],metrics[mode])
            actual = repaired_gate(baseline[mode],metrics[mode])
            compare_gate_record(actual,expected)
            if retained is not None:
                compare_gate_record(actual,retained[mode])
            rows.append(dict(state=label,mode=mode,all9_fields_exact_to_frozen_parent=True,
                five_numerical_fields_unchanged=True,retained_outcome_exact=retained is not None,
                gate=actual))
    return rows


def audit_saved_r1(root,repair,source,parent,replay,p,rp,old,data,receipt):
    assert sha(original.__file__) == ORIGINAL_SHA, 'Preserve sealed original auditor'
    assert p['assets_sha256']['audit_cctv_dgp_decoder_activation_v1_return.py'] == ORIGINAL_SHA
    assert p['source_required_sha256'][FROZEN_GATE_FILE] == FROZEN_GATE_SHA
    receipt = Path(receipt).resolve()
    sidecar = receipt.parent/(receipt.stem+'_schema_r1.json')
    assert not receipt.exists() and not sidecar.exists(), 'Retain every prior R1 attempt'
    gates = validate_retained_gate_schema(repair,source)
    norm_proofs,objective_proofs = [],[]
    module,derived_source = load_adapter_module(norm_proofs,objective_proofs)
    derived_path = sidecar.with_suffix('.derived.py')
    assert not derived_path.exists()
    with derived_path.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(derived_source+'\n')
    module.__file__ = str(derived_path)
    unchanged_write = module.write
    def write_labeled(path,value):
        assert Path(path).resolve()==receipt, 'Original checker writes only its final numerical receipt'
        assert len(norm_proofs)==1
        value = dict(value,original_checker_sha256=ORIGINAL_SHA,adapter_sha256=sha(__file__),
            transformed_local_checker_sha256=sha(derived_path),original_checker_passed_unmodified=False,
            derived_R1_numerical_checks_complete=True,exact_direction_norm_proof=norm_proofs[0],
            saved_norm_metadata_certified=False,saved_norm_metadata_excluded_as_approximate=True,
            gate_metadata_added_fields=copy.deepcopy(ADDED_FIELDS),scientific_gates_or_tolerances_changed=False,
            finite_objective_replay_R1=dict(original_exact_objective_check_passed=all(v['original_exact_objective_check_passed'] for v in objective_proofs),
                rows=objective_proofs,allowed_leaf_path='finite_arithmetic.objective',existing_absolute_bound=p['objective_replay_max_abs'],
                other_leaves_exact=True,new_or_wider_bound=False))
        unchanged_write(path,value)
    module.write = write_labeled
    module.audit_saved(root,repair,source,parent,replay,p,rp,old,data,receipt)
    assert len(norm_proofs)==1 and len(objective_proofs)==80
    failure_paths = [Path(root)/'saved_audit.log',Path(root)/'saved_audit_exit.json',receipt.parent/'audit_r0.log']
    assert all(path.is_file() for path in failure_paths), 'Bind original VM/local failed checks'
    enumeration = receipt.parent/'finite_arithmetic_leaf_differences.json'
    assert sha(enumeration)==REDUCTION_ENUMERATION_SHA
    test_path = Path(__file__).with_name('test_cctv_dgp_decoder_activation_v1_return_r1.py')
    record = dict(complete=True,adapter='Four historical gate metadata fields, exact dyadic radius proof, and one named objective reduction under its existing absolute bound',
        original_auditor_sha256=ORIGINAL_SHA,r1_auditor_sha256=sha(__file__),
        frozen_gate_source_sha256=FROZEN_GATE_SHA,protocol_sha256=sha(Path(root)/'protocol.json'),
        final_saved_audit_sha256=sha(receipt),all8_retained_state_mode_checks=gates,
        added_fields=copy.deepcopy(ADDED_FIELDS),numerical_gate_values_changed=False,
        exact_direction_norm_proof=norm_proofs[0],norm_assertion_replacement=dict(original_line=315,
            original_expression=NORM_ASSERTION,replacement='_r1_verify_exact_direction_radius(direction,direction_record,p)',
            purpose='Scientific radius is proved by exact dyadic squared sums; saved norm remains explicitly approximate/unverified.'),
        derived_local_checker_path=str(derived_path),derived_local_checker_sha256=sha(derived_path),
        objective_assertion_replacement=dict(original_line=381,original_expression=FINITE_CALL,
            allowed_leaf_path='finite_arithmetic.objective',existing_absolute_bound=p['objective_replay_max_abs'],
            relative_tolerance=0.,all_other_leaves_exact=True,new_or_wider_bound=False,
            original_exact_objective_check_passed=all(v['original_exact_objective_check_passed'] for v in objective_proofs),rows=objective_proofs),
        original_checker_passed_unmodified=False,derived_R1_numerical_checks_complete=True,
        saved_approximate_norm_metadata_certified=False,
        preserved_original_failure_evidence={str(path):dict(bytes=path.stat().st_size,sha256=sha(path)) for path in failure_paths},
        scientific_thresholds_changed=False,arithmetic_tolerances_changed=False,
        original_auditor_and_packet_preserved=True,no_model_or_gradient_calls=True,
        model_qualified=False,new_incumbent_selection=False,
        dependency_sha256={str(Path(original.__file__).resolve()):ORIGINAL_SHA,
            str((Path(source)/FROZEN_GATE_FILE).resolve()):FROZEN_GATE_SHA,
            str(test_path.resolve()):sha(test_path),str(enumeration):REDUCTION_ENUMERATION_SHA})
    write(sidecar,record)
    print(json.dumps(dict(complete=True,schema_only_R1=True,all8_gate_checks_exact=True,
                         model_qualified=False)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root','repair-root','source-root','parent-root','replay-root','receipt']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True)
    args = parser.parse_args()
    p,rp,old,data = verify_transfer(args.root,args.repair_root,args.source_root,args.parent_root,args.replay_root,args.protocol_sha)
    audit_saved_r1(args.root.resolve(),args.repair_root.resolve(),args.source_root.resolve(),args.parent_root.resolve(),
        args.replay_root.resolve(),p,rp,old,data,args.receipt.resolve())


if __name__=='__main__':
    main()
