"""Audit frozen cohort/source/state reports and Gram arithmetic; no CUDA replay."""
import argparse
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))
from cctv_dgp_pilot import read, write, sha
from cctv_dgp_perceptual_training_v3 import START_SHA, START_STATE
from cctv_dgp_objective_diagnostic_v4 import verify_recipe, summary_from_gram, TERM_WEIGHTS, TEACHERS
from audit_cctv_dgp_pilot_results import require, same, safe_result_path
from audit_cctv_dgp_normfix_results import save_receipt


def extract_return(archive, dest):
    require(not dest.exists(), 'Preserve existing returned/partial evidence')
    expected = (sha(archive) + '  ' + archive.name + '\n').encode('ascii')
    require(Path(str(archive) + '.sha256').read_bytes() == expected, 'Checksum/filename/LF differs')
    allowed = {'objective_diagnostic_protocol_v4.json', 'objective_diagnostic_protocol_v4.sha256',
               'objective_diagnostic_environment_v4.txt', 'objective_diagnostic_v4.log', 'cuda_runtime_before.txt'}
    prefix = 'outputs/cctv_dgp_objective_diagnostic_v4'
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        seen = set()
        total = 0
        require(len(members) <= 100, 'Finite return inventory exceeded')
        for member in members:
            name = member.name.rstrip('/')
            safe_result_path(dest, name)
            require((member.isfile() or member.isdir()) and not member.issym() and not member.islnk(), 'Special/linked return member')
            require(name in allowed or name in ('outputs', prefix) or name.startswith(prefix + '/'), 'Unexpected return root')
            require(name not in seen, 'Duplicate return member')
            seen.add(name)
            total += member.size
        require(total <= 12 * 1024**2, 'Finite diagnostic return size exceeded')
        dest.mkdir(parents=True)
        for member in members:
            target = safe_result_path(dest, member.name.rstrip('/'))
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, target.open('xb') as stream:
                    import shutil
                    shutil.copyfileobj(source, stream)
    return dest / prefix


def audit(root, bundle_dir, out, perceptual_bundle_dir=None, parent_return=None, v3_return=None):
    _, protocol = verify_recipe(root, bundle_dir, perceptual_bundle_dir, parent_return, v3_return)
    report, execution = read(out / 'results.json'), read(out / 'execution.json')
    require(report['complete'] and not (out / 'failure.json').exists(), 'Incomplete/failed diagnostic')
    require(report['protocol_sha256'] == execution['protocol_sha256'] == sha(bundle_dir / 'objective_diagnostic_protocol_v4.json'), 'Protocol lineage differs')
    require(execution['host'].split('.')[0] == 'forensic-dgp-thesis' and execution['device'] == 'cuda'
            and 'L4' in execution['gpu'] and not execution['optimizer_constructed'] and not execution['actual_training'], 'VM execution policy differs')
    require(execution['runtime_cap_seconds'] == 600 and execution['normalization_stats_cloned'],
            'Runtime/normalization execution policy differs')
    require(execution['starting_checkpoint_sha256'] == START_SHA and execution['starting_state_hash'] == START_STATE
            and report['student_state_before'] == report['student_state_after'] == START_STATE, 'Starting/unchanged student state differs')
    require(execution['teachers_before'] == report['teachers_before'] == report['teachers_after'] == TEACHERS, 'Frozen teacher state differs')
    require(report['optimizer_updates'] == 0 and not report['optimizer_constructed'] and not report['parameter_grads_accumulated']
            and not report['actual_training'] and not report['improved_model_claimed'] and not report['production_checkpoint_promoted'], 'Diagnostic/training claim differs')
    require(report['groups'] == 10 and report['training_references'] == 40 and report['student_forwards'] == 10
            and report['autograd_grad_calls'] == 50 and 0 < report['seconds'] <= 600
            and report['validation_references_used'] == report['native_cases_used'] == 0
            and not report['native_reserved_used'], 'Finite cohort/gradient/timing budget differs')
    expected_files = {'execution.json'} | {f'group_{i:02d}.json' for i in range(10)} | {'runtime_sources/' + n for n in protocol['assets_sha256']}
    require(set(report['artifacts_sha256']) == expected_files, 'Diagnostic artifact coverage differs')
    actual = {p.relative_to(out).as_posix() for p in out.rglob('*') if p.is_file()}
    require(actual - {'results.json', 'independent_audit.json'} == expected_files, 'Unexpected/partial diagnostic files')
    for name, pin in report['artifacts_sha256'].items():
        require(sha(safe_result_path(out, name)) == pin, 'Changed diagnostic artifact: ' + name)
    for name, pin in protocol['assets_sha256'].items():
        require(sha(out / 'runtime_sources' / name) == pin, 'Returned executable/capsule differs')
    gradients = []
    elapsed = 0.
    for index, group in enumerate(protocol['groups']):
        row = read(out / f'group_{index:02d}.json')
        require(row['group'] == group and row['optimizer_updates'] == 0 and row['state_unchanged'], 'Training-only group/state differs')
        require(row['autograd_grad_calls'] == 5 and row['cumulative_autograd_grad_calls'] == 5 * (index + 1)
                and elapsed <= row['elapsed_seconds'] <= report['seconds'], 'Gradient call/order/timing differs')
        elapsed = row['elapsed_seconds']
        require(set(row['unweighted_losses']) == set(row['weighted_losses']) == set(TERM_WEIGHTS), 'Objective term coverage differs')
        for name, weight in TERM_WEIGHTS.items():
            require(math.isfinite(row['unweighted_losses'][name]), 'Nonfinite objective loss')
            same(row['weighted_losses'][name], row['unweighted_losses'][name] * weight, 'Loss weight', 2e-6)
        require(row['input_gradient_support'] == 'Observed pixels only; uncaptured padding gradient zeroed', 'Gradient support policy differs')
        for space in ('student_parameters', 'restoration_input'):
            value = row[space]
            require(value['term_order'] == list(TERM_WEIGHTS), 'Gradient term order differs')
            expected_dimension = execution['student_parameter_dimension'] if space == 'student_parameters' else 4 * 3 * 256 * 256
            require(value['dimension'] == expected_dimension and expected_dimension > 0, 'Gradient vector dimension differs')
            rebuilt = summary_from_gram(value['term_order'], value['dimension'], value['gram_matrix'])
            same(value, rebuilt, 'Gram-derived summary', 1e-8)
            require(any(v['l2_norm'] > 0 for v in rebuilt['terms'].values()), 'Empty combined gradient diagnostic')
        gradients.append({'group': group['id'], 'parameter_summary': row['student_parameters'], 'input_summary': row['restoration_input']})
    return {'complete': True, 'protocol_sha256': report['protocol_sha256'], 'returned_results_sha256': sha(out / 'results.json'),
            'groups_checked': 10, 'training_references': 40, 'serialized_gram_summaries_rebuilt': 20,
            'recorded_vm_autograd_calls': 50, 'vm_optimizer_updates': 0, 'local_backward_calls': 0,
            'local_optimizer_updates': 0, 'native_reserved_used': False, 'gradients': gradients,
            'cuda_gradients_recomputed_by_this_audit': False, 'model_improvement_established': False,
            'limitation': 'Checks pinned execution sources/cohort and reported state/timing plus independent Gram arithmetic; it does not replay CUDA gradients locally or establish training/output usefulness.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--bundle-dir', type=Path, default=ROOT)
    parser.add_argument('--perceptual-bundle-dir', type=Path)
    parser.add_argument('--parent-return', type=Path)
    parser.add_argument('--v3-return', type=Path)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--extract-to', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    verify_recipe(args.root, args.bundle_dir, args.perceptual_bundle_dir, args.parent_return, args.v3_return)
    if args.archive:
        if not args.extract_to:
            parser.error('--archive requires a fresh --extract-to')
        out = extract_return(args.archive, args.extract_to)
        require(sha(args.extract_to / 'objective_diagnostic_protocol_v4.json') == sha(args.bundle_dir / 'objective_diagnostic_protocol_v4.json'), 'Transferred diagnostic protocol differs')
    else:
        out = args.results
    if out is None:
        parser.error('Supply --results or --archive')
    receipt = audit(args.root, args.bundle_dir, out, args.perceptual_bundle_dir, args.parent_return, args.v3_return)
    save_receipt(out, receipt, args.receipt)
    print({k: v for k, v in receipt.items() if k != 'gradients'})
