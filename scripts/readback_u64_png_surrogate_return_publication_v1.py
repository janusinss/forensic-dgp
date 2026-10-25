"""Independent completed-measurement document publication readback only."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit'
PREP = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_preparation'
PUBLICATION = AUDIT / 'publication_v1'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
GUIDE = 'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_VM.md'
PROTOCOL_SHA = '0a61c2d6433399e9e748f3782e4d87bf9da3089848b2502a761fcd029690797c'


def need(value, message):
    if not value:
        raise ValueError(message)


def main():
    consumed, checks = {}, []
    def pin(path):
        raw = path.read_bytes()
        return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    def read(path, expected=None):
        actual = pin(path)
        if expected is not None:
            need(actual == expected, 'Exact pin differs: ' + actual['path'])
        need(actual['path'] not in consumed or consumed[actual['path']] == actual, 'Consumed file changed')
        consumed[actual['path']] = actual
        return path.read_bytes()
    def load(path):
        return json.loads(read(path))
    def check(name, value, **detail):
        need(value, name)
        checks.append({'name': name, 'pass': True, **detail})
    publication = load(PUBLICATION / 'publication_receipt_v1.json')
    old = load(PREP / 'publication_v1/publication_receipt_v1.json')
    guide_review = load(PREP / 'guide_review_v1.json')
    oldpins = {row['after']['path']: row['after'] for row in old['changes']}
    oldpins[GUIDE] = guide_review['guide']
    expected_names = {*DOCS, GUIDE}
    changed = {row['after']['path']: row for row in publication['changes']}
    check('exact_five_document_scope', publication['complete'] and set(changed) == expected_names and len(changed) == 5)
    prefixes, document_rows = {}, []
    for name in (*DOCS, GUIDE):
        row = changed[name]
        current = read(ROOT / name, row['after'])
        backup = read(ROOT / row['backup']['path'], row['backup'])
        check('exact_prior_backup_and_current_suffix_' + name,
              row['before'] == oldpins[name] and len(backup) == oldpins[name]['bytes']
              and hashlib.sha256(backup).hexdigest() == oldpins[name]['sha256']
              and current[row['prefix_bytes']:] == backup and row['prior_bytes_preserved_exactly'] is True,
              prefix_bytes=row['prefix_bytes'])
        prefixes[name] = current[:row['prefix_bytes']]
        document_rows.append({'path': name, 'after': row['after'], 'backup': row['backup'],
                              'exact_historical_suffix': True, 'prefix_bytes': row['prefix_bytes']})
    check('four_authoritative_prefixes_identical', len({prefixes[name] for name in DOCS}) == 1)
    publisher = ROOT / publication['source']['path']
    publisher_raw = read(publisher, publication['source'])
    tree = ast.parse(publisher_raw, feature_version=(3, 10))
    literals = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in ('DOCS', 'GUIDE', 'PREFIX', 'GUIDE_PREFIX'):
                    literals[target.id] = ast.literal_eval(node.value)
    check('publisher_exact_docs_scope_and_literal_prefixes', literals['DOCS'] == DOCS and literals['GUIDE'] == GUIDE
          and prefixes[DOCS[0]] == literals['PREFIX'].encode('utf8') and prefixes[GUIDE] == literals['GUIDE_PREFIX'].encode('utf8'))
    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
    allowed = {'datetime', 'hashlib', 'json', 'os', 'pathlib'}
    check('publisher_stdlib_docs_only_imports', all(((node.module or '').split('.')[0] in allowed if isinstance(node, ast.ImportFrom)
          else all(alias.name.split('.')[0] in allowed for alias in node.names)) for node in imports))
    text = ' '.join(prefixes[DOCS[0]].decode().split())
    for token in ('EXECUTED_PNG_SURROGATE_MEASUREMENT_V1_AUDITED_ZERO_UPDATES', 'All575 initial/closing tensors remain exactly original trainedu64',
                  '72 forwards, 256 gradient queries,ZERO parameter applications,ZERO optimizer constructions',
                  '1,642 saved-pixel checks', '64 exactactualPNGdecodedFP32 surrogate forwards', '328 known operation pairs',
                  'no finalexport exit is inferred', 'full16direction has cosine0.99999338198',
                  'No near-identical16-solver training pilot is prepared or launched', '436forwards',
                  'nine explicittrial/restore/commit applications', '2e-5/1.5e-5/1e-5/5e-6',
                  '2.5e-5per-application and1e-4cumulative bounds withfixed.99headroom',
                  'NO runnable nextworker,sealedtransfer or nextmanualguide is ready',
                  'Originalu64 remains retained challenger;U96 remains experimental app incumbent', 'Full goal ACTIVE/incomplete'):
        check('current_status_boundary_' + token, token in text)
    completed_guide = ' '.join(prefixes[GUIDE].decode().split())
    check('completed_measurement_guide_historical_commands_not_repeatable',
          'Completed diagnostic: audited zero-update return; do not rerun' in completed_guide
          and 'commands below are historical' in completed_guide
          and 'No next runnable worker or manual transfer guide is ready' in completed_guide)
    # The rejected training pilot guide is outside the allowed publication scope.
    failed_publication = load(ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/publication_v1/publication_receipt_v1.json')
    failed_guide = next(row for row in failed_publication['changes'] if row['after']['path'] == 'CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_VM.md')
    failed_now = read(ROOT / failed_guide['after']['path'], failed_guide['after'])
    failed_before = read(ROOT / failed_guide['backup']['path'], failed_guide['backup'])
    check('failed_training_guide_untouched_warning_and_historical_suffix', failed_now[failed_guide['prefix_bytes']:] == failed_before
          and failed_now.startswith(b'# Executed pilot: quality stop at update two; do not rerun'))
    for row in publication['consumed_before_after_exact']:
        read(ROOT / row['path'], row)
    # Match immutable sources and the sealed21-static protocol without executing it.
    tests = load(PREP / 'source_tests_v1.json'); source_review = load(PREP / 'source_review_v1.json'); build = load(PREP / 'build_receipt_v1.json')
    check('old_source12_review_test_build_exact', tests['all_tests_pass'] and source_review['source_review_pass']
          and tests['sources'] == source_review['sources'] == build['sources'] and len(build['sources']) == 12)
    for row in build['sources']:
        read(ROOT / row['path'], row)
    protocol_raw = read(STAGE / 'protocol.json')
    check('sealed_measurement_protocol_unchanged', hashlib.sha256(protocol_raw).hexdigest() == PROTOCOL_SHA
          and len(protocol_raw) == build['protocol']['bytes'])
    protocol = json.loads(protocol_raw)
    for name, value in protocol['files'].items():
        raw = read(STAGE / name)
        need({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == value, 'Sealed static changed: ' + name)
    check('all21_measurement_static_bindings_unchanged', len(protocol['files']) == 21)
    for name in ('CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_RETURN_REVIEW.md', 'CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md'):
        read(ROOT / name)
    design = ' '.join((ROOT / 'CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_DESIGN.md').read_text().split())
    for token in ('Prospective design only: no runnable worker, sealed transfer or launch guide exists',
                  '2e-5, 1.5e-5, 1e-5, 5e-6', '0.99 numerical headroom', '2.5e-5 per-application bound',
                  '1e-4 original-relative cumulative bound', 'nine parameter applications total',
                  'actual RAW/PNG role-mean', 'No monitor output may leak into candidate selection'):
        check('next_source_only_design_' + token, token in design.replace('**', ''))
    check('publication_zero_local_execution_and_no_qualification', publication['all575_original_unchanged'] is True
          and publication['all_historical_bytes_retained'] is True and publication['actual_training_or_parameter_applications_local'] == 0
          and publication['VM_calls'] == 0 and publication['model_qualified'] is False
          and publication['next_worker_or_guide_ready'] is False and publication['automatic_goal_advancement_triggered'] is False
          and publication['full_goal'] == 'ACTIVE/incomplete')
    read(Path(__file__).resolve())
    for name, expected in consumed.items():
        need(pin(ROOT / name) == expected, 'Consumed bytes changed during final readback')
    result = {'complete': True, 'publication_readback_pass': True, 'unresolved_blockers': [],
              'checks': checks, 'document_count': 5, 'documents': document_rows,
              'status': publication['status'], 'all_prior_document_bytes_retained_exactly': True,
              'failed_training_guide_unchanged': True, 'sealed21_static_and_protocol_unchanged': True,
              'all12_approved_sources_unchanged': True, 'consumed_files': list(consumed.values()),
              'all_consumed_before_after_exact': True,
              'readback_scope': 'Exact5currentprefixes/backuphistoricalsuffixes, untouchedfailedpilotguide, allpublicationadmissions, sealed21static/protocol and12sources. No publisher, worker, inference, autograd, optimizer, parameter, VM or browser execution.',
              'actual_model_improvement_or_qualification': False, 'new_worker_or_manual_packet_ready': False,
              'original_u64_retained': True, 'U96_app_incumbent_retained': True, 'full_goal': 'ACTIVE/incomplete',
              'local_neural_gradient_optimizer_parameter_training_VM_browser_calls': 0}
    target = PUBLICATION / 'publication_readback_v1.json'
    with target.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({'complete': True, 'publication_readback_pass': True, 'checks': len(checks),
                      'consumed_files': len(consumed), 'receipt': pin(target)}, indent=2))


if __name__ == '__main__':
    main()
