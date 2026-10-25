"""Independent docs-only publication readback; no publisher/worker execution."""
from datetime import datetime, timezone
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u64_png_surrogate_measurement_v1_preparation'
OUT = PREP / 'publication_v1'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
PRIOR = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/publication_v1'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def main():
    consumed, checks = {}, []
    def pin(path):
        data = path.read_bytes()
        return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest()}
    def read(path, expected=None):
        binding = pin(path)
        if expected is not None:
            need(binding == expected, 'Exact readback pin differs: ' + str(path))
        need(binding['path'] not in consumed or consumed[binding['path']] == binding,
             'Consumed source changed during read')
        consumed[binding['path']] = binding
        return path.read_bytes()
    def load(path, expected=None):
        return json.loads(read(path, expected))
    def check(name, ok, **values):
        need(ok, name)
        checks.append({'name': name, 'pass': True, **values})
    publication = load(OUT / 'publication_receipt_v1.json')
    old = load(PRIOR / 'publication_receipt_v1.json')
    old_readback = load(PRIOR / 'publication_readback_v1.json')
    check('new_and_prior_publications_complete', publication['complete'] is True
          and old['complete'] is True and old_readback['publication_readback_pass'] is True,
          prior_document_count=old_readback['document_count'])
    need(len(publication['changes']) == 4, 'Exactly four current documents may change')
    previous = {row['after']['path']: row for row in old['changes']}
    actual = {row['after']['path']: row for row in publication['changes']}
    need(set(actual) == set(DOCS), 'Unexpected published document scope')
    prefixes, document_rows = [], []
    for name in DOCS:
        row = actual[name]
        now = read(ROOT / name, row['after'])
        before_path = ROOT / row['backup']['path']
        before = read(before_path, row['backup'])
        prefix = now[:row['prefix_bytes']]
        expected_before = previous[name]['after']
        check('exact_new_prefix_and_prior_historical_suffix_' + name,
              row['before'] == expected_before
              and len(before) == expected_before['bytes']
              and hashlib.sha256(before).hexdigest() == expected_before['sha256']
              and now[row['prefix_bytes']:] == before
              and row['prior_bytes_preserved_exactly'] is True,
              prefix_bytes=row['prefix_bytes'], total_bytes=len(now))
        # Preserve the prior fail-stop publication and its older suffix too.
        older = read(ROOT / previous[name]['backup']['path'], previous[name]['backup'])
        need(before[previous[name]['prefix_bytes']:] == older,
             'Prior fail-stop publication historical suffix changed: ' + name)
        prefixes.append(prefix)
        document_rows.append({'document': name, 'after': row['after'],
                              'new_prefix_bytes': row['prefix_bytes'],
                              'previous_failed_pilot_status_retained_in_suffix': True})
    check('all_four_current_prefixes_identical', all(p == prefixes[0] for p in prefixes),
          prefix_bytes=len(prefixes[0]))
    text = prefixes[0].decode('utf-8')
    normalized = ' '.join(text.split())
    # The failed pilot's launch guide is deliberately outside the four-document
    # scope. Its exact old fail-stop warning remains the current guide prefix.
    old_guide = 'CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_VM.md'
    old_row = previous[old_guide]
    old_guide_now = read(ROOT / old_guide, old_row['after'])
    old_guide_before = read(ROOT / old_row['backup']['path'], old_row['backup'])
    check('executed_failed_pilot_guide_warning_and_history_unchanged',
          old_guide_now[old_row['prefix_bytes']:] == old_guide_before,
          old_guide_current=old_row['after'])
    required = {'source_review_v1.json': 'source_review_pass',
                'source_tests_v1.json': 'all_tests_pass',
                'archive_readback_v1.json': 'archive_review_pass',
                'guide_review_v1.json': 'guide_review_pass'}
    evidence = {}
    for name, field in required.items():
        value = load(PREP / name)
        need(value.get('complete') is True and value.get(field) is True
             and not value.get('unresolved_blockers'), 'Verification incomplete: ' + name)
        evidence[name] = value
    build = load(PREP / 'build_receipt_v1.json')
    check('reviewed_tested_sealed_current_sources_exact',
          build['sources'] == evidence['source_review_v1.json']['sources']
          == evidence['source_tests_v1.json']['sources'] and len(build['sources']) == 12
          and evidence['source_tests_v1.json']['test_count'] == 51,
          source_count=12, pure_test_count=51)
    for row in build['sources']:
        read(ROOT / row['path'], row)
    archive, checksum = read(ROOT / build['archive']['path'], build['archive']), read(ROOT / build['checksum']['path'], build['checksum'])
    archive_review = evidence['archive_readback_v1.json']
    check('actual_sealed_archive_and_checksum_match_independent_readback',
          archive_review['archive'] == build['archive']
          and checksum == (build['archive']['sha256'] + '  ' + Path(build['archive']['path']).name + '\n').encode()
          and len(archive) == 243674, archive=build['archive'], checksum=build['checksum'])
    stage = ROOT / 'outputs/cctv_codeformer_u64_png_surrogate_measurement_v1_vm'
    protocol_raw = read(stage / 'protocol.json')
    need(len(protocol_raw) == build['protocol']['bytes'] == 188285
         and hashlib.sha256(protocol_raw).hexdigest() == build['protocol']['sha256'], 'Sealed protocol changed')
    protocol = json.loads(protocol_raw)
    for name, binding in protocol['files'].items():
        data = read(stage / name)
        need({'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()} == binding,
             'Staged static file changed: ' + name)
    check('all21_staged_static_bindings_and72_256_zero_execution_limits', len(protocol['files']) == 21
          and protocol['limits']['full_forward_attempts'] == 72
          and protocol['limits']['gradient_query_attempts'] == 256
          and protocol['limits']['parameter_applications'] == 0
          and protocol['limits']['optimizer_constructions'] == 0,
          static_files=21, archive_members=22, limits=protocol['limits'])
    guide = ROOT / 'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_VM.md'
    guide_raw = read(guide, evidence['guide_review_v1.json']['guide'])
    check('current_guide_exact_independently_reviewed',
          'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_VM.md' in text
          and b'not launched' in guide_raw.lower(), guide=pin(guide))
    expected_tokens = ('PREPARED_PNG_SURROGATE_MEASUREMENT_V1_UNRUN',
        'failed at update two', 'Neither that failed state nor its earlier update one is selected',
        'Do not resume or append epochs', 'original trained u64 remains retained',
        'U96 remains experimental app incumbent', 'All575 tensors must remain bitwise unchanged',
        'ZERO parameter applications', 'ZERO optimizer constructions', 'no common-direction solver',
        'biased identity surrogate', 'not the true derivative', 'every quality requirement stay intact',
        'No automatic retry', 'No VM connection', 'Broader DEV failures remain binding',
        'FINAL remains unopened', 'automatic matching and completion development remain deferred',
        'ACTIVE/incomplete')
    check('unrun_measurement_failed_pilot_and_no_quality_promotion_status_exact',
          all(token in normalized for token in expected_tokens)
          and build['archive']['sha256'] in text and build['protocol']['sha256'] in text
          and 'Transfer 243,674 bytes' in normalized and 'protocol 188,285 bytes' in normalized,
          status=publication['status'], failed_pilot_quality_stop_preserved=True,
          update1_promoted=False, new_model_trained=False)
    for key, expected in {'worker_seconds': 900, 'external_seconds': 960,
        'termination_grace_seconds': 20, 'per_phase_seconds': 180,
        'export_seconds': 300, 'export_external_seconds': 330,
        'initial_free_disk_bytes': 8 * 1024**3, 'continuing_free_disk_bytes': 4 * 1024**3,
        'maximum_RSS_bytes': 10 * 1024**3, 'maximum_CUDA_reserved_bytes': 18 * 1024**3,
        'maximum_output_bytes': 2 * 1024**3, 'maximum_archive_bytes': 2 * 1024**3}.items():
        need(protocol['limits'][key] == expected, 'Published resource bound differs from protocol: ' + key)
    check('zero_update_resource_and16vector_scope_in_current_prefix',
          all(token in normalized for token in ('72 full forwards', '256 separate gradient',
              '16 separately named pooled', 'Worker900s, external960s', '20s cleanup',
              'phases180s', 'Export300s internally/330s', 'RAM10GiB', 'reserved CUDA18GiB',
              'Require8GiB free', 'retain4GiB', 'outputs/archive2GiB each')),
          pooled_vectors=16, applied_parameters=0)
    for row in publication['consumed_before_after_exact']:
        read(ROOT / row['path'], row)
    publisher = ROOT / publication['source']['path']
    publisher_raw = read(publisher, publication['source'])
    ast.parse(publisher_raw, feature_version=(3, 10))
    check('publisher_docs_only_exact_preservation_and_no_execution_claims',
          publication['all_historical_bytes_retained'] is True
          and publication['neural_gradient_optimizer_training_VM_calls'] == 0
          and publication['actual_parameter_applications'] == 0
          and publication['model_qualified'] is False
          and publication['automatic_goal_advancement_triggered'] is False
          and publication['full_goal'] == 'ACTIVE/incomplete',
          publisher=publication['source'], AST_Python310_pass=True)
    read(Path(__file__).resolve())
    for name, expected in consumed.items():
        need(pin(ROOT / name) == expected, 'Consumed pin changed during final readback: ' + name)
    result = {'complete': True, 'publication_readback_pass': True,
        'unresolved_blockers': [], 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'readback_scope': 'Independent4currentprefixes/exactbeforebackups/priorhistoricalsuffixes and unchangedfailedpilotguide; current reviewed guide,12sources,21stagebindings,sealedarchive and all publication admissions. No publisher/worker/neural/model/gradient/optimizer/parameter/VM execution.',
        'checks': checks, 'document_count': 4, 'documents': document_rows,
        'all_historical_bytes_retained': True,
        'prior_failed_pilot_publication_5documents_preserved': True,
        'status': 'PREPARED_PNG_SURROGATE_MEASUREMENT_V1_UNRUN',
        'manual_guide_ready': True, 'actual_parameter_applications': 0,
        'new_model_trained_or_qualified': False, 'model_qualified': False,
        'automatic_goal_advancement_triggered': False, 'full_goal': 'ACTIVE/incomplete',
        'consumed_files': list(consumed.values()), 'all_consumed_before_after_exact': True,
        'neural_gradient_optimizer_training_VM_calls': 0}
    target = OUT / 'publication_readback_v1.json'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'complete': True, 'publication_readback_pass': True,
        'checks': len(checks), 'documents': 4, 'unresolved_blockers': []}, indent=2))


if __name__ == '__main__':
    main()
