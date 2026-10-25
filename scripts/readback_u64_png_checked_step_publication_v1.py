"""Independent bounded publication readback; saved bytes and JSON only.

No publisher, worker, framework, model, parameter or VM entry is imported.
The existing 63-check archive audit supplies archive framing evidence; this
readback authenticates its receipts and every consumed file again.
"""
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import stat

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_preparation'
PUB = PREP / 'publication_v1'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_png_checked_step_v1_vm'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
PUBLISHER = 'scripts/publish_u64_png_checked_step_v1.py'
PREVIOUS = 'outputs/codeformer_u64_png_surrogate_measurement_v1_return_audit/publication_v1/'
GUIDE = 'CCTV_CODEFORMER_U64_PNG_CHECKED_STEP_V1_VM.md'
OLD_MEASUREMENT_GUIDE = 'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_VM.md'
OLD_FAILED_GUIDE = 'CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_VM.md'


def need(value, message):
    if not value:
        raise ValueError(message)


def pin(path):
    path = Path(path).absolute()
    need(path.is_relative_to(ROOT), 'Readback file escapes workspace')
    for current in (path, *path.parents):
        info = current.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Ordinary readback ancestry required')
        need(stat.S_ISREG(info.st_mode) if current == path else stat.S_ISDIR(info.st_mode),
             'Ordinary readback file/directory required')
    before = path.stat()
    need(before.st_nlink == 1, 'Single-link readback file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'Readback file changed during hash')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': after.st_size,
            'sha256': digest.hexdigest()}


def strict_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def literal_prefix(node, build):
    """Render only the reviewed string/subscript/format AST, never publisher code."""
    if isinstance(node, ast.Constant):
        need(type(node.value) in (str, int), 'Literal prefix scalar required')
        return node.value
    if isinstance(node, ast.Name):
        need(node.id == 'build', 'Only admitted build data may format prefix')
        return build
    if isinstance(node, ast.Subscript):
        return literal_prefix(node.value, build)[literal_prefix(node.slice, build)]
    if isinstance(node, ast.JoinedStr):
        return ''.join(str(literal_prefix(value, build)) for value in node.values)
    if isinstance(node, ast.FormattedValue):
        need(node.conversion == -1, 'No arbitrary prefix conversion')
        spec = '' if node.format_spec is None else literal_prefix(node.format_spec, build)
        return format(literal_prefix(node.value, build), spec)
    if isinstance(node, ast.Call):
        need(isinstance(node.func, ast.Attribute) and node.func.attr == 'encode'
             and len(node.args) == 1 and not node.keywords
             and literal_prefix(node.args[0], build) == 'utf8', 'Only literal UTF8 prefix encoding allowed')
        return literal_prefix(node.func.value, build).encode('utf8')
    raise ValueError('Unsupported publisher prefix AST: ' + type(node).__name__)


def main():
    expected, checks = {}, []
    def admit(row):
        need(type(row) is dict and set(row) == {'path', 'bytes', 'sha256'}, 'Exact consumed pin schema required')
        old = expected.get(row['path'])
        need(old is None or old == row, 'Conflicting expected pins: ' + row['path'])
        expected[row['path']] = row
        need(pin(ROOT / row['path']) == row, 'Current consumed file differs: ' + row['path'])
    def read(relative):
        path = ROOT / relative
        admit(pin(path))
        return strict_json(path)
    def check(name, condition, **details):
        need(condition, name)
        checks.append({'name': name, 'pass': True, **details})

    pub_rel = PUB.relative_to(ROOT).as_posix() + '/'
    receipt = read(pub_rel + 'publication_receipt_v1.json')
    check('publication_completed_manual_unrun_only', receipt['complete'] is True
          and receipt['status'] == 'PREPARED_PNG_CHECKED_STEP_V1_UNRUN'
          and receipt['manual_guide_ready'] is True and receipt['model_qualified'] is False
          and receipt['actual_parameter_applications'] == 0
          and receipt['neural_gradient_optimizer_training_VM_calls'] == 0
          and receipt['automatic_goal_advancement_triggered'] is False
          and receipt['full_goal'] == 'ACTIVE/incomplete')
    admit(receipt['source'])
    check('exact_reviewed_publisher_source', receipt['source']['path'] == PUBLISHER)
    for row in receipt['consumed_before_after_exact']:
        admit(row)
    current = {}
    for name, flag in (('source_review_v1.json', 'source_review_pass'),
                       ('source_tests_v1.json', 'all_tests_pass'),
                       ('archive_readback_v1.json', 'archive_review_pass'),
                       ('guide_review_v1.json', 'guide_review_pass')):
        current[name] = read((PREP / name).relative_to(ROOT).as_posix())
        value = current[name]
        check('current_approval_' + name, value['complete'] is True and value[flag] is True
              and not value.get('unresolved_blockers'))
    review, tests, archive, guide_review = (current[name] for name in (
        'source_review_v1.json', 'source_tests_v1.json', 'archive_readback_v1.json', 'guide_review_v1.json'))
    build = read((PREP / 'build_receipt_v1.json').relative_to(ROOT).as_posix())
    check('exact16_sources_and78_tests', len(review['sources']) == 16
          and review['sources'] == tests['sources'] == build['sources']
          and tests['test_count'] == 78 and tests['all_sources_tests_unchanged_after'] is True)
    for row in review['sources']:
        admit(row)
    check('independent_archive63_checks', archive['checks_pass'] == 63
          and archive['framing']['regular_members'] == 34
          and archive['all16_sources_admitted_exact'] == review['sources'])
    for row in archive['consumed_files']:
        admit(row)
    check('all757_dependency_receipt_pins_current', len(archive['parent_files_rehashed']) == 469
          and len(archive['retained_measurement_files_rehashed']) == 288
          and len(archive['local_parent_model_aliases']) == 2
          and len(archive['explicit_builder_local_runtime_extraction_aliases']) == 267)
    for label in ('parent_files_rehashed', 'retained_measurement_files_rehashed'):
        for row in archive[label]:
            admit(row['actual_local'])
    stage_protocol = read((STAGE / 'protocol.json').relative_to(ROOT).as_posix())
    check('sealed33_static_files_and_protocol_current', len(stage_protocol['files']) == 33
          and pin(STAGE / 'protocol.json') == archive['stage_protocol']
          and archive['archive'] == build['archive'])
    for name, row in stage_protocol['files'].items():
        admit({'path': (STAGE / name).relative_to(ROOT).as_posix(), **row})
    admit(build['archive']); admit(build['checksum']); admit(guide_review['guide'])
    check('guide_bound_to_current_approval', guide_review['guide']['path'] == GUIDE)

    previous = read(PREVIOUS + 'publication_receipt_v1.json')
    previous_readback = read(PREVIOUS + 'publication_readback_v1.json')
    old_expected = {row['after']['path']: row['after'] for row in previous['changes']}
    admit(old_expected[OLD_MEASUREMENT_GUIDE])
    failed_pin = next(row for row in previous_readback['consumed_files'] if row['path'] == OLD_FAILED_GUIDE)
    admit(failed_pin)
    check('completed_measurement_guide_and_failed_pilot_guide_unchanged', True)

    tree = ast.parse((ROOT / PUBLISHER).read_bytes(), feature_version=(3, 10))
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'prefix' for target in node.targets)]
    check('one_reviewed_prefix_expression', len(assignments) == 1)
    prefix = literal_prefix(assignments[0].value, build)
    check('exact_four_document_scope', len(receipt['changes']) == 4
          and [row['after']['path'] for row in receipt['changes']] == list(DOCS))
    documents = []
    for row in receipt['changes']:
        name = row['after']['path']
        admit(row['after']); admit(row['backup'])
        prior = (ROOT / row['backup']['path']).read_bytes()
        actual = (ROOT / name).read_bytes()
        check('exact_prefix_backup_suffix_' + name,
              row['before'] == old_expected[name]
              and {k: row['backup'][k] for k in ('bytes', 'sha256')} ==
                  {k: row['before'][k] for k in ('bytes', 'sha256')}
              and actual == prefix + prior and row['prefix_bytes'] == len(prefix)
              and row['prior_bytes_preserved_exactly'] is True,
              prefix_bytes=len(prefix), historical_bytes=len(prior))
        documents.append({'path': name, 'current': row['after'], 'backup': row['backup'],
                          'prefix_bytes': len(prefix), 'historical_suffix_bytes': len(prior),
                          'exact_prefix_and_historical_suffix': True})
    text = prefix.decode('utf8')
    for name, phrases in (
        ('current_unrun_status_and_audited_history', ('PREPARED_PNG_CHECKED_STEP_V1_UNRUN',
            'It applied zero updates; all575 tensors remain', 'previous gate-aligned update2 remains failed')),
        ('finite_selection_and_terminal_quality', ('scales2e-5/1.5e-5/1e-5/5e-6',
            'all16 RAW/PNG rolemean', '436fullforwards andnine explicit',
            '380forwards/eightapplications', 'no nextcandidate or oldstate fallback')),
        ('manual_source_evidence_boundaries', ('no alias substitution', 'original2.5e-5per-step',
            'all539R2/U96 tensors exact', 'No VM connection,actualmodelcall,newautogradquery or parameterapplication')),
        ('retained_goal_and_incumbents', ('Originalu64 remains retained challenger;U96 remains experimental app incumbent.',
            'Full goal ACTIVE/incomplete.', 'statuses below are historical. Exact'))):
        check(name, all(phrase in text for phrase in phrases))
    admit(pin(Path(__file__).resolve()))
    before = [expected[name] for name in sorted(expected)]
    check('all_consumed_pins_stable_after_readback', all(pin(ROOT / row['path']) == row for row in before),
          files=len(before))
    result = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'publication_readback_pass': True, 'unresolved_blockers': [], 'checks': checks,
        'document_count': 4, 'documents': documents, 'status': receipt['status'],
        'all_prior_document_bytes_retained_exactly': True,
        'completed_measurement_guide_unchanged': True, 'failed_training_guide_unchanged': True,
        'all16_approved_sources_unchanged': True, 'sealed33_static_and_protocol_unchanged': True,
        'all757_dependency_pin_receipts_current': True, 'dependency_payloads_rehashed': 757,
        'consumed_files': before, 'all_consumed_before_after_exact': True,
        'readback_scope': 'Four current publication prefixes and exact old-byte suffixes/backups; protected historical guides; current source/test/archive/guide approvals, sixteen sources, thirty-three sealed static bindings/protocol, archive/checksum and all757 actual local dependency pin bindings. Existing63-check archive proof is authenticated; no scientific or neural audit is rerun.',
        'actual_model_improvement_or_qualification': False, 'manual_guide_ready': True,
        'actual_VM_run_performed': False, 'original_u64_retained': True,
        'U96_app_incumbent_retained': True, 'full_goal': 'ACTIVE/incomplete',
        'local_neural_gradient_optimizer_parameter_training_VM_browser_calls': 0,
        'source': pin(Path(__file__).resolve())}
    destination = PUB / 'publication_readback_v1.json'
    with destination.open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'publication_readback_pass': True, 'checks': len(checks),
                      'files': len(before), 'receipt': pin(destination)}, indent=2))


if __name__ == '__main__':
    main()
