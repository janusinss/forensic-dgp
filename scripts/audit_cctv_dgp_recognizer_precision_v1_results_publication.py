"""Independent closure of precision results, source histories and unchanged manual guide."""
from datetime import datetime, timezone
from pathlib import Path
import re

from import_cctv_dgp_recognizer_precision_v1_return import ANALYSIS, OUT, PIN, RETURN, ROOT, read, sha, write


def main():
    folder = ANALYSIS / 'publication'
    publication = read(folder / 'publication.json')
    assert publication['complete'] and publication['protocol_sha256'] == PIN
    assert len(publication['documents']) == 9 and len(publication['new_documents_sha256']) == 2
    assert all(sha(ROOT / n) == h for n, h in publication['evidence_sha256'].items())
    assert all(sha(ROOT / n) == h for n, h in publication['new_documents_sha256'].items())
    manifest = read(RETURN / 'export_manifest.json')
    assert len(manifest['files']) == 462 and all(sha(RETURN / n) == h for n, h in manifest['files'].items())
    audit = read(ANALYSIS / 'independent_audit.json'); summary = read(ANALYSIS / 'precision_activation_review.json')
    assert audit['full_diagnostic_returned'] and audit['failure'] is None and len(audit['precision_responses']) == 20
    failed = {s: sum(r['full_network_precision_response']['central_comparison_within_diagnostic_bound'] is False
                    for r in audit['precision_responses'] if r['precision'] == s) for s in ['FP32', 'FP64']}
    assert failed == summary['sampled_response_failure_counts'] == dict(FP32=6, FP64=5)
    assert len(audit['original_bound_parity_reports']) == 30 and all(r['original_parity_check_pass'] for r in audit['original_bound_parity_reports'])
    counts = []
    for precision in ['FP32', 'FP64']:
        for label in ['plus02', 'minus02', 'repeat']:
            selected = [r for r in audit['activation_transitions'] if r['precision'] == precision and r['label'] == label]
            expected = next(r for r in summary['activation_transition_totals'] if r['precision'] == precision and r['label'] == label)
            assert len(selected) == 50 and sum(r['units'] for r in selected) == expected['units'] == 37130240
            assert sum(r['changed_sign_units'] for r in selected) == expected['changed_sign_units']
            counts.append(expected['changed_sign_units'])
    assert counts == [1838, 1852, 0, 1845, 1844, 0]
    for n, binding in publication['documents'].items():
        old = Path(binding['backup'])
        assert sha(old) == binding['before_sha256'] and sha(ROOT / n) == binding['after_sha256']
        assert (ROOT / n).read_bytes()[binding['prefix_bytes']:] == old.read_bytes()
        prefix = (ROOT / n).read_bytes()[:binding['prefix_bytes']].decode('utf-8')
        assert prefix.startswith('# Precision diagnostic complete; nearby-gradient training design')
        assert 'USED/DIAGNOSTIC COMPLETE' in prefix and 'Goal incomplete.' in prefix
    guide = (ROOT / 'CCTV_DGP_RECOGNIZER_PRECISION_V1_VM.md').read_text(encoding='utf-8')
    old_commands = read(OUT / 'cctv_dgp_recognizer_precision_v1_publication/publication.json')['exact_command_blocks']
    assert re.findall(r'```(?:bat|bash)\n(.*?)\n```', guide, re.S) == list(old_commands.values())
    report = (ROOT / 'CCTV_DGP_RECOGNIZER_PRECISION_V1_RESULTS.md').read_text(encoding='utf-8')
    assert '41,576,988' in report and '6bbb53a45afaf6676675616c75775b35a16e3e4d9d5dbe3cb30f59b28963b54a' in report
    assert '| v9_tr_ffhq_59043_compound_lr24 | 6.7857% / fail | 6.8584% / fail |' in report
    assert '| v9_tr_asian_06782_lowlight_lr32 | 9.6853% / fail | 9.8010% / fail |' in report
    links = []
    for n in list(publication['documents']) + list(publication['new_documents_sha256']):
        length = publication['documents'][n]['prefix_bytes'] if n in publication['documents'] else None
        text = (ROOT / n).read_bytes()[:length].decode('utf-8')
        for target in re.findall(r'\]\(<([^>]+)>\)', text):
            assert Path(target).is_file(), target
            links.append(target)
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    assert publication['diagnostic_used'] and publication['next_training_is_design_only']
    assert not publication['automatic_VM_launch'] and publication['local_neural_or_training_calls'] == 0
    write(folder / 'publication_audit.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), protocol_sha256=PIN, publication_sha256=sha(folder / 'publication.json'),
        all462_return_bindings_rechecked=True, all9_history_suffixes_and2_new_reports_checked=True,
        failed_precision_comparisons_independently_recounted=failed,
        activation_changes_independently_recounted=counts, original30_parity_checks_retained=True,
        all5_old_manual_command_blocks_unchanged=True, verified_new_local_links=len(links),
        protected_files_exact=True, local_neural_or_training_calls=0, automatic_VM_launch=False,
        next_training_is_design_only=True, model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, history_documents=9, new_reports=2, response_failures=failed, protected=True))


if __name__ == '__main__':
    main()
