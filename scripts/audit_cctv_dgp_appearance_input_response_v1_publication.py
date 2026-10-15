"""Independent closure of published response counts, links, history and protected files."""
from datetime import datetime, timezone
from pathlib import Path
import re

from import_cctv_dgp_appearance_input_response_v1_return import ANALYSIS, OUT, PIN, RETURN, ROOT, read, sha, write


def main():
    stage = ANALYSIS / 'publication'
    published = read(stage / 'publication.json')
    assert published['complete'] and published['protocol_sha256'] == PIN
    audited = read(ANALYSIS / 'independent_audit.json')
    review = read(ANALYSIS / 'boundary_review.json')
    manifest = read(RETURN / 'export_manifest.json')
    assert all(sha(RETURN / n) == h for n, h in manifest['files'].items())
    assert all(sha(ROOT / n) == h for n, h in published['evidence_sha256'].items())
    assert all(sha(ROOT / n) == h for n, h in published['new_documents_sha256'].items())
    assert len(audited['per_case_gradient_parity_original_bounds']) == 100
    assert all(r['original_parity_check_pass'] for r in audited['per_case_gradient_parity_original_bounds'])
    assert len(review['group_records']) == 60 and len(review['case_records']) == 600
    classification = []
    for pair in range(3):
        groups = [r for r in audited['RGB_input_and_parameter_boundary_comparisons'] if r['pair'] == pair and r['repeat'] == 0]
        assert len(groups) == 10 and all(r['RGB_input_response']['resolved'] for r in groups)
        failed = sum(r['RGB_input_response']['central_comparison_within_diagnostic_bound'] is False for r in groups)
        canonical = next(r for r in review['summaries'] if r['pair'] == pair and r['repeat'] == 0)
        assert failed == canonical['input_canonical_fail'] == canonical['input_double_fail']
        classification.append(failed)
    assert classification == [9, 6, 3]
    assert len(published['documents']) == 8 and len(published['new_documents_sha256']) == 2
    for n, binding in published['documents'].items():
        before = Path(binding['backup'])
        assert sha(before) == binding['before_sha256'] and sha(ROOT / n) == binding['after_sha256']
        assert (ROOT / n).read_bytes()[binding['prefix_bytes']:] == before.read_bytes()
        text = (ROOT / n).read_text(encoding='utf-8')
        assert text.startswith('# Appearance-input diagnostic completed; training still unqualified')
        assert 'USED/DIAGNOSTIC COMPLETE' in text[:2400] and 'Goal incomplete.' in text[:3000]
    manual = read(OUT / 'cctv_dgp_appearance_input_response_v1_publication/publication.json')['exact_command_blocks']
    guide = (ROOT / 'CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'```(?:bat|bash)\n(.*?)\n```', guide, re.S) == list(manual.values())
    report = (ROOT / 'CCTV_DGP_APPEARANCE_INPUT_RESPONSE_V1_RESULTS.md').read_text(encoding='utf-8')
    assert '| 0 | 1 / 9 | 1 / 9 | 1 / 9 |' in report
    assert '| 1 | 4 / 6 | 4 / 6 | 5 / 5 |' in report
    assert '| 2 | 7 / 3 | 7 / 3 | 6 / 4 |' in report
    assert '40e96ceb18a67bd38813700320d834a97b37e41483f54940a77ce6e23524e361' in report
    paths = list(published['documents']) + list(published['new_documents_sha256'])
    links = []
    for n in paths:
        # Inspect the new prefix only; historical copied notes retain historical links.
        size = published['documents'][n]['prefix_bytes'] if n in published['documents'] else None
        text = (ROOT / n).read_bytes()[:size].decode('utf-8')
        for target in re.findall(r'\]\(<([^>]+)>\)', text):
            assert Path(target).is_file(), target
            links.append(target)
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    assert published['diagnostic_used'] and published['next_manual_diagnostic_is_design_only']
    assert not published['automatic_VM_launch'] and published['local_neural_or_training_calls'] == 0
    write(stage / 'publication_audit.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), publication_sha256=sha(stage / 'publication.json'),
        protocol_sha256=PIN, all193_return_bindings_freshly_rechecked=True,
        all8_history_suffixes_and2_reports_checked=True, all5_manual_command_blocks_unchanged=True,
        response_fail_counts_independently_recounted=classification, verified_new_links=len(links),
        protected_files_exact=True, local_neural_or_training_calls=0,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, history_documents=8, reports=2, failed_input_comparisons=classification, protected=True))


if __name__ == '__main__':
    main()
