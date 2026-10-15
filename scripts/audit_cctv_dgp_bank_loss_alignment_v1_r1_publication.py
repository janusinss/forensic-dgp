"""Independent publication, command, retained-history and pending-state closure."""
import ast
from datetime import datetime, timezone
import re
import tarfile

from import_cctv_dgp_bank_loss_alignment_v1_r1_return import ROOT, OUT, ANALYSIS, read, sha, write


def text(path):
    return path.read_text(encoding='utf-8')


def fences(value):
    return re.findall(rb'```[^\r\n]*\r?\n.*?```', value, re.S)


def main():
    receipt = ANALYSIS / 'publication_audit.json'
    assert not receipt.exists()
    publication = read(ANALYSIS / 'publication.json')
    correction = read(ANALYSIS / 'publication_wording_correction.json')
    assert publication['complete'] and correction['complete']
    assert correction['original_publication_sha256'] == sha(ANALYSIS / 'publication.json')
    documents = {}
    for name, original in publication['documents'].items():
        current = correction['documents'].get(name, original)
        path = ROOT / name
        backup = __import__('pathlib').Path(original['backup'])
        assert sha(path) == current['after_sha256'] and sha(backup) == original['before_sha256']
        assert path.read_bytes()[current['prefix_bytes']:] == backup.read_bytes()
        assert fences(path.read_bytes()) == fences(backup.read_bytes())
        documents[name] = sha(path)
    for name, digest in publication['new_documents_sha256'].items():
        current = correction['documents'].get(name)
        assert sha(ROOT / name) == (current['after_sha256'] if current else digest)
        documents[name] = sha(ROOT / name)
    for name, binding in correction['documents'].items():
        assert sha(binding['before_text_backup']) == binding['before_sha256']
        original = publication['documents'][name]['after_sha256'] if name in publication['documents'] else publication['new_documents_sha256'][name]
        assert original == binding['before_sha256']
    for name, digest in publication['evidence_sha256'].items():
        assert sha(ROOT / name) == digest and read(ROOT / name)['complete']
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())

    prep_path = OUT / 'cctv_dgp_bank_hinge_boundary_v1_preparation.json'
    prep = read(prep_path)
    transfer = read(OUT / 'cctv_dgp_bank_hinge_boundary_v1_transfer_audit.json')
    checker = read(OUT / 'cctv_dgp_bank_hinge_boundary_v1_return_checker_preparation.json')
    assert transfer['preparation_sha256'] == sha(prep_path)
    packet = OUT / prep['name']
    assert sha(packet / 'protocol.json') == prep['protocol_sha256']
    p = read(packet / 'protocol.json')
    assert p['initial_zero_guard_gradient_required'] and p['exact_scalar_guard_parity_required']
    assert p['bit_exact_control_check_retained'] and not p['automatic_follow_on']
    assert p['optimizer_updates'] == p['backwards'] == p['full_epochs'] == 0
    assert not (packet / 'outputs').exists()
    archive = OUT / (prep['stem'] + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256']
    assert (OUT / (archive.name + '.sha256')).read_bytes() == (prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    for name, digest in p['assets_sha256'].items():
        assert sha(packet / name) == digest
    with tarfile.open(archive, 'r:gz') as stream:
        expected = {prep['name']+'/'+name for name in p['assets_sha256']} | {prep['name']+'/protocol.json'}
        members = stream.getmembers()
        assert len(members) == 5 and {m.name for m in members} == expected
        for member in members:
            assert member.isfile() and stream.extractfile(member).read() == (OUT / member.name).read_bytes()
    for key, file in [('importer_sha256', 'import_cctv_dgp_bank_hinge_boundary_v1_return.py'),
                      ('auditor_sha256', 'audit_cctv_dgp_bank_hinge_boundary_v1_return.py')]:
        assert sha(ROOT / 'scripts' / file) == checker[key]
        ast.parse(text(ROOT / 'scripts' / file))
    assert checker['protocol_sha256'] == prep['protocol_sha256']
    assert checker['pure_initial_guard_validator_rejects_real_old_defect']
    assert checker['synthetic_summary_fixture_is_not_a_model_result']

    guide = text(ROOT / 'CCTV_DGP_BANK_HINGE_BOUNDARY_V1_VM.md')
    assert re.findall(r'^([1-5])\. ', guide, re.M) == list('12345')
    assert prep['protocol_sha256'] in guide and prep['archive_sha256'] in guide
    assert "tr -d '\\r'" in guide
    assert 'tmux new-session -A -s dgp_bank_hinge_boundary_v1' in guide
    assert 'tmux attach -t dgp_bank_hinge_boundary_v1' in guide
    assert 'scripts/cctv_dgp_bank_hinge_boundary_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha '+prep['protocol_sha256']+' --verify-transfer' in guide
    assert 'bash scripts/run_boundary.sh '+prep['protocol_sha256'] in guide
    calls = [line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(calls) == 5
    for call in calls:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in call
        assert call.count('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/') == 1
    for suffix in ['-results.tar.gz', '-results.tar.gz.sha256', '-export.json']:
        assert sum(prep['stem']+suffix+'"' in c for c in calls[2:]) == 1

    report = text(ROOT / 'CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_R1_RESULTS.md')
    review = read(ANALYSIS / 'loss_direction_review.json')
    assert '112.07 seconds' in report and '360 gradient queries' in report
    assert f"{prep['archive_bytes']:,}-byte packet" in report
    for state, title in [('initial', 'Pinned initial'), ('joint50', 'Rejected B50')]:
        r = review['group_summaries']['all'][state]
        row = f"| {title} | {r['scalar_zero_brightness_penalty_with_nonzero_gradient']} / 20 | {r['median_brightness_gradient_norm_over_detail_gradient_norm']:.2f}× | {r['nominal_single_reference_Adam_direction_detail_ascents']} / 20 | {r['nominal_single_reference_Adam_direction_degraded_MSE_ascents']} / 20 |"
        assert row in report
    prose = ' '.join(report.split())
    assert 'does **not** independently re-run all 40 neural derivatives' in prose
    assert 'not native CCTV outcomes' in prose and '15/20 initial and 20/20 B50' in prose
    assert 'More epochs are unqualified' in text(ROOT / 'PROJECT_HANDOFF.md')
    assert 'Goal **ACTIVE**' in report
    for stem in ['cctv-dgp-bank-loss-alignment-v1', 'cctv-dgp-bank-loss-alignment-v1-r1']:
        assert sha(OUT / (stem+'-results.tar.gz')) == read(OUT / (stem+'-export.json'))['archive_sha256']
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        publication_sha256=sha(ANALYSIS / 'publication.json'),
        wording_correction_sha256=sha(ANALYSIS / 'publication_wording_correction.json'),
        current_documents_sha256=documents, status_documents_checked=5, new_report_and_guide_checked=2,
        historical_suffixes_and_commands_retained=True, evidence_receipts_checked=len(publication['evidence_sha256']),
        protected_app_checkpoint_example_bindings_checked=len(protected), next_packet_members_checked=5,
        manual_guide_matches_protocol_and_sources=True, one_remote_source_per_SCP_call=True,
        prospective_import_and_return_auditor_hash_and_AST_checked=True,
        old_failure_and_completed_R1_archives_retained=True, R1_gradient_arithmetic_audit_bound=True,
        next_actual_L4_gradient_behavior_unrun=True, no_local_gradient_or_optimizer=True,
        no_agent_VM_launch=True, independent_final_review_and_all_seven_covering_families_pending=True,
        goal_complete=False))
    print(dict(complete=True, documents_checked=7, protected_bindings=12,
               next_manual_gradient_measurement_pending=True))


if __name__ == '__main__':
    main()
