"""Bind current prose, historical suffixes, commands, and prior scientific audits."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tarfile


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
ANALYSIS = OUT / 'cctv_dgp_bank_path_ablation_v1_analysis'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fences(payload):
    return re.findall(rb'```[^\r\n]*\r?\n.*?```', payload, re.S)


def main():
    receipt = ANALYSIS / 'publication_audit.json'
    assert not receipt.exists(), 'Retain the previous publication audit'
    published = read(ANALYSIS / 'publication.json')
    polished = read(ANALYSIS / 'publication_text_polish.json')
    assert published['complete'] and polished['complete']
    assert polished['original_publication_sha256'] == sha(ANALYSIS / 'publication.json')
    documents = {}
    for name, binding in polished['documents'].items():
        current = ROOT / name
        backup = Path(binding['before_text_backup'])
        payload = current.read_bytes()
        original = backup.read_bytes()
        assert sha(current) == binding['after_sha256'], name
        assert sha(backup) == binding['before_sha256'], name
        assert fences(payload) == fences(original), name + ' command block changed'
        if name in published['documents']:
            first = published['documents'][name]
            assert first['after_sha256'] == binding['before_sha256']
            historical = Path(first['backup'])
            assert sha(historical) == first['before_sha256']
            old = historical.read_bytes()
            assert payload[binding['prefix_bytes']:] == old, name + ' history changed'
        else:
            assert published['new_documents_sha256'][name] == binding['before_sha256']
            assert binding['prefix_bytes'] == len(payload)
        documents[name] = sha(current)

    for name, digest in published['evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
        assert read(ROOT / name)['complete'], name
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    for name, digest in protected.items():
        assert sha(ROOT / name) == digest, 'Protected binding changed: ' + name

    prep = read(OUT / 'cctv_dgp_bank_loss_alignment_v1_preparation.json')
    transfer = read(OUT / 'cctv_dgp_bank_loss_alignment_v1_transfer_audit.json')
    root = OUT / prep['name']
    protocol = read(root / 'protocol.json')
    archive = OUT / (prep['stem'] + '-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] == transfer['archive_sha256']
    assert archive.stat().st_size == prep['archive_bytes'] == 255429
    assert sha(root / 'protocol.json') == prep['protocol_sha256'] == transfer['protocol_sha256']
    checksum = archive.with_name(archive.name + '.sha256').read_bytes()
    assert checksum == (prep['archive_sha256'] + '  ' + archive.name + '\n').encode('ascii')
    assert not (root / 'outputs').exists(), 'Prepared packet unexpectedly executed'
    for name, digest in protocol['assets_sha256'].items():
        assert sha(root / name) == digest, name
    expected = {prep['name'] + '/' + name for name in protocol['assets_sha256']}
    expected.add(prep['name'] + '/protocol.json')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        assert len(members) == 4 and {m.name for m in members} == expected
        for member in members:
            assert member.isfile()
            local = OUT.joinpath(*PurePosixPath(member.name).parts)
            assert stream.extractfile(member).read() == local.read_bytes()

    guide = (ROOT / 'CCTV_DGP_BANK_LOSS_ALIGNMENT_V1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ', guide, re.M) == list('12345')
    assert prep['protocol_sha256'] in guide and prep['archive_sha256'] in guide
    assert "tr -d '\\r'" in guide
    assert 'tmux new-session -A -s dgp_bank_loss_alignment_v1' in guide
    assert 'tmux attach -t dgp_bank_loss_alignment_v1' in guide
    assert 'source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate' in guide
    assert 'scripts/cctv_dgp_bank_loss_alignment_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm' in guide
    assert 'bash scripts/run_alignment.sh ' + prep['protocol_sha256'] in guide
    calls = [line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(calls) == 5
    for call in calls:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in call
        assert call.count('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/') == 1
    for suffix in ['-results.tar.gz', '-results.tar.gz.sha256', '-export.json']:
        assert sum(prep['stem'] + suffix + '"' in call for call in calls[2:]) == 1
    assert protocol['optimizer_updates'] == protocol['backwards'] == protocol['full_epochs'] == 0
    assert protocol['maximum_gradient_queries'] == 360
    assert protocol['minimum_free_bytes'] == 4 * 1024**3
    assert not protocol['automatic_follow_on'] and not protocol['failed_run_resume_allowed']

    audit = read(ANALYSIS / 'independent_audit.json')
    counter = read(OUT / 'cctv_dgp_bank_partition_v1_analysis/independent_audit.json')
    visual = read(ANALYSIS / 'visual_review.json')
    counter_visual = read(OUT / 'cctv_dgp_bank_partition_v1_analysis/visual_review.json')
    report = (ROOT / 'CCTV_DGP_BANK_PATH_ABLATION_V1_RESULTS.md').read_text(encoding='utf-8')
    for label, gate in audit['partial_preview_gate_diagnostics'].items():
        raw, png = gate['raw'], gate['png']
        row = f"| {label} | {100*raw['gain']:.6f}% / {100*png['gain']:.6f}% | {len(raw['failures'])} / {len(png['failures'])} | {100*raw['brightness']:.6f}% / {100*png['brightness']:.6f}% |"
        assert row in report and not raw['pass'] and not png['pass']
    for label, gate in counter['partial_preview_gate_diagnostics'].items():
        raw, png = gate['raw'], gate['png']
        row = f"| {label} | {100*raw['gain']:.6f}% / {100*png['gain']:.6f}% | {len(raw['failures'])} / {len(png['failures'])} |"
        assert row in report and not raw['pass'] and not png['pass']
    assert visual['viewed_sheets'] == counter_visual['viewed_sheets'] == 26
    assert visual['sheet_cells_independently_verified'] == 1216
    assert counter_visual['sheet_cells_independently_verified'] == 720
    assert audit['VM_counts_recomputed']['optimizer_updates'] == 0
    assert 'not independently' in report and 'unpaired' in report
    assert 'independent final review is pending' in report
    assert 'Goal **ACTIVE and incomplete**' in report

    result = {
        'complete': True,
        'UTC': datetime.now(timezone.utc).isoformat(),
        'checker_sha256': sha(__file__),
        'publication_sha256': sha(ANALYSIS / 'publication.json'),
        'text_polish_sha256': sha(ANALYSIS / 'publication_text_polish.json'),
        'current_documents_sha256': documents,
        'unchanged_historical_suffixes': len(published['documents']),
        'unchanged_command_fence_documents': len(documents),
        'evidence_receipts_hash_checked': len(published['evidence_sha256']),
        'protected_app_checkpoint_and_example_bindings_checked': len(protected),
        'manual_guide_matches_sealed_archive_and_protocol': True,
        'single_remote_source_per_SCP_call': True,
        'new_packet_assets_checked': len(protocol['assets_sha256']),
        'archive_members_checked': 4,
        'all_seven_readout_and_three_counterfactual_table_rows_checked': True,
        'both_26_sheet_visual_closures_bound': True,
        'local_gradient_queries': 0,
        'local_optimizer_updates': 0,
        'agent_VM_launch': False,
        'next_manual_gradient_diagnostic_unrun': True,
        'goal_complete': False,
    }
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'complete': True, 'documents_checked': len(documents),
                      'protected_bindings_checked': len(protected),
                      'next_manual_diagnostic_unrun': True, 'receipt': str(receipt)}))


if __name__ == '__main__':
    main()
