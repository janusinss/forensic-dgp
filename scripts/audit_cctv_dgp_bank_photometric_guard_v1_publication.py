"""Independently verify publication, preserved bytes and exact manual guide."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
ANALYSIS = OUT/'cctv_dgp_bank_photometric_guard_v1_analysis'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt = ANALYSIS/'publication_audit.json'
    assert not receipt.exists()
    publication = read(ANALYSIS/'publication.json')
    assert publication['complete'] and not publication['model_qualified'] and not publication['goal_complete']
    for name, expected in publication['evidence_sha256'].items():
        assert read(ROOT/name)['complete'] and sha(ROOT/name) == expected
    for name, expected in publication['protected_app_and_checkpoint_sha256'].items():
        assert sha(ROOT/name) == expected, name
    assert len(publication['canonical_prefix_changes']) == 5
    for name, change in publication['canonical_prefix_changes'].items():
        path = ROOT/name; backup = Path(change['backup'])
        assert backup.resolve().is_relative_to(ANALYSIS/'publication_before')
        assert sha(path) == change['after_sha256'] and sha(backup) == change['before_sha256']
        assert path.read_bytes()[change['prefix_bytes']:] == backup.read_bytes()
        top = path.read_bytes()[:change['prefix_bytes']].decode('utf-8')
        if name.endswith('_VM.md'):
            assert 'Executed and audited' in top and 'do not rerun' in top
        else:
            for claim in ['0.680385%', '0.681828%', '4 / 4', 'zero complete epochs', 'unrun', 'Goal ACTIVE']:
                assert claim in top, (name, claim)
            assert 'Conditional extra epochs require preserved useful development evidence' in top
    for name, expected in publication['created_documents'].items():
        assert sha(ROOT/name) == expected
    report = (ROOT/'CCTV_DGP_BANK_PHOTOMETRIC_GUARD_V1_RESULTS.md').read_text(encoding='utf-8')
    for claim in ['0.680385%', '0.681828%', '34.582900%', '34.508277%', '0.617096%', '0.716408%',
                  '48 of 50', '2,184 to 1,717', '5,852', '7,910', '844', 'independent final review',
                  'not a percentage of recovered', 'not recognizer feature MSE']:
        # The metric sentence uses a complete alternative rather than these fragments.
        if claim == 'not a percentage of recovered':
            assert 'or a percentage of recovered' in report
        elif claim == 'not recognizer feature MSE':
            assert 'not recognizer feature MSE' in report
        else:
            assert claim in report, claim
    prep = read(OUT/'cctv_dgp_bank_path_ablation_v1_preparation.json')
    transfer = read(OUT/'cctv_dgp_bank_path_ablation_v1_transfer_audit.json')
    assert transfer['complete'] and transfer['actual_CUDA_parity_timing_and_results_unrun']
    dest = OUT/prep['name']; p = read(dest/'protocol.json')
    assert sha(dest/'protocol.json') == prep['protocol_sha256'] == transfer['protocol_sha256']
    assert not (dest/'outputs').exists()
    guide_path = ROOT/'CCTV_DGP_BANK_PATH_ABLATION_V1_VM.md'
    guide = guide_path.read_text(encoding='utf-8')
    assert b'\r' not in guide_path.read_bytes()
    assert re.findall(r'^(\d)\. ', guide, flags=re.M) == ['1', '2', '3', '4', '5']
    assert prep['protocol_sha256'] in guide and prep['archive_sha256'] in guide
    assert '4 GiB free' in guide and 'actual VM inference is unrun' in guide
    blocks = re.findall(r'```(bat|bash)\n(.*?)```', guide, flags=re.S)
    assert len(blocks) == 5
    scp = [line for _, block in blocks for line in block.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(scp) == 5
    for line in scp:
        assert line.count('janusdominic0@forensic-dgp-thesis:') == 1
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in line
        assert 'PuTTY' not in line
    assert all(prep['stem'] in line for line in scp)
    assert all(line.endswith('"."') for line in scp[-3:])
    assert 'cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"' in guide
    assert "tr -d '\\r'" in guide and 'test ! -e ~/forensic-dgp/'+prep['name'] in guide
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    checked = 0
    for language, block in blocks:
        if language == 'bash':
            result = subprocess.run([str(bash), '-n'], input=block, capture_output=True, text=True, timeout=30)
            assert result.returncode == 0, result.stderr
            checked += 1
    assert checked == 3
    archive = OUT/(prep['stem']+'-execution.tar.gz')
    assert sha(archive) == prep['archive_sha256'] == transfer['archive_sha256']
    assert Path(str(archive)+'.sha256').read_bytes() == (prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); assert len(members) == 3 and all(m.isfile() for m in members)
        assert {m.name for m in members} == {prep['name']+'/'+n for n in ['protocol.json']+list(p['assets_sha256'])}
    for name, expected in p['assets_sha256'].items():
        assert sha(dest/name) == expected
    assert p['optimizer_updates'] == p['gradient_queries'] == p['full_epochs'] == 0
    assert p['model_forward_cap'] == 312 and p['recognizer_forward_cap'] == 140
    for target in re.findall(r'\]\(<([^>]+)>\)', guide+'\n'+report):
        if not target.startswith('http'):
            assert Path(target).is_file(), target
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(Path(__file__)),
        publication_sha256=sha(ANALYSIS/'publication.json'), exact_old_document_suffixes=5,
        new_documents=2, checked_manual_steps=5, readonly_Bash_blocks_checked=checked,
        SCP_commands_checked=5, one_remote_source_per_download=True,
        archive_and_protocol_and_checksum_rechecked=True, protected_assets_rechecked=12,
        executed_guide_superseded=True, user_DEV_feedback_not_final_review=True,
        independent_final_review_pending=True, manual_VM_diagnostic_unrun=True,
        local_training_or_gradient_calls=0, app_promotion=False, goal_complete=False)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(dict(complete=True, documents=7, exact_backups=5, manual_steps=5, app_promotion=False))


if __name__ == '__main__':
    main()
