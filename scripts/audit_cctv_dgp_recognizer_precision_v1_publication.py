"""Close manual-command publication without launching any VM operation."""
from datetime import datetime, timezone
from pathlib import Path
import re

from import_cctv_dgp_recognizer_precision_v1_return import ROOT, OUT, NAME, STEM, PIN, read, sha, write


def main():
    folder = OUT / 'cctv_dgp_recognizer_precision_v1_publication'
    published = read(folder / 'publication.json'); assert published['complete'] and published['protocol_sha256'] == PIN
    prepared = read(OUT / 'cctv_dgp_recognizer_precision_v1_preparation.json')
    transfer = read(OUT / 'cctv_dgp_recognizer_precision_v1_transfer_audit.json')
    assert prepared['complete'] and transfer['complete'] and prepared['manual_UNRUN'] and transfer['manual_UNRUN']
    assert published['archive_sha256'] == prepared['archive_sha256'] == transfer['archive_sha256'] == sha(OUT / (STEM + '-execution.tar.gz'))
    assert PIN == sha(OUT / NAME / 'protocol.json') == transfer['protocol_sha256']
    assert all(sha(ROOT / n) == h for n, h in published['evidence_sha256'].items())
    assert all(sha(ROOT / n) == h for n, h in published['new_documents_sha256'].items())
    assert len(published['documents']) == 7 and len(published['new_documents_sha256']) == 2
    for n, b in published['documents'].items():
        before = Path(b['backup'])
        assert sha(before) == b['before_sha256'] and sha(ROOT / n) == b['after_sha256']
        assert (ROOT / n).read_bytes()[b['prefix_bytes']:] == before.read_bytes()
        text = (ROOT / n).read_bytes()[:b['prefix_bytes']].decode('utf-8')
        assert text.startswith('# Recognizer precision diagnostic verified, UNRUN') and 'Goal incomplete.' in text
        for target in re.findall(r'\]\(<([^>]+)>\)', text):
            assert Path(target).is_file(), target
    guide = (ROOT / 'CCTV_DGP_RECOGNIZER_PRECISION_V1_VM.md').read_text(encoding='utf-8')
    blocks = re.findall(r'```(?:bat|bash)\n(.*?)\n```', guide, re.S)
    assert blocks == list(published['exact_command_blocks'].values()) and len(blocks) == 5
    upload, install, tmux, launch, download = blocks
    assert upload.count('gcloud compute scp') == 2 and f'"{STEM}-execution.tar.gz"' in upload
    assert 'cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"' in upload and 'cd /d' in download
    assert f'sha256sum -c {STEM}-execution.tar.gz.sha256' in install and f'test ! -e ~/forensic-dgp/{NAME}' in install
    assert tmux == 'tmux new-session -A -s dgp_recognizer_precision_v1'
    assert launch.count(PIN) == 2 and '--verify-transfer' in launch and 'bash scripts/run_precision.sh' in launch
    scp = [line for line in download.splitlines() if line.startswith('gcloud compute scp')]
    assert len(scp) == 3 and all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in scp)
    for suffix in ['-results.tar.gz', '-results.tar.gz.sha256', '-export.json']:
        assert sum(STEM + suffix + '"' in line for line in scp) == 1
    assert all('--project=forensic-dgp-thesis --zone=us-central1-a' in line for line in scp)
    assert '12 minutes\n30 seconds' in guide and '**3 GiB free after installation**' in guide
    assert 'ca8d2fd69dffb19e040a36eb059e4f4124f00f05755cad976d24f8a00f4dbbe7' in guide
    assert not (OUT / NAME / 'outputs').exists() and not (OUT / NAME / 'review.log').exists()
    protected = read(OUT / 'cctv_dgp_appearance_input_response_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    assert published['manual_UNRUN'] and not published['automatic_VM_launch'] and published['local_neural_or_training_calls'] == 0
    write(folder / 'publication_audit.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), publication_sha256=sha(folder / 'publication.json'),
        protocol_sha256=PIN, archive_sha256=sha(OUT / (STEM + '-execution.tar.gz')),
        history_suffixes_verified=7, new_documents_verified=2, exact_command_blocks_verified=5,
        Windows_upload_and_single_remote_download_paths_checked=True, sealed_packet_unrun=True,
        protected_files_exact=True, local_neural_or_training_calls=0, VM_launch=False,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, history_documents=7, manual_command_blocks=5, protected=True, manual_UNRUN=True))


if __name__ == '__main__':
    main()
