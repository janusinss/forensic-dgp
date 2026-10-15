"""Check published status, commands and retained history without a neural call."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DESIGN = ROOT / 'outputs/cctv_dgp_spatial_path_replay_v1_design'
NAME = 'cctv_dgp_spatial_path_replay_v1_vm'
STEM = 'cctv-dgp-spatial-path-replay-v1'
PIN = '6f9a8757466578ef57457994ded7fe9b6b170657f2ee7dd3b8b23f0bd2cc1bc7'
ARCHIVE = '228e7bb9c1a957021c86be471f0eea1f3f0a5c38ada99dedf4fb69756b8c7f88'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    stage = DESIGN / 'verified_publication'
    destination = stage / 'independent_publication_audit.json'
    assert not destination.exists(), 'Retain completed audit'
    publication = read(stage / 'publication.json')
    assert publication['complete'] and publication['status'] == 'VERIFIED_PREPARED_UNRUN'
    packet = ROOT / 'outputs' / NAME
    protocol = read(packet / 'protocol.json')
    assert sha(packet / 'protocol.json') == PIN == publication['protocol_sha256']
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    assert sha(archive) == ARCHIVE == publication['archive_sha256']
    assert Path(str(archive) + '.sha256').read_bytes() == (
        ARCHIVE + '  ' + archive.name + '\n').encode('utf-8')
    for name, digest in protocol['assets_sha256'].items():
        assert sha(packet / name) == digest, 'Sealed asset changed: ' + name
    for name, binding in publication['documents'].items():
        path = ROOT / name
        before = Path(binding['backup'])
        assert sha(before) == binding['before_sha256']
        assert sha(path) == binding['after_sha256']
        current = path.read_bytes()
        assert current[binding['prefix_bytes']:] == before.read_bytes()
        prefix = current[:binding['prefix_bytes']].decode('utf-8')
        for required in ['PREPARED / UNRUN', '0.877397%', '0.132699%', 'zero new optimizer',
                         'not yet verified', 'No automatic launch', 'five' ]:
            assert required.lower() in prefix.lower(), (name, required)
        assert publication['app_promotion'] is False and publication['goal_complete'] is False
    for name, digest in publication['new_document_sha256'].items():
        assert sha(ROOT / name) == digest
    for name, digest in publication['unchanged_protected_files'].items():
        assert sha(ROOT / name) == digest
    guide = (ROOT / 'CCTV_DGP_SPATIAL_PATH_REPLAY_V1_VM.md').read_text(encoding='utf-8')
    steps = [line for line in guide.splitlines() if line[:3] in ['1. ', '2. ', '3. ', '4. ', '5. ']]
    assert len(steps) == 5
    upload = [line for line in guide.splitlines() if line.startswith('gcloud compute scp') and '-execution.' in line]
    download = [line for line in guide.splitlines() if line.startswith('gcloud compute scp') and '-execution.' not in line]
    assert len(upload) == 2 and len(download) == 3
    for line in upload + download:
        assert '--project=forensic-dgp-thesis' in line and '--zone=us-central1-a' in line
        assert line.count('janusdominic0@forensic-dgp-thesis:') == 1
    for suffix, line in zip(['-results.tar.gz', '-results.tar.gz.sha256', '-export.json'], download):
        assert ('/home/janusdominic0/' + STEM + suffix + '" "."') in line
    assert f'sha256sum -c {STEM}-execution.tar.gz.sha256' in guide
    assert f'test ! -e ~/forensic-dgp/{NAME}' in guide
    assert 'tmux new-session -A -s dgp_spatial_path_replay_v1' in guide
    assert f'cd ~/forensic-dgp/{NAME} &&' in guide
    assert f'--protocol-sha {PIN} --verify-transfer' in guide
    assert f'bash scripts/run_spatial_path.sh {PIN}' in guide
    assert '8 GiB free after install' in guide and '8 GiB available RAM before models' in guide
    assert '15–30 minutes' in guide and 'new runtime is unmeasured' in guide
    shell = (packet / 'scripts/run_spatial_path.sh').read_bytes()
    assert b'\r' not in shell and b'--supervise' in shell and b'${TMUX:-}' in shell
    help_run = subprocess.run([sys.executable, '-B', str(packet / 'scripts/cctv_dgp_spatial_path_replay_v1_vm.py'),
                               '--help'], cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert help_run.returncode == 0
    for flag in ['--verify-transfer', '--review', '--audit-saved', '--export', '--supervise']:
        assert flag in help_run.stdout
    assert '--train' not in help_run.stdout and '--resume' not in help_run.stdout
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    ledger_path = DESIGN / 'experiment_status_ledger.json'
    ledger = read(ledger_path)
    assert sha(ledger_path) == publication['experiment_ledger_sha256']
    old = read(ROOT / 'outputs/cctv_dgp_bank_group_direction_forward_v1_analysis/finite_metrics.json')['proposals']['balanced_0.02']['gates']
    new = read(ROOT / 'outputs/cctv_dgp_spatial_readout_v1_return/outputs/spatial_readout_v1/run_results.json')['outcomes']
    expected = [100 * old['png']['relative_feature_gain']] + [100 * row['gates']['png']['global_structure_gain'] for row in new]
    assert [row['PNG_gain_percent'] for row in ledger['rows']] == expected
    assert [row['PNG_failed_group_metric_checks'] for row in ledger['rows']] == [37, 8, 18]
    assert not any(row['accepted'] for row in ledger['rows']) and ledger['accepted_development_updates'] == 0
    compare = read(DESIGN / 'progress_comparison_audit.json')
    assert sha(DESIGN / 'progress_comparison_audit.json') == publication['progress_comparison_sha256']
    assert compare['complete'] and compare['same_scientific_thresholds'] and compare['baseline_RAW_controls'] == 100
    assert len(compare['baseline_batches']) == 20
    assert sha(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth') == ledger['original_checkpoint_sha256']
    result = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
                  published_documents=len(publication['documents']), new_documents=len(publication['new_document_sha256']),
                  old_document_bytes_preserved_as_exact_suffix=True,
                  same100_input_and_original_baseline_comparison_retained=True,
                  RAW_PNG_gain_and_failed_check_ledger_verified=True,
                  five_manual_steps_and_single_remote_source_downloads_checked=True,
                  actual_CLI_modes_checked=True, finite_limits_match_protocol=True,
                  archive_sha256=ARCHIVE, protocol_sha256=PIN,
                  original_checkpoint_unchanged=True, status='VERIFIED_PREPARED_UNRUN',
                  neural_calls=0, gradient_queries=0, optimizer_updates=0,
                  agent_VM_calls=0, app_promotion=False, model_qualified=False, goal_complete=False,
                  publication_sha256=sha(stage / 'publication.json'), checker_sha256=sha(__file__))
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
