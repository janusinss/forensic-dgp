"""Independent local evidence audit; no SSH, model imports or training."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v31_launch_import_v1'
PIN = 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'
ARCHIVE = 'bdf79346b10376b75328dfd2fab3ab3902935982cb9910b4478b401c0c55b9b8'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def transport(folder, stage, status):
    folder = ROOT / 'outputs' / folder
    receipt = read(folder / (stage + '_transport.json'))
    assert receipt['exit_code'] == status
    assert sha(folder / (stage + '.log')) == receipt['stdout_sha256']
    assert sha(folder / (stage + '_stderr.log')) == receipt['stderr_sha256']
    return folder


def main():
    failed = transport('cctv_dgp_v31_launch_status_v1', 'launch_status', 1)
    assert 'credentials.db' in (failed / 'launch_status_stderr.log').read_text()
    status_folder = transport('cctv_dgp_v31_launch_status_v1_r1', 'launch_status', 0)
    status = read(status_folder / 'launch_status.log')
    assert status['complete'] and status['read_only'] and not status['training_started_by_agent']
    assert status['protocol_sha256'] == PIN and not status['V31_processes']
    assert not status['trainer_log']['exists'] and not status['outputs_present']
    assert status['GPU']['exit_code'] == 0 and not status['GPU']['stdout'].strip()
    terminal = status['tmux_last_lines']['stdout']
    failure = "ModuleNotFoundError: No module named 'cctv_dgp_profile_batches_v31_schedule'"
    assert terminal.count(failure) == 2
    corrected_folder = transport('cctv_dgp_v31_launch_import_v1', 'corrected_transfer_check', 0)
    corrected = read(corrected_folder / 'corrected_transfer_check.log')
    assert corrected['complete'] and corrected['read_only_transfer_preflight']
    assert corrected['root_added_as_first_PYTHONPATH_entry'] and corrected['protocol_sha256'] == PIN
    assert corrected['packet_files_verified'] == 9 and not corrected['outputs_created']
    assert not corrected['trainer_log_created'] and not corrected['training_started_by_agent']
    pre = corrected['transfer_receipt']
    assert pre == {'complete': True, 'packet_assets': 8, 'original_assets': 246,
                   'initial_proof_cases': 50, 'training_cases': 3905, 'training_references': 781,
                   'optimizer_updates': 0, 'neural_or_gradient_calls': 0}
    bundle = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
    assert sha(bundle / 'protocol.json') == PIN
    p = read(bundle / 'protocol.json')
    for name, digest in p['assets_sha256'].items(): assert sha(bundle / name) == digest, name
    assert sha(ROOT / 'outputs/cctv-dgp-profile-batches-v31-execution.tar.gz') == ARCHIVE
    guide = ROOT / 'CCTV_DGP_PROFILE_BATCHES_V31_LAUNCH_IMPORT_V1.md'
    assert 'export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" &&' in guide.read_text()
    assert guide.read_text().count('gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a') == 3
    evidence = [ROOT / 'scripts/inspect_cctv_dgp_v31_launch_status_v1.py',
                ROOT / 'scripts/verify_cctv_dgp_v31_launch_import_v1.py', Path(__file__), guide]
    for folder in [failed, status_folder, corrected_folder]:
        evidence.extend(f for f in folder.iterdir() if f.is_file())
    evidence.extend(bundle / name for name in ['protocol.json', *p['assets_sha256']])
    evidence.append(ROOT / 'outputs/cctv-dgp-profile-batches-v31-execution.tar.gz')
    receipt = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
               'checker_sha256': sha(Path(__file__)), 'evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(set(evidence))},
               'original_missing_module_failures_retained': 2, 'corrected_transfer_check_passed': True,
               'packet_protocol_and_training_recipe_unchanged': True, 'protocol_sha256': PIN,
               'archive_sha256': ARCHIVE, 'VM_training_started_by_agent': False,
               'local_neural_or_gradient_calls': 0, 'app_promotion': False, 'reserved_final_used': False,
               'goal_complete': False}
    with (OUT / 'independent_launch_audit.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(receipt, stream, indent=2)
    print(json.dumps({'complete': True, 'bindings_verified': len(receipt['evidence_sha256']),
                      'corrected_transfer_check_passed': True, 'training_started_by_agent': False,
                      'goal_complete': False}, indent=2))


if __name__ == '__main__': main()
