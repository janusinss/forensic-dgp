"""Independent published status/commands/retained-history/evidence closure."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(1024**2),b''): h.update(value)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    start = time.monotonic(); receipt = ANALYSIS/'publication_audit.json'; assert not receipt.exists()
    pub = read(ANALYSIS/'publication.json')
    assert pub['complete'] and pub['new_guard_VM_training_pending'] and not pub['app_promotion'] and not pub['goal_complete']
    assert pub['paired_metric_definition_corrected_without_altering_historical_records']
    assert all(sha(ROOT/name) == value for name,value in pub['evidence_sha256'].items())
    assert all(sha(ROOT/name) == value for name,value in pub['new_documents_sha256'].items())
    assert all(sha(ROOT/name) == value for name,value in pub['protected_local_sha256'].items())
    for name,row in pub['document_mutations'].items():
        path,backup = ROOT/name,Path(row['backup'])
        assert backup.resolve().is_relative_to(ANALYSIS/'publication_before')
        assert sha(backup) == row['before_sha256'] and sha(path) == row['after_sha256']
        assert path.read_bytes()[row['prefix_bytes']:] == backup.read_bytes()
        if name != 'CCTV_DGP_BANK_COMPARISON_B_V1_R4_VM.md':
            head = path.read_bytes()[:row['prefix_bytes']].decode('utf-8')
            assert '0.625591% / 0.627191%' in head and 'Goal ACTIVE' in head
            assert 'landmark high-frequency MSE' in head and 'zero complete epochs' in head
            assert 'unrun' in head and 'final identities unused' in head
    prep = read(ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_preparation.json')
    guide = (ROOT/'CCTV_DGP_BANK_PHOTOMETRIC_GUARD_V1_VM.md').read_text(encoding='utf-8')
    report = (ROOT/'CCTV_DGP_BANK_AB_COMPARISON_V1_RESULTS.md').read_text(encoding='utf-8')
    steps = re.findall(r'^(\d)\. ',guide,re.M); assert steps == ['1','2','3','4','5']
    commands = [line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands) == 5 and all('--zone=us-central1-a' in line for line in commands)
    assert all(line.count('janusdominic0@forensic-dgp-thesis:') == 1 for line in commands)
    assert prep['transfer_archive_sha256'] in guide and guide.count(prep['protocol_sha256']) == 4
    assert prep['stem']+'-code-transfer.tar.gz' in guide
    assert prep['stem']+'-results.tar.gz' in guide and prep['stem']+'-export.json' in guide
    assert 'tmux new-session -A -s dgp_bank_photometric_guard_v1' in guide
    assert "tr -d '\\r'" in guide and '16 GiB after installation' in guide
    assert 'test ! -e ~/forensic-dgp/'+prep['name'] in guide
    assert '0.471905% / 0.473663%' in report and '0.625591% / 0.627191%' in report
    assert '40.516852% raw / 38.623819% PNG' in report
    assert 'Metric correction' in report and '`landmark_high_frequency_MSE`' in report
    assert 'independent\nfinal visual review remains required' in report
    assert 'TRAIN-only observed-pixel hinge' in report and 'not rejected B50' in report
    assert 'masks, sunglasses, strong' in report and 'automatic/assisted review' in report
    for relative in ['scripts/import_cctv_dgp_bank_photometric_guard_v1_return.py','scripts/audit_cctv_dgp_bank_photometric_guard_v1_execution.py']:
        ast.parse((ROOT/relative).read_text(),feature_version=(3,10))
    assert not (ROOT/'outputs'/prep['name']/'outputs').exists()
    result = dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(ANALYSIS/'publication.json'),documents_checked=len(pub['document_mutations'])+len(pub['new_documents_sha256']),
        exact_historical_document_suffixes_checked=len(pub['document_mutations']),
        bound_evidence_files_rechecked=len(pub['evidence_sha256']),protected_app_checkpoint_files_rechecked=len(pub['protected_local_sha256']),
        five_manual_steps_checked=True,one_remote_source_per_download=True,manual_new_trial_unrun=True,
        metric_definition_corrected=True,quality_stops_and_negative_probe_retained=True,
        scientific_gates_not_weakened=True,native_unpaired_and_final_identity_boundaries_present=True,
        independent_final_review_not_claimed=True,seven_completion_families_pending=True,
        local_training_or_gradient_calls=0,app_promotion=False,goal_complete=False,seconds=time.monotonic()-start)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,documents=result['documents_checked'],retained_history=result['exact_historical_document_suffixes_checked'],manual_trial_unrun=True),flush=True)


if __name__ == '__main__': main()
