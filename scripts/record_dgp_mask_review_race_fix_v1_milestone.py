"""Record the application review fix without changing historical research records."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/dgp_mask_review_race_fix_v1'
PREVIOUS = ROOT / 'outputs/dgp_sampling_review_and_profile_batches_v31_milestone'
DOCS = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md']


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    assert not (OUT / 'milestone.json').exists()
    old = read(PREVIOUS / 'milestone.json')
    assert sha(PREVIOUS / 'milestone.json') == 'a6dd026c8d343163ed7ba599cca7924af4085579960bece9568484c2280caa3f'
    for row in old['documents']: assert sha(ROOT / row['name']) == row['after_sha256']
    audit = read(OUT / 'independent_saved_output_audit.json'); assert audit['complete']
    launch = read(ROOT / 'outputs/cctv_dgp_v31_launch_import_v1/independent_launch_audit.json')
    assert launch['complete'] and launch['corrected_transfer_check_passed'] and not launch['VM_training_started_by_agent']
    assert audit['current_JS_sha256'] == sha(ROOT / 'static/face_workflow.js')
    assert not (ROOT / 'outputs/cctv-dgp-profile-batches-v31-results.tar.gz').exists()
    addition = '''
**Latest application milestone - 7 October 2026: imported masks require fresh review before generation.**

A reproduced frontend race allowed generation with the previously reviewed mask
while a replacement mask loaded, then displayed the old-area result after review
was cleared. Import now locks generation immediately and clears approval. Invalid
imports recover controls without approval; a superseded import cannot alter or
unlock a newer face upload. Three inline Playwright ordering regressions pass.

The real local app completed 11 generation and 11 proposal requests in 66.29 seconds
across all seven covering families, uncovered/clear-glasses controls and On/Off/Auto
selection. Two insufficient/unsupported input decisions submit zero generation
requests. All 11 PNG downloads match the prior audited neural processing exactly;
the independent audit also verifies 1,521,399 visible Off bytes and a complete
original/mask/result/raw-DGP bundle. Responsive 375/768/1280 checks have no overflow,
page exceptions or console errors. The existing design and backend are preserved.

This verifies review ordering and functional development flow. The cases are
previously exposed photographs, not native CCTV or independent final evidence.
Their legacy native suffix means original photo. Covering generation uses reviewed
operator masks; automatic-family quality remains unqualified. The prior finger,
scarf, gaze and DGP visible-softening defects are unchanged. No hidden-identity,
ethnicity or Zamboanga-performance claim is made. No training occurred.

The original frontend is archived at
outputs/dgp_mask_review_race_fix_v1/before/static/face_workflow.js.
Historical app22 source/evidence bindings remain verified against that archived
source where required; other bindings stay at their original locations. The
current frontend is separately bound. Original checkpoints, splits, scientific
caches, failure records, previous documents and the actual Windows backup remain.

The user's V31 launch then failed before training: the schedule helper in the
packet root was missing from Python's script import path. A read-only VM status
check retained two terminal tracebacks, confirmed no V31 process/log/outputs and
an idle GPU. A second read-only check verified the command-only PYTHONPATH fix:
all 3,905 cases/781 references pass transfer verification with zero neural,
gradient or optimizer calls. No training was started by the agent and no pilot
file was changed. The packet/protocol/gates are unchanged and no return is
present locally. The corrected launcher supersedes only step4 of the old guide.
Actual new training remains the user's manual tmux workflow on the existing L4 VM.
Useful native restoration, automatic/assisted covering quality and independent
final review remain outstanding. Goal active/incomplete.

[Application ordering fix and verification](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_MASK_REVIEW_RACE_FIX_V1.md>)
[V31 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_VM.md>)
[Corrected V31 tmux launch](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_PROFILE_BATCHES_V31_LAUNCH_IMPORT_V1.md>)

The complete previous document body is preserved below. Its unchanged-app-source
statements refer to the archived pre-fix frontend; neural processing is unchanged.

'''.encode('ascii')
    (OUT / 'before_docs').mkdir(); rows = []
    for name in DOCS:
        path = ROOT / name; before = path.read_bytes(); backup = OUT / 'before_docs' / name
        with backup.open('xb') as stream: stream.write(before)
        split = before.index(b'\n') + 1; after = before[:split] + addition + before[split:]
        path.write_bytes(after)
        rows.append({'name': name, 'before_path': backup.relative_to(ROOT).as_posix(), 'before_sha256': sha(backup),
                     'after_sha256': sha(path), 'addition_bytes': len(addition)})
    files = [ROOT / name for name in DOCS]
    files.extend(f for f in OUT.rglob('*') if f.is_file())
    scratch = ROOT / 'scratch/mask-import-race-v1'
    files.extend(f for f in scratch.rglob('*') if f.is_file() and f.suffix in ['.json', '.png', '.zip'])
    files.extend(ROOT / name for name in ['CCTV_DGP_MASK_REVIEW_RACE_FIX_V1.md', 'static/face_workflow.js',
                                         'scripts/audit_dgp_mask_review_race_fix_v1.py',
                                         'scripts/record_dgp_mask_review_race_fix_v1_milestone.py',
                                         'scripts/verify_dgp_mask_review_race_fix_v1_milestone.py'])
    files.extend(ROOT / name for name in ['CCTV_DGP_PROFILE_BATCHES_V31_LAUNCH_IMPORT_V1.md',
                                         'scripts/inspect_cctv_dgp_v31_launch_status_v1.py',
                                         'scripts/verify_cctv_dgp_v31_launch_import_v1.py',
                                         'scripts/audit_cctv_dgp_v31_launch_import_v1.py'])
    for name in ['cctv_dgp_v31_launch_status_v1', 'cctv_dgp_v31_launch_status_v1_r1', 'cctv_dgp_v31_launch_import_v1']:
        files.extend(f for f in (ROOT / 'outputs' / name).iterdir() if f.is_file())
    historical_JS = {'static/face_workflow.js': {'sha256': audit['original_JS_sha256'],
                                               'path': 'outputs/dgp_mask_review_race_fix_v1/before/static/face_workflow.js'}}
    m = {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(), 'documents': rows,
         'scope': 'Review-order fix and actual-model functional development flow; neural quality remains unqualified',
         'new_evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in sorted(set(files))},
         'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
         'previous_document_locations': {row['name']: row['before_path'] for row in rows},
         'historical_source_locations': historical_JS, 'original_app_record_sha256': sha(ROOT / 'outputs/dgp_app_v3_integration_record.json'),
         'ordering_regressions': 3, 'real_model_generated_outputs': 11, 'families_functionally_verified': 7,
         'quality_qualification': False, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
         'VM_actions': ['read-only V31 status', 'read-only corrected V31 transfer verification'],
         'V31_launch_import_audit_sha256': sha(ROOT / 'outputs/cctv_dgp_v31_launch_import_v1/independent_launch_audit.json'),
         'V31_original_launch_failures_preserved': 2, 'V31_command_only_import_fix_verified': True,
         'V31_protocol_sha256': 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e',
         'V31_archive_sha256': 'bdf79346b10376b75328dfd2fab3ab3902935982cb9910b4478b401c0c55b9b8',
         'V31_actual_training_started_by_agent': False, 'reserved_final_used': False, 'app_model_promotion': False,
         'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(m, stream, indent=2)
    print(json.dumps({'complete': True, 'milestone_sha256': sha(OUT / 'milestone.json'),
                      'new_bindings': len(m['new_evidence_sha256']), 'documents_updated': 3, 'goal_complete': False}, indent=2))


if __name__ == '__main__': main()
