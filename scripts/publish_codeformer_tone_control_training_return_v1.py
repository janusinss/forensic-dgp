"""Publish actual trained-but-failed return status with exact old document backups."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVIEW = ROOT / 'outputs/codeformer_tone_control_training_v1_return_review'
PUB = REVIEW / 'publication_v1'
CANDIDATE = ROOT / 'outputs/codeformer_tone_control_training_v1_candidate_unqualified'
NAMES = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'PROJECT_HANDOFF.md',
         'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_VM.md',
         'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN.md',
         'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_PREPARATION.md',
         'CCTV_CODEFORMER_NORMAL_TONE_TRAINING_V1_VM.md')
REPORTS = ('CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_RETURN_REVIEW.md',
           'CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN_REVIEW.md')
EVIDENCE = ('return_import_v1.json', 'independent_transport_v1.json',
            'state_v1/state_audit_v1.json', 'state_v1/training_response_review_v1.json',
            'pixels_v1/audit.json', 'root_visual_review_v1.json',
            'independent_candidate_export_v1.json', 'independent_return_report_v1.json')
PREFIX = '''# Actual trained CodeFormer composite exported; quality failed — 17 October 2026

**CURRENT: MODEL CREATED / 50 CONTROLLER UPDATES VERIFIED / NOT APP-QUALIFIED.**
[Actual return review](CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_RETURN_REVIEW.md).
[Trained composite and compatibility notes](outputs/codeformer_tone_control_training_v1_candidate_unqualified/README.md).
[All 35 reviewed comparison sheets](outputs/codeformer_tone_control_training_v1_return_review/pixels_v1/visual_sheets/INDEX.md).

The manual L4 run committed 50 Adam head updates and 200 microbatches. All four
head tensors changed; the 515 restorer tensors remain exactly the earlier
full-coarse fine-tuned CodeFormer. All 519 exported tensor values independently
match the actual stopped checkpoint. The composite is 376,657,585 bytes, SHA-256
`807f92a90960b559e2fa9ee7201ef6a1c00ca50c47c7f86dcdcaca0904f486a0`.
There were zero new CodeFormer updates. This is a credited existing restoration
architecture with a separately trained tone head, not a from-scratch model.

All 8,533 return members, four companions, complete RAW/PNG/base outputs, states,
Adam/RNG/sampler records and 3,090 events passed integrity readback. Root viewed
all 34 DEV sheets / 406 cases and the two-case known TRAIN sheet. Clear and
normal-blur preservation, paired dark retention and combined known TRAIN stress
FAILED. Clear PNG has five worst-case failures and mean regression 0.014620,
above the unchanged mean 0.01 rule. Both dark cohorts worsen against their base.

The head darkens all 406 DEV cases. Actual online TRAIN evidence also shows
unwanted negative shifts on 98/100 dark observations, despite correct targets
and lower aggregate recorded loss. Selective learning was an unproved hypothesis
that failed this run. No fixed-final all-TRAIN evaluation exists to isolate the
unique cause. Scalar tone control cannot repair frozen eye/mouth/geometry defects.
[Evidence-based next design review](CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN_REVIEW.md).
No new runnable packet or training recipe was created from that review.

The genuine inference artifact retains both base_model and head, with exact
features, image-support mask, precision and output policy. It is incompatible
with the current app and standard CodeFormer loaders; base-only fallback would
discard the trained head. Preserve the full source checkpoints and failures.
Do not repeat old commands, append epochs, resume this optimizer, relax gates or
switch the app. No neural training/inference/optimizer ran locally during audit.

The full goal remains useful 256x256 restoration of one cropped frontal/mild-turn
face, preservation of visible appearance and app design, clearer/less-covered
input requests when insufficient, credited untouched CodeFormer benchmark and
retained DGP/checkpoint/research assets. Keep sources, terms, provenance, overlap,
reserved FINAL identities and unpaired native versus paired synthetic evidence
separate. No ethnicity, Zamboanga performance or exact hidden-identity claim.
User acceptance, independent FINAL review and the app Auto/override/PNG flow with
bundled inline Playwright are still required. Seven-family covering detection,
reviewable removal masks and plausible completion stay deferred until restoration
acceptance. Storage offload was already complete and was not repeated. The full
goal remains active and incomplete; model export alone is not completion.

Earlier current-status statements and launch commands below are historical.
Every pre-publication byte is backed up and preserved as the exact suffix.

---

'''


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def new_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def main():
    evidence = {name: read(REVIEW / name) for name in EVIDENCE}
    assert all(row['complete'] is True for row in evidence.values())
    imported = evidence['return_import_v1.json']
    archive, protocol = imported['archive_sha256'], imported['protocol_sha256']
    for name in ('independent_transport_v1.json', 'state_v1/state_audit_v1.json',
                 'state_v1/training_response_review_v1.json', 'pixels_v1/audit.json',
                 'root_visual_review_v1.json'):
        assert evidence[name]['archive_sha256'] == archive and evidence[name]['protocol_sha256'] == protocol
    pixels = evidence['pixels_v1/audit.json']
    assert pixels['audit_pass'] is True and all(pixels[k] is False for k in
        ('preservation_pass', 'dark_retention_pass', 'known_stress_pass', 'worker_overall_quality_requirements_pass'))
    visual = evidence['root_visual_review_v1.json']
    assert visual['personal_visual_review'] is True and len(visual['sheets_reviewed']) == 35
    assert visual['DEV_case_rows_reviewed'] == 406 and visual['known_TRAIN_case_rows_reviewed'] == 2
    provenance = read(CANDIDATE / 'provenance.json')
    weights = CANDIDATE / 'codeformer_tone_head0050_composite_unqualified.pth'
    assert provenance['complete'] is True and provenance['all519exported_tensor_bytes_exact_source'] is True
    assert provenance['model_qualified'] is False and provenance['app_promoted'] is False
    assert provenance['exported_sha256'] == sha(weights) == '807f92a90960b559e2fa9ee7201ef6a1c00ca50c47c7f86dcdcaca0904f486a0'
    assert provenance['exported_bytes'] == weights.stat().st_size == 376657585
    assert provenance['completed_head_updates'] == 50 and provenance['restorer_updates_this_run'] == 0
    prefix = PREFIX.encode('utf-8')
    if not PUB.exists():
        PUB.mkdir()
        documents = {}
        for name in NAMES:
            before = (ROOT / name).read_bytes()
            with (PUB / (name + '.before')).open('xb') as stream: stream.write(before)
            documents[name] = {'before_sha256': hashlib.sha256(before).hexdigest(), 'before_bytes': len(before),
                'after_sha256': hashlib.sha256(prefix + before).hexdigest(), 'preserved_suffix_offset_bytes': len(prefix)}
        with (PUB / 'prefix.txt').open('xb') as stream: stream.write(prefix)
        new_json(PUB / 'plan.json', {'documents': documents, 'prefix_sha256': sha(PUB / 'prefix.txt'),
            'evidence_sha256': {name: sha(REVIEW / name) for name in EVIDENCE},
            'reports_sha256': {name: sha(ROOT / name) for name in REPORTS},
            'candidate_provenance_sha256': sha(CANDIDATE / 'provenance.json'), 'publisher_sha256': sha(__file__)})
    plan = read(PUB / 'plan.json')
    assert plan['publisher_sha256'] == sha(__file__) and (PUB / 'prefix.txt').read_bytes() == prefix
    assert plan['candidate_provenance_sha256'] == sha(CANDIDATE / 'provenance.json')
    for name, expected in plan['evidence_sha256'].items(): assert sha(REVIEW / name) == expected
    for name, expected in plan['reports_sha256'].items(): assert sha(ROOT / name) == expected
    for name, row in plan['documents'].items():
        path, backup = ROOT / name, PUB / (name + '.before')
        assert sha(backup) == row['before_sha256']
        if sha(path) == row['before_sha256']:
            temporary = PUB / (name + '.new')
            if not temporary.exists():
                with temporary.open('xb') as stream:
                    stream.write(prefix + backup.read_bytes()); stream.flush(); os.fsync(stream.fileno())
            assert sha(temporary) == row['after_sha256']; os.replace(temporary, path)
        assert sha(path) == row['after_sha256'] and path.read_bytes()[len(prefix):] == backup.read_bytes()
    if not (PUB / 'receipt.json').exists():
        new_json(PUB / 'receipt.json', {**plan, 'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
            'archive_sha256': archive, 'protocol_sha256': protocol,
            'status': 'ACTUAL50_COMPOSITE_EXPORTED_FAILED_PRESERVATION_AND_DARK_RETENTION',
            'model_created': True, 'model_qualified': False, 'app_promoted': False, 'overall_goal_complete': False,
            'documents_updated': len(NAMES), 'historical_bytes_preserved_exactly': True,
            'local_neural_gradient_optimizer_VM_calls': 0})
    print({'complete': True, 'model_created': True, 'model_qualified': False, 'documents_updated': len(NAMES)})


if __name__ == '__main__': main()
