"""Publish corrected manual receipt commands while preserving all prior evidence."""
from datetime import datetime, timezone

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, ANALYSIS, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new, finish_prepend


def main():
    stage = OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_publication'
    assert not (stage / 'publication.json').exists()
    previous = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_publication/publication.json')
    prepared = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_preparation.json')
    transfer = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_transfer_audit.json')
    regression = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_transfer_receipt_green.json')
    syntax = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_r1_bash_syntax.json')
    assert all(v['complete'] for v in [previous, prepared, transfer, regression, syntax])
    packet = OUT / prepared['name']
    pin = sha(packet / 'protocol.json')
    assert pin == prepared['protocol_sha256'] == transfer['protocol_sha256']
    assert sha(OUT / 'cctv-dgp-raw-conditioning-learning-v1-r1-execution.tar.gz') == prepared['archive_sha256'] == transfer['archive_sha256']
    assert transfer['archive_files'] == 17 and transfer['parent_assets_verified'] == 5852
    assert transfer['hash_only_transfer_preflight']['maximum_gradient_queries'] == 640
    assert syntax['exit_code'] == 0 and syntax['checked_shell_sha256'] == sha(packet / 'scripts/run_learning.sh')
    assert not (packet / 'outputs').exists() and not (packet / 'review.log').exists()
    old_prepared = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_preparation.json')
    old_guide = 'CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_VM.md'
    old_report = 'CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_PREPARATION.md'
    guide_name = 'CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_R1_VM.md'
    report_name = 'CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_R1_PREPARATION.md'
    assert sha(ROOT / old_guide) == previous['new_documents_sha256'][old_guide]
    assert sha(ROOT / old_report) == previous['new_documents_sha256'][old_report]
    def updated_text(name):
        text = (ROOT / name).read_text(encoding='utf-8')
        text = text.replace('cctv-dgp-raw-conditioning-learning-v1', 'cctv-dgp-raw-conditioning-learning-v1-r1')
        text = text.replace('raw_conditioning_learning_v1', 'raw_conditioning_learning_v1_r1')
        text = text.replace('CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_', 'CCTV_DGP_RAW_CONDITIONING_LEARNING_V1_R1_')
        text = text.replace(old_prepared['protocol_sha256'], pin).replace(old_prepared['archive_sha256'], prepared['archive_sha256'])
        text = text.replace(f"{old_prepared['archive_bytes']:,}", f"{prepared['archive_bytes']:,}")
        return text
    guide = updated_text(old_guide).replace('# RAW conditioning learning V1 —', '# RAW conditioning learning V1 R1 —')
    note = '''The final preparation check found an inherited transfer-summary count of
1,280 queries in V1 although its actual enforced limit is 640. R1 reads the
summary directly from the sealed protocol. Its fresh receipt agrees on 640
queries, 2 accepted updates, 100 cases and zero neural calls. Namespace and this
hash-only summary are the only changes; the training recipe and actual gates
remain fixed. Use these R1 commands. The earlier packet is preserved.

'''
    guide = guide.replace('1. Upload from **Windows Google Cloud SDK Shell**:', note + '1. Upload from **Windows Google Cloud SDK Shell**:')
    report = updated_text(old_report).replace('# RAW conditioning V1 preparation —', '# RAW conditioning V1 R1 preparation —')
    report = report.replace('checked all 16 archive files', 'checked all 17 archive files')
    report = report.replace('##', '##', 1)
    report = report.replace('**The optimizer comparison is verified and UNRUN.**',
        '**The optimizer comparison is verified and UNRUN.** The final receipt check\n'
        'caught V1’s inherited printed 1,280-query maximum; its enforced cap already\n'
        'was 640. The narrow R1 correction reads the summary directly from protocol.\n'
        'Fresh hash-only verification now agrees on every declared count. The used\n'
        'R4 and prepared V1 packet/checkers/archives/stopped regressions remain intact.\n'
        'No prior human VM execution is inferred from local files.\n\n')
    report += '\n' + link('Retained receipt mismatch and verified correction',
        'outputs/cctv_dgp_raw_conditioning_learning_v1_r1_transfer_receipt_green.json') + '.\n'
    retained_new(ROOT / guide_name, guide)
    retained_new(ROOT / report_name, report)
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
            'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md', 'CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    expected = {n: previous['documents'][n]['after_sha256'] for n in docs}
    expected.update({n: previous['new_documents_sha256'][n] for n in [old_guide, old_report]})
    prefix = f'''# Current manual RAW conditioning packet — R1 receipt correction, 14 October 2026

Final preparation caught an inherited printed1,280-query maximum in V1's
hash-only transfer receipt; its enforced protocol/worker cap was already640.
Fresh R1 reads receipt fields directly from protocol and passes exact count
agreement. Namespace and receipt metadata are the only changes. V1 packet,
checkers, archives and failed receipt regression are preserved; prior human
execution is unknown. Use the corrected R1 commands:
{link('Current manual upload, tmux and single-source downloads', guide_name)}.
{link('Preparation, unchanged recipe and resource limits', report_name)}.

R1 is verified and UNRUN. Same100 paired TRAIN cases,12 conditioning tensors,
initial full state,32 RAW optimizer groups and17 RAW/17 PNG actual acceptance
groups. At most2 accepted updates/3 trials each,640 derivative queries,
180 candidate/320 recognizer forwards. All original preservation/appearance,
brightness<=0.2,cumulative0.1% parameter distance and1% global/per-source RAW/PNG
endpoint gates remain. Require9GiB free; finite timing/VRAM/RSS/storage/output
and export stops remain.17 archive bindings,5,852 parent bindings,100 baseline
cases,synthetic solver/precision/pixel checks and retained rejected states pass.

R4's all-three-trial preservation failure and zero accepted updates remain
binding. No accepted incumbent,more epochs or useful new image is established
by preparation. Original model/app design,checkpoints,splits,caches,provenance,
full states and failures are retained. Native unpaired/paired TRAIN/final roles,
all five milestones and all seven automatic/assisted covering families remain
required. Actual training stays human-manual L4 tmux; no VM launch or local
neural model training occurred. Goal active/incomplete. Exact history follows.

'''
    banner = f'''# Superseded prepared V1 commands — use corrected R1

The enforced query limit was640; this packet's hash-only summary printed1,280.
R1 corrects receipt metadata and namespace only. Prior human execution is unknown;
the original packet and exact guide are retained. The recipe and gates remain.
{link('Current verified R1 manual commands', guide_name)}.
The previous document follows as historical evidence.

'''
    (stage / 'before').mkdir(parents=True, exist_ok=True)
    changed = {}
    for name in docs:
        changed[name] = finish_prepend(ROOT / name, prefix, stage / 'before' / name, expected[name])
    for name in [old_guide, old_report]:
        changed[name] = finish_prepend(ROOT / name, banner, stage / 'before' / name, expected[name])
    evidence_names = ['preparation', 'checker_preparation', 'transfer_audit', 'transfer_receipt_green',
                      'pixel_fixture_green', 'bash_syntax']
    evidence = {('outputs/cctv_dgp_raw_conditioning_learning_v1_r1_' + name + '.json'):
                sha(OUT / ('cctv_dgp_raw_conditioning_learning_v1_r1_' + name + '.json')) for name in evidence_names}
    evidence['outputs/cctv_dgp_raw_conditioning_learning_v1_transfer_receipt_red.json'] = sha(
        OUT / 'cctv_dgp_raw_conditioning_learning_v1_transfer_receipt_red.json')
    evidence['outputs/cctv_dgp_conditioning_learning_v1_r4_analysis/publication/publication_audit.json'] = sha(
        ANALYSIS / 'publication/publication_audit.json')
    write(stage / 'publication.json', dict(
        complete=True, UTC=datetime.now(timezone.utc).isoformat(), publisher_sha256=sha(__file__), documents=changed,
        new_documents_sha256={n:sha(ROOT / n) for n in [guide_name,report_name]}, evidence_sha256=evidence,
        protocol_sha256=pin, archive_sha256=prepared['archive_sha256'], archive_bytes=prepared['archive_bytes'],
        initial_V1_receipt_mismatch_and_all_old_files_preserved=True,
        old_human_execution_unknown=True, recipe_and_all_quality_stops_unchanged=True,
        manual_training_unrun=True, original_app_and_research_unchanged=True,
        model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, current_manual_guide=guide_name, history_documents=len(changed),
               archive_bytes=prepared['archive_bytes'], training_unrun=True))


if __name__ == '__main__':
    main()
