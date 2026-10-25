"""Publish reviewed manual readiness, preserving every previous document byte."""
import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_u64_next_design_v1'
PUB = PREP / 'publication_v1'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
        'CCTV_CODEFORMER_U64_CLEANANCHORS_CONTINUATION_V1_VM.md')
PREFIX = '''# 23 October 2026: original-u64 diagnostic ready for a manual VM run

[Five-step upload/install/tmux/download guide](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_VM.md).
The selected diagnostic starts from exact original trained u64 and measures
separate loss/fidelity gradients on16 admitted optimization photographs under
four matched roles. Maximum72 forwards and192 gradient queries; zero optimizers,
updates or parameter applications. All575 model values must stay bitwise exact.
The completed parent's source/data/weights are read-only dependencies; its
failed update8 training is never resumed. No new model has qualified.

The source-only archive is238,801bytes, SHA256
`457c324448c6e6baefdcde7797722e6ad85a8d81ba467692bbe9d4a934003994`;
protocol SHA256 `07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3`.
Root's45 pure tests pass with neural imports blocked and stable source pins.
Independent archive review passes592 checks; guide review passes29, no blockers.
All469 parent bindings were rehashed locally; two model-only aliases are
explicitly local evidence. Actual VM admission requires all469 installed paths.
Staged help returned0; no actual worker/model/gradient, optimizer, training,
upload or VM connection occurred during preparation.

Worker estimate3–8 minutes is unmeasured; limits900s internally/960 externally
plus20s cleanup, phases180s; export180/210s separately. Require8GiB free initially
and retain4GiB. RAM10GiB, reserved CUDA18GiB, output/archive2GiB each. Eighteen
planned exact-RGB sheets cover72 output slots once; partial/unattempted outputs
stay explicit. Independently audit returned states, journal, gradients, objectives
and RAW/PNG before choosing another training change. Hypothetical direction
arithmetic is local RAW evidence, not a finite-step or PNG quality guarantee.
No automatic projection, reweighting, LR selection or training follows.

STATUS: PREPARED_UNRUN_MANUAL_U64_ROLE_GRADIENTS_V1. Original u64 stays a retained
challenger; U96 stays experimental app incumbent. All prior quality stops and
broader DEV failures remain binding. Automatic goal advancement requires an
independently verified overall winner, preservation and agreed development/user
acceptance; this diagnostic cannot trigger it. Preserve checkpoints, states,
splits, failures and predecessors. Reviewed Protect/Apply/bypass stays separate;
automatic blur matching and later completion development remain deferred.
Native/wild remain unpaired; FINAL is unopened/execution-unfrozen with independent
reviewers pending. Full goal ACTIVE/incomplete. Exact prior bytes follow.

---

'''
OLD_GUIDE_PREFIX = '''# Executed and stopped: do not run this historical training guide again

The cleananchors continuation completed eight new updates and failed preservation.
Its retained quality stop is binding; do not resume, add updates or select an
earlier stopped state. For the new zero-update diagnostic, use the
[current manual guide](CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_VM.md).
The preparation-time guide below is preserved as historical evidence.

---

'''


def main():
    spec = importlib.util.spec_from_file_location('preparation_builder', ROOT / 'scripts/build_u64_role_gradients_v1.py')
    b = importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
    need = b.need
    need(not PUB.exists() or not list(PUB.iterdir()), 'Only a fresh empty publication directory is permitted')
    PUB.mkdir(exist_ok=True)
    sealed = b.doc(PREP / 'build_receipt_v1.json')
    tests = b.doc(PREP / 'source_tests_v1.json')
    need(tests['all_tests_pass'] and tests['tests_run'] == 45, 'Exact final pure tests required')
    need(b.pin(ROOT / sealed['archive']['path']) == sealed['archive'], 'Sealed archive changed')
    need(b.pin(ROOT / sealed['checksum']['path']) == sealed['checksum'], 'Checksum changed')
    stage = ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm'
    need(b.pin(stage / 'protocol.json', stage) == sealed['protocol'], 'Protocol changed')
    ar = b.doc(PREP / 'independent_archive_review_v1.json')
    gr = b.doc(PREP / 'independent_guide_review_v1.json')
    need(ar['complete'] and ar['archive_review_pass'] and ar['unresolved_blockers'] == [], 'Independent archive approval required')
    need(gr['complete'] and gr['guide_source_review_pass'] and
         gr['manual_guide_ready_at_exact_consumed_pins'] and gr['unresolved_blockers'] == [], 'Both independent guide approvals required')
    guide = b.pin(ROOT / 'CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_VM.md')
    need(guide in gr['consumed_files'], 'Guide changed after independent review')
    ready = {'complete': True, 'status': 'PREPARED_UNRUN_MANUAL_U64_ROLE_GRADIENTS_V1',
        'archive': sealed['archive'], 'checksum': sealed['checksum'], 'protocol': sealed['protocol'], 'guide': guide,
        'independent_archive_review': b.pin(PREP / 'independent_archive_review_v1.json'),
        'independent_guide_review': b.pin(PREP / 'independent_guide_review_v1.json'),
        'source_tests': b.pin(PREP / 'source_tests_v1.json'),
        'planned_visual_review': b.pin(PREP / 'planned_visual_review_v1.json'),
        'staged_help': b.pin(PREP / 'staged_help_readback_v1.json'),
        'maximum_forwards': 72, 'maximum_gradient_queries': 192, 'optimizer_updates': 0,
        'planned_sheets': 18, 'all575_values_planned_unchanged': True,
        'actual_neural_gradient_optimizer_training_VM_calls': 0, 'manual_run_required': True,
        'model_qualified': False, 'automatic_goal_winner_advancement_triggered': False,
        'original_u64_retained': True, 'U96_app_incumbent_unchanged': True,
        'previous_failures_remain_binding': True, 'full_goal_status': 'ACTIVE_INCOMPLETE'}
    b.write_new(PREP / 'ready_for_manual_run_v1.json', ready)
    records = []
    for name in DOCS:
        target = ROOT / name
        need(target.parent == ROOT and target.is_file(), 'Exact named workspace document required')
        before_pin = b.pin(target); before = target.read_bytes()
        backup = PUB / (name + '.before')
        with backup.open('xb') as stream:
            stream.write(before); stream.flush(); os.fsync(stream.fileno())
        need(backup.read_bytes() == before, 'Complete prior bytes must be backed up first')
        head = (OLD_GUIDE_PREFIX if name == DOCS[-1] else PREFIX).encode('utf-8')
        pending = target.with_name(target.name + '.u64_role_publication.pending')
        with pending.open('xb') as stream:
            stream.write(head + before); stream.flush(); os.fsync(stream.fileno())
        need(pending.read_bytes() == head + before, 'Pending publication readback required')
        os.replace(pending, target)
        need(target.read_bytes()[len(head):] == backup.read_bytes(), 'All prior document bytes must remain exact')
        records.append({'path': name, 'before': before_pin, 'backup': b.pin(backup), 'after': b.pin(target),
                        'preserved_suffix_offset': len(head), 'all_prior_bytes_preserved': True})
    b.write_new(PUB / 'publication_receipt_v1.json', {'complete': True,
        'status': 'PUBLISHED_READY_MANUAL_RUN_UNLAUNCHED', 'readiness': b.pin(PREP / 'ready_for_manual_run_v1.json'),
        'documents': records, 'all5_prior_documents_preserved_exactly': True,
        'actual_neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False,
        'full_goal_status': 'ACTIVE_INCOMPLETE'})
    print({'documents': len(records), 'manual_run_launched': False, 'full_goal_status': 'ACTIVE_INCOMPLETE'})


if __name__ == '__main__':
    main()
