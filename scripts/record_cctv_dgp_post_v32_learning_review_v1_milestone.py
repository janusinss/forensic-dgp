"""Record independently audited losses, primary research and an unlaunched VM diagnostic."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_post_v32_learning_review_v1_milestone'
PREVIOUS = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_milestone_v1'
LOSSES = ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1_r1'
PACKET = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm'
PREP = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_preparation'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    assert not OUT.exists()
    old = read(PREVIOUS / 'milestone.json')
    for name, digest in old['new_evidence_sha256'].items(): assert sha(ROOT / name) == digest, name
    loss, loss_audit = map(read, [LOSSES / 'analysis.json', LOSSES / 'independent_audit.json'])
    packet, preparation, packet_audit = map(read, [PACKET / 'protocol.json', PREP / 'preparation.json', PREP / 'independent_packet_audit.json'])
    assert loss['complete'] and loss_audit['complete'] and loss_audit['analysis_sha256'] == sha(LOSSES / 'analysis.json')
    assert packet_audit['complete'] and packet_audit['protocol_sha256'] == sha(PACKET / 'protocol.json') == preparation['protocol_sha256']
    assert packet_audit['adversarial_boundary_regressions_passed'] == 6 and packet_audit['read_only_local_Bash_syntax_pass']
    assert not packet_audit['VM_execution_started'] and not preparation['actual_VM_run_started']
    assert not (PACKET / 'outputs').exists() and packet['optimizer_updates'] == packet['epochs'] == 0
    addition = '''
**Latest research milestone - 7 October 2026: stopped-V32 raw-loss review audited; distinct zero-update diagnostic ready.**

The user's "apply the best approach and do research also if needed" answer satisfies
the required design discussion. The selected direction diagnoses current learning
first. Primary research is recorded with explicit limits; it does not establish
this model's failure cause or justify a blind frequency-loss/learning-rate change.

All50 frozen preflight/normalization TRAIN previews have an independently verified
raw-loss review. None was optimized by50; they are not unseen evaluation. The40
degraded previews improve raw feature error0.968684% and PNG error0.991558%.
The full3,905-case PNG stop remains0.970672% against the unchanged1% requirement.
Rounding does not explain the weak preview gain. Eyes/nose/mouth and remaining
observed interior show uneven small improvements, while useful added whole-face
clarity remains unestablished by the earlier full50 visual review.

Per-case protection is active:8 degraded and1 clear preview trigger raw identity
penalties, and1 clear control has3.802615% higher raw pixel error. All10 clear
anchors activate. Mean total raw objective on this fixed50 cohort changes from
1.2999999802 to1.3002432080: added preservation value0.0124378591 slightly exceeds
the restoration-term reduction0.0121946314. Values alone do not prove gradient
dominance or reconstruct unsaved AdamW moments. Passing17 delivered group-average
gates does not erase these per-case findings. No preservation term or gate is removed.

The45.94s local analysis makes100 frozen recognizer image calls and zero DGP,
gradient, backward or optimizer calls. The separate40.53s audit verifies387 bindings,
50 cases,400 regional/filter quantities and100 raw recognizer images. The first
schema lookup error and its plan remain intact; separate R1 review corrects only
the reference field and evidence routing. No original model or loss code changes.

A separate stopped-V32 loss-gradient packet is ready for the human's manual L4
launch. It measures the original and stopped states on two50-case TRAIN cohorts,
five references per photographic source with all five profiles. Exposed references
are the first five touched per source in the frozen schedule. The other cohort
uses all50 fixed previews, source/profile matched and unoptimized by50. Selection
uses existing metadata and the existing fixed set, not output rankings. Both
fusion and decoder actually changed at this endpoint, unlike the earlier V31
diagnostic. The next design depends on returned component conflict, directional
derivatives, partition magnitudes and batch coherence; no recipe is preselected.

The54,322-byte,3-file packet is independently verified against140 TRAIN files and
518 V32 dependencies. Six adversarial boundaries, Python3.10 syntax, pre-neural
Windows differentiation rejection and actual read-only Bash syntax pass. The
sandbox-denied Bash signal-pipe attempt, successful outside-sandbox parse and R1
readback are all retained; the packet and protocol are unchanged by that correction.
Protocol SHA256:61c04aef4ee184adc9830482477bf37ad9c936a8c0247e44d26eefc1a6bf483d.
Archive SHA256:22d76f3637a3ac580bc18fbac79ff5e3d24a61ec0a00ce082fde78c7069c6052.

The diagnostic permits280 gradient queries,0 optimizer updates,0 epochs and no
new checkpoint. Require an idle existing L4/g2-standard-4 with6GiB free. Worker600s,
external900s plus30s grace, export300s/external330s plus30s grace, VRAM20GiB and
uncompressed return1.5GiB are enforced. Estimate1-5 minutes plus1-3 minutes export.
No automatic historical/follow-on launch, cleanup, upload or VM run occurs here.
The guide has exact gcloud upload, tmux launch and three separate PuTTY downloads.
Returned data require independent audit before a justified new finite training pilot.

Source names are not ethnicity. This is paired synthetic photographic TRAIN
evidence, not native CCTV or Zamboanga performance. Native development/reserved
pixels remain unopened in this milestone. The own-DGP app checkpoint, selector,
256 processing and design remain unchanged. Original checkpoints, splits,
research caches, every failure and the actual Windows backup receipt remain.
The converted-MAT comparison remains unqualified; all seven completion families
still need separate automatic/assisted quality, useful native restoration and
independent final review. Goal active/incomplete; actual training remains manual.

[Audited learning and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V32_LEARNING_REVIEW_V1.md>)
[Five manual VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V32_LOSS_GRADIENT_V1_VM.md>)

The complete previous handoff is retained below. Its cause-under-investigation
wording remains valid; this milestone adds measurements and an unlaunched diagnostic.

'''
    OUT.mkdir(); (OUT / 'before_docs').mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'
    before = handoff.read_bytes()
    before_path = OUT / 'before_docs/PROJECT_HANDOFF.md'
    with before_path.open('xb') as stream: stream.write(before)
    split = before.index(b'\n') + 1
    handoff.write_bytes(before[:split] + addition.encode('utf-8') + before[split:])
    files = [Path(__file__), ROOT / 'scripts/verify_cctv_dgp_post_v32_learning_review_v1_milestone.py',
             ROOT / 'CCTV_DGP_POST_V32_LEARNING_REVIEW_V1.md', ROOT / 'CCTV_DGP_V32_LOSS_GRADIENT_V1_VM.md',
             ROOT / 'scripts/review_cctv_dgp_post_v32_saved_losses_v1.py', ROOT / 'scripts/review_cctv_dgp_post_v32_saved_losses_v1_r1.py',
             ROOT / 'scripts/audit_cctv_dgp_post_v32_saved_losses_v1.py', ROOT / 'scripts/prepare_cctv_dgp_v32_loss_gradient_v1.py',
             ROOT / 'scripts/verify_cctv_dgp_v32_loss_gradient_v1_packet.py', ROOT / 'scripts/verify_cctv_dgp_v32_loss_gradient_v1_packet_audit_r1.py',
             ROOT / 'scripts/cctv_dgp_v32_loss_gradient_v1_vm.py', ROOT / 'scripts/cctv_dgp_v32_loss_gradient_v1_return_audit_template.py',
             ROOT / 'scripts/audit_cctv_dgp_v32_loss_gradient_v1_return.py', ROOT / 'outputs/cctv-dgp-v32-loss-gradient-v1-execution.tar.gz',
             ROOT / 'outputs/cctv-dgp-v32-loss-gradient-v1-execution.tar.gz.sha256', handoff, before_path,
             PREVIOUS / 'milestone.json', PREVIOUS / 'independent_closure_audit.json', PREVIOUS / 'final_readback.json']
    for folder in [LOSSES, ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1', ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1', PACKET, PREP]:
        files.extend(path for path in sorted(folder.rglob('*')) if path.is_file())
    m = {'complete': True, 'datetime_UTC': datetime.now(timezone.utc).isoformat(), 'recorder_sha256': sha(Path(__file__)),
         'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
         'new_evidence_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in files},
         'previous_document_locations': {'PROJECT_HANDOFF.md': before_path.relative_to(ROOT).as_posix()},
         'document': {'name': handoff.name, 'before_path': before_path.relative_to(ROOT).as_posix(),
                      'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': sha(handoff), 'addition_bytes': len(addition.encode('utf-8'))},
         'user_design_discussion_satisfied': True, 'loss_review_independently_audited': True, 'primary_research_used_with_limits': True,
         'fixed50_TRAIN_previews_reviewed': True, 'source_balanced100_case_gradient_packet_verified': True,
         'diagnostic_gradient_query_cap': 280, 'diagnostic_optimizer_update_cap': 0, 'VM_execution_started': False,
         'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'native_or_reserved_used': False,
         'original_one_percent_gate_failure_preserved': True, 'preservation_terms_and_gates_removed': False,
         'app_changes': False, 'app_promotion': False, 'independent_final_review': False, 'manual_VM_training_required': True,
         'training_started_by_agent': False, 'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(m, stream, indent=2)
    print(json.dumps({'complete': True, 'new_bindings': len(m['new_evidence_sha256']), 'milestone_sha256': sha(OUT / 'milestone.json'), 'VM_execution_started': False}, indent=2))


if __name__ == '__main__':
    main()
