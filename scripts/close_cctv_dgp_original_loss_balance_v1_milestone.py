"""Publish audited status prefixes with exact historical suffixes and hash guards."""
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cctv_dgp_group_conflicts_v1_contract import sha, read, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_closure'
PREP = ROOT/'outputs/cctv_dgp_group_conflicts_v1_preparation'
PROTOCOL = ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm/protocol.json'
PIN = 'bc0dcbd92b876c53ebe4f484cc5743fad7c8e7cdce6ca8222e04de3d485ec4b2'
CHECKPOINTS = {
    'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth': '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b',
    'checkpoints/dgp_zamboanga_final.pth': 'b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c'}

HANDOFF = '''**DGP diagnostic milestone — 10 October 2026: loss-balance return audited and visually reviewed; group-conflicts diagnostic verified for manual execution.**

The downloaded original-loss-balance return is verified:869,859,156 bytes,
SHA256 a18bf64a1db7abac2aaa7885aa8882463f59415841681d273722fe45433d0e2b.
The independent checker verifies all4,030 regular members,1,000 raw/PNG metric
records, nine saved-vector trial formulas and36 unchanged scientific decisions.
Two hundred fresh CPU inference cases meet the prospectively declared bounds:
raw2.712012e-6, PNG1 byte, embedding3.091991e-7. All60 exact-cell pages and
1,000 unique outputs are visually reviewed as primary-assistant development
evidence; a separate checker verifies1,800 exact256-pixel cells and20 notes.
This is not independent final review. Earlier creation-time pending flags are
closed by the later hash-bound visual_review.json.

All nine trials remain unqualified. The strongest ratio-2 trial has delivered
structure gains1.386497% and1.099287% in the two small photographic TRAIN cohorts,
but13 and12 delivered preservation failure records. Clear delivered MSE worsens
about9.67% and8.73%, with clear SSIM/recognition regressions too. Balanced outputs
remain soft; strong feature-only steps wash out clear detail. Failure records
count group/metric entries, not people or identification mistakes. Raw failures
persist, so PNG processing alone does not explain the limitation. All100 baseline
raw/PNG outputs and six matched decoder portions equal the prior pure-decoder
controls. Mean identity protection does not protect every source/profile output.

V42's full3,905-case0.00692364% structure failure remains binding. These100 TRAIN
cases do not establish useful native CCTV, paired DEV or final performance.
Current and original checkpoints retain their exact hashes; no optimizer,
epoch, trained checkpoint, app change or completion generation is added.

The user selected the evidence-based approach before training. A distinct
group-conflicts diagnostic measures32 source/profile losses with160 bounded
manual-VM gradient queries and zero optimizer updates/epochs. It conditionally
tests three reset displacements only if every sampled group has a certified
local descent direction. No certificate means export without trials, not proof
of global infeasibility. All17 raw/PNG source/profile scientific groups retain
the existing structure, MSE, SSIM, recognition and brightness requirements.
The SSIM derivative is a declared surrogate; finite scientific checks are not
replaced. Aggregate gradients and metadata are exported, but individual query
vectors/autograd are not independently recomputed locally; saved-vector
arithmetic and fresh forward outputs are audited on return.

Manual guide: CCTV_DGP_GROUP_CONFLICTS_V1_VM.md.
Design: CCTV_DGP_GROUP_CONFLICTS_V1_DESIGN.md.
Reviewed result: CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_RESULTS.md.
Packet audit: outputs/cctv_dgp_group_conflicts_v1_preparation/independent_packet_audit.json.
Protocol: bc0dcbd92b876c53ebe4f484cc5743fad7c8e7cdce6ca8222e04de3d485ec4b2.
Execution archive:220,267,830 bytes; SHA256
cfe829808d9280744d45633bcdfb9bcdd0dbc6a70549d6f86d0a73acf2f96ff5.
Independent preparation verifies174 archive members,39 source bindings,
19 Python3.10 AST files, four mathematical fixtures,100 cached loss checks and
four initial CPU parity cases. All four local worker modes and three local
candidate derivative/trial guards refuse execution. States/buffers stay exact.
The first loss target-rounding parity failure is preserved; the corrected check
matches the retained target definition without changing scientific thresholds.

This new packet is prepared and verified, not run. Require4GiB free after
installation; estimated diagnostic8–20 minutes and export1–5 minutes. Worker1500s,
external1530s plus30s grace, cache120s, gradients480s, solver120s, trials300s,
export300s/external330s plus30s grace,20GiB allocated VRAM,1.5GiB uncompressed
return and512MiB reserve are enforced. There is no bundled cleanup or follow-on
training. Launch manually inside tmux and download one remote source per gcloud
command. No VM connection or actual training occurs during preparation/closure.

The native QMUL comparison remains unpaired development evidence: no convincing
useful current-DGP gain on its six predeclared core cases; seven insufficient
cases remain insufficient and32 reserved final pixels remain unviewed. Source
labels imply neither ethnicity nor Zamboanga performance. A subsequent epoch
recipe still requires full-corpus capacity, paired DEV preservation, useful
native DEV outputs and independent final review. No failed recipe is resumed.
All five restoration milestones, seven automatic/assisted completion families
and bundled inline Playwright app verification remain required. The full goal
is active/incomplete. This status supersedes pending-return statements below
while retaining their exact bytes. Before/status hashes are preserved in
outputs/cctv_dgp_original_loss_balance_v1_closure/.


'''

SYSTEM = '''**Latest diagnostic status — 10 October 2026: loss-balance return reviewed; group-conflicts diagnostic verified, not launched.**

All nine reviewed loss-balance trials fail preservation. Some exceed1% structure
gain on two small photographic TRAIN cohorts, but clear appearance regresses and
outputs remain soft. All1,000 unique outputs are visually reviewed. The mean
recognition-protection assumption failed; further epochs under that recipe are
not justified. V42's full-corpus failure and all original preservation gates remain.

The user requests the best evidence-based approach before training. A separate
manual diagnostic measures32 source/profile objectives before any optimizer.
There are160 finite VM-only gradient queries, zero optimizer updates and zero
epochs. Only a certified improving direction permits three reset trials. A
failed certificate completes the diagnostic without training and proves no
global infeasibility. Independent transfer verification passes; the packet is
prepared and verified, not run. See CCTV_DGP_GROUP_CONFLICTS_V1_VM.md and
CCTV_DGP_GROUP_CONFLICTS_V1_DESIGN.md for exact commands and finite stop rules.
The declared SSIM surrogate does not replace scientific evaluation; saved group
vectors are audited without locally replaying individual autograd queries.

Current/Phase3 checkpoints, splits, research caches and failed gates are preserved.
No failed pilot is resumed and no automatic training or app promotion occurs.
All five milestones, native DEV usefulness, independent final identities, seven
automatic/assisted covering families and bundled inline Playwright remain required.
The full goal is active/incomplete. This status supersedes historical pending
statements below without changing the goal or manual execution boundary.
Exact preceding bytes: outputs/cctv_dgp_original_loss_balance_v1_closure/before/.


'''

PRACTICAL = '''**Latest diagnostic status — 10 October 2026: loss-balance evidence reviewed; agreed restoration and covering scope remains incomplete.**

All1,000 loss-balance outputs were reviewed; no trial meets preservation. Small
TRAIN structure gains do not establish clear facial detail, useful native CCTV,
independent final performance or successful completion. The app design and
retained DGP weights remain intact. See CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_RESULTS.md.

The next verified manual packet measures source/profile loss conflicts before
training: CCTV_DGP_GROUP_CONFLICTS_V1_VM.md. It has160 gradient queries, zero
optimizer updates and zero epochs, with conditional reset trials and unchanged
scientific gates. It is prepared, not run. No failed recipe is resumed.

All seven covering families, automatic removal preview and optional correction,
preservation of visible appearance/clear glasses/non-obstructing hair, clearer or
less-covered requests, original/mask/result, PNG/bundle downloads and separate
automatic/assisted reviews remain required. Independent final review and bundled
inline Playwright app flow remain required. The full goal is active/incomplete.
Read the latest PROJECT_HANDOFF.md for precise evidence and execution boundaries.
Exact preceding bytes: outputs/cctv_dgp_original_loss_balance_v1_closure/before/.


'''


def bindings(p):
    assert sha(PROTOCOL) == PIN
    for name, digest in p['local_sources_sha256'].items(): assert sha(ROOT/name) == digest, name
    for name, digest in CHECKPOINTS.items(): assert sha(ROOT/name) == digest, name


def main():
    started = time.monotonic(); assert not OUT.exists()
    p = read(PROTOCOL); bindings(p)
    audited_packet = read(PREP/'independent_packet_audit.json'); prepared = read(PREP/'prepared.json')
    assert audited_packet['complete'] and audited_packet['protocol_sha256'] == PIN
    assert audited_packet['archive_sha256'] == prepared['archive_sha256']
    assert audited_packet['archive_members_verified'] == 174 and audited_packet['source_bindings_verified'] == 39
    assert audited_packet['states_before'] == audited_packet['states_after']
    for key in ['local_gradient_queries', 'local_optimizer_updates', 'local_trial_assignments', 'VM_connections']:
        assert audited_packet[key] == 0
    assert not audited_packet['VM_diagnostic_launched']
    review_path = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1/visual_review.json'
    review = read(review_path); assert review['complete'] and review['unique_model_outputs_reviewed'] == 1000
    assert review['all60_pages_viewed_at_original_resolution'] and not review['independent_final_reviewer']
    result_path = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_vm_return/outputs/results.json'
    result = read(result_path)
    decisions = [stage for trial in result['trial_summaries'] for stages in trial['comparisons'].values() for stage in stages.values()]
    assert len(decisions) == 36 and all(not stage['pass'] for stage in decisions)
    docs = {'PROJECT_HANDOFF.md': HANDOFF, 'SYSTEM_WORKFLOW_AND_GOAL.md': SYSTEM, 'PRACTICAL_OUTPUT_SCOPE.md': PRACTICAL}
    assert not set(docs).intersection(p['local_sources_sha256'])
    originals = {name: (ROOT/name).read_bytes() for name in docs}
    OUT.mkdir(); (OUT/'before').mkdir(); published = {}
    for name, prefix in docs.items():
        (OUT/'before'/name).write_bytes(originals[name])
        path = ROOT/name; assert path.read_bytes() == originals[name]
        prefix_bytes = prefix.encode('utf-8'); path.write_bytes(prefix_bytes+originals[name])
        assert path.read_bytes() == prefix_bytes+(OUT/'before'/name).read_bytes()
        published[name] = {'before_sha256': sha(OUT/'before'/name), 'published_sha256': sha(path),
            'prefix_bytes': len(prefix_bytes), 'original_suffix_exact': True}
    bindings(p)
    write(OUT/'closure.json', {'complete': True, 'return_review_complete': True,
        'return_independent_audit_sha256': sha(ROOT/'outputs/cctv_dgp_original_loss_balance_v1_independent_audit.json'),
        'return_results_sha256': sha(result_path), 'visual_review_sha256': sha(review_path),
        'reviewed_unique_model_outputs': 1000, 'all36_scientific_trial_decisions_failed': True,
        'next_manual_protocol_sha256': PIN, 'next_execution_archive_sha256': prepared['archive_sha256'],
        'next_execution_archive_bytes': prepared['archive_bytes'],
        'next_independent_packet_audit_sha256': sha(PREP/'independent_packet_audit.json'),
        'manual_guide_sha256': sha(ROOT/'CCTV_DGP_GROUP_CONFLICTS_V1_VM.md'),
        'checkpoint_sha256': CHECKPOINTS, 'status_documents': published,
        'closure_script_sha256': sha(Path(__file__)), 'closure_neural_calls': 0,
        'closure_gradient_queries': 0, 'closure_optimizer_updates': 0, 'VM_connections': 0,
        'VM_diagnostic_launched': False, 'app_changed': False, 'native_or_final_used_in_diagnostic': False,
        'model_qualification': False, 'goal_complete': False, 'seconds': time.monotonic()-started})
    print({'complete': True, 'status_documents': 3, 'goal_complete': False})


if __name__ == '__main__': main()
