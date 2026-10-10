"""Bind the completed return review and pending manual diagnostic to project status."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_original_feature_probe_v1_closure'
PREPARATION = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_preparation'
PROTOCOL = ROOT / 'outputs/cctv_dgp_original_loss_balance_v1_vm/protocol.json'
PROTOCOL_SHA = '8e28ab1de4873bca2aff806a21ae173bc039c9463d5b2acf04848b031791d2d2'
CHECKPOINTS = {
    'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth':
        '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b',
    'checkpoints/dgp_zamboanga_final.pth':
        'b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c',
}

HANDOFF = '''**Original-DGP diagnostic milestone — 9 October 2026: return audited and all outputs reviewed; a distinct loss-balance packet is verified for manual execution.**

The downloaded original-feature return is present and verified: 1,051,691,000
bytes, SHA256 dbba13dac96fe0bcdf58c03083f6bdfc1e939383b3996b6039ba5c8e93b0f290.
The independent R2 checker verifies all 4,042 regular members, 1,000 raw/PNG
metric records, 30 saved derivative vectors, nine reset displacements and all
36 unchanged scientific decisions. Two hundred fresh CPU outputs meet the
prospective replay tolerances. R1's derived brightness-ratio arithmetic failure
and exact failure log remain preserved; R2 bounds arithmetic propagation only
and changes no scientific threshold or trial decision.

All 60 exact-cell pages and 1,000 unique model outputs are visually reviewed by
the primary assistant as development evidence. The strongest original-decoder
trial reaches 1.34283% and 1.17117% delivered structure gain in the two small
photographic TRAIN cohorts, but has 23 and 18 delivered preservation failure
records. All nine trials are unqualified. Decoder outputs remain soft without
convincing useful clarity; stronger feature/joint steps wash out facial detail.
These 100 cases are not V42's full 3,905-case corpus, native CCTV, DEV or final
identities. V42's 0.00692364% failure remains binding. No quality fix is claimed.

Saved derivatives show that original-decoder structure descent increases mean
identity loss, whereas original-feature identity descent provides a protective
direction. A distinct nine-trial loss-balance diagnostic tests coordination of
these disjoint partitions. First-order mean predictions do not guarantee finite
image or source/profile preservation. The same 1% structure, raw/PNG appearance,
source/profile and brightness requirements apply, with every failure retained.
No failed pilot or epoch trajectory is resumed.

Manual guide: CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_VM.md.
Design: CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_DESIGN.md.
Return report: CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_RESULTS.md.
Packet audit: outputs/cctv_dgp_original_loss_balance_v1_preparation/independent_packet_audit.json.
Protocol: 8e28ab1de4873bca2aff806a21ae173bc039c9463d5b2acf04848b031791d2d2.
Execution archive: 245,615,832 bytes; SHA256
da08ec66d881495efbcfbfbd794ab179df8a1fbff69342b96b3f1e4dd582bede.
Independent preparation verifies 170 archive members, 23 local source bindings,
17 Python 3.10 AST files, all nine saved-vector formulas, both denied local
trial guards, and 100 initial CPU parity cases with unchanged states/buffers.

The new diagnostic reuses the audited derivatives: zero new gradient queries,
zero optimizer updates and zero epochs. It is prepared and verified, not run.
Require 4 GiB free after installation. Estimated diagnostic 5–10 minutes and
export 1–5 minutes; worker/trial/cache/export/storage/VRAM limits are enforced.
All launch commands are manual inside tmux; downloads use one remote source
per gcloud command. No VM connection, cleanup, actual training, app change,
native/final exposure or completion generation occurs during this closure.
Current and original checkpoints retain their exact SHA256 hashes.

QMUL's completed current-checkpoint comparison remains unpaired development
evidence, with no convincing useful current-DGP gain on its six predeclared
core cases. Reserved final pixels remain unviewed. Source labels imply neither
ethnicity nor Zamboanga performance. Any later epoch recipe needs full-corpus,
paired DEV and useful native DEV evidence; independent final review, app flow
and all seven automatic/assisted covering families still require qualification.
The full goal remains active/incomplete. This status supersedes pending-return
statements below while preserving their exact bytes as historical evidence.
Before/status hashes are in outputs/cctv_dgp_original_feature_probe_v1_closure/.


'''

SYSTEM = '''**Latest diagnostic status — 9 October 2026: original-feature return verified; loss-balance diagnostic prepared and verified, not launched.**

All nine original-feature trials fail preservation, including the decoder trial
that exceeds 1% structure gain on two small photographic TRAIN cohorts. All
1,000 outputs have received assistant development visual review. This does not
solve V42's full-corpus 0.00692364% failure or establish useful native restoration.
The retained current DGP and original Phase 3 checkpoint are unchanged.

A distinct finite diagnostic tests original-decoder structure descent combined
with original-feature identity descent, using audited saved derivatives. It has
zero new gradient queries, zero optimizer updates and zero epochs. Its independent
packet audit verifies all transferred members, source bindings, nine formulas,
local trial denials and 100 initial-output parity cases. Manual tmux instructions:
CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_VM.md. No VM execution is launched automatically.
Scientific requirements remain unchanged; every failed trial is retained. More
epochs require a justified recipe and full-corpus, DEV and native review.

Read CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_RESULTS.md and the latest PROJECT_HANDOFF.md.
All five restoration milestones, all seven automatic/assisted covering families,
independent final review and bundled inline Playwright remain required. The full
goal is active/incomplete. This execution status supersedes pending-return
statements below without changing the agreed goal or manual training workflow.
Exact preceding bytes: outputs/cctv_dgp_original_feature_probe_v1_closure/before/.


'''

PRACTICAL = '''**Latest restoration diagnostic status — 9 October 2026: return audited and reviewed; completion requirements remain unchanged.**

The original-feature diagnostic's nine trials remain unqualified on preservation.
All 1,000 outputs were reviewed as photographic TRAIN development evidence.
Its small-cohort structure gain does not solve V42 or qualify native CCTV,
restoration quality, final identities or completion. Current app weights and
design remain intact. No new completion generation or app change occurs.

A verified finite original-loss-balance diagnostic is ready for manual tmux
execution: CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_VM.md. It reuses saved derivatives
with zero new gradient queries, optimizer updates or epochs. No failed recipe
is resumed and the existing preservation requirements remain binding.

All seven covering families, automatic mask preview with optional correction,
visible appearance and clear-glasses/hair preservation, insufficient-information
handling, original/mask/result, PNG/bundle downloads and separate automatic versus
assisted review remain required. The full goal, independent final review and
bundled inline Playwright remain active/incomplete. Read the latest handoff and
CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_RESULTS.md for the exact evidence boundary.
Exact preceding bytes: outputs/cctv_dgp_original_feature_probe_v1_closure/before/.


'''


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def assert_bindings(protocol):
    assert sha(PROTOCOL) == PROTOCOL_SHA
    for name, digest in protocol['local_sources_sha256'].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in CHECKPOINTS.items():
        assert sha(ROOT / name) == digest, name


def main():
    start = time.monotonic()
    assert not OUT.exists(), 'Closure already exists; retain its evidence'
    protocol = read(PROTOCOL)
    assert_bindings(protocol)
    audit_path = PREPARATION / 'independent_packet_audit.json'
    audit = read(audit_path)
    prepared = read(PREPARATION / 'prepared.json')
    assert audit['complete'] and audit['protocol_sha256'] == PROTOCOL_SHA
    assert audit['archive_sha256'] == prepared['archive_sha256']
    assert audit['archive_members_verified'] == 170 and audit['source_bindings_verified'] == 23
    assert audit['initial_CPU_parity_cases'] == 100
    assert audit['states_before'] == audit['states_after']
    for name in ('local_gradient_queries', 'local_optimizer_updates', 'local_trial_assignments', 'VM_connections'):
        assert audit[name] == 0, name
    assert not audit['VM_diagnostic_launched']
    review_path = ROOT / 'outputs/cctv_dgp_original_feature_probe_v1_return_review_v1/visual_review.json'
    review = read(review_path)
    assert review['complete'] and review['unique_model_outputs_reviewed'] == 1000
    assert review['all60_pages_viewed_at_original_resolution']
    assert not review['model_qualification'] and not review['independent_final_reviewer']
    analysis = read(ROOT / 'outputs/cctv_dgp_original_feature_probe_v1_return_review_v1/partition_analysis.json')
    decisions = [stage for trial in analysis['trial_comparisons']
        for stages in trial['comparisons'].values() for stage in stages.values()]
    assert len(decisions) == 36 and all(not stage['pass'] for stage in decisions)
    docs = {'PROJECT_HANDOFF.md': HANDOFF, 'SYSTEM_WORKFLOW_AND_GOAL.md': SYSTEM,
        'PRACTICAL_OUTPUT_SCOPE.md': PRACTICAL}
    assert not set(docs).intersection(protocol['local_sources_sha256'])
    original = {name: (ROOT / name).read_bytes() for name in docs}
    OUT.mkdir()
    (OUT / 'before').mkdir()
    for name, content in original.items():
        (OUT / 'before' / name).write_bytes(content)
    bindings = {}
    for name, prefix in docs.items():
        path = ROOT / name
        assert path.read_bytes() == original[name], name
        prefix_bytes = prefix.encode('utf-8')
        path.write_bytes(prefix_bytes + original[name])
        published = path.read_bytes()
        assert published == prefix_bytes + (OUT / 'before' / name).read_bytes()
        bindings[name] = {'before_sha256': sha(OUT / 'before' / name),
            'published_sha256': sha(path), 'prefix_bytes': len(prefix_bytes),
            'original_suffix_exact': True}
    assert_bindings(protocol)
    receipt = {'complete': True, 'return_review_complete': True,
        'independent_return_audit_sha256': sha(ROOT / 'outputs/cctv_dgp_original_feature_probe_v1_independent_audit_r2.json'),
        'visual_review_sha256': sha(review_path), 'reviewed_unique_model_outputs': 1000,
        'all36_scientific_trial_decisions_failed': True,
        'next_manual_protocol_sha256': PROTOCOL_SHA,
        'next_execution_archive_sha256': audit['archive_sha256'],
        'next_execution_archive_bytes': prepared['archive_bytes'],
        'next_independent_packet_audit_sha256': sha(audit_path),
        'manual_guide_sha256': sha(ROOT / 'CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_VM.md'),
        'checkpoint_sha256': CHECKPOINTS,
        'status_documents': bindings, 'closure_script_sha256': sha(Path(__file__)),
        'closure_neural_calls': 0, 'closure_gradient_queries': 0,
        'closure_optimizer_updates': 0, 'VM_connections': 0,
        'VM_diagnostic_launched': False, 'app_changed': False,
        'native_or_final_used_in_diagnostic': False,
        'model_qualification': False, 'goal_complete': False,
        'seconds': time.monotonic() - start}
    with (OUT / 'closure.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print({'complete': True, 'status_documents': len(bindings),
        'protocol_sha256': PROTOCOL_SHA, 'goal_complete': False}, flush=True)


if __name__ == '__main__':
    main()
