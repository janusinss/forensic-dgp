"""Publish audited return status with exact instruction/guide history preserved."""
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_loss_response_v1 import contract as c

REVIEW = ROOT / 'outputs/codeformer_loss_response_v1_return_review'
REPORT = ROOT / 'CCTV_CODEFORMER_LOSS_RESPONSE_V1_RETURN_REVIEW.md'
CORE = ['SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'PROJECT_HANDOFF.md']
HISTORY = ['CCTV_CODEFORMER_LOSS_RESPONSE_V1_VM.md', 'CCTV_CODEFORMER_LOSS_RESPONSE_V1_PREPARATION.md',
           'CCTV_CODEFORMER_LOSS_RESPONSE_V1_DESIGN.md']

PREFIX = '''# CodeFormer loss response V1 returned and audited — 17 October 2026

**CURRENT: FINITE MEASUREMENT SUCCEEDED / NO NEW MODEL CHECKPOINT OR PROMOTION.**
[Actual return review](CCTV_CODEFORMER_LOSS_RESPONSE_V1_RETURN_REVIEW.md).
[All16 exact256 sheets](outputs/codeformer_loss_response_v1_return_review/pixels_v1/visual_sheets/INDEX.md).
[Saved-pixel evidence](outputs/codeformer_loss_response_v1_return_review/pixels_v1/SCIENTIFIC_READBACK.md).
[Next independent design review](outputs/codeformer_loss_response_v1_return_review/next_design_independent_review_v1.md).

All4,609 returned regular files /2,881,853,749 payload bytes passed hash-bound
import and independent readback. Archive SHA-256:
`b0e2b6d0aa245fcebe334c21c1b50cfbfde3ca65d494bb1d1fd8a9d17730de86`.
Actual manual L4 work was425 neural forwards,193 gradient queries and18 REAL
DISPOSABLE Adam steps; zero committed training updates. Worker585.444 seconds,
preflight16.966 and internal export203.374 seconds. Every branch restored its
declared original0/stopped50 base; these were not additional training epochs.

All352 native-to-RAW conversions and floor-PNG checks match exactly. Independent
saved-component composition matches all6,300 assigned-gradient digests. Root
personally viewed all16 planned sheets; all416 exact256 cells were independently
checked. This TRAIN-only study uses the same16 observations for gradients and
probes, not DEV/final acceptance. No consistent visual preservation fix emerged.

The previous AdaIN50 clear/normal-blur preservation failure stays binding; these
full61-clear/53-blur guards were not rerun. Deleting coarse RGB improves some
normal errors but weakens dark benefit. Deleting LPIPS harms all8normal cases
at both states; deleting behavior RGB harms all8normal cases at stopped50.
CE-deletion effects change across states. Stopped-state zero-current-gradient
momentum changes all350 trainable rows and much of the finite output response.
No universal deletion, sole cause, unchanged resume or extra-epoch recipe is proven.

The next bounded question holds dark coarse supervision fixed and compares
full, normal-coarse omission and explicitly defined normal-luminance centering,
with baseline/momentum controls at both saved states. Preserve chroma constraints,
LPIPS and behavior anchors. L1 is not an additive DC/residual decomposition;
per-channel centering also removes chroma DC. This objective hypothesis is new
and unmeasured, not the rejected inference brightness gate. No next execution
packet or training launch is prepared by this audit.

Runtime success and initial resource projections are verified at their proper
scope; measured peak RAM/CUDA/minimum free disk were not exported. Camera parity
and repeat flags remain source-bound VM controls, not independent local neural
replay. Root's two old-checkpoint weights-only CPU audits verify base identities;
NumPy equation after hashes match6,607/9,270 tensors, not every CUDA result.
All2,970 frozen comparisons match. Nonexact trainable hashes stay nonexact;
close scalar norms establish no missing per-element delta bound. Final active
export.log/post-export finished.json bytes are unsupplied; home checksum and
export receipt are present. Every failed checker/source remains retained.

The full restoration-first goal remains active: accurately credited CodeFormer
fine-tuned by us for useful256x256 restoration of one cropped frontal/mild-turn
face, preserving visible appearance and the existing app design. Untouched
CodeFormer remains initializer/benchmark; retain original DGP/checkpoints,
splits, caches, provenance, failures and migration backups. Useful moderate
brightening is allowed with structure/appearance protected. Keep public native
CCTV unpaired and source-separated, with audited terms/resolution/overlap,
separate paired synthetic evidence and untouched FINAL. No ethnicity, exact
identity recovery or Zamboanga performance claim follows. User acceptance,
independent final review and full app Auto/override/PNG flow verified through
bundled inline Playwright remain required. No app/checkpoint promotion occurred.

The input-only usable-face refusal threshold remains a recorded requirement,
not an implemented/calibrated automatic gate. Judge original visible structure,
effective resolution, pose, blocking, clipping and coverage; request clearer,
more frontal/less-covered crops when information is insufficient. Brightness
or blur alone must not reject usable input. Preserve all DEV cases; use DEV-only
input labels/independent review, no output-based relabeling or FINAL tuning.
Confirmed hard insufficiency must not be bypassed by restoration On/Off or masks.
Covered mouth/nose alone does not reject enough visible upper-face context.

Separate detection/completion improvement remains deferred until restoration
acceptance. Keep the later seven families, automatic removal-area preview and
optional correction, visible/clear-glasses/non-obstructing-hair preservation,
one plausible estimate, original/mask/result and PNG/optional bundle downloads,
separate automatic/assisted evidence and no exact hidden-identity claim.

All actual training remains the user's finite manual L4/tmux workflow with
verified transfers and independently audited returns. This local audit ran no
neural/gradient/optimizer/VM calls or new training, app change, cleanup, Drive
transfer or deletion. Prior storage offload remains complete and separate.
Old commands below are historical; do not relaunch this completed diagnostic.
All earlier instruction/guide/design bytes are retained exactly below and in
publication backups. The overall goal remains incomplete.

---

'''


def pin(path):
    path = c.no_links(path)
    return {'sha256': c.sha256(path), 'bytes': path.stat().st_size}


def main():
    publication = REVIEW / 'publication_v1'
    c.require(not publication.exists(), 'Preserve prior publication')
    imported = c.read_json(REVIEW / 'return_import_v1.json')
    independent = c.read_json(REVIEW / 'root_report_independent_review_v1.json')
    c.require(imported['complete'] and independent['complete'] and
              independent['report_sha256'] == c.sha256(REPORT), 'Current independent report closure required')
    c.require(imported['archive_sha256'] == 'b0e2b6d0aa245fcebe334c21c1b50cfbfde3ca65d494bb1d1fd8a9d17730de86',
              'Returned archive identity differs')
    for name in ['root_visual_review_v1.json', 'root_fixture_readback_v1.json',
                 'state_v1/state_audit_v2.json', 'pixels_v1/audit.json',
                 'pixels_v1/independent_sheet_readback.json', 'independent_transport_runtime_v1.json',
                 'CPU_Adam_v1/receipt.json', 'independent_CPU_Adam_review_v1.json',
                 'next_design_independent_review_v1.json']:
        c.require(c.read_json(REVIEW / name)['complete'] is True, 'Missing completed audit: ' + name)
    publication.mkdir()
    before = {}
    for name in CORE + HISTORY:
        data = (ROOT / name).read_bytes()
        target = publication / 'prior' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
        c.require(target.read_bytes() == data, 'Exact publication backup differs')
        before[name] = pin(target)
    evidence = sorted(path for path in REVIEW.rglob('*') if path.is_file() and publication not in path.parents)
    evidence += [REPORT, Path(__file__), ROOT / 'scripts/import_codeformer_loss_response_return_v1.py',
                 ROOT / 'scripts/audit_codeformer_loss_response_return_state_v1.py',
                 ROOT / 'scripts/audit_codeformer_loss_response_CPU_Adam_v1.py',
                 ROOT / 'tests/test_codeformer_loss_response_return_state_v1.py',
                 Path(imported['archive']),
                 ROOT / 'outputs/cctv-codeformer-loss-response-v1-export.json',
                 ROOT / 'outputs/cctv-codeformer-loss-response-v1-results.tar.gz.sha256']
    c.write_new_json(publication / 'evidence_chain.json', {'complete': True,
        'scope': 'All local return-audit/report/visual artifacts and exact prior documentation',
        'files': {p.relative_to(ROOT).as_posix(): pin(p) for p in evidence},
        'prior_documentation': before, 'archive_sha256': imported['archive_sha256'],
        'protocol_sha256': imported['protocol_sha256'], 'model_qualified': False})
    prefix = PREFIX.encode('utf-8')
    with (publication / 'prefix.md').open('xb') as stream:
        stream.write(prefix)
    changed = {}
    for name in CORE + HISTORY:
        path, prior = ROOT / name, publication / 'prior' / name
        c.require(pin(path) == before[name], 'Documentation changed after backup')
        payload = prefix + prior.read_bytes()
        path.write_bytes(payload)
        c.require(path.read_bytes() == payload and path.read_bytes()[len(prefix):] == prior.read_bytes(),
                  'Exact preserved suffix differs')
        changed[name] = {'after': pin(path), 'prior': before[name], 'suffix_offset': len(prefix),
                         'prior_backup': str(prior)}
    receipt = {'complete': True, 'status': 'RETURN_AUDITED_NO_MODEL_PROMOTION',
        'UTC': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': imported['protocol_sha256'],
        'archive_sha256': imported['archive_sha256'], 'documents': changed,
        'evidence_chain': pin(publication / 'evidence_chain.json'), 'prefix': pin(publication / 'prefix.md'),
        'publisher_sha256': c.sha256(__file__), 'actual_local_neural_gradient_optimizer_VM_calls': 0,
        'actual_manual_disposable_Adam_steps': 18, 'committed_training_updates': 0,
        'new_training_packet_prepared': False, 'model_qualified': False, 'overall_goal_complete': False}
    c.write_new_json(publication / 'receipt.json', receipt)
    print({'complete': True, 'updated_documents': len(changed), 'status': receipt['status'],
           'receipt_sha256': c.sha256(publication / 'receipt.json')})


if __name__ == '__main__':
    main()
