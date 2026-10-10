"""Bind the independently audited failed human pilot and completion ablation."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import time
from completion_feature_fusion_off_v1_common import ROOT, OUT as COMPLETION, sha, read, write, verify_bindings

OUT = ROOT/'outputs/cctv_dgp_v40_return_completion_milestone'
PRIOR = ROOT/'outputs/cctv_dgp_v39_return_v40_prepared_milestone'
RETURNED = ROOT/'outputs/cctv_dgp_spatial_fit_v40_return'
REVIEW = ROOT/'outputs/cctv_dgp_spatial_fit_v40_failure_review'


def main():
    start = time.monotonic(); assert not OUT.exists()
    parent = read(PRIOR/'milestone.json'); parent_audit = read(PRIOR/'independent_closure_audit.json')
    assert parent_audit['complete'] and parent_audit['milestone_sha256'] == sha(PRIOR/'milestone.json')
    for name, digest in parent['new_evidence_sha256'].items():
        assert sha(ROOT/name) == digest, name
        assert time.monotonic()-start < 600
    p = read(COMPLETION/'protocol.json'); verify_bindings(p)
    completion = read(COMPLETION/'independent_review_audit.json'); v = read(COMPLETION/'visual_review.json')
    a = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json'); reviewed = read(REVIEW/'independent_review_audit.json')
    imported = read(ROOT/'outputs/cctv_dgp_spatial_fit_v40_return_import.json'); failed = read(RETURNED/'outputs/failure.json')
    assert a['complete'] and a['failure_retained'] and not a['necessary_capacity_pass'] and not a['training_finished800']
    assert a['members_verified'] == imported['members'] == 27451 and failed['optimizer_updates'] == 50
    assert reviewed['complete'] and reviewed['all50_case_notes_and10_actual_page_records_verified']
    assert completion['complete'] and completion['original_checker_and_failure_retained'] and v['all32_outputs_and32_w1_baselines_actually_viewed']
    assert not v['app_adoption'] and not v['automatic_quality_qualification'] and not v['assisted_quality_qualification']
    stopped_record = read(COMPLETION/'milestone_record_stop.json')
    assert stopped_record['handoff_unchanged_verified'] and stopped_record['milestone_folder_not_created']
    assert not (ROOT/'outputs/completion_feature_fusion_off_v1_milestone').exists()
    OUT.mkdir(); (OUT/'before_docs').mkdir()
    before = (ROOT/'PROJECT_HANDOFF.md').read_bytes(); (OUT/'before_docs/PROJECT_HANDOFF.md').write_bytes(before)
    section = '''
**Research milestone — 8 October 2026: V40 human return independently audited, early structure failed; completion feature-fusion variant also rejected.**

V40 ran manually on the existing L4 and stopped at 50/800 updates. Its unchanged
prospective independent checker confirms 0.0106079234% structure gain against
the retained 1% requirement. All 17 preservation groups pass. This is a learning
quality stop; export exit code 0 packaged failure evidence. No 800 result,
resume, unchanged repeat, development qualification or application promotion.
The original trained DGP remains primary in the unchanged local application.

The 1,138,304,426-byte return, sidecar and export receipt match SHA256:
58706c14674e34a2a60308b94338f02b8f083d02a9fc0b2a94af8d6344745ac6.
Protocol b21597d7c50dc39ee7cb2bd70d7d0ad9f5cd967661b6f5e78413a7e4eb121010.
All 27,451 regular files are safely imported with exact manifest hashes;
returned code is not executed. The 751.24s audit recomputes all 3,905 PNG,
mean-only PNG and vector comparisons per complete snapshot 0/50, all groups and
50 saved raw preview compositions per state. The other 3,855 raw stages per
state are hashed on the VM but not retained; no all-raw audit is claimed.

One hundred frozen CPU replays pass: raw error <=2.2947788e-6, PNG difference
one byte and vector error <=2.7939677e-7 within prospective limits. Original DGP,
fixed decoder and recognizer states stay unchanged; the separate learned decoder
changes. VM has 50 backwards/updates, 1,027.71s total, 492.89s fit including its
update50 snapshot and 9,504,983,552 bytes allocated VRAM within finite limits.
The agent performs no local gradients, optimizer updates, actual training or
VM operation. The stopped checkpoint, original packet and failed gate remain.

All 50 fixed TRAIN previews and 250 exact comparison cells were actually viewed
on 10 sheets at original 256-cell detail. Eyes, nose, mouth, outline and overall
appearance remain in scope. Stopped50 stays near the retained DGP, with slight
tone/texture changes and no convincing new facial definition. No output-based
input relabeling hides the failure. These are TRAIN, not final evaluation.

Fifty paired batches optimize 250 cases from 50 TRAIN references. The 200
degraded optimized cases gain only 0.007422% structure; 2,924 not-yet-used TRAIN
cases gain 0.010873%. Forty degraded fixed previews gain 0.014370%; none of the
50 fixed previews is directly optimized in these updates. Weak learning is not
confined to unseen cases. This does not establish a unique architecture/loss/
optimizer cause. Per-update component losses and optimizer moments are absent;
they cannot be reconstructed from two checkpoints. V39's valid nonzero-gradient
proof does not guarantee useful learning under this recipe. Further learning
needs a separate justified diagnostic, retaining gates and the manual workflow.

The human selected “apply the best approach” for completion architecture review.
A single frozen w=0 ablation disables direct encoder-feature fusion. Initial
w=1 cached parity is exact; the matched neural input, encoder, prediction scores,
codes and quantized latent are identical. Actual candidate fusion calls are
zero. Same two assisted conditioning masks, same final removal support/two-pixel
margin, no output-driven annotation, eligibility, seed/weight or source search.

The completion worker takes 139.81561s (external 145.19614s): 29 frozen forwards,
28 estimates, four exact empty controls and four unchanged input-only exclusions.
There are no DGP/detector, gradient/backward/optimizer or VM calls. All original
states stay unchanged. The 108 artifacts total 164,378,715 bytes within 256MiB;
worker600s/external630s+30s limits remain. Protocol:
53c038ea900c1b8d0c602679435cf794b6770708196183ec6a1773437becf95b.

The original checker stops at replay scores because its saved C-contiguous
input changed the actual channels-last layout. Raw RGB/codes stay exact; scores
differ by up to0.0034141541. A distinct12.05021s two-call frozen diagnostic proves
the layout cause and exact replay with the actual layout. R1 changes only
layout/receipt filename, inverse source/AST exact and every exact rule unchanged.
It passes in13.02887s. Original checker/failure are preserved, without tolerance
changes or replacement estimates. All520 sources/108 artifacts,28 compositions,
160 sheet cells,5,483,088 visible and1,075,314 protected bytes are verified.

All32 completion outputs and32 w=1 counterparts were actually inspected on eight
sheets. Pale eye patches/clouded glare and hand/scarf joins remain; w=0 adds or
strengthens red/bright lower-face patches. It is rejected for adoption. Automatic
outputs:zero; automatic and assisted quality remain separately unqualified.
The legacy native suffix means original photograph, not CCTV. Assistance is
reused for synthetic pairs; pretraining overlap and true hidden appearance are
unknown. Exterior copied fragments are separate from generated defects. Clear
glasses/ordinary hair stay intact. All seven covering families remain required.

The first documentation recorder correctly stops before handoff edits when the
new V40 return begins arriving. Its original source/stop receipt are retained;
the completed human download and independent return audit supersede absence.
Both reports reflect the current evidence. The full previous handoff is archived
and its complete body preserved below, with earlier unrun statements historical.

All14 app/checkpoint bindings/design stay unchanged: original DGP primary,
Auto/On/Off override, input and removal-area review, original/mask/estimate,
PNG and bundle downloads. No new browser-quality claim is made. Checkpoints,
splits, provenance, source terms, research caches/local backup and all failed
gates remain. V38's development failure and weak native clarity remain binding.
Paired synthetic TRAIN and native unpaired CCTV remain separate. No new native
acquisition, hidden identity, ethnicity, native PSNR/SSIM or Zamboanga performance
is claimed. Reserved45 identities/58 crops remain unopened. Useful reviewed
development restoration, all covering families, independent final review and
full qualified DGP-led flow remain outstanding. Goal active and incomplete.

[V40 audited failure](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FIT_V40_RESULTS.md>)
[All50 visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_fit_v40_failure_review/visual_review.json>)
[Completion architecture findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_FEATURE_FUSION_OFF_V1_RESULTS.md>)

'''
    addition = section.encode('utf-8'); at = before.index(b'\n')+1
    (ROOT/'PROJECT_HANDOFF.md').write_bytes(before[:at]+addition+before[at:])
    bindings = dict(p['sources_sha256']); bindings.update(p['app_preservation_sha256'])
    for folder in [COMPLETION, REVIEW]:
        for path in sorted(folder.rglob('*')):
            if path.is_file(): bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, digest in imported['files_sha256'].items(): bindings[(RETURNED/name).relative_to(ROOT).as_posix()] = digest
    names = ['PROJECT_HANDOFF.md', 'CCTV_DGP_SPATIAL_FIT_V40_RESULTS.md', 'CCTV_DGP_COMPLETION_FEATURE_FUSION_OFF_V1_RESULTS.md',
             'outputs/cctv-dgp-spatial-fit-v40-results.tar.gz', 'outputs/cctv-dgp-spatial-fit-v40-results.tar.gz.sha256',
             'outputs/cctv-dgp-spatial-fit-v40-export.json', 'outputs/cctv_dgp_spatial_fit_v40_return_import.json',
             'outputs/cctv_dgp_spatial_fit_v40_independent_audit.json', 'outputs/cctv_dgp_spatial_fit_v40_independent_audit.log']
    scripts = ['diagnose_completion_feature_fusion_off_v1_replay_layout', 'prepare_completion_feature_fusion_off_v1_audit_r1',
               'audit_completion_feature_fusion_off_v1_r1', 'record_completion_feature_fusion_off_v1_review',
               'verify_completion_feature_fusion_off_v1_review', 'record_completion_feature_fusion_off_v1_milestone',
               'verify_completion_feature_fusion_off_v1_milestone', 'record_completion_feature_fusion_off_v1_milestone_stop',
               'prepare_cctv_dgp_spatial_fit_v40_failure_review', 'audit_cctv_dgp_spatial_fit_v40_failure_review',
               'record_cctv_dgp_spatial_fit_v40_visual_review', 'verify_cctv_dgp_spatial_fit_v40_visual_review',
               'record_cctv_dgp_v40_return_completion_milestone', 'verify_cctv_dgp_v40_return_completion_milestone']
    names.extend('scripts/'+s+'.py' for s in scripts)
    for name in names: bindings[name] = sha(ROOT/name)
    for name, digest in bindings.items():
        assert sha(ROOT/name) == digest, name
        assert time.monotonic()-start < 600
    write(OUT/'milestone.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
          'previous_milestone_sha256': sha(PRIOR/'milestone.json'),
          'previous_handoff_path': (OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
          'document': {'before_sha256': hashlib.sha256(before).hexdigest(), 'addition_bytes': len(addition)},
          'new_evidence_sha256': bindings, 'V40_human_return_independently_audited': True, 'V40_updates': 50,
          'V40_structure_requirement_failed': True, 'all50_TRAIN_and32_completion_outputs_actually_reviewed': True,
          'original_audit_and_documentation_stops_retained': True, 'layout_only_R1_exact_audit_pass': True,
          'DGP_primary_app_bindings_unchanged': True, 'VM_calls': 0, 'local_gradient_calls': 0,
          'local_optimizer_updates': 0, 'training_started_here': False, 'app_promotion': False,
          'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
          'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'bindings': len(bindings), 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
