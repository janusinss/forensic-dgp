"""Extend the closed research history with an audited negative completion diagnostic."""
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import time
from completion_feature_fusion_off_v1_common import ROOT, OUT as COMPARISON, sha, read, write, verify_bindings

OUT = ROOT/'outputs/completion_feature_fusion_off_v1_milestone'
PRIOR = ROOT/'outputs/cctv_dgp_v39_return_v40_prepared_milestone'


def main():
    start = time.monotonic(); assert not OUT.exists()
    p = read(COMPARISON/'protocol.json'); verify_bindings(p)
    previous = read(PRIOR/'milestone.json'); closed = read(PRIOR/'independent_closure_audit.json')
    assert closed['complete'] and closed['milestone_sha256'] == sha(PRIOR/'milestone.json')
    for name, digest in previous['new_evidence_sha256'].items():
        assert sha(ROOT/name) == digest, name
        assert time.monotonic()-start < 600
    a, v, review = [read(COMPARISON/name) for name in ['independent_saved_output_audit_r1.json', 'visual_review.json', 'independent_review_audit.json']]
    assert a['complete'] and v['complete'] and review['complete'] and review['R1_inverse_source_and_AST_exact']
    assert not v['app_adoption'] and not v['automatic_quality_qualification'] and not v['assisted_quality_qualification']
    assert not (ROOT/'outputs/cctv-dgp-spatial-fit-v40-results.tar.gz').exists()
    OUT.mkdir(); (OUT/'before_docs').mkdir()
    before = (ROOT/'PROJECT_HANDOFF.md').read_bytes(); (OUT/'before_docs/PROJECT_HANDOFF.md').write_bytes(before)
    section = '''
**Completion milestone — 8 October 2026: feature-fusion ablation independently audited and visually rejected; primary DGP preserved.**

The human answered the post-processing architecture question with “apply the
best approach”. A single prospective frozen CodeFormer inpainting ablation
disables direct encoder-feature fusion at w=0. Its initial w=1 union-conditioned
parity is exact. The first matched case retains identical neural input, encoder
latent, prediction scores, selected codes and quantized latent; actual fusion
calls are zero for candidates. It remains an explicitly nondefault supporting
completion comparison. Our original trained DGP remains the primary restorer.

All36 existing exposed photographic requests remain:28 estimates, four exact
empty-mask controls and four unchanged input-only exclusions. The worker takes
139.81561s; external timing145.19614s. There are29 frozen completion forwards,
zero DGP/detector calls, zero gradients/backwards/optimizer updates/checkpoints
or VM operations. The same pretrained state is unchanged. Worker600s/external
630s+30s termination and256MiB artifacts are finite. The108 artifacts occupy
164,378,715 bytes. Protocol SHA256:
53c038ea900c1b8d0c602679435cf794b6770708196183ec6a1773437becf95b.

The original prospective auditor stopped during its last exact replay: saved
C-contiguous input changed the actual adapter's channels-last layout. Raw RGB
and selected codes stayed exact, but131,004 prediction scores differed by up to
0.0034141541. Original source/failure are retained. A separate12.05021s two-call
frozen diagnostic confirms equal input values and the layout cause. Correct
layout reproduces RGB, scores and encoder features exactly. The distinct R1
checker changes only replay layout/receipt filename, with inverse source/AST
identical and every exact comparison/quality criterion unchanged. It passes
in13.02887s. No replacement estimate or new tolerance was introduced.

The independent R1 audit verifies520 source bindings,108 artifacts,28 actual
input/codebook/raw512-to256/final PNG compositions and160 exact page cells.
All5,483,088 visible-source bytes and1,075,314 protected bytes are unchanged.
All32 outputs and32 w=1 counterparts were actually viewed on eight sheets at
original256 cell detail. Pale eye patches, cloudy glare and hand/scarf joins
remain; w=0 introduces or strengthens red/bright/patterned lower-face patches.
Some central estimates are plausible, but the whole scope does not qualify.
The separate0.75830s review verifier confirms case scope and repair provenance;
it does not replace independent final visual review. No app adoption.

Same two historical assisted masks define conditioning; same final area and
documented two-pixel margin define delivery. Original-photo assistance is reused
for synthetic pairs. There is no output-driven annotation, eligibility change,
mask expansion, seed/weight search or narrowing of covering scope. All seven
families remain required. Clear glasses/ordinary hair are preserved. Exterior
flowers/leaf/hands/neck scarf outside final area are copied, separate from neural
defects. The legacy native suffix means original photo, not CCTV. Pretraining
overlap is unknown; no aligned hidden reference or hidden PSNR/SSIM exists.
No ethnicity, exact hidden identity or Zamboanga performance is inferred.

New automatic estimates:zero. Automatic and assisted quality remain separately
unqualified. The existing14 app/checkpoint bindings and design are unchanged,
including DGP primary, Auto/On/Off override, reviewed removal area, plausible
estimate/original/mask, PNG and bundle downloads. No new browser claim is made.
Input-only requests for clearer/less-covered crops remain only for insufficient
information; visible generated defects do not justify relabeling usable input.

V39's zero-update spatial gradient proof remains independently audited. The
verified V40 spatial capacity packet and five manual L4 transfer/tmux commands
remain ready. No V40 result archive is present locally at this milestone; no
current VM status or human training activity is inferred. Actual learning stays
manual on the existing L4/g2-standard-4 at ~/forensic-dgp. Another feature-fusion
setting or unchanged failed processing recipe is not justified. Further learned
completion needs a separately justified prospective finite VM design.

All original checkpoints, splits, source/terms/provenance/caches/local backup,
saved failures and closed history remain retained. V38 development preservation
failure and weak native clarity remain binding. Paired synthetic evidence and
24 unpaired native development crops stay separate; no new native acquisition
or real Zamboanga data is claimed. Reserved45 identities/58 crops stay unopened.
Useful reviewed development restoration, all covering families, independent
final review and full qualified DGP-led app flow remain outstanding. Goal active
and incomplete; neither generation completion nor audit implies qualification.

[Completion diagnostic results](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_FEATURE_FUSION_OFF_V1_RESULTS.md>)
[All32 development observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/completion_feature_fusion_off_v1/visual_review.json>)
[V40 five manual VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FIT_V40_VM.md>)

The complete preceding handoff follows. Its earlier unrun observations are
historical statements about the corresponding artifact review time.

'''
    addition = section.encode('utf-8'); at = before.index(b'\n')+1
    (ROOT/'PROJECT_HANDOFF.md').write_bytes(before[:at]+addition+before[at:])
    bindings = dict(p['sources_sha256']); bindings.update(p['app_preservation_sha256'])
    for path in sorted(COMPARISON.rglob('*')):
        if path.is_file():
            assert not path.is_symlink(); name = path.relative_to(ROOT).as_posix(); digest = sha(path)
            assert name not in bindings or bindings[name] == digest; bindings[name] = digest
    extra = ['scripts/diagnose_completion_feature_fusion_off_v1_replay_layout.py',
             'scripts/prepare_completion_feature_fusion_off_v1_audit_r1.py', 'scripts/audit_completion_feature_fusion_off_v1_r1.py',
             'scripts/record_completion_feature_fusion_off_v1_review.py', 'scripts/verify_completion_feature_fusion_off_v1_review.py',
             'scripts/record_completion_feature_fusion_off_v1_milestone.py', 'scripts/verify_completion_feature_fusion_off_v1_milestone.py',
             'CCTV_DGP_COMPLETION_FEATURE_FUSION_OFF_V1_RESULTS.md', 'PROJECT_HANDOFF.md']
    for name in extra: bindings[name] = sha(ROOT/name)
    for name, digest in bindings.items(): assert sha(ROOT/name) == digest, name
    assert time.monotonic()-start < 600
    write(OUT/'milestone.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
          'previous_milestone_sha256': sha(PRIOR/'milestone.json'),
          'previous_handoff_path': (OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
          'document': {'before_sha256': hashlib.sha256(before).hexdigest(), 'addition_bytes': len(addition)},
          'new_evidence_sha256': bindings, 'all32_development_outputs_actually_reviewed': True,
          'original_audit_failure_retained_and_layout_only_R1_passed': True, 'exact_criteria_unchanged': True,
          'DGP_primary_app_bindings_unchanged': True, 'V40_return_present_locally': False, 'VM_status_observed': False,
          'local_optimizer_updates': 0, 'local_gradient_calls': 0, 'VM_calls': 0, 'training_started_here': False,
          'app_promotion': False, 'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
          'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'bindings': len(bindings), 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
