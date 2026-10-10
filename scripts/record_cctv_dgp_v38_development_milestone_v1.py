"""Preserve prior handoff and bind the audited V38 development/completion evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v38_return_development_milestone_v1'
PRIOR = ROOT / 'outputs/cctv_dgp_v37_return_v38_probe_milestone'
NATIVE = ROOT / 'outputs/cctv_dgp_v38_quarter_native_development_v1'
PAIRED = ROOT / 'outputs/cctv_dgp_v38_quarter_paired_development_v1'
ANALYSIS = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1'
COMPLETION = ROOT / 'outputs/completion_margin_feather_v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    start = time.monotonic(); assert not OUT.exists()
    previous = read(PRIOR / 'milestone.json'); closure = read(PRIOR / 'independent_closure_audit.json')
    assert previous['complete'] and closure['complete']
    assert closure['milestone_sha256'] == sha(PRIOR / 'milestone.json')
    for name, digest in previous['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
        assert time.monotonic() - start < 300
    audit = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_independent_audit.json')
    a = read(ANALYSIS / 'analysis.json'); av = read(ANALYSIS / 'visual_review.json')
    n = read(NATIVE / 'results.json'); nv = read(NATIVE / 'visual_review.json')
    p = read(PAIRED / 'results.json'); pv = read(PAIRED / 'visual_review.json')
    for folder, receipt in [(ANALYSIS, 'independent_analysis_page_audit.json'),
                            (NATIVE, 'saved_output_audit.json'), (PAIRED, 'saved_output_audit.json'),
                            (COMPLETION, 'independent_saved_output_audit.json')]:
        assert read(folder / receipt)['complete']
    assert audit['complete'] and audit['finite_probe_complete'] and not audit['training_capacity_pass']
    assert a['jointly_eligible_subset_variants'] == ['margin_quarter', 'margin_eighth']
    assert av['complete'] and nv['complete'] and nv['cases_reviewed'] == 24
    assert p['complete'] and len(p['rows']) == 520 and pv['complete']
    cv = read(COMPLETION / 'visual_review.json')
    assert cv['complete'] and not cv['automatic_quality_qualification'] and not cv['assisted_quality_qualification']
    failures = len(p['diagnostic_preservation_failures'])
    gains = {key: 100 * value for key, value in p['degraded_structure_gain_fraction'].items()}
    addition = f'''
**DGP milestone — 8 October 2026: V38 returned and independently audited; frozen quarter-step development check completed.**

The human returned V38's 388,426,632-byte archive, SHA256
c1403d67ba0a66a1200285bcab291cd87dc9d6d7bf0518534f3b07a12ebff876.
The prospective checker verifies all2,135 regular files, all500 raw/PNG/vector
measurements,170 group aggregates,210 projection rows and100 frozen CPU outputs.
Local audit runtime is240.167 seconds inside a241.612-second supervised run.
The111.690-second L4 probe makes four reset trials, zero new gradients, optimizer
updates, committed trajectory updates or epochs, and no new checkpoint.

The quarter and eighth steps pass the unchanged measured PNG/source/brightness
checks on both50-case TRAIN cohorts. Full/half failures remain: six V38 PNG
failures in total, alongside V36's11. The quarter gains are0.068478% and0.116937%,
not recovered-identity percentages. All100 original raw/PNG pairs equal V36.
All20 comparison pages were actually reviewed:100 cases,500 model output cells
and200 input/target cells. A separate checker verifies all700 cells. Visible
eyelids, nostrils, lips and teeth remain weak; passing small steps do not establish
useful additional clarity. TRAIN preservation is insufficient for adoption.

The largest passing TRAIN step,0.25, was fixed before new development inference.
A disposable frozen copy has state
b7aad53d93d4fa58ca826be6162667fff2faf85578da4f421e938b8711bc49ba.
Single-input CPU parity was checked on all100 already-exposed TRAIN cases against
the VM trial. No local fitting, autograd, optimizer or checkpoint writing occurs.
This is a finite displacement, not an800-update trained replacement.

All24 frozen ChokePoint C1 native crops were reviewed on six pages, including
resize, retained Phase3, original DGP, declared CodeFormer and quarter/Auto arms
on identical256 inputs. The saved-output audit verifies all144 cells and every
raw composition/padding/Auto alias. Auto chooses11 quarter and13 resize outputs;
that input-only suggestion is not structure qualification. Quarter remains very
similar to original DGP and adds no convincing native clarity. All24 input-only
usable labels remain; weak model outputs do not make these inputs unusable.
Native evidence is unpaired: no PSNR, SSIM or recovered-identity accuracy.

The separate paired photographic development check retains all520 cases/104
references, with resize and original/quarter DGP. Its runtime is{p['seconds']:.3f}
seconds under1200 internal/1290 external seconds. All520 raw/PNG/vector/metric
rows,17 groups and200 prospective preview cells are independently recomputed.
It records{failures} preservation failures. Degraded high-frequency gain is
{gains['degraded']:.6f}% overall; source-specific figures remain separately reported.
The actual image review covers{pv['cases_reviewed']} prospective preview cases;
all520 numeric checks do not imply all520 images were visually reviewed.
No development-selected scale is substituted and no final identity is opened.

The separate completion display comparison changes only the existing two-pixel
margin. It lowers the measured boundary jump in all28 nonempty assisted cases,
but does not fix central glare, eyewear/covering remnants or anatomy. All32
delivered variants on eight pages were actually inspected, with four prior
input-only rejections retained. The independent audit verifies517 sources,
72 artifacts,160 cells,28 old raw compositions and exact core/visible/protected
bytes. Automatic outputs generated here: zero. Automatic and assisted whole-scope
quality remain separately unqualified. The default-Python missing-cv2 audit
failure is retained; only execution in the project venv corrected that runtime.

The14 current app/checkpoint bindings remain unchanged. The original own-trained
DGP is still primary with Auto/On/Off, mask review, original/mask/result and PNG/
bundle downloads. Existing bundled inline Playwright functional proof is retained;
no UI edit or new browser claim is made here. All original checkpoints, splits,
source/provenance/terms records, caches/local backup and failed gates remain.
The1%-at50/10%-at800 capacity gates are unchanged and are not passed by V38.

These are exposed development photographs or native ChokePoint capture, reported
separately. Source labels are not ethnicity. There are no real Zamboanga samples.
Reserved45 identities/58 crops remain unopened. Useful native restoration, all
seven covering families with separate automatic/assisted quality, independent
final review and the complete reviewed DGP-led app remain required. Goal active.
No VM connection, cleanup, new training launch or historical pilot occurs here.

[V38 return findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_DELIVERED_MARGIN_PROBE_V38_RESULTS.md>)
[Frozen quarter development findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md>)
[Completion margin findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_COMPLETION_MARGIN_FEATHER_V1_RESULTS.md>)

The complete preceding handoff follows. Its V38-prepared-only language records
the earlier milestone and is superseded by this audited human return.

'''
    bindings = {}
    def merge(mapping):
        for name, digest in mapping.items():
            assert name not in bindings or bindings[name] == digest, name
            bindings[name] = digest
    imported = read(ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_return_import.json')
    merge({'outputs/cctv_dgp_delivered_margin_probe_v38_return/' + k: v for k, v in imported['files_sha256'].items()})
    merge(a['bindings_sha256'])
    for folder in [ROOT / 'outputs/cctv_dgp_v38_quarter_single_input_parity_v1', NATIVE, PAIRED]:
        merge(read(folder / 'plan.json')['sources_sha256'])
    merge(read(COMPLETION / 'protocol.json')['sources_sha256'])
    folders = [ANALYSIS, ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_audit_run_v1',
               ROOT / 'outputs/cctv_dgp_v38_quarter_single_input_parity_v1', NATIVE, PAIRED,
               ROOT / 'outputs/cctv_dgp_v38_quarter_paired_development_run_v1', COMPLETION]
    for folder in folders:
        for path in sorted(folder.rglob('*')):
            if path.is_file(): merge({path.relative_to(ROOT).as_posix(): sha(path)})
    names = ['outputs/cctv_dgp_delivered_margin_probe_v38_return_import.json',
             'outputs/cctv_dgp_delivered_margin_probe_v38_independent_audit.json',
             'outputs/cctv-dgp-delivered-margin-probe-v38-results.tar.gz',
             'outputs/cctv-dgp-delivered-margin-probe-v38-results.tar.gz.sha256',
             'outputs/cctv-dgp-delivered-margin-probe-v38-export.json',
             'CCTV_DGP_DELIVERED_MARGIN_PROBE_V38_RESULTS.md', 'CCTV_DGP_V38_QUARTER_DEVELOPMENT_RESULTS.md',
             'CCTV_DGP_COMPLETION_MARGIN_FEATHER_V1_RESULTS.md',
             'scripts/supervise_cctv_dgp_v38_return_audit_v1.py',
             'scripts/analyze_cctv_dgp_delivered_margin_probe_v38_return_v1.py',
             'scripts/record_cctv_dgp_v38_visual_review_v1.py',
             'scripts/verify_cctv_dgp_v38_analysis_and_pages_v1.py',
             'scripts/cctv_dgp_v38_quarter_inference_v1.py',
             'scripts/verify_cctv_dgp_v38_quarter_single_input_parity_v1.py',
             'scripts/run_cctv_dgp_v38_quarter_native_development_v1.py',
             'scripts/verify_cctv_dgp_v38_quarter_native_development_v1.py',
             'scripts/record_cctv_dgp_v38_quarter_native_development_review_v1.py',
             'scripts/run_cctv_dgp_v38_quarter_paired_development_v1.py',
             'scripts/verify_cctv_dgp_v38_quarter_paired_development_v1.py',
             'scripts/supervise_cctv_dgp_v38_quarter_paired_development_v1.py',
             'scripts/record_cctv_dgp_v38_quarter_paired_development_review_v1.py',
             'scripts/completion_margin_feather_v1.py', 'scripts/prepare_completion_margin_feather_v1.py',
             'scripts/run_completion_margin_feather_v1.py', 'scripts/audit_completion_margin_feather_v1.py',
             'scripts/record_completion_margin_feather_v1_review.py',
             'scripts/record_cctv_dgp_v38_development_milestone_v1.py',
             'scripts/verify_cctv_dgp_v38_development_milestone_v1.py',
             (PRIOR / 'milestone.json').relative_to(ROOT).as_posix(),
             (PRIOR / 'independent_closure_audit.json').relative_to(ROOT).as_posix()]
    for name in names: merge({name: sha(ROOT / name)})
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14
    for name, digest in app.items(): assert sha(ROOT / name) == digest
    merge(app)
    # Validate all other evidence before changing the handoff.
    for name, digest in bindings.items():
        assert sha(ROOT / name) == digest, name
        assert time.monotonic() - start < 300
    OUT.mkdir(); (OUT / 'before_docs').mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'; before = handoff.read_bytes()
    saved = OUT / 'before_docs/PROJECT_HANDOFF.md'; saved.write_bytes(before)
    at = before.index(b'\n') + 1; added = addition.encode('utf-8')
    handoff.write_bytes(before[:at] + added + before[at:])
    merge({saved.relative_to(ROOT).as_posix(): sha(saved), 'PROJECT_HANDOFF.md': sha(handoff)})
    assert not any(name in sys.modules for name in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    m = {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
         'previous_milestone_sha256': sha(PRIOR / 'milestone.json'),
         'previous_handoff_path': saved.relative_to(ROOT).as_posix(),
         'new_evidence_sha256': bindings, 'previous_bindings': len(previous['new_evidence_sha256']),
         'document': {'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': sha(handoff), 'addition_bytes': len(added)},
         'V38_independently_audited': True, 'V38_optimizer_updates': 0,
         'V38_passing_TRAIN_displacements': a['jointly_eligible_subset_variants'],
         'native_cases_actually_reviewed': 24, 'paired_numeric_cases': 520,
         'paired_cases_actually_reviewed': pv['cases_reviewed'], 'paired_preservation_failures': failures,
         'completion_display_outputs_reviewed': 32, 'automatic_completion_outputs_generated': 0,
         'V36_and_V38_failed_gates_retained': True, 'training_capacity_pass': False,
         'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls_here': 0,
         'app_changes': False, 'app_adoption': False, 'automatic_quality_qualification': False,
         'assisted_quality_qualification': False, 'independent_final_review': False,
         'goal_complete': False, 'seconds': time.monotonic() - start, 'cap_seconds': 300}
    assert m['seconds'] < 300
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(m, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'bindings': len(bindings), 'seconds': m['seconds'],
                      'milestone_sha256': sha(OUT / 'milestone.json')}), flush=True)


if __name__ == '__main__':
    main()
