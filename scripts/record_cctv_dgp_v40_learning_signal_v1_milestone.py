"""Close saved-weight analysis and prepared-only diagnostic without replacing history."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from completion_feature_fusion_off_v1_common import ROOT, sha, read, write

OUT = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_milestone'
PRIOR = ROOT/'outputs/cctv_dgp_v40_return_completion_milestone'
PREP = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_preparation'
BUNDLE = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm'
ANALYSIS = ROOT/'outputs/cctv_dgp_v40_saved_learning_analysis'


def main():
    start = time.monotonic(); assert not OUT.exists()
    previous = read(PRIOR/'milestone.json'); prior_audit = read(PRIOR/'independent_closure_audit.json')
    assert prior_audit['complete'] and prior_audit['milestone_sha256'] == sha(PRIOR/'milestone.json')
    for name, digest in previous['new_evidence_sha256'].items():
        assert sha(ROOT/name) == digest, name
        assert time.monotonic()-start < 600
    p = read(BUNDLE/'protocol.json'); prepared = read(PREP/'prepared.json'); packet = read(PREP/'independent_packet_audit.json')
    saved = read(ANALYSIS/'results.json'); checked = read(ANALYSIS/'independent_analysis_audit.json')
    assert prepared['complete'] and packet['complete'] and packet['protocol_sha256'] == sha(BUNDLE/'protocol.json') == prepared['protocol_sha256']
    assert packet['V40_decoder_inverse_source_and_AST_exact'] and packet['V40_losses_filters_normalizers_and_gates_exact']
    assert saved['complete'] and checked['complete'] and checked['results_sha256'] == sha(ANALYSIS/'results.json')
    assert saved['tensors_changed'] == 57 and saved['stop_and_snapshot50_tensors_exact'] and saved['normalizers_exact']
    assert p['component_gradient_calls'] == 280 and p['optimizer_updates'] == p['parameter_updates'] == p['epochs'] == p['backwards'] == 0
    assert not (ROOT/'outputs/cctv-dgp-v40-learning-signal-v1-results.tar.gz').exists()
    assert not (ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return').exists()
    assert not (BUNDLE/'outputs').exists()
    OUT.mkdir(); (OUT/'before_docs').mkdir()
    before = (ROOT/'PROJECT_HANDOFF.md').read_bytes(); (OUT/'before_docs/PROJECT_HANDOFF.md').write_bytes(before)
    section = f'''
**Research milestone — 8 October 2026: V40 saved-weight analysis audited; endpoint diagnostic prepared, not run.**

V40's failed50-update structure requirement remains unchanged:0.0106079234%
gain against1%. Every original/stopped checkpoint, failed gate and visual review
remains. No unchanged rerun, resume, weaker gate or new app model is adopted.

Independent saved-state arithmetic verifies all17,952 parameter differences;
all57 tensors changed and the stopped checkpoint equals snapshot50 exactly.
The float64 displacement norm is0.21028403907940274. Initial fixed50 landmark,
observed-detail and pixel gradient alignment with the finite change is weak
(0.0204007062,0.0300057107,0.0598935766); initial preservation gradients are zero.
This establishes weight movement, not useful learning or a unique cause. No
optimizer moments/per-step losses were saved; they are not reconstructed.
Analysis0.792s and independent arithmetic0.079s use no neural/gradient calls.

A distinct endpoint diagnostic is now independently verified but unrun. It
compares original/stopped50 decoder states on100 photographic TRAIN cases:
fixed50 not used by the first50 updates and firstfive optimized references per
source in the frozen schedule, each with clear plus four degraded profiles.
Selection is metadata-only; these cohorts are not held-out evaluation or CCTV.
Source labels do not establish ethnicity. No native or final pixels are opened.

Architecture inverse source/AST, all seven losses, initial50 normalization,
original/fixed decoder and recognizer and every preservation gate stay exact.
There are280 component-gradient queries, zero optimizer/parameter updates,
backwards or epochs. Endpoint norms/conflicts/directions are diagnostic, not an
AdamW trajectory, qualification or the next actual-training recipe. All57 tensor
partitions, all200 raw/300 PNG compositions and40 frozen CPU replay cases have a
prospective independent return checker; it performs no local differentiation.

Protocol SHA256:{prepared['protocol_sha256']}
Execution archive SHA256:{prepared['archive_sha256']}
Archive {prepared['archive_bytes']:,}bytes / {packet['archive_files']}regular files.
The verifier checks metadata/portable assets, Python3.10 and read-only Bash,
Windows rejection before neural imports, three wrong scopes, nine unsafe
archives, nine corrupted saved-array cases and five single-source gcloud commands.
Synthetic fixtures are boundary tests, not VM evidence. No VM connection or
diagnostic/training execution occurs here. Human upload/install/tmux/run/download
remains required on the existing L4/g2-standard-4; only its venv is needed.

Require2GiB free. Worker600s/external630s+30s kill grace; export300s/external330s
+30s grace; allocated VRAM20GiB; return512MiB uncompressed. Estimated diagnostic
2–6minutes plus1–3minutes export. Historical free space is not a current reading;
no cleanup occurs. Any diagnostic failure must remain and be downloaded too.
The original V40 failure and completion fusion-off rejection remain binding.

All14 DGP-primary app/checkpoint/design bindings remain exact. Prior documents,
original checkpoints, data/splits/provenance, caches/local backup and all gate
failures remain. The full earlier handoff is archived and preserved below.
V38 development preservation/native-quality failures remain; reserved45
identities/58 crops remain unopened. No native PSNR/SSIM, exact hidden identity,
ethnicity or Zamboanga performance is claimed. Useful native restoration and
all visible features, seven covering families with separate automatic/assisted
quality, independent final review and the qualified DGP-led app flow are still
required. Goal active/incomplete.

[Five manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM.md>)
[Evidence and diagnostic limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_REVIEW.md>)

'''
    addition = section.encode('utf-8'); at = before.index(b'\n')+1
    (ROOT/'PROJECT_HANDOFF.md').write_bytes(before[:at]+addition+before[at:])
    bindings = dict(p['local_basis_sha256'])
    bindings.update(read(ROOT/'outputs/completion_feature_fusion_off_v1/protocol.json')['app_preservation_sha256'])
    for folder in [BUNDLE, PREP, ANALYSIS]:
        for path in sorted(folder.rglob('*')):
            if path.is_file(): bindings[path.relative_to(ROOT).as_posix()] = sha(path)
    names = ['PROJECT_HANDOFF.md', 'CCTV_DGP_V40_LEARNING_SIGNAL_V1_REVIEW.md', 'CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM.md',
             'outputs/cctv-dgp-v40-learning-signal-v1-execution.tar.gz', 'outputs/cctv-dgp-v40-learning-signal-v1-execution.tar.gz.sha256',
             'scripts/record_cctv_dgp_v40_learning_signal_v1_milestone.py', 'scripts/verify_cctv_dgp_v40_learning_signal_v1_milestone.py']
    for name in names: bindings[name] = sha(ROOT/name)
    for name, value in bindings.items():
        assert sha(ROOT/name) == value, name
        assert time.monotonic()-start < 600
    write(OUT/'milestone.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'previous_milestone_sha256': sha(PRIOR/'milestone.json'), 'previous_handoff_path': (OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
        'document': {'before_sha256': hashlib.sha256(before).hexdigest(), 'addition_bytes': len(addition)}, 'new_evidence_sha256': bindings,
        'saved57_tensor17952_value_analysis_independently_audited': True, 'diagnostic_prepared_not_run': True,
        'V40_failure_preserved': True, 'DGP_primary_app_bindings_unchanged': True, 'VM_calls': 0, 'local_neural_calls_here': 0,
        'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'training_started_here': False, 'app_promotion': False,
        'automatic_quality_qualification': False, 'assisted_quality_qualification': False, 'independent_final_review': False,
        'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600})
    print({'complete': True, 'bindings': len(bindings), 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
