"""Archive the preceding handoff and bind V39 initial proof and unrun packet."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v39_prepared_milestone_v1'
PRIOR = ROOT / 'outputs/cctv_dgp_v38_return_development_milestone_v1'
INITIAL = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_initial_review_v1'
PREP = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_preparation'
BUNDLE = ROOT / 'outputs/cctv_dgp_spatial_decoder_vm_v39'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    start = time.monotonic(); assert not OUT.exists()
    old = read(PRIOR / 'milestone.json'); prior_audit = read(PRIOR / 'independent_closure_audit.json')
    assert old['complete'] and prior_audit['complete'] and prior_audit['milestone_sha256'] == sha(PRIOR / 'milestone.json')
    for name, digest in old['new_evidence_sha256'].items():
        assert sha(ROOT / name) == digest, name
        assert time.monotonic() - start < 300
    r = read(INITIAL / 'results.json'); audit = read(INITIAL / 'independent_initial_audit.json')
    p = read(BUNDLE / 'protocol.json'); packet = read(PREP / 'independent_packet_audit.json')
    runner = read(PREP / 'independent_packet_audit_r1_execution.json')
    assert r['complete'] and audit['complete'] and audit['all50_raw_pairs_exact'] and audit['all100_PNG_compositions_exact']
    assert packet['complete'] and runner['complete'] and runner['all_scientific_and_archive_and_command_check_AST_unchanged']
    assert packet['archive_files'] == 142 and packet['assets_verified'] == 141 and packet['local_basis_verified'] == 152
    assert p['optimizer_updates'] == p['parameter_updates'] == p['epochs'] == 0
    assert not (BUNDLE / 'outputs').exists() and not (ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_return').exists()
    addition = '''
**DGP milestone — 8 October 2026: distinct V39 spatial decoder initial proof audited; zero-update manual VM diagnostic prepared.**

The V38 quarter-step development failure remains binding: one FFHQ-thumbnail
compound-profile ArcFace regression on520 paired cases, and no convincing added
clarity in the24 native crops. Smaller steps are not selected from development
outcomes. All original V36/V38 failures and prior capacity stops remain. The
following V38 milestone preserves that evidence and the separate completion
display comparison; neither is an adoption or independent-final quality pass.

The next hypothesis tests a different learned spatial path within our own DGP.
Observed RGB, retained DGP RGB,64 lateral and32 reconstruction feature channels
form102 channels at256. A new16/32-channel two-scale decoder uses local gates,
channel normalization and a spatial shortcut. It subtracts a separately frozen
initial decoder response, mean-centers the half-tanh residual on observed pixels,
adds it to the original DGP and clamps. Outside-support camera pixels are exact.
Original weights/statistics stay frozen. No pretrained restorer weights/targets,
display sharpening or input/source/person/target-conditioned routing is used.

The17,952 parameters in57 new tensors are untrained. A separately transferred
CPU seed asset is initialization, with zero local learning. The46.123-second
initial proof covers all50 exposed V27 TRAIN inputs: exact fresh raw and PNG
parity, partial-support preservation, five pre-neural invalid-input rejections
and actual Windows rejection of learning enablement. It makes101 original DGP
and51 forwards through each decoder copy. No local gradient, optimizer or trained
checkpoint occurs. The9.282-second independent checker verifies90 sources,
203 artifacts, all50 raw pairs/100 PNG compositions and57 seed tensors. Ten
metadata-selected frozen CPU replays are exact; historical cache maximum raw
difference is2.38419e-6 within1e-5. Gradient connectivity is not established.

Primary NAFNet research motivates testing simple gated restoration blocks; it
does not guarantee forensic identity, CCTV usefulness or the cause of our
failures. This is a different experimental spatial decoder within the retained
own-trained feedforward DGP, with disclosed lineage. It uses no NAFNet weights
and makes no published-architecture or transferred benchmark-performance claim.

V39 is a separate70-query, zero-update gradient proof on the same50 photographic
TRAIN cases, ten paired-five-profile batches and seven original loss terms.
Every57 new tensor must have finite nonzero summed improvement gradients.
All four initial preservation values/gradients must be exactly zero; original
DGP, frozen initial decoder and recognizer must remain unchanged. There are no
optimizer/parameter updates, epochs or trained checkpoint writers. A returned
proof and independent audit are needed before a separate finite training recipe.
The1%-at50/10%-at800 and17-group/source/brightness gates remain unchanged.

The self-contained packet has142 regular files,235,212,632 uncompressed bytes
and213,203,706 compressed bytes. It copies verified original weights, code,
untrained seed and existing TRAIN input/target/support/cache assets. Only the
existing VM venv is needed; no deleted historical worker or cache is recreated
or executed. Protocol SHA256:
d1775d67c8c80ddad9372e5af47be6ce987ca3fed7749defcdbdc190fb1153a4.
Execution archive SHA256:
d2093fc136d5305db5071a53be7fbdb0722e5505026051c68501b57959cc29bc.

The4.874-second packet checker verifies141 assets/152 local bindings, every
archive member, all50 cases/10 references/57 tensor layouts, AST-exact original
loss/filter definitions and unchanged quality gates. Three scope rejections,
actual Windows pre-neural rejection, nine unsafe archive cases and seven saved
gradient rejection fixtures pass. Nineteen Python3.10 sources, read-only Bash
syntax and five single-remote-source gcloud commands are checked. Synthetic
fixture arrays are not VM gradient or training evidence.

Two local audit-environment failures remain: Git Bash's sandbox signal pipe
restriction and Python313 TemporaryDirectory access denial. The Bash -n parser
passes without executing the script. A separate5.025-second R1 runner changes
only fixture directory creation to inherited workspace access and partial-log
routing; inverse AST comparison verifies all scientific checks unchanged.
Original checker, packet, protocol, gates and failure receipts are preserved.
The retained fixture directory is explicitly synthetic, separate from VM data.

V39 is PREPARED, NOT RUN. Follow the five manual Google Cloud SDK/SSH/tmux steps
on the existing idle NVIDIA L4/g2-standard-4 at ~/forensic-dgp. Require2GiB free.
Prospective estimate2–6 minutes diagnostic plus1–2 minutes export; this is not
measured V39 VM timing. Worker600s/external630s with30s kill grace, export300s/
external330s with30s grace, allocated VRAM20GiB and uncompressed return192MiB
are bounded. Any stop exports evidence; no automatic follow-on or retry.

The14 app/checkpoint bindings and existing design are unchanged. The original
trained DGP remains primary with Auto/On/Off, mask review, original/mask/result,
PNG and bundle downloads. Prior inline Playwright functional proof is retained;
no new UI or browser claim is made. Completion remains separate and unqualified
for automatic/assisted masks, sunglasses, strong lens glare, hands, obstructing
hair, scarves and objects. Preserve clear glasses/non-obstructing hair and
visible appearance; request clearer/less-covered crops when information is
insufficient, not as an explanation for poor model outputs.

TRAIN photographs, paired520 development and native24 unpaired CCTV are separate.
No native PSNR/SSIM, ethnicity, hidden-identity or Zamboanga-performance claim is
made. There are no real Zamboanga samples. Reserved45 identities/58 crops remain
unopened. Checkpoints, source/split/terms/provenance records, caches/local backup
and gate failures remain. Useful development outputs, independent final review,
all seven covering families and the reviewed DGP-led app remain required. Goal
active/incomplete. No VM connection, cleanup, diagnostic or training was launched
by the agent in this milestone.

[V39 five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_DECODER_V39_VM.md>)
[V39 spatial design and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V38_SPATIAL_DECODER_REVIEW.md>)
[Independent packet audit and environment correction](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_spatial_decoder_v39_preparation/independent_packet_audit_r1_execution.json>)

The complete preceding handoff follows. No preceding failure or scope is removed.

'''
    bindings = {}
    def merge(mapping):
        for name, digest in mapping.items():
            assert name not in bindings or bindings[name] == digest, name
            bindings[name] = digest
    merge(read(INITIAL / 'plan.json')['sources_sha256']); merge(p['local_basis_sha256'])
    for folder in [INITIAL, BUNDLE, PREP / 'r1_synthetic_fixture_workspace']:
        for path in sorted(folder.rglob('*')):
            if path.is_file(): merge({path.relative_to(ROOT).as_posix(): sha(path)})
    # The first Python temporary directory is inaccessible under this sandbox;
    # retain its failure/path receipt, and enumerate only the known audit files.
    prep_names = ['prepared.json', 'Bash_sandbox_syntax_failure.json', 'Bash_readonly_syntax.json',
                  'packet_audit_sandbox_failure.json', 'Windows_pre_neural_rejection.txt',
                  'Windows_pre_neural_rejection_r1.txt', 'independent_packet_audit.json',
                  'independent_packet_audit_r1_execution.json']
    for name in prep_names:
        path = PREP / name; merge({path.relative_to(ROOT).as_posix(): sha(path)})
    names = ['CCTV_DGP_POST_V38_SPATIAL_DECODER_REVIEW.md', 'CCTV_DGP_SPATIAL_DECODER_V39_VM.md',
             'scripts/cctv_dgp_spatial_decoder_v39.py', 'scripts/cctv_dgp_spatial_decoder_v39_vm.py',
             'scripts/review_cctv_dgp_spatial_decoder_v39_initial_v1.py', 'scripts/verify_cctv_dgp_spatial_decoder_v39_initial_v1.py',
             'scripts/prepare_cctv_dgp_spatial_decoder_v39.py', 'scripts/verify_cctv_dgp_spatial_decoder_v39_packet.py',
             'scripts/verify_cctv_dgp_spatial_decoder_v39_packet_r1.py', 'scripts/audit_cctv_dgp_spatial_decoder_v39_return.py',
             'scripts/record_cctv_dgp_v39_prepared_milestone_v1.py', 'scripts/verify_cctv_dgp_v39_prepared_milestone_v1.py',
             'outputs/cctv-dgp-spatial-decoder-v39-execution.tar.gz', 'outputs/cctv-dgp-spatial-decoder-v39-execution.tar.gz.sha256',
             (PRIOR / 'milestone.json').relative_to(ROOT).as_posix(),
             (PRIOR / 'independent_closure_audit.json').relative_to(ROOT).as_posix()]
    for name in names: merge({name: sha(ROOT / name)})
    app = read(ROOT / 'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']
    assert len(app) == 14
    for name, digest in app.items(): assert sha(ROOT / name) == digest
    merge(app)
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
         'previous_milestone_sha256': sha(PRIOR / 'milestone.json'), 'previous_handoff_path': saved.relative_to(ROOT).as_posix(),
         'new_evidence_sha256': bindings, 'previous_bindings': len(old['new_evidence_sha256']),
         'document': {'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': sha(handoff), 'addition_bytes': len(added)},
         'V38_development_failure_retained': True, 'V39_initial_parity_exact_cases': 50,
         'V39_spatial_decoder_untrained_parameters': 17952, 'V39_gradient_connectivity_established': False,
         'V39_packet_prepared_and_verified': True, 'V39_prepared_only': True, 'V39_VM_launches': 0,
         'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls_here': 0,
         'app_changes': False, 'app_adoption': False, 'training_capacity_pass': False,
         'automatic_quality_qualification': False, 'assisted_quality_qualification': False,
         'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic() - start, 'cap_seconds': 300}
    assert m['seconds'] < 300
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(m, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'bindings': len(bindings), 'seconds': m['seconds'],
                      'milestone_sha256': sha(OUT / 'milestone.json')}), flush=True)


if __name__ == '__main__':
    main()
