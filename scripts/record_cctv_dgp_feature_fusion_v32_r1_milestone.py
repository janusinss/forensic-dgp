"""Publish the separately verified metadata-corrected V32 R1 transfer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREVIOUS = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_milestone'
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_milestone'
PACKET = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r1'
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r1_preparation'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def verify(mapping, aliases=None):
    for name, digest in mapping.items():
        path = (ROOT / (aliases or {}).get(name, name)).resolve()
        assert path.is_relative_to(ROOT) and path.is_file() and not path.is_symlink() and sha(path) == digest, name


def main():
    assert not OUT.exists()
    previous = read(PREVIOUS / 'milestone.json'); verify(previous['new_evidence_sha256'])
    closure = read(PREVIOUS / 'independent_closure_audit.json')
    assert closure['complete'] and closure['milestone_sha256'] == sha(PREVIOUS / 'milestone.json')
    checked = read(PREP / 'independent_packet_audit.json'); prepared = read(PREP / 'preparation.json')
    p = read(PACKET / 'protocol.json')
    assert checked['complete'] and checked['all23_descriptions_match_actual_partition']
    assert checked['unchanged_mathematical_training_behavior'] and checked['first_packet_preserved_unlaunched']
    assert checked['protocol_sha256'] == prepared['protocol_sha256'] == sha(PACKET / 'protocol.json')
    verify(p['local_basis_sha256']); verify(p['superseded_packet_evidence_sha256'])
    assert p['metadata_revision'] == 1 and p['selected_tensors'] == 23 and p['selected_parameters'] == 978243
    assert p['optimizer_updates'] == 800 and p['budgets']['minimum_free_disk_bytes'] == 8 * 1024**3
    assert not (PACKET / 'outputs').exists() and not checked['actual_VM_training_started']
    addition = f'''
**Latest transfer milestone - 7 October 2026: use V32 r1, with consistent23-tensor protocol descriptions.**

The original V32 packet's runtime correctly selected23 tensors, but four inherited
protocol descriptions still referred to12 selected tensors. That unlaunched
packet, archive, successful checks and original documents are retained unchanged.
Use the separately verified V32 r1 packet below. It corrects initial-parity,
normalization, gradient-policy and optimizer descriptions, clarifies the manual
training boundary and routes a distinct directory/export. This is a metadata
revision of the same new finite fusion-partition experiment, never a rerun of
V31 or a stopped candidate. No loss, learning rate, weight decay, clipping rule,
selected tensor, schedule, forward, normalization buffer or scientific gate changes.

R1's11-file/756,048-byte archive passes exact source/protocol/archive comparison,
Python3.10 syntax, its Windows rejection before neural imports, actual prospective
return gradient-schema verification and failed1% gate receipt verification.
The unchanged candidate, policy, cache, training and shell bytes inherit the
previous11 passing regressions, four exact CPU parity comparisons and read-only
Bash parse. Those are inherited proof, not new neural or parser calls. R1 preparation
and audit perform zero neural, gradient or optimizer calls. No VM action occurs.

V32 r1 protocol SHA256: {checked['protocol_sha256']}
V32 r1 execution archive SHA256: {checked['archive_sha256']}

The audited feature-fusion diagnostic remains complete:756 files,280 VM gradient
queries,301,298,844 saved values and200 independent CPU inference outputs; zero
training updates. All11 original fusion plus12 decoder tensors are connected in
both measured TRAIN cohorts/states. This supports a finite fresh-copy test, not
an improved-output claim. Keep V31's50-update/0.694525% failed1% requirement,
stopped checkpoint and every earlier failed gate. Source labels are not ethnicity;
native/unpaired evidence stays separate from paired synthetic measurements.

The same experiment enables eight original fusion convolutions/11 tensors beside
12 active decoder tensors,978,243 parameters total. Backbone, inactive head4 and
all buffers remain fixed. Maximum800 updates; stop at50 unless structure gain
reaches1%. Final TRAIN gates retain10%, all17 preservation groups, both source
gains>=0 and mean-only fraction<=20%. Require8GiB free, idle existing L4/g2-standard-4
and the manual tmux flow at ~/forensic-dgp. Preflight300s/cache900s/fit3600s/
worker4500s, external4800s +30s grace, export900s/external930s +30s grace,
VRAM20GiB and return3GiB remain enforced. Preserve every partial/time/gate failure.

The agent has not uploaded, installed or launched either V32 packet, has performed
no local training and has deleted no research files. The app's original own-DGP
primary checkpoint/design are unchanged. Useful native development outputs,
separate automatic/assisted quality for all seven covering families, independent
final review and the full DGP-led goal remain incomplete. Only passing capacity
and all50 TRAIN preview review can justify the fixed520 paired DEV/24 unpaired
native DEV review; reserved-final pixels stay unopened.

[V32 r1 five manual upload/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_V32_R1_VM.md>)
[Audited feature-fusion findings](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_RESULTS.md>)

The complete previous handoff follows. Its original V32 transfer recommendation
is superseded by this separately verified R1 archive and guide.

'''
    OUT.mkdir(); before_folder = OUT / 'before_docs'; before_folder.mkdir()
    handoff = ROOT / 'PROJECT_HANDOFF.md'; before = handoff.read_bytes()
    (before_folder / handoff.name).write_bytes(before)
    split = before.index(b'\n') + 1
    handoff.write_bytes(before[:split] + addition.encode('utf-8') + before[split:])
    files = [Path(__file__), ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_r1_milestone.py',
             ROOT / 'scripts/prepare_cctv_dgp_feature_fusion_v32_metadata_r1.py',
             ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_metadata_r1.py',
             ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_r1_return.py',
             ROOT / 'scripts/cctv_dgp_feature_fusion_v32_r1_return_audit_template.py',
             ROOT / 'CCTV_DGP_FEATURE_FUSION_V32_R1_VM.md',
             ROOT / 'outputs/cctv-dgp-feature-fusion-v32-r1-execution.tar.gz',
             ROOT / 'outputs/cctv-dgp-feature-fusion-v32-r1-execution.tar.gz.sha256',
             ROOT / 'PROJECT_HANDOFF.md', before_folder / handoff.name,
             PREVIOUS / 'milestone.json', PREVIOUS / 'independent_closure_audit.json']
    files.extend(f for folder in [PACKET, PREP] for f in sorted(folder.rglob('*')) if f.is_file())
    m = {'complete': True, 'datetime_UTC': datetime.now(timezone.utc).isoformat(),
         'recorder_sha256': sha(Path(__file__)), 'previous_milestone_sha256': sha(PREVIOUS / 'milestone.json'),
         'previous_document_locations': {'PROJECT_HANDOFF.md': (before_folder / handoff.name).relative_to(ROOT).as_posix()},
         'new_evidence_sha256': {f.relative_to(ROOT).as_posix(): sha(f) for f in files},
         'document': {'name': handoff.name, 'before_path': (before_folder / handoff.name).relative_to(ROOT).as_posix(),
                      'before_sha256': hashlib.sha256(before).hexdigest(), 'after_sha256': sha(handoff),
                      'addition_bytes': len(addition.encode('utf-8')), 'full_previous_body_preserved': True},
         'protocol_sha256': checked['protocol_sha256'], 'archive_sha256': checked['archive_sha256'],
         'metadata_revision': 1, 'first_packet_preserved': True, 'scientific_behavior_unchanged': True,
         'finite_updates': 800, 'manual_VM_required': True, 'actual_VM_training_started_by_agent': False,
         'local_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
         'app_changes': False, 'native_or_reserved_used': False, 'quality_qualification': False,
         'app_promotion': False, 'goal_status': 'active', 'goal_complete': False}
    with (OUT / 'milestone.json').open('x', encoding='utf-8', newline='\n') as stream: json.dump(m, stream, indent=2)
    print(json.dumps({'complete': True, 'new_bindings': len(m['new_evidence_sha256']), 'milestone_sha256': sha(OUT / 'milestone.json'),
                      'protocol_sha256': m['protocol_sha256'], 'manual_VM_required': True}, indent=2))


if __name__ == '__main__':
    main()
