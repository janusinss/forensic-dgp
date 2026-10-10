"""Record verified preparation, preserving the preceding handoff byte-for-byte."""
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_residual_epochs_v42_milestone'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text())


def main():
    assert not OUT.exists(); base = ROOT/'outputs/cctv_dgp_residual_epochs_v42_preparation'
    p = read(base/'prepared.json'); a = read(base/'independent_packet_audit.json')
    s = read(base/'all_reference_scale_support.json'); bash = read(base/'bash_syntax.json')
    assert p['complete'] and a['complete'] and s['complete'] and bash['complete']
    assert p['protocol_sha256'] == a['protocol_sha256'] == s['protocol_sha256']
    assert a['CPU_initial_parity_cases'] == 50 and a['training_or_gradient_calls'] == 0
    assert a['initial_checker_failure_retained'] and bash['external_read_only_result']['exit_code'] == 0
    history = read(ROOT/'outputs/cctv_dgp_current_training_review_v1/training_history.json')
    for name, digest in history['source_bindings'].items(): assert sha(ROOT/name) == digest
    names = ['CCTV_DGP_RESIDUAL_SUPERVISION_V1_REVIEW.md', 'CCTV_DGP_RESIDUAL_EPOCHS_V42_VM.md',
        'CCTV_DGP_RESIDUAL_EPOCHS_V42_PREPARATION_REVIEW.md',
        'outputs/cctv_dgp_residual_supervision_v1/analysis.json',
        'outputs/cctv_dgp_residual_supervision_v1/independent_audit.json',
        'outputs/cctv_dgp_residual_epochs_vm_v42/protocol.json',
        'outputs/cctv-dgp-residual-epochs-v42-execution.tar.gz',
        'outputs/cctv-dgp-residual-epochs-v42-execution.tar.gz.sha256',
        'scripts/audit_cctv_dgp_residual_epochs_v42_return.py',
        'scripts/verify_cctv_dgp_residual_epochs_v42_packet_r1.py']
    names += [q.relative_to(ROOT).as_posix() for q in base.glob('*.json')]
    bindings = {name: sha(ROOT/name) for name in names}
    assert bindings['outputs/cctv_dgp_residual_epochs_vm_v42/protocol.json'] == p['protocol_sha256']
    assert bindings['outputs/cctv-dgp-residual-epochs-v42-execution.tar.gz'] == p['archive_sha256']
    OUT.mkdir(); (OUT/'before').mkdir()
    handoff = ROOT/'PROJECT_HANDOFF.md'; before = handoff.read_bytes()
    shutil.copyfile(handoff, OUT/'before/PROJECT_HANDOFF.md')
    prefix = f'''**Training preparation milestone — 9 October 2026: V42 direct correction study ready for manual transfer; no training started.**

The 145-case pre-clamp correction arithmetic audit passes without local training.
Its NumPy/OpenCV checker rechecks1,740 terms and356 source bindings. The distinct
V42 recipe supervises observed, landmark and RGB-pyramid corrections directly,
with zero correction for clear inputs, identity supervision and retained
regression penalties. The current app DGP, stored normalization and fixed initial
decoder remain frozen; only our own17,952-parameter spatial decoder learns.
These are spatial-decoder epochs, not five extra epochs of the frozen encoder.

The prepared study has five complete781-reference epochs/3,905 updates and
19,525 photographic TRAIN exposures, comparing0/50/781/1,562/3,905 snapshots.
Current checkpoint initialization, all original splits and failures remain.
No native/DEV/final input is optimized. Unchanged1% early/10% final structure,
all17 preservation groups, both sources and20% brightness-only limit apply
separately to raw and PNG outputs. Stop and export the first failed requirement.
AdamW3e-4 is reduced to9e-5 after update1,562. Rate/weight choices are declared,
not established optimal; new improvement gradients are checked on the VM before
constructing an optimizer. No failed recipe is resumed or retried automatically.

Independent preparation verifies all5,493 assets/archive contents, five complete
schedules and50 exact initial CPU inference outputs. Every781 reference passes
three scale-support checks. The R1 checker corrects an AST-variable test; the
original checker/failure remain and the packet/recipe/archive do not change.
Read-only Bash syntax passes outside the Windows signal-pipe sandbox restriction;
both outcomes are retained. No local gradient, backward or optimizer call ran.

Manual training requires the idle existing L4/g2-standard-4 inside tmux and8GiB
free after installation. Cache900s, fit6,300s, worker7,200s/external7,230s,
export900s/external930s,30s kill grace,20GiB peak allocated VRAM,3.5GiB return and
512MiB protected disk reserve are enforced. Timing and output/export projections
have1.25 safety factors. Full optimizer/scheduler/RNG/schedule state is exported
at snapshots/stops for a later reviewed migration, never automatic failed resume.

The prospective auditor checks all delivered PNG pixel metrics, raw aggregates,
saved vectors and50 raw CPU replays per snapshot. Other raw arrays are hashed,
so a full raw-value audit is not claimed. Useful native development outputs,
independent final review and all seven automatic/assisted covering families
remain unqualified. The existing app and accepted weights are unchanged.
No VM connection, training or promotion occurred. Full goal active/incomplete.

Manual five-step guide: CCTV_DGP_RESIDUAL_EPOCHS_V42_VM.md
Preparation review: CCTV_DGP_RESIDUAL_EPOCHS_V42_PREPARATION_REVIEW.md
Protocol SHA256: {p['protocol_sha256']}
Execution archive SHA256: {p['archive_sha256']}
Execution archive bytes: {p['archive_bytes']:,}
Independent packet audit: outputs/cctv_dgp_residual_epochs_v42_preparation/independent_packet_audit.json
The exact preceding handoff is archived at
outputs/cctv_dgp_residual_epochs_v42_milestone/before/PROJECT_HANDOFF.md.
Its complete bytes are retained below.


'''
    handoff.write_bytes(prefix.encode('utf-8')+before)
    record = {'complete': True, 'date': '2026-10-09', 'bindings_sha256': bindings,
        'historical_source_bindings': history['source_bindings'],
        'preceding_handoff_sha256': sha(OUT/'before/PROJECT_HANDOFF.md'),
        'new_handoff_sha256': sha(handoff), 'exact_previous_handoff_suffix': True,
        'training_launched': False, 'VM_connections': 0, 'local_optimizer_updates': 0,
        'app_promoted': False, 'goal_complete': False}
    with (OUT/'record.json').open('x', encoding='utf-8') as f: json.dump(record, f, indent=2); f.write('\n')
    print({'complete': True, 'milestone': str(OUT), 'training_launched': False})


if __name__ == '__main__': main()
