"""Publish the audited update-two quality stop; keep historical bytes exact."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/publication_v1'
DOCS = ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'PROJECT_HANDOFF.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
GUIDE = 'CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_VM.md'
PREFIX = '''# 23 October 2026: original-u64 pilot audited; quality stop at update two

[Audited return](CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_RETURN_REVIEW.md).
[Next source-only measurement design](CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_DESIGN.md).

STATUS: EXECUTED_GATE_ALIGNED_PILOT_V1_AUDITED_QUALITY_STOP_AT2.
The user's manual L4 run applied two updates, then stopped at its scheduled
quality bank. Worker exit2, no timeout, about256 seconds. The complete actual
prefix authenticates:300 full forwards,256 gradient queries,two parameter
applications,558 operation pairs and1,679 journal events,with no pending unknown
operations. Archive1,166,765,803 bytes SHA256
`005167e9be87c37efab55382652b9e5776466a61ef4f201f485fdafd527281d0`;
protocolSHA256 `35a5cfd877a97355622c6c69d9792fd9238ccf312335dd679081a8636ba23b48`.
All1,643 archived members and26 static bindings authenticate. Final export
process exit receipts are outside the archive by design; no finalexit claim
is inferred from valid download/transport.

Update1 passed its bank. Update2 failed the localized central128 PNG adjacent
endpoint against originalu64 and prioru1:0.01697144654169071 versus
0.01697094396257641 and0.016970657491969073. This is +0.0029614% against
originalu64,+0.00464949% againstu1. RAW localized error decreased; original
inherited guards passed. The additional PNG nonincrease requirement correctly
stopped the run; its zero threshold remains unchanged. Terminal4 was not reached.
Do not rerun/resume this pilot,append epochs,or select its earlieru1 retrospectively.

All256 actual gradients and both rounded FP32 updates independently reconstruct;
all539 frozen R2/U96 tensors stay bitwise exact. Complete575 initial/failed states,
head0/1/2,RNG,sampler,null optimizer and failures are retained. Independent pixel
review passes4,138 checks,including300 exactRAW/floorPNG pairs,256 scalar receipts
andthree56-case banks. Root inspected all20 exactRGB sheets;296 distinct output
cells plusfour hash-identical repeat aliases cover all300 calls. These small
updates do not establish useful visible improvement.

Sparse PNG code crossings explain the measured failure:only1,132/393,216 local
RGB components changed fromu1 tou2,each byone code. Their residualedge increments
sum to+2,321 and decrements-2,164,net+157. The savedPNG mismatch is independently
reconstructed without changing quantizer,images,gates or quality tolerance.

Next work measures RAW and PNG-surrogate signals at exactoriginalu64 before
choosing any training change:72 forwards,256 queries,16 pooled vectors,zero
parameter applications/optimizers. Any custom identity backward is explicitly a
biased surrogate,not the true PNG derivative; forward must match actualsavedPNG
decodedFP32 exactly. No new measurement packet or manual guide is ready at this
publication. New source implementation remains unsealed until independently
reviewed. Existing8-vector solver must not silently drop domains from16vectors.

Originalu64 remains retained challenger;U96 remains experimental app incumbent.
No winner or automatic goal advancement. BroaderDEV failures,checkpoints,splits,
provenance and every gate failure remain binding. ReviewedProtect/Apply/bypass
stays separate;automatic blurmatching and completion development remain deferred.
Native/wild staysunpaired;FINAL remainsunopened/execution-unfrozen,reviewerspending.
All actual training remains manualL4. Full goal ACTIVE/incomplete. Historical
prepared statuses below are superseded; exact prior bytes follow.

---

'''
GUIDE_PREFIX = '''# Executed pilot: quality stop at update two; do not rerun

23 October 2026. This manual run is completed and independently audited.
Worker exit2:300 forwards,256 gradient queries,two actual updates; localized
PNG structure retention failed. Both failed and earlier states remain retained;
neither is selected as a successful successor. Do not resume or append updates.

Read the [audited return](CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_RETURN_REVIEW.md)
and [next source-only design](CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_DESIGN.md).
The upload/tmux instructions below are historical provenance for an executed
pilot. They must not be repeated. No next measurement transfer guide is ready.
U96 remains app incumbent; originalu64 is retained. Full goal ACTIVE/incomplete.
Exact preceding guide bytes follow.

---

'''


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    if (OUT/'publication_receipt_v1.json').exists():
        raise ValueError('Retain prior publication; do not overwrite')
    requirements = {
        'outputs/codeformer_u64_gate_aligned_pilot_v1_return_transport/audit_v1.json': 'complete',
        'outputs/codeformer_u64_gate_aligned_pilot_v1_return_updates/receipt_v1.json': 'saved_state_gradient_update_audit_pass',
        'outputs/codeformer_u64_gate_aligned_pilot_v1_return_pixels/audit_v1.json': 'independent_saved_pixel_integrity_pass',
        'outputs/codeformer_u64_gate_aligned_pilot_v1_png_diagnosis/receipt_v1.json': 'saved_PNG_threshold_diagnosis_pass',
        'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/visual_readback_v1.json': 'independent_visual_cell_readback_pass',
        'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/root_visual_review_v1.json': 'root_personally_inspected_all_planned_sheets',
        'outputs/codeformer_u64_gate_aligned_pilot_v1_return_audit/report_design_readback_v1.json': 'review_pass',
    }
    consumed = []
    for name, key in requirements.items():
        path = ROOT / name
        value = json.loads(path.read_bytes())
        if value.get('complete') is not True or value.get(key) is not True or value.get('unresolved_blockers'):
            raise ValueError('Unresolved audit: ' + name)
        consumed.append(pin(path))
    for name in ('CCTV_CODEFORMER_U64_GATE_ALIGNED_PILOT_V1_RETURN_REVIEW.md',
                 'CCTV_CODEFORMER_U64_PNG_SURROGATE_MEASUREMENT_V1_DESIGN.md'):
        consumed.append(pin(ROOT / name))
    previous=json.loads((ROOT/'outputs/codeformer_u64_gate_aligned_pilot_v1_preparation/publication_v1/publication_receipt_v1.json').read_bytes())
    oldpins={row['after']['path']:row['after'] for row in previous['changes']}
    guidepin=json.loads((ROOT/'outputs/codeformer_u64_gate_aligned_pilot_v1_preparation/guide_review_v1.json').read_bytes())['guide']
    oldpins[GUIDE]=guidepin
    pending=[]
    for name in (*DOCS,GUIDE):
        path=ROOT/name;backup=OUT/(name+'.before')
        before=backup.read_bytes() if backup.exists() else path.read_bytes()
        expected=oldpins[name]
        if len(before)!=expected['bytes'] or hashlib.sha256(before).hexdigest()!=expected['sha256']:
            raise ValueError('Exact prior publication required before recovering: '+name)
        prefix=(GUIDE_PREFIX if name==GUIDE else PREFIX).encode('utf8')
        current=path.read_bytes()
        if current not in (before,prefix+before):
            raise ValueError('Concurrent/partial document differs; preserve without overwrite: '+name)
        pending.append((path,before,expected,prefix,current))
    OUT.mkdir(exist_ok=True)
    changes = []
    for path, before, before_pin,prefix,current in pending:
        if path.read_bytes()!=current:
            raise ValueError('Concurrent document change')
        backup = OUT / (path.name + '.before')
        if not backup.exists():
            with backup.open('xb') as stream:
                stream.write(before)
        if current==before:
            staged=OUT/(path.name+'.new')
            with staged.open('xb')as stream:
                stream.write(prefix+before);stream.flush();os.fsync(stream.fileno())
            if staged.read_bytes()!=prefix+before:raise ValueError('Staged document differs')
            os.replace(staged,path)
        if path.read_bytes()[len(prefix):] != backup.read_bytes():
            raise ValueError('Historical byte suffix differs')
        changes.append({'before': before_pin, 'backup': pin(backup), 'after': pin(path),
                        'prefix_bytes': len(prefix), 'prior_bytes_preserved_exactly': True})
    if any(pin(ROOT / row['path']) != row for row in consumed):
        raise ValueError('Consumed evidence changed during publication')
    result = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(),
              'status': 'EXECUTED_GATE_ALIGNED_PILOT_V1_AUDITED_QUALITY_STOP_AT2',
              'changes': changes, 'consumed_before_after_exact': consumed,
              'all_historical_bytes_retained': True, 'failed_states_and_prior_models_retained': True,
              'new_measurement_packet_ready': False, 'no_local_training_or_VM_calls': True,
              'interrupted_R0_recovered_from_exact_prefix_and_prior_byte_guards':True,
              'model_qualified': False, 'automatic_goal_advancement_triggered': False,
              'full_goal': 'ACTIVE/incomplete', 'source': pin(Path(__file__).resolve())}
    with (OUT / 'publication_receipt_v1.json').open('x', encoding='utf8', newline='\n') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print({'complete': True, 'documents': len(changes), 'model_qualified': False})


if __name__ == '__main__':
    main()
