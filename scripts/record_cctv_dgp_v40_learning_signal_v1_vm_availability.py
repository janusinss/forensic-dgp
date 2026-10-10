"""Retain a stopped-VM observation and manual next action without changing research."""
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import time
from completion_feature_fusion_off_v1_common import ROOT, sha, read, write

OUT = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm_availability_milestone'
PRIOR = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_milestone'
OBS = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm_readiness'


def main():
    start = time.monotonic(); assert not OUT.exists()
    previous = read(PRIOR/'milestone.json'); closed = read(PRIOR/'independent_closure_audit.json')
    assert closed['complete'] and closed['milestone_sha256'] == sha(PRIOR/'milestone.json')
    for name, digest in previous['new_evidence_sha256'].items(): assert sha(ROOT/name) == digest, name
    status = read(OBS/'instance_status.json'); transport = read(OBS/'transport.json'); api = read(OBS/'instance_status_transport.json')
    assert status['status'] == 'TERMINATED' and status['name'] == 'forensic-dgp-thesis' and status['machineType'].endswith('/g2-standard-4')
    assert api['exit_code'] == 0 and api['read_only'] and not api['VM_started'] and not api['training_launched']
    assert transport['exit_code'] == 1 and not transport['complete'] and transport['TLS_validation_enabled']
    assert transport['stdout_sha256'] == sha(OBS/'stdout.log') and transport['stderr_sha256'] == sha(OBS/'stderr.log')
    assert transport['files_removed'] == transport['model_or_gradient_calls'] == transport['optimizer_updates'] == 0
    assert not transport['diagnostic_or_training_launched'] and not (OBS/'readiness.json').exists()
    OUT.mkdir(); (OUT/'before_docs').mkdir()
    before = (ROOT/'PROJECT_HANDOFF.md').read_bytes(); (OUT/'before_docs/PROJECT_HANDOFF.md').write_bytes(before)
    section = '''
**VM availability — 8 October 2026: existing VM is stopped; diagnostic remains manual and unrun by the agent.**

The fresh read-only Google Cloud API query reports forensic-dgp-thesis in
us-central1-a as TERMINATED, machine type g2-standard-4. The initial maintenance
connection stops at TLS validation. A bounded retry uses the existing Windows
CA bundle and pinned historical host key with TLS enabled, then SSH closes
through IAP. The separate API status query succeeds. Both failures are retained;
no VM start, guest deletion, diagnostic or actual training is performed here.

Fresh disk/GPU/tmux/venv and guest research-file integrity cannot be verified
while stopped. No cleanup or recovered-space claim is made. The next action is
manual VM start when the human is ready, followed by the existing five verified
upload/install/tmux/run/download steps. The supplementary availability guide
includes the exact startup, temporary SDK Shell CA setting and status command.
No global SDK setting or frozen packet/guide/protocol is changed.

The 266-file/239.5MiB endpoint packet remains independently verified. It has
280 component-gradient queries, zero optimizer updates/backwards/epochs, an
existing-L4/idle-GPU/2GiB-free guard,600s worker and finite export/VRAM/size stops.
New actual learning continues under the human's manual VM workflow. The full
prepared-diagnostic closure passes in112.23s and retains578 new bindings plus
the ten prior milestone binding counts. All14 DGP-primary app bindings remain.

V40's failed50-update structure gate, every original/stopped checkpoint, split,
cache/local backup and failure remain locally preserved. There is no new native,
development, reserved-final or completion quality qualification. Useful native
restoration, all seven covering families with separate automatic/assisted
review, independent final review and the qualified DGP-led flow remain required.
The entire previous handoff is archived and preserved below. Goal active/incomplete.

[Manual start and availability evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM_AVAILABILITY.md>)
[Five verified manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM.md>)

'''
    addition = section.encode('utf-8'); at = before.index(b'\n')+1
    (ROOT/'PROJECT_HANDOFF.md').write_bytes(before[:at]+addition+before[at:])
    names = ['PROJECT_HANDOFF.md', 'CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM_AVAILABILITY.md',
        'scripts/inspect_cctv_dgp_v40_learning_signal_v1_vm_readiness.py', 'scripts/record_cctv_dgp_v40_learning_signal_v1_vm_availability.py',
        'scripts/verify_cctv_dgp_v40_learning_signal_v1_vm_availability.py', 'outputs/cctv_dgp_v40_learning_signal_v1_readiness_tls_failure.json',
        'outputs/cctv_dgp_v40_learning_signal_v1_milestone/milestone.json', 'outputs/cctv_dgp_v40_learning_signal_v1_milestone/independent_closure_audit.json']
    names.extend(path.relative_to(ROOT).as_posix() for path in OBS.rglob('*') if path.is_file())
    bindings = {name: sha(ROOT/name) for name in names}
    write(OUT/'milestone.json', {'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'previous_milestone_sha256':sha(PRIOR/'milestone.json'),
        'previous_handoff_path':(OUT/'before_docs/PROJECT_HANDOFF.md').relative_to(ROOT).as_posix(),
        'document':{'before_sha256':hashlib.sha256(before).hexdigest(),'addition_bytes':len(addition)},'new_evidence_sha256':bindings,
        'live_VM_status':'TERMINATED','fresh_guest_inventory_available':False,'files_removed':0,'VM_started':False,'learning_launched':False,
        'original_prepared_packet_and_guide_unchanged':True,'prior_full_history_closure_retained':True,'app_promotion':False,
        'goal_complete':False,'seconds':time.monotonic()-start,'cap_seconds':120})
    print({'complete':True,'status':'TERMINATED','bindings':len(bindings),'seconds':time.monotonic()-start},flush=True)


if __name__ == '__main__': main()
