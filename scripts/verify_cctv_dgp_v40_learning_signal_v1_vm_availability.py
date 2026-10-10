"""Independently verify availability observation and exact prior research retention."""
import ast
import hashlib
from pathlib import Path
import sys
import time
from completion_feature_fusion_off_v1_common import ROOT, sha, read, write

OUT = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm_availability_milestone'
OBS = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_vm_readiness'
PRIOR = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_milestone'


def main():
    start = time.monotonic(); m = read(OUT/'milestone.json'); assert m['complete']
    for name, digest in m['new_evidence_sha256'].items(): assert sha(ROOT/name) == digest, name
    before = (ROOT/m['previous_handoff_path']).read_bytes(); current = (ROOT/'PROJECT_HANDOFF.md').read_bytes(); at = before.index(b'\n')+1
    assert hashlib.sha256(before).hexdigest() == m['document']['before_sha256']
    assert current[:at] == before[:at] and current[at+m['document']['addition_bytes']:] == before[at:]
    parent = read(PRIOR/'milestone.json'); closed = read(PRIOR/'independent_closure_audit.json')
    assert m['previous_milestone_sha256'] == closed['milestone_sha256'] == sha(PRIOR/'milestone.json') and closed['complete']
    for name, digest in parent['new_evidence_sha256'].items():
        path = ROOT/m['previous_handoff_path'] if name == 'PROJECT_HANDOFF.md' else ROOT/name
        assert sha(path) == digest, name
    assert len(parent['new_evidence_sha256']) == closed['new_bindings_verified'] == 578
    assert closed['historical_bindings_preserved'] == [28136,11283,536,9157,734,2562,539,2230,316,5961]
    status = read(OBS/'instance_status.json'); api = read(OBS/'instance_status_transport.json'); transport = read(OBS/'transport.json')
    assert status['status'] == m['live_VM_status'] == 'TERMINATED' and status['name'] == 'forensic-dgp-thesis'
    assert status['machineType'].endswith('/projects/forensic-dgp-thesis/zones/us-central1-a/machineTypes/g2-standard-4')
    assert api['exit_code'] == 0 and api['read_only'] and not api['VM_started'] and not api['training_launched']
    import json
    assert json.loads(api['output']) == status and api['CA_setting_process_only']
    assert transport['exit_code'] == 1 and not transport['complete'] and transport['seconds'] < 120
    assert transport['TLS_validation_enabled'] and transport['hostkey_pinned'] == 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
    assert sha(OBS/'remote_source.py') == transport['source_sha256'] and sha(ROOT/'scripts/inspect_cctv_dgp_v40_learning_signal_v1_vm_readiness.py') == transport['local_inspector_sha256']
    assert sha(OBS/'stdout.log') == transport['stdout_sha256'] and sha(OBS/'stderr.log') == transport['stderr_sha256']
    assert not (OBS/'stdout.log').read_text().strip() and 'Remote side unexpectedly closed network connection' in (OBS/'stderr.log').read_text()
    assert not (OBS/'readiness.json').exists() and not m['fresh_guest_inventory_available']
    for key in ['files_removed','model_or_gradient_calls','optimizer_updates']: assert transport[key] == 0
    assert not transport['diagnostic_or_training_launched'] and not m['VM_started'] and not m['learning_launched']
    tree = ast.parse((OBS/'remote_source.py').read_text()); commands = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'command': commands.append(ast.literal_eval(node.args[0]))
    assert {c[0] for c in commands} == {'nvidia-smi','tmux'} and len(commands) == 3
    assert 'shutil.disk_usage' in (OBS/'remote_source.py').read_text()
    assert not any(s in (OBS/'remote_source.py').read_text() for s in ['torch','unlink','rmtree','train_cctv','write_text','write_bytes'])
    # urllib.request.urlopen is the read-only metadata GET, not a file writer.
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'open' for n in ast.walk(tree))
    guide = (ROOT/'CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM_AVAILABILITY.md').read_text(encoding='utf-8')
    assert 'gcloud compute instances start forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a' in guide
    assert 'set "CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE=C:\\xampp\\htdocs\\YEAR 4\\Testing\\scratch\\gcloud_windows_trust.pem"' in guide
    app = read(ROOT/'outputs/completion_feature_fusion_off_v1/protocol.json')['app_preservation_sha256']; assert len(app) == 14
    for name, digest in app.items(): assert sha(ROOT/name) == digest
    assert not m['app_promotion'] and not m['goal_complete'] and not any(n in sys.modules for n in ['torch','dgp_frozen_inference_v2'])
    assert time.monotonic()-start < 120
    write(OUT/'independent_closure_audit.json',{'complete':True,'milestone_sha256':sha(OUT/'milestone.json'),'checker_sha256':sha(Path(__file__)),
        'new_bindings_verified':len(m['new_evidence_sha256']),'prior578_bindings_preserved':True,'prior_full_history_audit_retained':True,
        'entire_previous_handoff_preserved':True,'live_status_query_verified':'TERMINATED','guest_inventory_unavailable_reported':True,
        'original_TLS_and_pinned_transport_failures_retained':True,'original_packet_protocol_guide_and_gates_unchanged':True,
        'all14_app_bindings_unchanged':True,'VM_started':False,'files_removed':0,'neural_or_gradient_calls':0,'optimizer_updates':0,
        'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start,'cap_seconds':120})
    print({'complete':True,'status':'TERMINATED','prior_bindings':578,'seconds':time.monotonic()-start},flush=True)


if __name__ == '__main__': main()
