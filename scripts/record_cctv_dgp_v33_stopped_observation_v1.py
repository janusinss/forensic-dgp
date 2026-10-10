"""Record already obtained read-only transport/API evidence; no network calls."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_v33_probe_status_v1_r1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    transport=json.loads((OUT/'status_transport.json').read_text())
    assert transport['exit_code']==1 and 0<transport['seconds']<=60
    assert transport['stdout_sha256']==sha(OUT/'status.log') and transport['stderr_sha256']==sha(OUT/'status_stderr.log')
    assert not (OUT/'status.log').read_bytes() and 'Remote side unexpectedly closed' in (OUT/'status_stderr.log').read_text()
    instance=json.loads((OUT/'instance_state.json').read_text())
    assert instance['name']=='forensic-dgp-thesis' and instance['status']=='TERMINATED'
    assert instance['machineType'].endswith('/g2-standard-4')
    assert not (OUT/'instance_state_stderr.log').read_bytes()
    first=ROOT/'outputs/cctv_dgp_v33_probe_status_v1/failure_record.json'
    initial=json.loads(first.read_text());assert not initial['complete']
    sources=[Path(__file__),ROOT/'scripts/inspect_cctv_dgp_v33_probe_status_v1.py',
        ROOT/'scripts/inspect_cctv_dgp_v33_probe_status_v1_r1.py',first]
    sources.extend(q for q in sorted(OUT.iterdir()) if q.is_file())
    record={'complete':True,'recorded_UTC':datetime.now(timezone.utc).isoformat(),
        'sources_sha256':{q.relative_to(ROOT).as_posix():sha(q) for q in sources},
        'SSH_read_completed':False,'SSH_failure_exit_code':1,'SSH_seconds':transport['seconds'],
        'control_plane_API_exit_code':0,'control_plane_API_tool_chunk':'5fa6c0',
        'control_plane_API_query':['compute','instances','describe','forensic-dgp-thesis','--project=forensic-dgp-thesis',
            '--zone=us-central1-a','--format=json(name,status,machineType,networkInterfaces)'],
        'control_plane_observed_state':'TERMINATED','VM_machine_type':'g2-standard-4',
        'control_plane_json_file_written_UTC':datetime.fromtimestamp((OUT/'instance_state.json').stat().st_mtime,timezone.utc).isoformat(),
        'exact_API_response_clock_not_retained':True,'no_external_IP_present_in_API_result':True,
        'V33_packet_process_export_contents_observed':False,'V33_historical_execution_known':False,
        'local_V33_result_archive_present':(ROOT/'outputs/cctv-dgp-loss-cone-probe-v33-results.tar.gz').exists(),
        'agent_started_VM':False,'agent_started_training':False,'remote_writes':0,'cleanup_deletions':0,
        'snapshot_only_not_a_live_future_state_claim':True,'manual_VM_optimizer_workflow_preserved':True,
        'initial_sandbox_credential_access_failure_retained':True,'model_or_training_calls':0,'goal_complete':False}
    with (OUT/'inspection_review.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'complete':True,'Cloud_API_state':'TERMINATED','remote_V33_execution':'unobserved','VM_started':False}))


if __name__=='__main__':main()
