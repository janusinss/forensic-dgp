"""Publish corrected namespaces; preserve both original packet and documents."""
from datetime import datetime, timezone

from import_cctv_dgp_joint_raw_png_forward_v1_return import ROOT, OUT, ANALYSIS, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import link
from publish_cctv_dgp_png_group_gradients_v1_r2 import retained_new, finish_prepend


def main():
    stage=ANALYSIS/'publication_live_conditioning_v1_r1';assert not (stage/'publication.json').exists()
    previous=read(ANALYSIS/'publication/publication.json')
    prep=read(OUT/'cctv_dgp_live_conditioning_v1_r1_preparation.json')
    paths=[OUT/('cctv_dgp_live_conditioning_v1_r1_'+suffix+'.json') for suffix in
        ['preparation','transfer_audit','gate_fixtures','bash_syntax']]
    assert all(read(path)['complete'] for path in paths)
    pin=prep['protocol_sha256'];packet=OUT/'cctv_dgp_live_conditioning_v1_r1_vm'
    assert sha(packet/'protocol.json')==pin and not (packet/'outputs').exists() and not (packet/'review.log').exists()
    assert all(read(path)['protocol_sha256']==pin for path in paths)
    assert sha(OUT/'cctv-dgp-live-conditioning-v1-r1-execution.tar.gz')==prep['archive_sha256']
    oldprep=read(OUT/'cctv_dgp_live_conditioning_v1_preparation.json')
    original=ROOT/'CCTV_DGP_LIVE_CONDITIONING_V1_VM.md'
    assert sha(original)==previous['new_documents_sha256'][original.name]
    text=original.read_text(encoding='utf-8').replace('live_conditioning_v1','live_conditioning_v1_r1').replace(
        'cctv-dgp-live-conditioning-v1','cctv-dgp-live-conditioning-v1-r1')
    text=text.replace(oldprep['protocol_sha256'],pin).replace(oldprep['archive_sha256'],prep['archive_sha256'])
    text=text.replace(f"{oldprep['archive_bytes']:,}",f"{prep['archive_bytes']:,}")
    text=text.replace('# Live conditioning diagnostic — manual L4 commands',
        '# Live conditioning R1 diagnostic — manual L4 commands')
    text=text.replace('**Verified and UNRUN. This is an architecture/gradient diagnostic, not training.**',
        '**Corrected R1, verified and UNRUN. This is an architecture/gradient diagnostic, not training.**\n\n'
        'R1 fixes an inherited export filename prefix. Its worker, download and importer\n'
        'names agree. The original V1 packet and failed command audit are retained.\n'
        'Architecture, cases, initialization, gradients and all stop/quality limits are\n'
        'unchanged. Original human VM execution is unknown; this fresh directory preserves it.')
    guide='CCTV_DGP_LIVE_CONDITIONING_V1_R1_VM.md';retained_new(ROOT/guide,text)
    stage.mkdir(exist_ok=True);backup=stage/'before';backup.mkdir(exist_ok=True)
    docs=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
          'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md','CCTV_DGP_DELIVERED_LOSS_LEARNING_REVIEW_V1.md']
    expected={name:previous['documents'][name]['after_sha256'] for name in docs}
    result='CCTV_DGP_JOINT_RAW_PNG_FORWARD_V1_RESULTS.md'
    expected[original.name]=previous['new_documents_sha256'][original.name]
    expected[result]=previous['new_documents_sha256'][result]
    assert all(sha(ROOT/name)==digest for name,digest in expected.items())
    prefix=f'''# Corrected manual diagnostic — live conditioning R1; 13 October 2026

Final command audit caught an inherited bank- prefix in V1's export/import
filename. The original packet, staged/publication records and failed audit are
retained. Fresh R1 fixes namespaces only; its worker, export, download and
importer agree. Original human VM execution is unknown. No new agent VM run,
neural inference, gradient or optimizer update occurred.
{link('Use these exact R1 upload, tmux and download commands',guide)}.

R1 is verified and UNRUN. The anchored same-input feature-path hypothesis,
initial state,100 paired TRAIN cases,32 groups,320 gradient queries, zero
optimizer/committed updates or checkpoints, and all original scientific/resource
limits are unchanged. CUDA starting-output parity and live gradients in all12
conditioning tensors remain unproven until the manual L4 run and return audit.
Activation does not qualify more epochs or model adoption.

The preceding joint result remains rejected:0.865441% raw/0.886155% PNG detail
gain,37 preservation failures each,20 visual sheets reviewed and no convincing
whole-face clarity advantage. The original app checkpoint/design, data/failure
records and all five milestones/seven covering families remain. Full goal
ACTIVE and incomplete. The exact previous document follows.

'''
    published={name:finish_prepend(ROOT/name,prefix,backup/name,expected[name]) for name in docs}
    banner=f'''# Superseded commands — use corrected R1

The final command audit found that V1's worker exports with an inherited bank-
prefix while this guide expected filenames without it. All original bytes are
retained. {link('Verified R1 commands with matching upload/export/download/import names',guide)}.
Original human VM execution is unknown; the fresh R1 root preserves any used V1.
The original commands below are historical.

'''
    published[original.name]=finish_prepend(original,banner,backup/original.name,expected[original.name])
    note=f'''# Command correction — fresh R1 available

The audited joint-image findings below are unchanged. The selected live-path
diagnostic now has a fresh R1 that fixes an inherited export filename prefix.
Architecture and every quality/resource requirement are unchanged; it is
verified and UNRUN. {link('Exact corrected manual commands',guide)}.
The original V1 packet, command audit stop and document bytes remain retained.

'''
    published[result]=finish_prepend(ROOT/result,note,backup/result,expected[result])
    write(stage/'publication.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        publisher_sha256=sha(__file__),evidence_sha256={path.relative_to(ROOT).as_posix():sha(path) for path in paths},
        original_publication_sha256=sha(ANALYSIS/'publication/publication.json'),
        documents=published,new_documents_sha256={guide:sha(ROOT/guide)},
        new_protocol_sha256=pin,new_archive_sha256=prep['archive_sha256'],
        original_packet_and_failed_command_audit_retained=True,namespace_only_repair=True,
        all_scientific_and_resource_requirements_preserved=True,manual_diagnostic_unrun=True,
        no_agent_VM_connection_upload_or_launch=True,training_launch_ready=False,model_qualified=False,
        more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,manual_guide=guide,histories_preserved=7,namespace_only_repair=True,unrun=True))


if __name__=='__main__':main()
