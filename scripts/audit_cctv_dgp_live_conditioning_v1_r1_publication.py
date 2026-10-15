"""Final command/export and exact-history checks for corrected manual R1."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import re

from import_cctv_dgp_joint_raw_png_forward_v1_return import ROOT, OUT, ANALYSIS, read, sha, write


def constant(path,name):
    matches=[node.value.value for node in ast.parse(path.read_text(encoding='utf-8')).body
             if isinstance(node,ast.Assign) and isinstance(node.value,ast.Constant)
             and any(isinstance(t,ast.Name) and t.id==name for t in node.targets)]
    assert len(matches)==1;return matches[0]


def main():
    stage=ANALYSIS/'publication_live_conditioning_v1_r1';target=stage/'publication_audit.json'
    assert not target.exists();published=read(stage/'publication.json')
    assert published['complete'] and sha(ROOT/'scripts/publish_cctv_dgp_live_conditioning_v1_r1.py')==published['publisher_sha256']
    for name,digest in published['evidence_sha256'].items():assert sha(ROOT/name)==digest
    for name,record in published['documents'].items():
        path=ROOT/name;backup=Path(record['backup'])
        assert sha(path)==record['after_sha256'] and sha(backup)==record['before_sha256']
        assert path.read_bytes()[record['prefix_bytes']:]==backup.read_bytes()
    for name,digest in published['new_documents_sha256'].items():assert sha(ROOT/name)==digest
    packet=OUT/'cctv_dgp_live_conditioning_v1_r1_vm';pin=sha(packet/'protocol.json')
    worker=packet/'scripts/cctv_dgp_live_conditioning_v1_r1_vm.py'
    importer=ROOT/'scripts/import_cctv_dgp_live_conditioning_v1_r1_return.py'
    stem='cctv-dgp-live-conditioning-v1-r1'
    assert all(constant(path,'STEM')==stem and constant(path,'NAME')==packet.name for path in [worker,importer])
    assert constant(importer,'PIN')==pin==published['new_protocol_sha256']
    prep=read(OUT/'cctv_dgp_live_conditioning_v1_r1_preparation.json')
    assert sha(OUT/(stem+'-execution.tar.gz'))==prep['archive_sha256']==published['new_archive_sha256']
    namespace=read(OUT/'cctv_dgp_live_conditioning_v1_r1_namespace_audit.json')
    assert namespace['complete'] and namespace['all_names_agree'] and namespace['protocol_sha256']==pin
    assert sha(worker)==namespace['worker_sha256'] and sha(importer)==namespace['importer_sha256']
    guide=ROOT/'CCTV_DGP_LIVE_CONDITIONING_V1_R1_VM.md';text=guide.read_text(encoding='utf-8')
    blocks=re.findall(r'```[^\n]*\n(.*?)```',text,re.S);assert len(blocks)==5
    for index in [0,4]:assert blocks[index].startswith('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"\n')
    uploads=[s for s in blocks[0].splitlines() if s.startswith('gcloud compute scp ')]
    downloads=[s for s in blocks[4].splitlines() if s.startswith('gcloud compute scp ')]
    assert len(uploads)==2 and len(downloads)==3
    for command in uploads+downloads:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in command
        assert command.count('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/')==1
    for suffix,command in zip(['-execution.tar.gz','-execution.tar.gz.sha256'],uploads,strict=True):
        assert f'"{stem+suffix}"' in command and (OUT/(stem+suffix)).is_file()
    for suffix,command in zip(['-results.tar.gz','-results.tar.gz.sha256','-export.json'],downloads,strict=True):
        assert f'/home/janusdominic0/{stem+suffix}"' in command and command.endswith('"'+stem+suffix+'"')
    assert f'sha256sum -c {stem}-execution.tar.gz.sha256' in blocks[1]
    assert f'test ! -e ~/forensic-dgp/{packet.name}' in blocks[1]
    assert blocks[2].strip()=='tmux new-session -A -s dgp_live_conditioning_v1_r1'
    assert worker.name in blocks[3] and f'--protocol-sha {pin} --verify-transfer' in blocks[3]
    assert f'bash scripts/run_conditioning.sh {pin}' in blocks[3]
    assert 'outputs/live_conditioning_v1_r1' in worker.read_text(encoding='utf-8')
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    old=OUT/'cctv_dgp_live_conditioning_v1_vm';oldprep=read(OUT/'cctv_dgp_live_conditioning_v1_preparation.json')
    assert sha(old/'protocol.json')==oldprep['protocol_sha256']
    assert sha(OUT/'cctv-dgp-live-conditioning-v1-execution.tar.gz')==oldprep['archive_sha256']
    assert all(sha(old/name)==digest for name,digest in read(old/'protocol.json')['assets_sha256'].items())
    assert (ANALYSIS/'publication/failed_command_audit.py').is_file() and (ANALYSIS/'publication/failed_publisher.py').is_file()
    original_guide=(ROOT/'CCTV_DGP_LIVE_CONDITIONING_V1_VM.md').read_text(encoding='utf-8')
    assert original_guide.startswith('# Superseded commands') and guide.name in original_guide
    result=(ROOT/'CCTV_DGP_JOINT_RAW_PNG_FORWARD_V1_RESULTS.md').read_text(encoding='utf-8')
    bound=read(ANALYSIS/'review_binding_audit.json');finite=read(ANALYSIS/'finite_metrics.json')
    assert bound['complete'] and bound['all_four_proposals_rejected_for_adoption']
    for row in bound['summary']:
        assert f"{row['raw_structure_gain_percent']:.6f}%" in result and f"{row['PNG_structure_gain_percent']:.6f}%" in result
        assert not finite['proposals'][row['proposal']]['all_preview_requirements_pass']
    assert '47 of80' in result and 'not a clean target' in result and 'no new trained model exists' in result
    assert read(ANALYSIS/'visual_review.json')['all20_sheets_and800_cells_inspected']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    write(target,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(stage/'publication.json'),namespace_audit_sha256=sha(OUT/'cctv_dgp_live_conditioning_v1_r1_namespace_audit.json'),
        instruction_and_report_histories_preserved=7,new_guide_checked=True,command_blocks_checked=5,
        upload_export_download_and_importer_names_agree=True,single_remote_source_per_PuTTY_download=True,
        original_misnamed_packet_and_audit_stops_retained=True,namespace_only_repair=True,
        actual_joint_findings_and_all20_visual_reviews_bound=True,original_model_app_and_research_preserved=True,
        new_manual_diagnostic_verified_and_unrun=True,new_CUDA_parity_and_activation_not_claimed=True,
        no_agent_VM_upload_or_launch=True,training_launch_ready=False,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,command_blocks=5,all_names_agree=True,histories_preserved=7,
        original_packet_retained=True,new_VM_diagnostic_unrun=True))


if __name__=='__main__':main()
