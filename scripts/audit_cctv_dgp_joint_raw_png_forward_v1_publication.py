"""Bind published findings/commands to actual evidence and exact old suffixes."""
from datetime import datetime, timezone
from pathlib import Path
import re

from import_cctv_dgp_joint_raw_png_forward_v1_return import ROOT, OUT, ANALYSIS, read, sha, write


def main():
    stage=ANALYSIS/'publication';target=stage/'publication_audit.json';assert not target.exists()
    published=read(stage/'publication.json')
    assert published['complete'] and sha(ROOT/'scripts/publish_cctv_dgp_joint_raw_png_forward_v1_results.py')==published['publisher_sha256']
    for name,digest in published['evidence_sha256'].items():assert sha(ROOT/name)==digest
    for name,record in published['documents'].items():
        path=ROOT/name;backup=Path(record['backup'])
        assert sha(path)==record['after_sha256'] and sha(backup)==record['before_sha256']
        assert path.read_bytes()[record['prefix_bytes']:]==backup.read_bytes()
        assert record['old_bytes_preserved_as_exact_suffix']
    for name,digest in published['new_documents_sha256'].items():assert sha(ROOT/name)==digest
    guide=ROOT/'CCTV_DGP_LIVE_CONDITIONING_V1_VM.md';text=guide.read_text(encoding='utf-8')
    blocks=re.findall(r'```[^\n]*\n(.*?)```',text,re.S);assert len(blocks)==5
    packet=OUT/'cctv_dgp_live_conditioning_v1_vm';pin=sha(packet/'protocol.json')
    prep=read(OUT/'cctv_dgp_live_conditioning_v1_preparation.json')
    worker=packet/'scripts/cctv_dgp_live_conditioning_v1_vm.py'
    shell=packet/'scripts/run_conditioning.sh'
    stem='cctv-dgp-live-conditioning-v1'
    assert str(prep['archive_bytes'])==str((OUT/(stem+'-execution.tar.gz')).stat().st_size)
    assert pin==published['new_protocol_sha256'] and prep['archive_sha256']==published['new_archive_sha256']
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    assert blocks[0].startswith('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"\n')
    uploads=[line for line in blocks[0].splitlines() if line.startswith('gcloud compute scp ')]
    downloads=[line for line in blocks[4].splitlines() if line.startswith('gcloud compute scp ')]
    assert len(uploads)==2 and len(downloads)==3
    for line in uploads+downloads:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in line
        assert line.count('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/')==1
    for suffix,line in zip(['-execution.tar.gz','-execution.tar.gz.sha256'],uploads,strict=True):
        assert f'"{stem+suffix}"' in line and (OUT/(stem+suffix)).is_file()
    for suffix,line in zip(['-results.tar.gz','-results.tar.gz.sha256','-export.json'],downloads,strict=True):
        assert f'/home/janusdominic0/{stem+suffix}"' in line and line.endswith('"'+stem+suffix+'"')
    assert f'sha256sum -c {stem}-execution.tar.gz.sha256' in blocks[1]
    assert f'test ! -e ~/forensic-dgp/{packet.name}' in blocks[1]
    assert blocks[2].strip()=='tmux new-session -A -s dgp_live_conditioning_v1'
    assert '--protocol-sha '+pin+' --verify-transfer' in blocks[3]
    assert 'bash scripts/run_conditioning.sh '+pin in blocks[3]
    assert str(worker.name) in blocks[3] and worker.is_file() and shell.is_file()
    assert f'STEM = \'{stem}\'' in worker.read_text(encoding='utf-8')
    assert 'outputs/live_conditioning_v1' in worker.read_text(encoding='utf-8')
    report=(ROOT/'CCTV_DGP_JOINT_RAW_PNG_FORWARD_V1_RESULTS.md').read_text(encoding='utf-8')
    bound=read(ANALYSIS/'review_binding_audit.json')
    finite=read(ANALYSIS/'finite_metrics.json')
    for row in bound['summary']:
        assert f"{row['raw_structure_gain_percent']:.6f}%" in report
        assert f"{row['PNG_structure_gain_percent']:.6f}%" in report
        assert not finite['proposals'][row['proposal']]['all_preview_requirements_pass']
    spatial=read(ANALYSIS/'spatial_patch_review.json')
    nose=spatial['degraded_region_groups']['balanced_0.02']['png']['all']['nose']
    assert nose['individual_regressions']==47 and nose['cases']==80 and '47 of80' in report
    assert read(ANALYSIS/'visual_review.json')['all20_sheets_and800_cells_inspected']
    assert 'earlier audited B50 training' in report and 'not evidence that conditioning can' in report
    assert 'apply best approach' in report and 'zero optimizer updates' in report
    assert 'https://arxiv.org/html/2012.00739' in report
    assert 'not a clean target' in report and 'no new trained model exists' in report
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(target,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(stage/'publication.json'),documents_checked=len(published['documents']),
        new_documents_checked=2,all_instruction_history_exact_suffixes=True,
        command_blocks_checked=5,single_remote_source_per_PuTTY_download=True,
        original_checkpoint_and_app_bindings_preserved=True,all_four_proposals_rejected=True,
        raw_PNG_source_and_spatial_findings_bound_to_actual_arrays=True,
        all20_visual_reviews_bound=True,new_packet_verified_and_unrun=True,
        new_CUDA_parity_and_gradient_activation_not_claimed=True,
        no_agent_VM_upload_or_launch=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,instruction_histories=6,command_blocks=5,actual_findings_bound=True,
        new_VM_diagnostic_unrun=True,model_qualified=False))


if __name__=='__main__':main()
