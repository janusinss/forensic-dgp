"""Exact history, evidence and final manual command namespace audit."""
import ast
from datetime import datetime,timezone
from pathlib import Path
import re
from import_cctv_dgp_live_conditioning_v1_r1_return import ROOT,OUT,ANALYSIS,RETURN,read,sha,write


def constant(path,key):
    nodes=ast.parse(path.read_text(encoding='utf-8')).body
    values=[n.value.value for n in nodes if isinstance(n,ast.Assign) and isinstance(n.value,ast.Constant)
            and any(isinstance(t,ast.Name) and t.id==key for t in n.targets)]
    assert len(values)==1;return values[0]


def main():
    folder=ANALYSIS/'publication';record=read(folder/'publication.json')
    assert record['complete'] and sha(ROOT/'scripts/publish_cctv_dgp_live_conditioning_v1_r1_results.py')==record['publisher_sha256']
    for name,digest in record['evidence_sha256'].items():assert sha(ROOT/name)==digest
    for name,row in record['documents'].items():
        path=ROOT/name;before=Path(row['backup'])
        assert sha(path)==row['after_sha256'] and sha(before)==row['before_sha256']
        assert path.read_bytes()[row['prefix_bytes']:]==before.read_bytes()
    assert len(record['documents'])==6
    for name,digest in record['new_documents_sha256'].items():assert sha(ROOT/name)==digest
    packet=OUT/'cctv_dgp_conditioning_learning_v1_r2_vm';p=read(packet/'protocol.json');pin=sha(packet/'protocol.json')
    assert pin==record['next_manual_protocol_sha256']
    worker=packet/'scripts/cctv_dgp_conditioning_learning_v1_r2_vm.py'
    importer=ROOT/'scripts/import_cctv_dgp_conditioning_learning_v1_r2_return.py'
    stem='cctv-dgp-conditioning-learning-v1-r2'
    assert all(constant(file,'NAME')==packet.name and constant(file,'STEM')==stem for file in [worker,importer])
    assert constant(importer,'PIN')==pin and constant(worker,'STAGE')=='outputs/conditioning_learning_v1_r2'
    assert constant(packet/'conditioning_model.py','NAME')==packet.name
    assert all(sha(packet/name)==digest for name,digest in p['assets_sha256'].items())
    transfer=read(OUT/'cctv_dgp_conditioning_learning_v1_r2_transfer_audit.json')
    assert transfer['complete'] and transfer['future_importer_sha256']==sha(importer)
    assert transfer['future_auditor_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_conditioning_learning_v1_r2_return.py')
    assert transfer['all100_saved_raw_and_PNG_checker_baselines_replayed'] and transfer['NaN_and_infinity_metrics_rejected']
    assert len(transfer['actual_prior_rejected_joint_proposals_still_rejected'])==4
    prep=read(OUT/'cctv_dgp_conditioning_learning_v1_r2_preparation.json')
    archive=OUT/(stem+'-execution.tar.gz')
    assert sha(archive)==record['next_manual_archive_sha256']==prep['archive_sha256']
    assert Path(str(archive)+'.sha256').read_bytes()==f"{prep['archive_sha256']}  {archive.name}\n".encode('ascii')
    text=(ROOT/'CCTV_DGP_CONDITIONING_LEARNING_V1_R2_VM.md').read_text(encoding='utf-8')
    blocks=re.findall(r'```[^\n]*\n(.*?)```',text,re.S);assert len(blocks)==5
    for i in [0,4]:assert blocks[i].startswith('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"\n')
    uploads=[s for s in blocks[0].splitlines() if s.startswith('gcloud compute scp ')]
    downloads=[s for s in blocks[4].splitlines() if s.startswith('gcloud compute scp ')]
    assert len(uploads)==2 and len(downloads)==3
    for command in uploads+downloads:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in command
        assert command.count('janusdominic0@forensic-dgp-thesis:/home/janusdominic0/')==1
    for suffix,command in zip(['-execution.tar.gz','-execution.tar.gz.sha256'],uploads):
        assert f'"{stem+suffix}"' in command and (OUT/(stem+suffix)).is_file()
    for suffix,command in zip(['-results.tar.gz','-results.tar.gz.sha256','-export.json'],downloads):
        assert f'/home/janusdominic0/{stem+suffix}"' in command and command.endswith('"'+stem+suffix+'"')
    assert f'sha256sum -c {stem}-execution.tar.gz.sha256' in blocks[1]
    assert f'test ! -e ~/forensic-dgp/{packet.name}' in blocks[1]
    assert blocks[2].strip()=='tmux new-session -A -s dgp_conditioning_learning_v1_r2'
    assert worker.name in blocks[3] and f'--protocol-sha {pin} --verify-transfer' in blocks[3]
    assert f'bash scripts/run_learning.sh {pin}' in blocks[3]
    assert f'{prep["archive_bytes"]:,}bytes' in text and '9GiB free after installation' in text
    assert 'at2 accepted updates' in text and '1% structure gain' in text
    assert not (packet/'outputs').exists() and not (packet/'review.log').exists()
    run=read(RETURN/'outputs/live_conditioning_v1_r1/run_results.json')
    assert run['complete'] and run['all12_conditioning_tensors_live_for_both_sources']
    assert run['counts']['optimizer_updates']==run['counts']['parameter_displacements']==0
    report=(ROOT/'CCTV_DGP_LIVE_CONDITIONING_V1_R1_RESULTS.md').read_text(encoding='utf-8')
    assert f"{run['seconds']:.2f}s" in report and read(ANALYSIS/'import.json')['archive_sha256'] in report
    assert 'native DEV qualification remain' in report and 'seven completion families' in report
    for name in ['live_conditioning_v1','live_conditioning_v1_r1','conditioning_learning_v1','conditioning_learning_v1_r1']:
        old=OUT/('cctv_dgp_'+name+'_vm')
        meta=read(old/'protocol.json')
        assert all(sha(old/n)==h for n,h in meta['assets_sha256'].items())
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(folder/'publication_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),publication_sha256=sha(folder/'publication.json'),
        exact_history_suffixes_and_backups_checked=6,command_blocks_checked=5,
        upload_export_import_download_and_model_root_names_agree=True,single_remote_source_per_PuTTY_download=True,
        LF_ASCII_checksum_exact=True,all_original_local_drafts_and_diagnostic_packets_preserved=True,
        returned_activation_and_all100_controls_verified=True,original_app_checkpoint_bindings_unchanged=True,
        actual_VM_storage_inventory_used_without_deletion_or_training=True,
        next_manual_training_recipe_verified_and_unrun=True,no_agent_training_launch=True,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,commands_checked=5,history_documents_preserved=6,all_names_agree=True,manual_training_unrun=True))


if __name__=='__main__':main()
