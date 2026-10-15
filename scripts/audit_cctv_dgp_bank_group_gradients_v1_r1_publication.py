"""Check published bytes, historical suffixes and exact manual commands."""
from datetime import datetime, timezone
from pathlib import Path
import re
import tarfile
from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, read, sha, write


def main():
    dest=ANALYSIS/'publication_audit.json'
    assert not dest.exists()
    receipt=read(ANALYSIS/'publication.json')
    assert receipt['complete'] and receipt['goal_status']=='active' and not receipt['goal_complete']
    for name,h in receipt['evidence_sha256'].items():
        assert sha(ROOT/name)==h and read(ROOT/name)['complete']
    documents={}
    for name,row in receipt['documents'].items():
        file=ROOT/name
        assert sha(file)==row['after_sha256']
        backup=Path(row['backup'])
        assert backup.resolve().is_relative_to((ANALYSIS/'publication_before').resolve())
        assert sha(backup)==row['before_sha256']
        assert file.read_bytes()[row['prefix_bytes']:]==backup.read_bytes()
        text=file.read_text(encoding='utf-8')[:row['prefix_bytes']]
        if name in ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']:
            assert all(term.casefold() in text.casefold() for term in ['Goal ACTIVE','UNRUN','zero updates/displacements/epochs','five milestones','seven completion',
                'paired photographic TRAIN','reserved final identities','maintenance only'])
        else:
            assert 'Superseded used measurement packet' in text and 'do not relaunch' in text
        documents[name]=dict(after_sha256=row['after_sha256'],before_sha256=row['before_sha256'],exact_historical_suffix_preserved=True)
    for name,h in receipt['new_documents_sha256'].items():assert sha(ROOT/name)==h
    base='cctv_dgp_bank_group_direction_forward_v1'
    prep=read(OUT/(base+'_preparation.json'))
    transfer=read(OUT/(base+'_transfer_audit.json'))
    packet=OUT/prep['name'];p=read(packet/'protocol.json')
    assert sha(packet/'protocol.json')==prep['protocol_sha256']==transfer['protocol_sha256']
    assert not (packet/'outputs').exists()
    archive=OUT/(prep['stem']+'-execution.tar.gz')
    assert sha(archive)==prep['archive_sha256']
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==5 and {m.name for m in members}=={prep['name']+'/'+n for n in p['assets_sha256']}|{prep['name']+'/protocol.json'}
        for m in members:
            assert m.isfile() and not m.islnk() and not m.issym()
            assert tar.extractfile(m).read()==(OUT/m.name).read_bytes()
    guide=(ROOT/'CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ',guide,re.M)==['1','2','3','4','5']
    commands=[line for line in guide.splitlines() if line.startswith('gcloud compute scp ')]
    assert len(commands)==5
    for line in commands:assert '--project=forensic-dgp-thesis' in line and '--zone=us-central1-a' in line
    assert all(line.count('janusdominic0@forensic-dgp-thesis:')==1 for line in commands)
    for line,suffix in zip(commands[2:],['-results.tar.gz','-results.tar.gz.sha256','-export.json'],strict=True):
        filename=prep['stem']+suffix
        assert line.endswith('"'+filename+'"') and '/home/janusdominic0/'+filename+'"' in line
    assert guide.count(prep['protocol_sha256'])==3 and prep['archive_sha256'] in guide
    assert 'tmux new-session -A -s dgp_group_direction_v1' in guide
    assert 'sha256sum -c '+prep['stem']+'-execution.tar.gz.sha256' in guide
    assert f'--parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {prep["protocol_sha256"]} --verify-transfer' in guide
    assert f'bash scripts/run_forward.sh {prep["protocol_sha256"]}' in guide
    assert (OUT/(archive.name+'.sha256')).read_bytes()==(prep['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    audit=read(ANALYSIS/'independent_audit.json')
    assert sha(RETURN/'outputs/group_gradients_v1_r1/run_results.json')==audit['run_results_sha256']
    assert audit['counts_recomputed']['gradient_queries']==320 and audit['counts_recomputed']['optimizer_updates']==0
    direction=read(ANALYSIS/'common_direction/direction_review.json')
    assert sha(ANALYSIS/'common_direction/directions.npz')==direction['direction_file_sha256']
    assert direction['old_directions_initially_regressing_group_counts']==dict(ordinary=10,constrained=7)
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(dest,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(ANALYSIS/'publication.json'),documents=documents,new_document_hashes_checked=2,
        original_history_preserved=True,exact_sealed_members_checked=5,protocol_and_LF_checksum_checked=True,
        manual_steps=5,single_source_upload_commands=2,single_remote_source_download_commands=3,
        explicit_download_destinations_checked=True,exact_manual_launch_PIN_and_paths_checked=True,
        measured_diagnostic_not_training_or_quality_claim=True,manual_VM_probe_prepared_unrun=True,
        all_five_milestones_seven_families_and_manual_VM_boundary_retained=True,
        protected_bindings_checked=len(protected),model_qualified=False,goal_complete=False))
    print(dict(complete=True,updated_documents=6,new_documents=2,manual_commands_checked=5,goal_status='active'))


if __name__=='__main__':
    main()
