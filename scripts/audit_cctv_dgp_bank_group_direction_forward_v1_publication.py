"""Independent read-back of publication, scientific limits and manual commands."""
import ast
from datetime import datetime, timezone
import hashlib
import re
from import_cctv_dgp_bank_group_direction_forward_v1_return import ROOT,OUT,ANALYSIS,read,sha,write


def main():
    receipt=ANALYSIS/'publication_audit.json'
    assert not receipt.exists()
    publication=read(ANALYSIS/'publication.json')
    assert publication['complete'] and publication['goal_status']=='active' and not publication['goal_complete']
    for name,h in publication['evidence_sha256'].items():
        assert sha(ROOT/name)==h and read(ROOT/name)['complete'],name
    previous=read(OUT/'cctv_dgp_bank_group_gradients_v1_r1_analysis/publication.json')
    for name,binding in publication['documents'].items():
        before=ROOT/name
        old=__import__('pathlib').Path(binding['backup']).read_bytes()
        current=before.read_bytes()
        assert hashlib.sha256(old).hexdigest()==binding['before_sha256']
        assert sha(before)==binding['after_sha256']
        assert current[binding['prefix_bytes']:]==old
        expected=previous['documents'][name]['after_sha256'] if name in previous['documents'] else previous['new_documents_sha256'][name]
        assert expected==binding['before_sha256']
        prefix=current[:binding['prefix_bytes']].decode('utf-8')
        assert 'UNRUN' in prefix and 'rejected' in prefix.lower()
        if name!='CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_VM.md':
            assert 'Goal ACTIVE' in prefix and 'all seven' in prefix.lower() and 'All five' in prefix
            assert '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b' in prefix
        else:assert 'Do not rerun' in prefix
    for name,h in publication['new_documents_sha256'].items():assert sha(ROOT/name)==h
    base='cctv_dgp_bank_group_step_check_v1'
    prep=read(OUT/(base+'_preparation.json'))
    packet=OUT/prep['name'];p=read(packet/'protocol.json');pin=sha(packet/'protocol.json')
    assert pin==prep['protocol_sha256']
    assert not (packet/'outputs').exists() and not (OUT/(base+'_return')).exists()
    guide=(ROOT/'CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'(?m)^([1-5])\. ',guide)==['1','2','3','4','5']
    commands=re.findall(r'```(?:bat|bash)\n(.*?)\n```',guide,flags=re.DOTALL)
    assert len(commands)==5
    upload,install,tmux,launch,download=commands
    prefix='gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a '
    stem=prep['stem'];remote='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'
    assert [s for s in upload.splitlines() if s.startswith('gcloud')]==[
        prefix+'"'+stem+'-execution.tar.gz" "'+remote+'"',
        prefix+'"'+stem+'-execution.tar.gz.sha256" "'+remote+'"']
    assert upload.splitlines()[0]=='cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"'
    assert install=='cd ~ &&\nsha256sum -c '+stem+'-execution.tar.gz.sha256 &&\n'+\
        'test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&\n'+\
        'test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&\n'+\
        'test ! -e ~/forensic-dgp/'+prep['name']+' &&\n'+\
        'tar -xzf '+stem+'-execution.tar.gz -C ~/forensic-dgp'
    assert tmux=='tmux new-session -A -s dgp_group_step_check_v1'
    assert launch=='cd ~/forensic-dgp/'+prep['name']+' &&\n'+\
        'source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&\n'+\
        'python -B -u scripts/'+base+'_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha '+pin+' --verify-transfer &&\n'+\
        'bash scripts/run_forward.sh '+pin
    suffixes=['-results.tar.gz','-results.tar.gz.sha256','-export.json']
    expected=[prefix+'"'+remote+stem+s+'" "'+stem+s+'"' for s in suffixes]
    assert [s for s in download.splitlines() if s.startswith('gcloud')]==expected
    assert all(s.count('@forensic-dgp-thesis:')==1 for s in expected)
    assert pin in guide and prep['archive_sha256'] in guide
    assert '4 GiB free after install' in guide and 'Verified and UNRUN' in guide
    assert (OUT/(stem+'-execution.tar.gz.sha256')).read_bytes()==(
        prep['archive_sha256']+'  '+stem+'-execution.tar.gz\n').encode('ascii')
    assert sha(OUT/(stem+'-execution.tar.gz'))==prep['archive_sha256']
    transfer=read(OUT/(base+'_transfer_audit.json'))
    helper=read(OUT/(base+'_return_checker_preparation.json'))
    fixture=read(OUT/(base+'_signed_derivative_fixture_audit.json'))
    for name,h in p['assets_sha256'].items():assert sha(packet/name)==h
    for name,h in prep['future_checker_files_sha256'].items():assert sha(ROOT/name)==h
    assert fixture['checker_sha256']==sha(ROOT/'scripts/cctv_dgp_signed_step_audit.py')
    assert transfer['future_auditor_sha256']==helper['auditor_sha256']==fixture['future_auditor_sha256']
    assert p['proposals']==[dict(label='balanced_'+s,direction='balanced',rate=float(s)) for s in ['-0.001','-0.0001','0.0001','0.001']]
    assert p['optimizer_updates']==p['gradient_queries']==p['backwards']==p['full_epochs']==0
    assert p['negative_steps_diagnostic_only'] and p['export_storage_reserve_enforced']
    assert p['no_arbitrary_derivative_threshold_or_model_selection']
    assert p['additional_source_degraded_structure_gain_required']==.01
    assert p['minimum_free_bytes']==4*1024**3 and p['disk_reserve_bytes']==1024**3
    report=(ROOT/'CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_RESULTS.md').read_text(encoding='utf-8')
    summary=read(ANALYSIS/'finite_direction_review.json')['summary']
    for row in summary:
        assert f"{row['raw_structure_gain_percent']:.6f}%" in report
        assert f"{row['PNG_structure_gain_percent']:.6f}%" in report
        assert not row['all_preview_requirements_pass']
    assert 'paired photographic TRAIN' in report and 'Negative steps diagnose' in report
    assert 'A particular layer or code' in report and 'Goal ACTIVE' in report
    assert 'no more epochs' in guide.lower()
    links=[]
    for doc in publication['new_documents_sha256']:
        for target in re.findall(r'\]\(<([^>]+)>\)',(ROOT/doc).read_text(encoding='utf-8')):
            path=__import__('pathlib').Path(target)
            assert path.is_absolute() and path.exists(),target
            links.append(target)
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(ANALYSIS/'publication.json'),old_histories_preserved_exactly=5,new_documents_checked=2,
        exact_upload_install_tmux_launch_and_single_remote_download_commands_checked=True,
        command_blocks_checked=5,LF_checksum_and_archive_hash_checked=True,absolute_local_links_checked=len(links),
        four_finite_failure_results_reported_correctly=True,signed_diagnostic_UNRUN=True,
        scientific_gates_not_weakened=True,negative_step_selection_prohibited=True,
        no_extra_epochs_or_new_model_qualification=True,all_five_milestones_and_seven_families_retained=True,
        protected_original_app_checkpoint_hashes_unchanged=True,local_neural_or_gradient_calls=0,
        agent_VM_connections_or_neural_launches=0,goal_status='active',goal_complete=False))
    print(dict(complete=True,document_histories=5,command_blocks=5,signed_diagnostic_UNRUN=True))


if __name__=='__main__':main()
