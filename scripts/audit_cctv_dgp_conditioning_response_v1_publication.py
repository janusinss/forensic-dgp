"""Independently bind five manual command blocks and retained project history."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import re

from import_cctv_dgp_conditioning_response_v1_return import ROOT,OUT,NAME,STEM,PIN,read,sha,write


def main():
    stage=OUT/'cctv_dgp_conditioning_response_v1_publication';r=read(stage/'publication.json')
    assert r['complete'] and r['manual_diagnostic_unrun'] and not r['automatic_VM_launch']
    assert r['protocol_sha256']==PIN==sha(OUT/NAME/'protocol.json')
    assert len(r['documents'])==6 and len(r['new_documents_sha256'])==2
    for name,v in r['documents'].items():
        before=Path(v['backup']);assert before.resolve().is_relative_to(stage.resolve()) and before.name==name
        assert sha(before)==v['before_sha256'] and sha(ROOT/name)==v['after_sha256']
        assert (ROOT/name).read_bytes()[v['prefix_bytes']:]==before.read_bytes()
    for name,h in {**r['new_documents_sha256'],**r['evidence_sha256']}.items():assert sha(ROOT/name)==h,name
    p=read(OUT/NAME/'protocol.json');prep=read(OUT/'cctv_dgp_conditioning_response_v1_preparation.json');transfer=read(OUT/'cctv_dgp_conditioning_response_v1_transfer_audit.json')
    assert transfer['complete'] and transfer['exact_archive_bindings']==20 and transfer['parent_bindings']==5852
    assert transfer['protocol_sha256']==prep['protocol_sha256']==PIN
    assert sha(OUT/(STEM+'-execution.tar.gz'))==prep['archive_sha256']==r['archive_sha256']
    assert prep['archive_bytes']==10290328 and p['gradient_repeats']==2 and p['temporary_displacements']==6
    assert not (OUT/NAME/'outputs').exists() and not (OUT/NAME/'review.log').exists()
    guide=(ROOT/'CCTV_DGP_CONDITIONING_RESPONSE_V1_VM.md').read_text(encoding='utf-8')
    blocks=re.findall(r'```(?:bat|bash)\n(.*?)\n```',guide,re.S);assert len(blocks)==5
    windows='cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"'
    remote='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'
    scp='gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a'
    assert blocks[0].splitlines()==[windows,f'{scp} "{STEM}-execution.tar.gz" "{remote}"',f'{scp} "{STEM}-execution.tar.gz.sha256" "{remote}"']
    assert blocks[1].splitlines()==['cd ~ &&',f'sha256sum -c {STEM}-execution.tar.gz.sha256 &&','test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&',f'test ! -e ~/forensic-dgp/{NAME} &&',f'tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp']
    assert blocks[2]=='tmux new-session -A -s dgp_conditioning_response_v1'
    assert blocks[3].splitlines()==[f'cd ~/forensic-dgp/{NAME} &&','source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&',
        f'python -B -u scripts/cctv_dgp_conditioning_response_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {PIN} --verify-transfer &&',f'bash scripts/run_response.sh {PIN}']
    assert blocks[4].splitlines()==[windows]+[f'{scp} "{remote}{STEM}{suffix}" "."' for suffix in ['-results.tar.gz','-results.tar.gz.sha256','-export.json']]
    assert all(line.count(remote)==1 for line in blocks[4].splitlines()[1:])
    assert PIN in guide and prep['archive_sha256'] in guide and '9 GiB' in guide and '29.5' in guide and 'UNRUN' in guide
    worker=(OUT/NAME/'scripts/cctv_dgp_conditioning_response_v1_vm.py').read_text(encoding='utf-8')
    importer=(ROOT/'scripts/import_cctv_dgp_conditioning_response_v1_return.py').read_text(encoding='utf-8')
    assert f"STEM='{STEM}'" in worker and f"STEM='{STEM}'" in importer and f"PIN='{PIN}'" in importer
    assert p['maximum_gradient_queries']==640 and p['model_forward_cap']==200 and p['recognizer_forward_cap']==360
    assert p['optimizer_updates']==p['accepted_updates']==0
    bash=read(OUT/'cctv_dgp_conditioning_response_v1_bash_syntax.json')
    assert bash['complete'] and bash['read_only_syntax_check'] and not bash['VM_training_launched']
    assert bash['checked_shell_sha256']==sha(OUT/NAME/'scripts/run_response.sh')
    stop=read(OUT/'cctv_dgp_conditioning_response_v1_preparation_checker_stop/stop.json')
    a=ROOT/'scripts/audit_cctv_dgp_conditioning_response_v1_preparation.py';b=ROOT/'scripts/audit_cctv_dgp_conditioning_response_v1_preparation_r1.py'
    assert sha(a)==stop['original_checker_sha256'] and sha(b)==stop['replacement_checker_sha256']==transfer['auditor_sha256']
    change=stop['source_replacement'];assert a.read_text(encoding='utf-8').replace(change['before'],change['after'],1)==b.read_text(encoding='utf-8')
    report=(ROOT/'CCTV_DGP_CONDITIONING_RESPONSE_V1_PREPARATION.md').read_text(encoding='utf-8')
    assert 'diagnostic5%' in report and 'not an image preservation' in report and 'no new trained image' in report
    assert '640 derivative queries' in report and '800 RAW' in report and '1,000 cells' in report
    for name in [*r['new_documents_sha256'],*r['documents']]:
        for target in re.findall(r'\]\(<(C:/[^>]+)>\)',(ROOT/name).read_text(encoding='utf-8')[:14000]):assert Path(target).is_file(),target
    for file in [Path(__file__),ROOT/'scripts/publish_cctv_dgp_conditioning_response_v1.py',ROOT/'scripts/audit_cctv_dgp_conditioning_response_v1_return.py']:
        ast.parse(file.read_text(encoding='utf-8'),feature_version=(3,10))
    protected=read(OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(stage/'publication_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
        publication_sha256=sha(stage/'publication.json'),exact_manual_command_blocks=5,single_remote_source_per_download=True,
        exact_history_suffixes_and_backups_verified=6,all_local_evidence_links_verified=True,
        worker_export_import_download_namespaces_and_protocol_and_ASCII_LF_checksum_match=True,
        original_failed_model_qualification_and_preparatory_checker_stops_retained=True,
        original_model_and_app_files_unchanged=True,manual_diagnostic_unrun=True,automatic_VM_launch=False,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,command_blocks=5,history_documents=6,manual_diagnostic_unrun=True,model_qualified=False))


if __name__=='__main__':main()
