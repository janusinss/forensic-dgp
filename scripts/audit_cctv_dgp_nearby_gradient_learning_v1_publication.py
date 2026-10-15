"""Close published commands, exact histories and protected files; no VM calls."""
from datetime import datetime,timezone
from pathlib import Path
import re

from import_cctv_dgp_nearby_gradient_learning_v1_return import ROOT,OUT,NAME,STEM,PIN,read,sha,write


def main():
    folder=OUT/'cctv_dgp_nearby_gradient_learning_v1_publication';published=read(folder/'publication.json')
    prepared=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_preparation.json');transfer=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_transfer_audit.json')
    assert all(p['complete'] and p['manual_UNRUN'] for p in [published,prepared,transfer])
    assert PIN==published['protocol_sha256']==prepared['protocol_sha256']==transfer['protocol_sha256']==sha(OUT/NAME/'protocol.json')
    assert published['archive_sha256']==prepared['archive_sha256']==transfer['archive_sha256']==sha(OUT/(STEM+'-execution.tar.gz'))
    assert all(sha(ROOT/n)==h for n,h in published['evidence_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in published['new_documents_sha256'].items())
    assert len(published['documents'])==8 and len(published['new_documents_sha256'])==2
    for n,b in published['documents'].items():
        before=Path(b['backup']);assert sha(before)==b['before_sha256'] and sha(ROOT/n)==b['after_sha256']
        assert (ROOT/n).read_bytes()[b['prefix_bytes']:]==before.read_bytes()
        text=(ROOT/n).read_bytes()[:b['prefix_bytes']].decode('utf-8')
        assert text.startswith('# Nearby-gradient training comparison verified, UNRUN') and 'Goal incomplete.' in text
        for target in re.findall(r'\]\(<([^>]+)>\)',text):assert Path(target).is_file(),target
    guide=(ROOT/'CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_VM.md').read_text(encoding='utf-8')
    blocks=re.findall(r'```(?:bat|bash)\n(.*?)\n```',guide,re.S)
    assert blocks==list(published['exact_command_blocks'].values()) and len(blocks)==5
    upload,install,tmux,launch,download=blocks
    assert upload.count('gcloud compute scp')==2 and f'"{STEM}-execution.tar.gz"' in upload
    assert 'cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"' in upload and 'cd /d' in download
    assert f'sha256sum -c {STEM}-execution.tar.gz.sha256' in install and f'test ! -e ~/forensic-dgp/{NAME}' in install
    assert tmux=='tmux new-session -A -s dgp_nearby_gradient_learning_v1'
    assert launch.count(PIN)==2 and '--verify-transfer' in launch and 'bash scripts/run_learning.sh' in launch
    assert '--parent-root ../cctv_dgp_bank_photometric_guard_v1_vm' in launch
    scp=[line for line in download.splitlines() if line.startswith('gcloud compute scp')]
    assert len(scp)==3 and all(line.count('janusdominic0@forensic-dgp-thesis:')==1 for line in scp)
    assert all('--project=forensic-dgp-thesis --zone=us-central1-a' in line for line in scp)
    for suffix in ['-results.tar.gz','-results.tar.gz.sha256','-export.json']:
        assert sum(STEM+suffix+'"' in line for line in scp)==1
    assert '**9 GiB free after installation**' in guide and '**34-minute external\nceiling**' in guide
    assert PIN in guide and prepared['archive_sha256'] in guide
    assert 'Below 1%' in guide and 'at most one' in guide.lower()
    report=(ROOT/'CCTV_DGP_NEARBY_GRADIENT_LEARNING_V1_PREPARATION.md').read_text(encoding='utf-8')
    assert 'All **25 packet bindings**' in report and 'all 96 constraint rows' in report
    assert 'five existing PNG failures' in report and '2,040s (34 minutes)' in report
    assert not (OUT/NAME/'outputs').exists() and not (OUT/NAME/'review.log').exists()
    protected=read(OUT/'cctv_dgp_recognizer_precision_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(folder/'publication_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),publication_sha256=sha(folder/'publication.json'),protocol_sha256=PIN,
        archive_sha256=sha(OUT/(STEM+'-execution.tar.gz')),history_suffixes_verified=8,new_documents_verified=2,
        exact_command_blocks_verified=5,Windows_upload_and_single_remote_download_paths_checked=True,
        sealed_packet_unrun=True,protected_files_exact=True,local_neural_or_training_calls=0,VM_launch=False,
        model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,history_documents=8,manual_command_blocks=5,protected=True,manual_UNRUN=True))


if __name__=='__main__':main()
