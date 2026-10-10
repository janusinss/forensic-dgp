"""Current verified maintenance snapshot and pinned PuTTY transfer commands."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'outputs/cctv_dgp_head4_capacity_vm_v1'))
from cctv_dgp_head4_capacity_contract_v1 import read,write,sha


def main():
    source=ROOT/'CCTV_DGP_HEAD4_CAPACITY_V1_VM.md';dest=ROOT/'CCTV_DGP_HEAD4_CAPACITY_V1_VM_CURRENT.md'
    assert not dest.exists()
    prep=ROOT/'outputs/cctv_dgp_head4_capacity_v1_preparation';packet=read(prep/'preparation.json')
    assert sha(source)==packet['manual_guide_sha256'] and read(prep/'independent_packet_audit.json')['complete']
    cleanup=ROOT/'outputs/cctv_dgp_head4_archive_cleanup_v1';live=read(cleanup/'post_apply_inventory.json')
    assert live['complete'] and live['GPU_compute_idle'] and live['projection_after_upload_install_GiB']>=14
    key=live['hostkey_pinned'];assert key=='SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'
    text=source.read_text()
    old='The last API check found the existing VM stopped. Start it manually if needed\nfrom **Windows Google Cloud SDK Shell**; this starts billed VM runtime:'
    assert old in text
    text=text.replace(old,'The existing VM is running in the fresh maintenance inventory. If you stop it\nlater, start it manually from **Windows Google Cloud SDK Shell**; this starts\nbilled VM runtime:')
    lines=text.splitlines();text='\n'.join(line.replace('gcloud compute scp ','gcloud compute scp --scp-flag=-hostkey --scp-flag='+key+' ') if line.startswith('gcloud compute scp ') else line for line in lines)+'\n'
    prefix=f'''Current verified instructions —10 October2026. The gradient return audit and
new transfer audit pass. Authorized cleanup removed15 locally backed-up archive
copies, recovering6.21GiB. Fresh inventory at{live['UTC']} reports
**{live['free_GiB']:.2f}GiB free**, an idle GPU and the new training packet not installed.
The conservative upload/install projection leaves**{live['projection_after_upload_install_GiB']:.2f}GiB**;
step2 checks the required14GiB again. The transfer commands pin the retained
verified VM public SSH key, so its new IP needs no blind host-key acceptance.
No training is launched by this document or the maintenance connection.

The original prepared guide is preserved. Only the current VM status and transfer
host-key arguments differ; protocol, training code, archive and scientific gates
remain identical. Follow the five numbered steps below.

---

'''
    with dest.open('x',encoding='utf-8',newline='\n') as f:f.write(prefix+text)
    scps=[line for line in text.splitlines() if line.startswith('gcloud compute scp ')];assert len(scps)==5
    for line in scps:
        assert '--scp-flag=-hostkey --scp-flag='+key in line
        assert '--project=forensic-dgp-thesis' in line and '--zone=us-central1-a' in line and line.count('janusdominic0@forensic-dgp-thesis:')==1
    assert sum(line.endswith('"."') for line in scps)==3
    assert packet['protocol_sha256'] in text and packet['packet_sha256'] in text
    assert 'tmux new-session -A -s dgp_head4_capacity_v1' in text
    assert 'python -B -u scripts/supervise_cctv_dgp_head4_capacity_v1.py --root . --protocol-sha '+packet['protocol_sha256'] in text
    write(prep/'current_manual_commands_audit.json',{'complete':True,'current_guide_sha256':sha(dest),
        'original_guide_sha256':sha(source),'protocol_sha256':packet['protocol_sha256'],'packet_sha256':packet['packet_sha256'],
        'post_cleanup_inventory_sha256':sha(cleanup/'post_apply_inventory.json'),'hostkey_pinned':key,
        'one_remote_source_per_download':True,'scp_commands_verified':5,'training_recipe_unchanged':True,
        'goal_complete':False,'training_launched':False})
    print({'complete':True,'current_commands':str(dest),'verified_scp_lines':5,'training_launched':False})


if __name__=='__main__':main()
