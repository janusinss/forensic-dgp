"""Fresh read-only closure of exact deletions, source assets and launch storage."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import urllib.request

HOME = Path('/home/janusdominic0')
ROOT = HOME/'forensic-dgp'
BANK = ROOT/'cctv_dgp_bank_comparison_v1_r4_B_vm'
PIN = '8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85'
PLAN = HOME/'cctv_dgp_bank_B4_output_cleanup_v2_plan.json'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(1024**2),b''): h.update(value)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    start = time.monotonic()
    signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('Finite300s read-only postcheck stop')))
    signal.alarm(300)
    assert Path.home().resolve() == HOME and os.getuid() == 1001 and socket.gethostname().split('.')[0] == 'forensic-dgp-thesis'
    req = urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/id',headers={'Metadata-Flavor':'Google'})
    assert urllib.request.urlopen(req,timeout=5).read().decode() == '4410777042005672095'
    assert sha(PLAN) == 'f622a1e1511ea4f00ec810f2558f87e20fad7fa48f6b717c358e20b9ab7c615f'
    plan = read(PLAN)
    cleanup = read(HOME/'dgp_bank_B4_output_cleanup_v2_receipts/cleanup_receipt.json')
    assert cleanup['complete'] and cleanup['files_removed'] == len(plan['candidates']) == 17347
    assert all(not Path(row['path']).exists() and not Path(row['path']).is_symlink() for row in plan['candidates'])
    assert sha(BANK/'protocol.json') == PIN
    p = read(BANK/'protocol.json')
    for name,expected in p['assets_sha256'].items(): assert sha(BANK/name) == expected, name
    hashes = {}
    for file in [BANK/'outputs/bank_comparison_v1/quality_stop_B.json',
        BANK/'outputs/bank_comparison_v1/state_B50/full_state.pth',BANK/'supervisor_receipt.json',
        BANK/'export.log',BANK/'export_exit_code.txt',
        ROOT/'cctv_dgp_residual_epochs_vm_v42/outputs/stopped_training_state.pt',
        ROOT/'cctv_dgp_residual_epochs_vm_v42/outputs/failure.json',
        ROOT/'cctv_dgp_residual_epochs_vm_v42/outputs/capacity_update50.json']:
        assert file.is_file() and not file.is_symlink(); hashes[file.relative_to(ROOT).as_posix()] = sha(file)
    gpu = subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=15)
    assert not gpu.stdout.strip(), 'GPU active; report no launch-ready claim'
    tmux = subprocess.run(['tmux','list-panes','-a','-F','#S|#{pane_current_command}|#{pane_dead}'],capture_output=True,text=True,timeout=15)
    assert tmux.returncode == 0 or (tmux.returncode == 1 and ('no server running' in tmux.stderr or 'No such file or directory' in tmux.stderr))
    for line in tmux.stdout.splitlines():
        fields = line.split('|'); assert fields[1] in {'bash','sh','zsh','fish'} and fields[2] == '0'
    assert not (ROOT/'cctv_dgp_bank_photometric_guard_v1_vm').exists(), 'New pilot already installed; do not duplicate commands'
    free = shutil.disk_usage(ROOT).free
    # These are pinned by the independently audited local full/reuse packets.
    packet_bytes, copied_install_reserve = 907131639, 64*1024**2
    transfer_archive_bytes, extracted_transfer_bytes = 3756519, 27204290
    post = free-packet_bytes-copied_install_reserve-transfer_archive_bytes-extracted_transfer_bytes
    result = dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),instance_id='4410777042005672095',
        hostname=socket.gethostname(),uid=os.getuid(),free_bytes=free,
        exact_deleted_paths_absent=len(plan['candidates']),reusable_B4_assets_rehashed=len(p['assets_sha256']),
        retained_stop_state_and_post_export_log_hashes=hashes,GPU_idle=True,tmux=tmux.stdout,
        expected_new_packet_bytes=packet_bytes,copy_reserve_bytes=copied_install_reserve,
        pending_transfer_archive_bytes=transfer_archive_bytes,pending_transfer_extracted_bytes=extracted_transfer_bytes,
        conservative_post_install_free_bytes=post,minimum_post_install_free_bytes=16*1024**3,
        manual_guard_space_requirement_met=post >= 16*1024**3,current_snapshot_not_future_launch_guarantee=True,
        all_scientific_cache_bytes_rehashed=False,new_pilot_installed_or_launched=False,
        model_gradient_or_training_calls=0,VM_power_change=False,worker_sha256=sha(__file__),seconds=time.monotonic()-start)
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
