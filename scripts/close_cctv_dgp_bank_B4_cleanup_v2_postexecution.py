"""Bind fresh live postcheck and separately retained post-export logs."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_bank_B4_output_cleanup_v2'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for value in iter(lambda:stream.read(1024**2),b''): h.update(value)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    path = OUT/'post_execution.json'; assert not path.exists()
    inventory = read(OUT/'inventory_after.json')
    assert inventory['worker_sha256'] == sha(ROOT/'scripts/cctv_dgp_bank_B4_cleanup_v2_postcheck_vm.py')
    assert inventory['complete'] and inventory['exact_deleted_paths_absent'] == 17347
    assert inventory['reusable_B4_assets_rehashed'] == 5630-1
    assert inventory['GPU_idle'] and not inventory['model_gradient_or_training_calls']
    assert not inventory['VM_power_change'] and not inventory['new_pilot_installed_or_launched']
    prep = read(ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_preparation.json')
    packet = ROOT/'outputs'/prep['name']; p = read(packet/'protocol.json')
    expected_packet = sum((packet/name).stat().st_size for name in p['assets_sha256'])+(packet/'protocol.json').stat().st_size
    patch = ROOT/'outputs'/prep['transfer_name']
    extracted = sum(file.stat().st_size for file in patch.rglob('*') if file.is_file())
    assert expected_packet == inventory['expected_new_packet_bytes'] and extracted == inventory['pending_transfer_extracted_bytes']
    assert inventory['pending_transfer_archive_bytes'] == prep['transfer_archive_bytes']
    post = inventory['free_bytes']-expected_packet-64*1024**2-extracted-prep['transfer_archive_bytes']
    assert post == inventory['conservative_post_install_free_bytes']
    assert inventory['manual_guard_space_requirement_met'] == (post >= 16*1024**3)
    before = read(OUT/'remote_receipts/cleanup_receipt.json')
    assert before['complete'] and before['files_removed'] == inventory['exact_deleted_paths_absent']
    preserved = inventory['retained_stop_state_and_post_export_log_hashes']
    import gzip
    with gzip.open(OUT/'remote_receipts/protected_after.json.gz','rb') as stream:
        protected_after = json.load(stream)['bank_retained_bytes_sha256']
    assert all(protected_after[name] == value for name,value in preserved.items())
    logs = {}
    for name in ['export.log','export_exit_code.txt']:
        file = OUT/'post_execution_files'/name
        key = 'cctv_dgp_bank_comparison_v1_r4_B_vm/'+name
        assert sha(file) == preserved[key]
        logs[str(file)] = sha(file)
    bindings = read(ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/name) == value for name,value in bindings.items())
    result = dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        inventory_after_sha256=sha(OUT/'inventory_after.json'),deleted_paths_absent=17347,
        all_reusable_B4_assets_freshly_rehashed=5629,post_export_logs_preserved_locally=logs,
        retained_stops_states_unchanged=True,protected_local_app_and_original_checkpoint_unchanged=True,
        current_free_bytes=inventory['free_bytes'],conservative_post_install_free_bytes=post,
        minimum_space_gate_met=inventory['manual_guard_space_requirement_met'],snapshot_not_future_launch_guarantee=True,
        new_pilot_installed_or_launched=False,model_gradient_or_training_calls=0,goal_complete=False)
    with path.open('x',encoding='utf-8',newline='\n') as stream: stream.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,free_GiB=inventory['free_bytes']/1024**3,post_install_GiB=post/1024**3,space_gate_met=result['minimum_space_gate_met']),flush=True)


if __name__ == '__main__': main()
