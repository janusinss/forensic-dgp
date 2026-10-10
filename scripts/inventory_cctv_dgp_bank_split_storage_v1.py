"""Fresh pinned read-only VM storage/workload inventory; no model execution."""
from pathlib import Path
import argparse
import cctv_dgp_bank_archive_cleanup_v1 as t
import cctv_dgp_multiscale_archive_cleanup_v1 as transport


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--credential-access-retry', action='store_true')
    group.add_argument('--vm-restarted', action='store_true')
    args = parser.parse_args()
    suffix = '_r2' if args.vm_restarted else ('_r1' if args.credential_access_retry else '')
    dest = t.ROOT / ('outputs/cctv_dgp_bank_split_storage_v1' + suffix)
    assert not dest.exists(), 'Preserve maintenance evidence'
    dest.mkdir(); (dest / 'transport').mkdir()
    transport.OUT = dest
    t.OUT = dest
    t.write(dest / 'instance_before.json', t.api('api_before'))
    source = t.INVENTORY_SOURCE
    (dest / 'inventory_source.py').write_text(source, encoding='utf-8', newline='\n')
    guest = t.read(t.ssh_source(source, 'inventory_before'))
    assert guest['complete'] and guest['gpu']['exit_code'] == guest['gpu_processes']['exit_code'] == 0
    assert 'NVIDIA L4' in guest['gpu']['stdout']
    t.write(dest / 'inventory_before.json', guest)
    t.write(dest / 'inventory_binding.json', dict(complete=True, source_sha256=t.sha(dest/'inventory_source.py'),
        driver_sha256=t.sha(Path(__file__)), hostkey_pinned=t.HOSTKEY, files_removed=0,
        model_gradient_or_training_calls=0, VM_started=False))
    print(dict(complete=True, free_GiB=guest['disk']['free_bytes']/1024**3,
        GPU_compute_idle=not guest['gpu_processes']['stdout'].strip(), files_removed=0), flush=True)


if __name__ == '__main__':
    main()
