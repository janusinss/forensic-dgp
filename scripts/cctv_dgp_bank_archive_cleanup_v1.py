"""Fresh, pinned maintenance transport for the bank comparison; no model work."""
from pathlib import Path
import cctv_dgp_multiscale_archive_cleanup_v1 as transport

ROOT = transport.ROOT
OUT = ROOT / 'outputs/cctv_dgp_bank_archive_cleanup_v1'
REMOTE_OUT = '/home/janusdominic0/dgp_bank_archive_cleanup_v1_receipts'
HOSTKEY = transport.HOSTKEY
INSTANCE_ID = transport.INSTANCE_ID
CLOUD = transport.CLOUD
BASE = transport.BASE
FLAGS = transport.FLAGS
sha = transport.sha
read = transport.read
write = transport.write

# Retain historical workers/receipts. All new operations write a separate ledger.
transport.OUT = OUT
INVENTORY_SOURCE = transport.INVENTORY_SOURCE.replace(
    "current_multiscale_installed=(root/'cctv_dgp_multiscale_calibration_vm_v1').exists()",
    "current_bank_installed=(root/'cctv_dgp_bank_comparison_v1_vm').exists()"
)
INVENTORY_SOURCE = INVENTORY_SOURCE.replace(
    "disk=shutil.disk_usage(root)",
    """root_archives=[]
for folder in (root,root/'outputs'):
 if not folder.is_dir():continue
 for path in sorted(folder.iterdir()):
  if not path.name.endswith('.tar.gz'):continue
  s=path.lstat()
  root_archives.append(dict(path=str(path),regular_file=stat.S_ISREG(s.st_mode),symlink=path.is_symlink(),bytes=s.st_size,allocated_bytes=s.st_blocks*512,uid=s.st_uid,nlink=s.st_nlink,inode=s.st_ino,mtime_ns=s.st_mtime_ns))
disk=shutil.disk_usage(root)"""
).replace("archives=archives,other_large_home_files", "archives=archives,root_archives=root_archives,other_large_home_files")
INVENTORY_SOURCE = INVENTORY_SOURCE.replace(
    "home_usage=call(['du','-x','-B1','--max-depth=1',str(home)],120)",
    "home_usage=call(['du','-x','-B1','--max-depth=1',str(home)],120),research_usage=call(['du','-x','-B1','--max-depth=1',str(root)],120)"
)
transport.INVENTORY_SOURCE = INVENTORY_SOURCE
invoke = transport.invoke
api = transport.api
ssh_source = transport.ssh_source


def inventory():
    assert not OUT.exists(), 'Preserve prior maintenance attempt'
    OUT.mkdir()
    (OUT / 'transport').mkdir()
    trust = ROOT / 'outputs/cctv_dgp_vm_storage_cleanup_20261007_v2/hostkey_verification.json'
    previous = read(trust)
    assert previous['complete'] and previous['known_historical_hostkey'] == HOSTKEY and previous['current_offered_key_exact_match']
    write(OUT / 'instance_before.json', api('api_before'))
    (OUT / 'inventory_source.py').write_text(INVENTORY_SOURCE, encoding='utf-8', newline='\n')
    guest = read(ssh_source(INVENTORY_SOURCE, 'inventory_before'))
    assert guest['complete'] and guest['gpu']['exit_code'] == guest['gpu_processes']['exit_code'] == 0
    assert 'NVIDIA L4' in guest['gpu']['stdout']
    write(OUT / 'inventory_before.json', guest)
    write(OUT / 'inventory_transport_binding.json', dict(complete=True,
          source_sha256=sha(OUT / 'inventory_source.py'), local_driver_sha256=sha(Path(__file__)),
          trust_receipt_sha256=sha(trust), hostkey_pinned=HOSTKEY, VM_started=False,
          files_removed=0, model_gradient_or_training_calls=0))
    print(dict(complete=True, free_GiB=guest['disk']['free_bytes']/1024**3,
               GPU_compute_idle=not guest['gpu_processes']['stdout'].strip(),
               home_archives=len(guest['archives']), root_archives=len(guest['root_archives']),
               current_bank_installed=guest['current_bank_installed'], files_removed=0), flush=True)


if __name__ == '__main__':
    inventory()
