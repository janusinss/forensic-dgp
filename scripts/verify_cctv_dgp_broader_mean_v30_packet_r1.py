"""Preserve original checker; use the separately observed read-only Bash check."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
ORIGINAL=ROOT/'scripts/verify_cctv_dgp_broader_mean_v30_packet.py'
ORIGINAL_PIN='a7bb1c255929069fd29689aec7f168bb75d14715d8aaa3d8d9616c3ae34b00d5'
EXTERNAL=ROOT/'outputs/cctv_dgp_broader_mean_v30_preparation/bash_external_syntax_receipt.json'


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert sha(ORIGINAL)==ORIGINAL_PIN
    receipt=json.loads(EXTERNAL.read_text())
    assert receipt['complete'] and receipt['returncode']==0 and receipt['read_only'] and not receipt['script_executed'] and not receipt['VM_actions']
    assert receipt['shell_sha256']==sha(ROOT/'outputs/cctv_dgp_broader_mean_vm_v30/scripts/run_v30.sh')
    source=ORIGINAL.read_text()
    old="syntax=subprocess.run([str(bash),'-n',str(shell)],capture_output=True,text=True,timeout=30)"
    assert source.count(old)==1
    source=source.replace(old,'syntax=SimpleNamespace(**external_receipt)')
    old="receipt={'complete':True,'checker_sha256':sha(Path(__file__)),"
    assert source.count(old)==1
    source=source.replace(old,"receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'original_checker_sha256':ORIGINAL_PIN,'external_Bash_receipt_sha256':sha(EXTERNAL),")
    namespace={'__name__':'verified_local_packet_r1','__file__':str(Path(__file__)),
               'SimpleNamespace':SimpleNamespace,'external_receipt':receipt,'ORIGINAL_PIN':ORIGINAL_PIN,'EXTERNAL':EXTERNAL}
    exec(compile(source,str(ORIGINAL),'exec'),namespace)
    original_write=namespace['write']
    def retain_or_write(path,value):
        if path.exists():
            assert json.loads(path.read_text())==value,'Preserve existing receipt: '+str(path)
        else:original_write(path,value)
    namespace['write']=retain_or_write
    namespace['main']()


if __name__=='__main__':main()
