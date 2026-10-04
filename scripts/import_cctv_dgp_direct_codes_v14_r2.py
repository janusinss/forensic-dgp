"""Verify a fresh V14 return, safely extract it, run an independent local audit."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block)
    return h.hexdigest()


def import_return(archive, completion, destination):
    assert archive.resolve().is_relative_to(ROOT/'outputs') and destination.resolve().is_relative_to(ROOT/'outputs')
    assert not destination.exists(), 'Preserve previous/partial extraction'
    terminal=json.loads(completion.read_text());assert terminal['complete'] and terminal['optimizer_updates']==1000
    bundle=ROOT/'outputs/cctv_dgp_direct_codes_vm_v14_r2'
    pin=sha(bundle/'direct_code_protocol_v14.json')
    assert terminal['protocol_sha256']==pin and archive.stat().st_size==terminal['bytes'] and sha(archive)==terminal['archive_sha256']
    assert Path(str(archive)+'.sha256').read_text().split()==[terminal['archive_sha256'],archive.name]
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();assert len({m.name for m in members})==len(members)
        assert sum(m.size for m in members)<2*1024**3
        for m in members:
            p=PurePosixPath(m.name)
            assert p.parts and not p.is_absolute() and '..' not in p.parts and ':' not in m.name and '\\' not in m.name
            assert m.isfile() or m.isdir()
            assert (destination/m.name).resolve().is_relative_to(destination.resolve())
        destination.mkdir();stream.extractall(destination,filter='data')
    assert sha(destination/'direct_code_protocol_v14.json')==pin
    plan=json.loads((bundle/'direct_code_protocol_v14.json').read_text())
    for name, expected in plan['sources_sha256'].items():assert sha(destination/name)==expected
    out=destination/'outputs/cctv_dgp_direct_codes_v14_r2'
    assert sha(out/'results.json')==terminal['results_sha256']
    subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'scripts/audit_cctv_dgp_direct_codes_v14_r2.py'),
        '--root',str(bundle),'--parent-bundle',str(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2'),
        '--parent-results',str(ROOT/'outputs/cctv_dgp_face_code_fit_return_v12_r2_verified/outputs/cctv_dgp_face_code_fit_v12'),
        '--expected-protocol-sha',pin,'--results',str(out),'--receipt',str(destination/'local_independent_audit.json')],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--completion',type=Path,required=True)
    parser.add_argument('--extract-to',type=Path,required=True)
    args=parser.parse_args();import_return(args.archive.resolve(),args.completion.resolve(),args.extract_to.resolve())
