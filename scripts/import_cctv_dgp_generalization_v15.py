"""Verify/extract one fresh return and independently audit it without neural calls."""
import argparse
import json
from pathlib import Path,PurePosixPath
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import cctv_dgp_generalization_v15 as v


def collect(archive,completion,destination):
    assert archive.resolve().is_relative_to(ROOT/'outputs') and destination.resolve().is_relative_to(ROOT/'outputs')
    assert not destination.exists(),'Preserve prior/partial extraction'
    terminal=v.read(completion);bundle=ROOT/'outputs/cctv_dgp_generalization_vm_v15';pin=v.sha(bundle/v.PLAN)
    assert terminal['complete'] and terminal['protocol_sha256']==pin and terminal['optimizer_updates']==terminal['backward_calls']==0
    assert v.sha(archive)==terminal['archive_sha256'] and archive.stat().st_size==terminal['bytes']
    assert Path(str(archive)+'.sha256').read_text().split()==[terminal['archive_sha256'],archive.name]
    with tarfile.open(archive,'r:gz') as t:
        members=t.getmembers();assert len({m.name for m in members})==len(members) and sum(m.size for m in members)<1024**3
        for m in members:
            p=PurePosixPath(m.name)
            assert (m.isfile() or m.isdir()) and p.parts and not p.is_absolute() and '..' not in p.parts and ':' not in m.name and '\\' not in m.name
            assert (destination/m.name).resolve().is_relative_to(destination)
        destination.mkdir();t.extractall(destination,filter='data')
    assert v.sha(destination/v.PLAN)==pin
    for name,expected in v.read(bundle/v.PLAN)['assets_sha256'].items():
        if name.endswith('.py') or name.startswith('lineage/'):assert v.sha(destination/name)==expected
    out=destination/'outputs/generalization_v15';assert v.sha(out/'results.json')==terminal['results_sha256']
    subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'scripts/audit_cctv_dgp_generalization_v15.py'),
        '--root',str(bundle),'--parent',str(ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2'),
        '--capacity',str(ROOT/'outputs/cctv_dgp_direct_codes_return_v14_r2/outputs/cctv_dgp_direct_codes_v14_r2'),
        '--expected-sha',pin,'--results',str(out),'--receipt',str(destination/'local_independent_audit.json')],cwd=ROOT,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['archive','completion','extract-to']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();collect(a.archive.resolve(),a.completion.resolve(),a.extract_to.resolve())
