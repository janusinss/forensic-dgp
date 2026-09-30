"""New-files-only package for the bounded trained-weight continuation."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.package_face_occlusion_vm import build_bundle,sha


def main():
    source=ROOT/'outputs/downloaded_face_occlusion/outputs/face_occlusion_pilot_vm/pretrained/epoch_10.pth'
    if sha(source)!='cdd1752bce8e4a087ce2aac5af73ba81b6316ac9118e47f12acd7a24fcb62669':
        raise ValueError('Audited starting detector changed')
    files=['scripts/train_face_occlusion_continuation_vm.py','scripts/package_face_occlusion_continuation_vm.py',
           'FACE_OCCLUSION_CONTINUATION.md','FACE_OCCLUSION_CONTINUATION_VM.md','FACE_OCCLUSION_REPLAY_RESULTS.md',
           'tests/test_face_occlusion_continuation.py']
    inventories=[ROOT/'outputs/coverage_protocol_v1/vm_inventory.json',ROOT/'outputs/face_occlusion_bundle_v1/inventory.json']
    expected=['c831635933c9b615b4849ad23811a3b82ebde1883aa30a5c2249b8879fa95623',
              'cb8485ef260bf83f8f4f3e6ce2f6fd45dc1a8726980130ddf4dfea956e83a4c1']
    for path,digest in zip(inventories,expected):
        if sha(path)!=digest or set(files)&set(json.loads(path.read_text())):
            raise ValueError('Existing inventory changed or new package overlaps old inputs')
    inventory=ROOT/'outputs/face_occlusion_continuation_bundle_v1/inventory.json'
    archive=ROOT/'outputs/face-occlusion-continuation-code.tar.gz'
    report=build_bundle(ROOT,files,inventory,archive)
    (inventory.parent/'build.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
