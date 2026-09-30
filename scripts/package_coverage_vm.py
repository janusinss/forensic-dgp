"""Build a self-contained VM payload from verified local comparison inputs."""
import hashlib
import json
from pathlib import Path
import tarfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path.cwd().resolve()
    protocol_path = Path('outputs/coverage_protocol_v1/protocol.json')
    protocol = json.loads(protocol_path.read_text())
    files = set(protocol['input_hashes'])
    files.add(protocol_path.as_posix())
    files.update(r['path'] for r in protocol['sources'])
    files.update(['scripts/train_coverage_vm.py','coverage_protocol.py','completion.py','completion_data.py',
                  'completion_inference.py','detector_training.py','detector_replay.py'])
    parent = 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    assert sha(parent) == 'c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93'
    files.add(parent)
    for manifest in ('dataset/detector_glare_review_v3/manifest.json','dataset/detector_training_extension_v2/manifest.json'):
        files.add(manifest)
        for r in json.loads(Path(manifest).read_text())['records']:
            files.update((Path(manifest).parent/r[k]).as_posix() for k in ('image','mask'))
    benchmark = Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm')
    for r in json.loads((benchmark/'manifest.json').read_text())['cases']:
        files.update((benchmark/folder/r['file']).as_posix() for folder in ('input','mask'))
    inventory = {}
    for relative in sorted(files):
        path = (root/relative.replace('\\','/')).resolve()
        assert path.is_relative_to(root)
        inventory[path.relative_to(root).as_posix()] = sha(path)
    inventory_path = root/'outputs/coverage_protocol_v1/vm_inventory.json'
    inventory_path.write_text(json.dumps(inventory,indent=2)+'\n')
    archive = root/'outputs/coverage-vm-bundle.tar.gz'
    if archive.exists():
        raise ValueError('Preserve existing archive; choose a versioned rebuild')
    with tarfile.open(archive,'w:gz') as tar:
        for relative in inventory:
            tar.add(root/relative,arcname='coverage_vm_bundle/'+relative,recursive=False)
        tar.add(inventory_path,arcname='coverage_vm_bundle/outputs/coverage_protocol_v1/vm_inventory.json',recursive=False)
    expected = {**inventory,'outputs/coverage_protocol_v1/vm_inventory.json':sha(inventory_path)}
    with tarfile.open(archive,'r:gz') as tar:
        assert len(tar.getmembers()) == len(expected)
        for member in tar.getmembers():
            relative = member.name.removeprefix('coverage_vm_bundle/')
            assert member.isfile() and relative in expected
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest() == expected[relative]
    checksum = sha(archive)
    archive.with_name(archive.name+'.sha256').write_bytes((checksum+'  '+archive.name+'\n').encode())
    print({'files':len(expected),'bytes':archive.stat().st_size,'sha256':checksum,'archive_verified':True})


if __name__=='__main__':
    main()
