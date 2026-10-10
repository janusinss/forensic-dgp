"""Real archive-member scope regression; imports auditors without running them."""
import argparse
import importlib.util
from pathlib import Path
import sys
import tarfile

from cctv_dgp_spatial_fit_v40_contract import read, sha, write

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'cctv_dgp_spatial_fit_v40_return/'


def check(path):
    spec = importlib.util.spec_from_file_location('prospective_scope_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    protocol = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41/protocol.json')
    diagnostic = read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_archive_scope_diagnostic.json')
    assert diagnostic['prefixes'] == {PREFIX.rstrip('/'): 27551}
    first = diagnostic['first_members'][0]
    member = tarfile.TarInfo(first['name']); member.size = first['bytes']
    accepted, total = module.safe_members([member], protocol)
    assert accepted == [(member, 'outputs/cache_receipt.json')] and total == member.size

    def reject(members):
        try:
            module.safe_members(members, protocol)
        except AssertionError:
            return
        raise AssertionError('Unsafe or wrong-scope member was accepted')

    mutations = []
    for name, kind, size in [
        (PREFIX+'../protocol.json', tarfile.REGTYPE, 1),
        ('other_return/protocol.json', tarfile.REGTYPE, 1),
        ('cctv_dgp_pcgrad_fit_v41_return/protocol.json', tarfile.REGTYPE, 1),
        (PREFIX+'protocol.json', tarfile.SYMTYPE, 1),
        (PREFIX+'protocol.json', tarfile.LNKTYPE, 1),
        (PREFIX+'protocol.json/', tarfile.DIRTYPE, 0),
        (PREFIX+'protocol.json', tarfile.REGTYPE, 16*1024**2+1),
        (PREFIX+'C:/protocol.json', tarfile.REGTYPE, 1),
        (PREFIX+'outputs\\failure.json', tarfile.REGTYPE, 1),
        (PREFIX+'scripts/returned_code.py', tarfile.REGTYPE, 1),
    ]:
        invalid = tarfile.TarInfo(name); invalid.type = kind; invalid.size = size
        reject([invalid]); mutations.append(name)
    reject([member, member])
    assert 'torch' not in sys.modules
    return {'complete': True, 'auditor_sha256': sha(path),
            'actual_first_legacy_member_accepted_with_relative_name': True,
            'unsafe_or_wrong_scope_members_rejected': len(mutations)+1,
            'no_extraction_or_neural_calls': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--auditor', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    result = check(args.auditor.resolve())
    write(args.receipt, result)
    print(result, flush=True)
