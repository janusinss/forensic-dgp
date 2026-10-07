"""Retain the first verifier; correct only its metadata test fixture, not the packet."""
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_preparation'
OUT = PREP / 'independent_audit_r1'
ORIGINAL = ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_routing_r2.py'


def sha(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = json.loads((ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2/protocol.json').read_text())
    assert sha(ORIGINAL) == p['local_basis_sha256'][ORIGINAL.relative_to(ROOT).as_posix()]
    assert not OUT.exists()
    OUT.mkdir()
    for name in ['preparation.json', 'bash_syntax.json']:
        with (OUT/name).open('xb') as stream: stream.write((PREP/name).read_bytes())
    spec = importlib.util.spec_from_file_location('immutable_original_R2_packet_verifier', ORIGINAL)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    source = inspect.getsource(module.verify_actual_transfer_branch)
    before = "'json':json,'p':{}"
    after = "'json':json,'p':{'assets_sha256':{}}"
    assert source.count(before) == 1
    # The packet contains this real field. The first mock omitted it, so its
    # final transfer-status print failed after valid guards/dependency checks.
    exec(compile(source.replace(before,after), '<corrected-metadata-fixture-only>', 'exec'), module.__dict__)
    module.PREP = OUT
    module.__file__ = __file__
    module.main()
    receipt = {'complete':True,'fixture_checker_sha256':sha(Path(__file__)),
               'original_verifier_sha256':sha(ORIGINAL),
               'original_verifier_preserved_byte_identical':True,
               'correction':'Provide assets_sha256 in the simulated protocol for the actual transfer-status print.',
               'only_test_fixture_changed':True,'packet_and_protocol_unchanged':True,
               'original_failed_verifier_receipt_sha256':sha(PREP/'verifier_attempt1.json'),
               'independent_packet_audit_sha256':sha(OUT/'independent_packet_audit.json'),
               'model_imports':0,'gradient_calls':0,'optimizer_updates':0,'VM_writes':0,
               'goal_complete':False}
    with (OUT/'fixture_correction_receipt.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(receipt,stream,indent=2)
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__':
    main()
