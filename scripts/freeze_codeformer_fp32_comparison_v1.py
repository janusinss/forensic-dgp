"""Freeze reviewed sources and parity inputs; never run a neural framework."""
import hashlib
import json
import os
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from research.codeformer_fp32_comparison_v1 import worker as w


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(1024**2), b''):
            digest.update(data)
    return digest.hexdigest()


def put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data); stream.flush(); os.fsync(stream.fileno())


def main():
    evidence = ROOT / 'outputs/codeformer_fp32_comparison_v1_preparation'
    packet = ROOT / 'outputs/cctv_codeformer_fp32_comparison_v1_vm'
    parent = ROOT / 'outputs/cctv_codeformer_adaptation_v2_vm'
    readiness_path = evidence / 'source_readiness.json'
    readiness = read(readiness_path)
    w.require(readiness['complete'] and readiness['approved_for_protocol_freeze'] and
              readiness['training_authorized'] is False, 'Independent approved source readiness required')
    data_audit = evidence / 'independent_data_resource_v3/preparation_audit.json'
    inputs = read(data_audit)
    w.require(inputs['complete'] and inputs['approved_data_and_prospective_resource_contract'],
              'Full actual data/resource readback required')
    for name, digest in readiness['review_evidence_sha256'].items():
        w.require(sha(ROOT / name) == digest, 'Readback evidence changed: ' + name)
    stage = read(evidence / 'stage_inputs.json')
    p = read(evidence / 'protocol_template_unsealed.json')
    w.require(p['parent_protocol_sha256'] == sha(parent / 'protocol.json'), 'Parent protocol changed')
    w.require(set(p['files_sha256']) == set(stage['files_sha256']) | set(stage['dependency_files_sha256']),
              'Unsealed input namespace differs')
    w.require(not (packet / 'protocol.json').exists(), 'Preserve previous frozen protocol')
    source_hashes = readiness['source_sha256']
    for name, digest in source_hashes.items():
        w.require(name not in p['files_sha256'] and sha(ROOT / name) == digest, 'Reviewed source changed/collides: ' + name)
        put(packet / name, (ROOT / name).read_bytes())
        p['files_sha256'][name] = digest
        p['export_files'].append(name)
    extra = {
        'source_readiness.json': readiness_path,
        'provenance/resource_ledger.json': evidence / 'final_resource_ledger.json',
        'provenance/independent_input_readback.json': data_audit,
    }
    for name, path in extra.items():
        w.require(name not in p['files_sha256'], 'Approval metadata collision')
        put(packet / name, path.read_bytes())
        p['files_sha256'][name] = sha(path)
        p['export_files'].append(name)
    p['source_readiness'] = {'path': 'source_readiness.json', 'sha256': sha(readiness_path)}
    p['resource_ledger'] = {'path': 'provenance/resource_ledger.json',
                            'sha256': sha(extra['provenance/resource_ledger.json'])}
    p['export_files'] = sorted(set(p['export_files']))
    w.validate_protocol(p)
    # Verify all original input/dependency bytes before committing the protocol.
    for name, digest in p['files_sha256'].items():
        path = parent / name if name in p['dependency_files_sha256'] else packet / name
        w.require(sha(path) == digest, 'Final pre-freeze binding differs: ' + name)
    payload = (json.dumps(p, indent=2, allow_nan=False) + '\n').encode('utf-8')
    put(packet / 'protocol.json', payload)
    protocol_sha = hashlib.sha256(payload).hexdigest()
    _, receipt = w.verify_transfer(packet, protocol_sha, dependency_root=parent)
    w.require(receipt['model_calls'] == receipt['gradient_queries'] == receipt['optimizer_updates'] == 0,
              'Static verification unexpectedly performs neural work')
    result = {'complete': True, 'protocol_sha256': protocol_sha, 'files': len(p['files_sha256']),
              'overlay_files': len(p['files_sha256']) - len(p['dependency_files_sha256']) + 1,
              'dependency_files_not_reuploaded': len(p['dependency_files_sha256']),
              'source_readiness_sha256': sha(readiness_path), 'static_transfer_readback': receipt,
              'freezer_sha256': sha(__file__), 'local_neural_gradient_optimizer_VM_calls': 0,
              'actual_training_run': False, 'model_qualified': False}
    put(evidence / 'sealed_sources.json', (json.dumps(result, indent=2) + '\n').encode('utf-8'))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
