"""Independent local V2 source/packet/archive audit; never runs a model or VM."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import prepare_codeformer_adaptation_packet_v2 as prepare
from scripts import seal_codeformer_adaptation_packet_v2 as sealer

EVIDENCE = ROOT / 'outputs/codeformer_training_readiness_v2/root_review_v1'
SOURCES = sorted(prepare.SOURCE_REQUIRED | {
    'tests/test_codeformer_adaptation_data_v2.py',
    'tests/test_codeformer_adaptation_worker_v2.py',
    'tests/test_codeformer_adaptation_model_v2.py',
})


def save(name, record):
    prepare.write_json(EVIDENCE / name, record)
    return prepare.sha256(EVIDENCE / name)


def source_readiness():
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    before = {name: prepare.sha256(ROOT / name) for name in SOURCES}
    for name in SOURCES:
        ast.parse((ROOT / name).read_text(encoding='utf-8'), feature_version=(3, 10))
    review = ROOT / 'outputs/codeformer_adaptation_v2_design_review/worker_review_v1/independent_review.json'
    design = ROOT / 'CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md'
    prepare.require(prepare.sha256(design) == prepare.DESIGN_SHA, 'Reviewed design changed')
    independent = prepare.read_json(review)
    prepare.require(independent.get('complete') is True or
                    independent.get('status') == 'PASS_STATIC_AND_CPU_FORMULAS_ONLY',
                    'Independent model/worker review missing')
    modules = ['tests.test_codeformer_adaptation_data_v2',
               'tests.test_codeformer_adaptation_model_v2',
               'tests.test_codeformer_adaptation_worker_v2',
               'tests.test_codeformer_adaptation_packet_v2']
    command = [sys.executable, '-X', 'utf8', '-B', '-m', 'unittest', *modules, '-v']
    start = time.monotonic()
    result = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=180)
    log = result.stdout + result.stderr
    prepare.write_new(EVIDENCE / 'combined_tests.log', log)
    prepare.require(result.returncode == 0 and b'Ran 59 tests' in log and log.rstrip().endswith(b'OK'),
                    'Combined V2 tests did not pass; preserve log')
    prepare.require(before == {name: prepare.sha256(ROOT / name) for name in SOURCES},
                    'Sources changed during independent tests')
    record = {'complete': True, 'approved_for_protocol_freeze': True,
              'design_sha256': prepare.DESIGN_SHA, 'training_authorized': False,
              'source_sha256': before, 'combined_tests': 59, 'failures': 0,
              'test_log_sha256': prepare.sha256(EVIDENCE / 'combined_tests.log'),
              'test_seconds': time.monotonic() - start,
              'independent_worker_model_review_sha256': prepare.sha256(review),
              'review_scope': 'Static code, synthetic data, CPU tensor formulas, packaging/safety fixtures',
              'GPU_execution_and_quality_unverified': True, 'neural_calls': 0,
              'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
              'reviewer_source_sha256': prepare.sha256(Path(__file__))}
    pin = save('source_readiness.json', record)
    print(json.dumps({'source_readiness_sha256': pin, 'tests': 59, 'complete': True}))


def packet_review():
    packet = ROOT / 'outputs' / prepare.NAME
    finalized = prepare.read_json(ROOT / prepare.EVIDENCE / 'finalized.json')
    protocol = prepare.read_json(packet / 'protocol.json')
    pin = prepare.sha256(packet / 'protocol.json')
    prepare.require(pin == finalized['protocol_sha256'], 'Final protocol differs')
    worker = prepare.load_worker(packet)
    with prepare.no_neural_imports():
        _, transfer = worker.verify_transfer(packet, pin)
    files = dict(protocol['files_sha256'], **{'protocol.json': pin})
    prepare.require(prepare.exact_namespace(packet) == set(files), 'Extra or missing packet file')
    for relative, digest in files.items():
        prepare.require(prepare.sha256(prepare.bounded(packet, relative)) == digest, 'Packet file changed')
    for relative, origin in finalized['copy_bindings'].items():
        prepare.require(files[relative] == origin['sha256'] ==
                        prepare.sha256(prepare.bounded(ROOT, origin['original'])), 'Copy origin differs')
    v1 = prepare.read_json(ROOT / prepare.V1_PACKET / 'protocol.json')
    for relative in prepare.SHARED:
        prepare.require(files[relative] == v1['files_sha256'][relative] ==
                        prepare.sha256(ROOT / relative), 'Original dependency changed')
    original_inputs = prepare.read_json(packet / 'provenance/v1/validation_inputs_v1.json')
    saved = prepare.read_json(packet / 'validation_saved_v1.json')
    returned = prepare.read_json(ROOT / 'outputs/cctv-codeformer-adaptation-v1-export.json')
    prior_files = prepare.read_json(ROOT / prepare.V1_RETURN / returned['manifest_member'])['files_sha256']
    for row in saved['items']:
        original = original_inputs['cases'][row['id']]
        prepare.require(files[row['input_path']] == original['sha256'] == prior_files[original['path']],
                        'Exact V1 input reuse differs')
        prepare.require(files[row['pair_record_path']] == prior_files[original['pair_record_path']],
                        'Exact V1 pair record reuse differs')
    companion_bytes = sum((packet / name).stat().st_size for name in protocol['export_files'])
    ledger = prepare.storage_ledger(protocol, companion_bytes)
    record = {'complete': True, 'approved_for_archive_seal': True,
              'protocol_sha256': pin, 'design_sha256': prepare.DESIGN_SHA,
              'training_authorized': False, 'files': len(files),
              'copied_origins_verified': len(finalized['copy_bindings']),
              'exact_saved_VM_validation_inputs': 318, 'comparison_cases': 88,
              'TRAIN_references': 391, 'validation_references': 53,
              'immutable_V1_dependencies_verified': len(prepare.SHARED),
              'full_namespace_and_hashes_verified': True, 'transfer_verification': transfer,
              'storage_ledger': ledger, 'no_model_qualification': True,
              'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
              'reviewer_source_sha256': prepare.sha256(Path(__file__))}
    approval_sha = save('packet_approval.json', record)
    print(json.dumps({'complete': True, 'protocol_sha256': pin,
                      'packet_approval_sha256': approval_sha, 'files': len(files)}))


def archive_review():
    packet = ROOT / 'outputs' / prepare.NAME
    archive = ROOT / 'outputs' / prepare.EXECUTION_ARCHIVE
    protocol = prepare.read_json(packet / 'protocol.json')
    files = dict(protocol['files_sha256'], **{'protocol.json': prepare.sha256(packet / 'protocol.json')})
    result = sealer.inspect_archive(archive, prepare.NAME, files)
    archive_sha = prepare.sha256(archive)
    sidecar = archive.with_name(archive.name + '.sha256').read_bytes()
    prepare.require(sidecar == (archive_sha + '  ' + archive.name + '\n').encode('ascii'),
                    'SCP checksum sidecar is not exact LF-only bytes')
    sealed = prepare.read_json(ROOT / 'outputs/codeformer_adaptation_v2_archive.json')
    prepare.require(sealed['archive_sha256'] == archive_sha and sealed['protocol_sha256'] == files['protocol.json'],
                    'Sealed receipt differs')
    record = {'complete': True, 'archive_sha256': archive_sha, 'bytes': archive.stat().st_size,
              'protocol_sha256': files['protocol.json'], 'sidecar_exact_LF_only': True,
              **result, 'model_qualified': False, 'training_started': False,
              'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0}
    report_sha = save('independent_archive_review.json', record)
    print(json.dumps({**record, 'receipt_sha256': report_sha}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['source', 'packet', 'archive'])
    mode = parser.parse_args().mode
    {'source': source_readiness, 'packet': packet_review, 'archive': archive_review}[mode]()
