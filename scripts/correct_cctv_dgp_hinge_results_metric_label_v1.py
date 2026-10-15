"""Correct one report definition while retaining the failed R0 publication."""
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import hashlib
import json
import msvcrt
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PUB = ROOT/'outputs/cctv_dgp_decoder_hinge_repair_v1_analysis/publication'
OLD = ('Structure gain is the frozen recognizer-feature error proxy, not a percentage of\n'
       'human-visible clarity. The unchanged qualification threshold is at least 1%\n')
NEW = ('Structure gain is the relative reduction in landmark-supported high-frequency\n'
       'luminance MSE against the baseline, not a percentage of human-visible clarity.\n'
       'ArcFace recognition similarity is a separate preservation check. The unchanged\n'
       'qualification threshold is at least 1%\n')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    receipt = PUB/'metric_definition_correction_r1.json'
    assert not receipt.exists()
    publication = json.loads((PUB/'publication.json').read_text(encoding='utf-8'))
    before = PUB/'metric_definition_r0'
    before.mkdir(exist_ok=True)
    for name in ['publication.json', 'independent_audit.json']:
        original = (PUB/name).read_bytes()
        with (before/name).open('xb') as stream:
            stream.write(original)
    failure = json.loads((PUB/'independent_audit.json').read_text(encoding='utf-8'))
    assert failure['complete'] is False
    expected = publication['new_document_sha256']['CCTV_DGP_DECODER_HINGE_REPAIR_V1_RESULTS.md']
    paths = [(ROOT/'CCTV_DGP_DECODER_HINGE_REPAIR_V1_RESULTS.md', expected),
             (ROOT/'scripts/publish_cctv_dgp_decoder_hinge_repair_v1_results.py',
              sha(ROOT/'scripts/publish_cctv_dgp_decoder_hinge_repair_v1_results.py'))]
    api = ctypes.WinDLL('kernel32', use_last_error=True)
    api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
        wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    api.CreateFileW.restype = wintypes.HANDLE
    changes = []
    for path, digest in paths:
        handle = api.CreateFileW(str(path), 0xC0000000, 0, None, 3, 0x80, None)
        assert handle != ctypes.c_void_p(-1).value, str(path)
        fd = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
        with os.fdopen(fd, 'r+b') as stream:
            original = stream.read()
            assert hashlib.sha256(original).hexdigest() == digest
            assert original.count(OLD.encode()) == 1
            corrected = original.replace(OLD.encode(), NEW.encode(), 1)
            backup = before/path.name
            with backup.open('xb') as saved:
                saved.write(original); saved.flush(); os.fsync(saved.fileno())
            stream.seek(0); stream.write(corrected); stream.truncate()
            stream.flush(); os.fsync(stream.fileno())
        assert path.read_bytes() == corrected and sha(backup) == digest
        changes.append(dict(path=str(path), before_sha256=digest,
                            after_sha256=sha(path), backup=str(backup), exact_text_only=True))
    metric_root = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
    sources = {name: sha(metric_root/name) for name in ['frozen_raw_metrics.py', 'frozen_capacity_contract.py']}
    for name, row in publication['documents'].items():
        assert sha(ROOT/name) == row['after_sha256']
    for name, digest in publication['evidence_sha256'].items():
        assert sha(PUB.parent/name) == digest
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        supersedes='publication.json new_document_sha256 for results report only; all other bindings remain current',
        retained_failed_publication_sha256=sha(before/'publication.json'),
        retained_failed_independent_audit_sha256=sha(before/'independent_audit.json'),
        changes=changes, old_text=OLD, corrected_text=NEW,
        authoritative_metric_source_sha256=sources,
        metric='1 - candidate_degraded_mean_landmark_high_frequency_MSE / baseline_degraded_mean_landmark_high_frequency_MSE',
        luminance_weights=[.299, .587, .114], high_pass='subtract separable 13-tap sigma-2 Gaussian smoothing',
        numerical_values_and_verdict_unchanged=True, all10_instruction_prefixes_unchanged=True,
        evidence_and_model_files_unchanged=True, model_qualified=False,
        neural_calls=0, gradient_queries=0, optimizer_updates=0, agent_VM_calls=0)
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(record, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps(dict(complete=True, corrected_files=2, numbers_and_failed_verdict_unchanged=True)))


if __name__ == '__main__':
    main()
