"""Independent packet audit of Conditioned SFT V2 execution archive in an isolated temp stage."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
TESTING_ROOT = Path(r'C:\xampp\htdocs\YEAR 4\Testing')
WEIGHTS_SRC = TESTING_ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights'
BUILD_RECEIPT = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_source_preparation/build_receipt_v2.json'
ARCHIVE = ROOT / 'outputs/cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz'
ARCHIVE_SHA = ROOT / 'outputs/cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz.sha256'
AUDIT_OUT = ROOT / 'outputs/codeformer_conditioned_sft_learning_v2_source_preparation/independent_packet_audit_v2.json'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    receipt = json.loads(BUILD_RECEIPT.read_text(encoding='utf-8'))
    protocol_sha = receipt['protocol_sha256']
    actual_archive_sha = sha256_file(ARCHIVE)
    assert actual_archive_sha == receipt['archive_sha256'], 'Archive SHA256 mismatch vs build receipt'
    sidecar_text = ARCHIVE_SHA.read_text(encoding='ascii').strip()
    assert sidecar_text == f'{actual_archive_sha}  {ARCHIVE.name}', 'Sidecar .sha256 mismatch'

    with tempfile.TemporaryDirectory(prefix='v2_packet_audit_') as tmpdir:
        tmp_path = Path(tmpdir)
        with tarfile.open(ARCHIVE, mode='r:gz') as tar:
            members = tar.getmembers()
            for m in members:
                assert not m.name.startswith('/') and '..' not in m.name.split('/'), f'Unsafe tar path: {m.name}'
                assert '__pycache__' not in m.name and not m.name.endswith(('.pyc', '.pyo')), f'Bytecode in tar: {m.name}'
            tar.extractall(tmp_path, filter='data')

        extracted_stage = tmp_path / 'cctv_codeformer_conditioned_sft_learning_v2_vm'
        assert extracted_stage.is_dir(), 'Extracted stage directory missing'
        assert not (extracted_stage / 'weights/codeformer_r2.pth').exists(), 'R2 weight should be omitted from upload tar'
        assert not (extracted_stage / 'weights/codeformer_u96_adapter.pth').exists(), 'U96 weight should be omitted from upload tar'

        # Simulate install_weights_v2.py copying R2 & U96 weights into extracted stage
        shutil.copyfile(WEIGHTS_SRC / 'codeformer_r2.pth', extracted_stage / 'weights/codeformer_r2.pth')
        shutil.copyfile(WEIGHTS_SRC / 'codeformer_u96_adapter.pth', extracted_stage / 'weights/codeformer_u96_adapter.pth')

        # Run standalone bootstrap --verify-transfer in isolated subprocess
        proc = subprocess.run(
            [
                sys.executable, '-B', '-u',
                str(extracted_stage / 'scripts/run_conditioned_sft_learning_v2.py'),
                '--root', str(extracted_stage),
                '--protocol-sha', protocol_sha,
                '--verify-transfer',
            ],
            cwd=str(extracted_stage),
            capture_output=True,
            text=True,
            check=True,
        )
        transfer_out = json.loads(proc.stdout.strip())
        assert transfer_out['complete'] is True
        assert transfer_out['v2_optimization_cases'] == 336
        assert transfer_out['v2_monitors'] == 48
        assert transfer_out['v1_monitors'] == 40
        assert transfer_out['retained_old_quality'] == 56
        assert transfer_out['native_cctv_dev_cases'] == 4
        assert transfer_out['neural_gradient_optimizer_training_calls'] == 0

    audit_payload = {
        'complete': True,
        'audited_UTC': datetime.now(timezone.utc).isoformat(),
        'status': 'INDEPENDENT_V2_PACKET_AUDIT_PASSED',
        'protocol_sha256': protocol_sha,
        'archive_name': ARCHIVE.name,
        'archive_bytes': ARCHIVE.stat().st_size,
        'archive_sha256': actual_archive_sha,
        'pax_member_count': len(members),
        'verify_transfer_subprocess_output': transfer_out,
        'zero_bytecode_verified': True,
        'weights_omitted_from_upload_tar_verified': True,
        'post_install_full_static_verification_passed': True,
    }
    AUDIT_OUT.write_bytes((json.dumps(audit_payload, indent=2, sort_keys=True) + '\n').encode('utf-8'))
    print(json.dumps(audit_payload, indent=2))


if __name__ == '__main__':
    main()
