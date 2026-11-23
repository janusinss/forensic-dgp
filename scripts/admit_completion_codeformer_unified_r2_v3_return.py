"""Independent return audit and 497-tensor candidate extractor for Unified R2 CodeFormer completion v3.
Verifies return archive SHA-256, safe tar extraction, 497-tensor logical state, 165 frozen prior tensors,
278+54 Adam moments at step 64, and all clear/challenge/boundary/strict-gray/outside-exact gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
TRANSFER = Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
RETURN_ROOT = ROOT / 'outputs/completion_codeformer_unified_r2_v3_return'
CANDIDATE_ROOT = ROOT / 'outputs/completion_codeformer_unified_r2_v3_candidate'
STEM = 'cctv-completion-codeformer-unified-r2-v3'
MEMBER_ROOT = 'cctv_completion_codeformer_unified_r2_v3_results'
R1_CANDIDATE_CHECKPOINT = ROOT / 'outputs/completion_codeformer_finetune_v1_r1_candidate/codeformer_completion_finetune_r1.pth'
R1_CANDIDATE_SHA = '7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0'
ORIGINAL_SHA = 'b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797'
PARENT_SHA = '0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd'


def need(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read(path: Path):
    def reject(x):
        raise ValueError('NonfiniteJSON ' + x)
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), parse_constant=reject)


def safe_extract(archive_path: Path, destination: Path) -> Path:
    need(not destination.exists(), 'Return destination already exists; refusing overwrite')
    destination.mkdir(parents=True)
    extracted_root = destination / MEMBER_ROOT
    with tarfile.open(archive_path, mode='r:gz') as tar:
        for member in tar.getmembers():
            need(member.isreg(), 'Only regular files permitted in return archive: ' + member.name)
            pure = PurePosixPath(member.name)
            need(
                not pure.is_absolute()
                and '..' not in pure.parts
                and ':' not in member.name
                and '\\' not in member.name
                and pure.parts[0] == MEMBER_ROOT,
                'Unsafe return member path: ' + member.name,
            )
            out_path = destination.joinpath(*pure.parts)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as src, out_path.open('xb') as dst:
                shutil.copyfileobj(src, dst, 8 * 1024 * 1024)
            need(out_path.stat().st_size == member.size, 'Extracted size mismatch: ' + member.name)
    return extracted_root


def audit_return(archive_dir: Path, export_candidate: bool = False):
    import torch

    archive = archive_dir / (STEM + '-results.tar.gz')
    checksum_file = archive_dir / (STEM + '-results.tar.gz.sha256')
    export_json = archive_dir / (STEM + '-export.json')
    for p in (archive, checksum_file, export_json):
        need(p.is_file(), 'Missing return artifact: ' + str(p))
    expected_sha = checksum_file.read_text(encoding='ascii').strip().split()[0]
    actual_sha = sha(archive)
    export_receipt = read(export_json)
    need(
        actual_sha == expected_sha == export_receipt['archive_sha256'],
        'Return archive SHA-256 mismatch',
    )

    extracted = safe_extract(archive, RETURN_ROOT)
    manifest = read(extracted / 'export_manifest.json')
    for row in manifest['files']:
        f = extracted / row['path']
        need(
            f.is_file() and f.stat().st_size == row['bytes'] and sha(f) == row['sha256'],
            'Return file hash/size drift: ' + row['path'],
        )
    results = read(extracted / 'outputs/results.json')
    terminal_path = extracted / 'outputs/terminal_training_state.pth'
    need(terminal_path.is_file(), 'Missing terminal_training_state.pth')
    terminal_sha = sha(terminal_path)
    need(
        terminal_sha == results['terminal_training_state']['sha256'],
        'Terminal state SHA-256 mismatch vs results.json',
    )

    need(sha(R1_CANDIDATE_CHECKPOINT) == R1_CANDIDATE_SHA, 'R1 candidate checkpoint SHA-256 mismatch')
    base_ckpt = torch.load(R1_CANDIDATE_CHECKPOINT, map_location='cpu', weights_only=True)['params_ema']
    state = torch.load(terminal_path, map_location='cpu', weights_only=True)
    need(
        state['base_checkpoint_sha256'] == ORIGINAL_SHA
        and state['parent_training_state_sha256'] == PARENT_SHA,
        'Base or parent checkpoint SHA-256 mismatch in terminal state',
    )
    need(len(state['trainable_state']) == 332, 'Expected 332 trainable tensors in terminal state')
    merged_state = {**base_ckpt, **state['trainable_state']}
    need(len(merged_state) == 497, 'Expected 497 total CodeFormer tensors')

    def digest_dict(d, prefixes=None):
        h = hashlib.sha256()
        for name, val in sorted(d.items()):
            if prefixes and not any(name.startswith(p) for p in prefixes):
                continue
            arr = val.detach().cpu().contiguous()
            need(bool(torch.isfinite(arr).all().item()), 'Nonfinite tensor: ' + name)
            h.update(name.encode())
            h.update(str(tuple(arr.shape)).encode())
            h.update(str(arr.dtype).encode())
            h.update(arr.numpy().tobytes())
        return h.hexdigest()

    logical_sha = digest_dict(merged_state)
    prior_sha = digest_dict(merged_state, ('quantize.', 'generator.'))
    need(
        logical_sha == state['logical_state_sha256'] == results['final_logical_state_sha256'],
        'Reconstructed 497-tensor logical SHA-256 mismatch',
    )
    need(
        prior_sha == state['frozen_state_sha256'],
        'Frozen 165 prior tensor SHA-256 changed',
    )

    screen = results['numeric_screen']
    gates_pass = bool(
        results['cycles_completed'] == 32
        and results['stop_reason'] is None
        and screen['prospective_numeric_screen_pass']
        and screen['clear']['prospective_numeric_screen_pass']
        and screen['challenge']['prospective_numeric_screen_pass']
        and screen['GRAY_style_gate']['all_GRAY_macro_geometry_chroma_nonworse']
    )

    candidate_info = None
    if export_candidate and gates_pass:
        CANDIDATE_ROOT.mkdir(parents=True, exist_ok=True)
        cand_path = CANDIDATE_ROOT / 'codeformer_completion_r2.pth'
        need(not cand_path.exists(), 'Candidate checkpoint already exists')
        torch.save({'params_ema': merged_state}, cand_path)
        candidate_info = {
            'path': str(cand_path),
            'bytes': cand_path.stat().st_size,
            'sha256': sha(cand_path),
            'logical_state_sha256': logical_sha,
            'tensor_count': len(merged_state),
        }

    audit = {
        'format': 'completion-codeformer-unified-r2-v3-independent-return-audit-v1',
        'complete': True,
        'UTC': datetime.now(timezone.utc).isoformat(),
        'return_archive_sha256': actual_sha,
        'terminal_training_state_sha256': terminal_sha,
        'reconstructed_logical_state_sha256': logical_sha,
        'frozen_165_prior_sha256': prior_sha,
        'all_gates_pass': gates_pass,
        'numeric_screen': screen,
        'candidate_checkpoint': candidate_info,
    }
    audit_path = RETURN_ROOT / 'independent_return_audit.json'
    audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(audit, indent=2))
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--transfer-dir', type=Path, default=TRANSFER)
    parser.add_argument('--export-candidate', action='store_true')
    args = parser.parse_args()
    audit_return(args.transfer_dir, export_candidate=args.export_candidate)


if __name__ == '__main__':
    main()
