"""Automatically sync RESTORATION branch commits and Conditioned SFT V1/V2 outputs into main (Testing)."""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path


RESTORATION_ROOT = Path(__file__).resolve().parents[1]
MAIN_ROOT = RESTORATION_ROOT.parent / 'Testing'

SYNC_OUTPUT_PATHS = (
    'outputs/cctv-codeformer-conditioned-sft-learning-v1-execution.tar.gz',
    'outputs/cctv-codeformer-conditioned-sft-learning-v1-execution.tar.gz.sha256',
    'outputs/codeformer_conditioned_sft_learning_v1_feasibility',
    'outputs/codeformer_conditioned_sft_learning_v1_source_preparation',
    'outputs/cctv_codeformer_conditioned_sft_learning_v1_vm',
    'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/transport.json',
    'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/pixels.json',
    'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/member_index.json',
    'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/visuals',
    'outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/checkpoints',
    'outputs/cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz',
    'outputs/cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz.sha256',
    'outputs/codeformer_conditioned_sft_learning_v2_feasibility',
    'outputs/codeformer_conditioned_sft_learning_v2_preparation',
    'outputs/codeformer_conditioned_sft_learning_v2_source_preparation',
    'outputs/cctv_codeformer_conditioned_sft_learning_v2_vm',
    'outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1',
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sync_file(src: Path, dst: Path) -> bool:
    if dst.exists() and dst.stat().st_size == src.stat().st_size and sha256_file(dst) == sha256_file(src):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def sync_outputs() -> int:
    copied = 0
    for rel in SYNC_OUTPUT_PATHS:
        src = RESTORATION_ROOT / rel
        dst = MAIN_ROOT / rel
        if not src.exists():
            continue
        if src.is_file():
            if sync_file(src, dst):
                copied += 1
        elif src.is_dir():
            for child in sorted(src.rglob('*')):
                if child.is_file():
                    sub_rel = child.relative_to(src)
                    if sync_file(child, dst / sub_rel):
                        copied += 1
    return copied


def sync_git_main() -> str:
    clean_env = {
        k: v for k, v in os.environ.items()
        if not k.startswith('GIT_')
    }
    subprocess.run(
        ['git', '-C', str(MAIN_ROOT), 'merge', '-X', 'theirs', '--no-edit', 'RESTORATION', '-m',
         'chore: auto-sync RESTORATION into main'],
        env=clean_env,
        check=True,
    )
    head = subprocess.check_output(
        ['git', '-C', str(MAIN_ROOT), 'rev-parse', '--short', 'HEAD'],
        env=clean_env,
        text=True,
    ).strip()
    return head


def main():
    head = sync_git_main()
    copied = sync_outputs()
    print(f'{{"main_head": "{head}", "outputs_synced": {copied}}}')


if __name__ == '__main__':
    main()
