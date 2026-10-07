"""Static-only source, license, and adapter prerequisite readback."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_mat_source_review_v1_r1'
COMMIT = 'd273d891ecdad2e1df106516423a75bc45b2d800'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    acquired = json.loads((OUT / 'acquisition.json').read_text())
    assert acquired['complete'] and acquired['source_commit'] == COMMIT
    assert acquired['source_only'] and not acquired['source_executed'] and not acquired['checkpoint_downloaded']
    assert acquired['model_or_gradient_calls'] == acquired['optimizer_updates'] == 0
    assert [r['file'] for r in acquired['files']] == ['README.md', 'LICENSE', 'generate_image.py', 'legacy.py', 'requirements.txt']
    for row in acquired['files']:
        assert row['source_url'] == 'https://raw.githubusercontent.com/fenglinglwb/MAT/' + COMMIT + '/' + row['file']
        assert (OUT / row['file']).stat().st_size == row['bytes'] and sha(OUT / row['file']) == row['sha256']
    source = (OUT / 'generate_image.py').read_text(encoding='utf-8-sig')
    legacy = (OUT / 'legacy.py').read_text(encoding='utf-8-sig')
    readme = (OUT / 'README.md').read_text(encoding='utf-8-sig')
    license_text = (OUT / 'LICENSE').read_text(encoding='utf-8-sig')
    trees = [ast.parse(value) for value in [source, legacy]]
    assert 'Attribution-NonCommercial 4.0 International' in license_text
    assert 'code and models in this repo are for research purposes only' in readme
    assert '0 and 1 values in a mask refer to masked and remained pixels' in readme
    assert 'CelebA-HQ-256' in readme and 'FFHQ-512' in readme
    assert "device = torch.device('cuda')" in source
    assert "seed = 240" in source and "if resolution != 512:\n        noise_mode = 'random'" in source
    assert 'legacy.load_network_pkl(f)' in source and '_LegacyUnpickler(f).load()' in legacy
    assert 'class _LegacyUnpickler(pickle.Unpickler)' in legacy and 'return super().find_class(module, name)' in legacy
    assert all(isinstance(tree, ast.Module) for tree in trees)
    failed = ROOT / 'outputs/completion_mat_source_review_v1'
    failure = json.loads((failed / 'acquisition_failure.json').read_text())
    assert failure['downloaded_source_files'] == 0 and not failure['checkpoint_downloaded']
    assert sha(failed / 'acquirer_at_transport_failure.py') == failure['source_sha256']
    assert sha(ROOT / 'scripts/acquire_completion_mat_source_review_v1.py') == failure['source_sha256']
    receipt = {'complete': True, 'checker_sha256': sha(Path(__file__)),
               'acquisition_sha256': sha(OUT / 'acquisition.json'), 'source_commit': COMMIT,
               'official_source_files_verified': 5, 'source_bytes': sum(r['bytes'] for r in acquired['files']),
               'code_parsed_not_executed': 2, 'license': 'CC-BY-NC-4.0; research-only README; dependency terms need separate audit',
               'mask_zero_is_estimated_one_is_retained': True, 'default_CUDA_entrypoint': True,
               'official_seed': 240, 'non512_forces_random_noise': True,
               'whole_network_pickle_loader_needs_separate_review': True,
               'model_download_provenance_verified': False, 'local_runtime_or_parity_verified': False,
               'initial_restricted_network_failure_preserved': True,
               'checkpoint_downloaded': False, 'source_executed': False, 'model_or_gradient_calls': 0,
               'optimizer_updates': 0, 'app_changes': False, 'quality_acceptance': False, 'goal_complete': False}
    with (OUT / 'independent_source_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
