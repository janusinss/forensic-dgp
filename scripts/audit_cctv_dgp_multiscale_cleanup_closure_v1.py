"""Independent documentation closure and preserved-state byte checks."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_multiscale_archive_cleanup_v1'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    assert not (OUT / 'independent_closure_audit.json').exists()
    closure = read(OUT / 'closure.json')
    plan = read(OUT / 'documentation_plan.json')
    audit = read(OUT / 'independent_audit.json')
    guest = read(OUT / 'inventory_after.json')
    assert closure['complete'] and plan['complete'] and audit['complete'] and guest['complete']
    assert sha(OUT / 'documentation_plan.json') == closure['documentation_plan_sha256']
    assert sha(OUT / 'plan.json') == closure['cleanup_plan_sha256'] == plan['cleanup_plan_sha256'] == audit['plan_sha256']
    assert sha(OUT / 'independent_audit.json') == closure['independent_audit_sha256'] == plan['independent_audit_sha256']
    assert sha(OUT / 'inventory_after.json') == closure['final_inventory_sha256'] == plan['final_inventory_sha256']
    assert sha(ROOT / 'scripts/close_cctv_dgp_multiscale_archive_cleanup_v1.py') == closure['closure_script_sha256']
    assert sha(ROOT / 'CCTV_DGP_MULTISCALE_STORAGE_CLEANUP_V1.md') == closure['report_sha256']
    docs = {'PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md'}
    assert set(closure['before_sha256']) == docs | {'CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md'}
    for name, before_hash in closure['before_sha256'].items():
        before = (OUT / 'before_docs' / name).read_bytes()
        assert sha(OUT / 'before_docs' / name) == before_hash == plan['before_sha256'][name]
        current = (ROOT / name).read_bytes()
        assert sha(ROOT / name) == closure['after_sha256'][name]
        if name in docs:
            assert current.endswith(before) and current[:len(current)-len(before)] == plan['prefix_utf8'].encode('utf-8')
        else:
            expected = before.decode('utf-8')
            for old, new in plan['guide_replacements']:
                assert expected.count(old) == 1
                expected = expected.replace(old, new, 1)
            assert current == expected.encode('utf-8')
    assert sha(OUT / 'local_protected_sha256.json') == plan['protected_manifest_sha256']
    protected = read(OUT / 'local_protected_sha256.json')
    assert protected == closure['protected_sha256']
    for name, digest in protected.items():
        assert sha(ROOT / name) == digest, name
    assert guest['target_archives_remaining'] == 0 and guest['conservative_post_install_GiB'] >= 7
    assert not guest['current_multiscale_installed'] and not guest['gpu_processes']['stdout'].strip()
    assert read(OUT / 'instance_post.json')['status'] == 'RUNNING'
    assert closure['model_gradient_or_training_calls'] == 0 and closure['manual_VM_run_pending']
    assert not closure['model_qualification'] and not closure['goal_complete']
    result = dict(complete=True, checker_sha256=sha(Path(__file__)), protected_bindings_verified=len(protected),
                  historical_document_bytes_retained=3, guide_changes_exactly_declared=True,
                  VM_running=True, GPU_idle_at_final_snapshot=True, manual_VM_run_pending=True,
                  free_GiB=guest['free_GiB'], conservative_post_install_GiB=guest['conservative_post_install_GiB'],
                  model_gradient_or_training_calls=0, model_qualification=False, goal_complete=False)
    with (OUT / 'independent_closure_audit.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(result)


if __name__ == '__main__':
    main()
