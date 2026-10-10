"""Record verified maintenance and current manual instructions without promotion."""
from datetime import datetime, timezone
from pathlib import Path
import cctv_dgp_multiscale_archive_cleanup_v1 as t


def main():
    out = t.OUT
    assert not (out / 'closure.json').exists() and not (t.ROOT / 'CCTV_DGP_MULTISCALE_STORAGE_CLEANUP_V1.md').exists()
    audit = t.read(out / 'independent_audit.json')
    guest = t.read(out / 'inventory_after.json')
    plan = t.read(out / 'plan.json')
    receipt = t.read(out / 'remote_receipts/cleanup_receipt.json')
    assert audit['complete'] and guest['complete'] and t.read(out / 'post_execution.json')['complete']
    assert guest['target_archives_remaining'] == 0 and guest['conservative_post_install_GiB'] >= 7
    assert guest['gpu_processes']['exit_code'] == 0 and not guest['gpu_processes']['stdout'].strip()
    assert not guest['current_multiscale_installed'] and not guest['VM_started']
    protected = t.read(out / 'local_protected_sha256.json')
    for name, digest in protected.items():
        assert t.sha(t.ROOT / name) == digest
    old_closure = t.read(t.ROOT / 'outputs/cctv_dgp_multiscale_calibration_v1_milestone/closure.json')
    docs = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md']
    guide_name = 'CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md'
    before_dir = out / 'before_docs'
    assert not before_dir.exists()
    before_dir.mkdir()
    before_sha = {}
    for name in docs + [guide_name]:
        path = t.ROOT / name
        expected = old_closure['after_sha256'][name] if name in docs else old_closure['guide_sha256']
        assert t.sha(path) == expected
        (before_dir / name).write_bytes(path.read_bytes())
        before_sha[name] = t.sha(path)
    free = guest['free_GiB']
    post = guest['conservative_post_install_GiB']
    recovered = audit['observed_recovered_GiB']
    prefix = (
        '**Latest maintenance status — 10 October 2026: backup-verified VM cleanup complete; multiscale diagnostic still awaits the user\'s manual run. Full goal incomplete.**\n\n'
        f'Two obsolete Head4 capacity home transfer archives were removed after complete local SHA-256/GZIP backup checks and fresh workload checks. '
        f'Observed recovery: {recovered:.3f} GiB. Fresh guest inventory: {free:.3f} GiB free; '
        f'conservative projection after the new packet upload/install: {post:.3f} GiB, above the retained 7 GiB requirement.\n\n'
        f'Independent checks preserve {audit["research_metadata_entries_unchanged"]:,} research metadata entries, '
        f'{audit["protected_byte_hashes_unchanged"]:,} critical/evidence byte hashes, '
        f'{len(protected)} local bindings and all251 current packet assets. Scientific caches were not deleted; '
        'their full bytes were not all rehashed. The deletion proof is exact regular nlink1 files outside the research tree, with unchanged research metadata and critical hashes. '
        'All unpacked failed-pilot outputs, states, checkpoints, splits and provenance remain.\n\n'
        'The existing verified L4 instance is RUNNING, with an idle GPU and no tmux session at the final snapshot. '
        'Maintenance started no VM, model, gradient or training job and did not stop the VM. '
        'Upload, install and launch remain manual: CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md. '
        'Recheck storage/workload before launching; this snapshot is not a future availability guarantee. '
        'The previous quality failures and all five restoration/completion milestones remain unchanged. '
        'Report: CCTV_DGP_MULTISCALE_STORAGE_CLEANUP_V1.md. Evidence: outputs/cctv_dgp_multiscale_archive_cleanup_v1/.\n\n---\n\n'
    ).encode('utf-8')
    old_guide = (before_dir / guide_name).read_bytes().decode('utf-8')
    old_status = ('The existing VM was API-verified **TERMINATED** on 10 October 2026, instance ID\n'
                  '4410777042005672095. Its current guest free space is unknown. Require **7 GiB\n'
                  'free after installation**.')
    new_status = (f'The existing L4 VM is maintenance-verified **RUNNING**, instance ID\n'
                  f'4410777042005672095. After verified archive cleanup, **{free:.3f} GiB** is\n'
                  f'free; conservative projection after upload/install is **{post:.3f} GiB**.\n'
                  'The GPU is idle and no tmux session is active at the final snapshot. Recheck\n'
                  'availability before launching. Require **7 GiB free after installation**.')
    old_start = ('1. Start the existing VM manually:\n   ```bat\n'
                 '   gcloud compute instances start forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a\n   ```')
    new_start = ('1. Check that the existing VM is running:\n   ```bat\n'
                 '   gcloud compute instances describe forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a --format="value(status)"\n   ```\n'
                 '   Expected: `RUNNING`. Maintenance left it running; no model job was launched.')
    assert old_guide.count(old_status) == old_guide.count(old_start) == 1
    guide_bytes = old_guide.replace(old_status, new_status, 1).replace(old_start, new_start, 1).encode('utf-8')
    update_plan = dict(complete=True, before_sha256=before_sha, protected_manifest_sha256=t.sha(out / 'local_protected_sha256.json'),
                       prefix_utf8=prefix.decode(), guide_replacements=[[old_status, new_status], [old_start, new_start]],
                       cleanup_plan_sha256=t.sha(out / 'plan.json'), independent_audit_sha256=t.sha(out / 'independent_audit.json'),
                       final_inventory_sha256=t.sha(out / 'inventory_after.json'), model_qualification=False, manual_run_pending=True)
    t.write(out / 'documentation_plan.json', update_plan)
    for name in docs:
        assert t.sha(t.ROOT / name) == before_sha[name]
        (t.ROOT / name).write_bytes(prefix + (before_dir / name).read_bytes())
    assert t.sha(t.ROOT / guide_name) == before_sha[guide_name]
    (t.ROOT / guide_name).write_bytes(guide_bytes)
    report = (
        '# Verified L4 VM archive cleanup — 10 October 2026\n\n'
        f'Cleanup is complete and independently audited. It recovered **{recovered:.3f} GiB**; '
        f'the final guest snapshot has **{free:.3f} GiB free**. Conservative upload/install '
        f'projection is **{post:.3f} GiB**, exceeding the next diagnostic\'s unchanged 7 GiB requirement.\n\n'
        'Removed only these two obsolete, inactive, individually linked home copies:\n\n'
        '- `/home/janusdominic0/cctv-dgp-head4-capacity-v1-execution.tar.gz`\n'
        '- `/home/janusdominic0/cctv-dgp-head4-capacity-v1-results.tar.gz`\n\n'
        'Complete matching backups remain in the Windows project `outputs/` directory. Both passed full '
        'SHA-256 and GZIP integrity checks before deletion; remote size, allocated blocks, owner, inode, '
        'mtime, nlink1 and SHA-256 matched the frozen plan. Active GPU, tmux, research processes and '
        'open archive readers were checked again before removal. No recursive removal was used.\n\n'
        f'Independent return checks verify all six receipt members, the exact deletion ledger, unchanged '
        f'{audit["research_metadata_entries_unchanged"]:,} research metadata entries and '
        f'{audit["protected_byte_hashes_unchanged"]:,} critical/evidence hashes, '
        f'{len(protected)} local evidence bindings and all251 current packet assets. '
        'Original/current checkpoints, optimizer/scheduler/RNG state, failed gates, raw/PNG outputs, '
        'datasets, splits, logs, provenance and scientific caches remain. Cache bytes were not all '
        'rehashed: preservation is established by disjoint nlink1 home-file deletion plus the full '
        'research metadata comparison and critical byte hashes.\n\n'
        'Verified VM: project/instance `forensic-dgp-thesis`, zone `us-central1-a`, '
        'instance ID `4410777042005672095`, `g2-standard-4` with NVIDIA L4. Host key remains pinned '
        'and TLS validation remains enabled. The VM was already running and remains running. '
        'No model, gradient, diagnostic or training job was launched; there is no active GPU task '
        'or tmux session at the final snapshot. This is not a future launch availability guarantee.\n\n'
        'Next action remains the user\'s manual upload/install/tmux launch in '
        '`CCTV_DGP_MULTISCALE_CALIBRATION_V1_VM.md`. The execution archive and protocol hashes are '
        'unchanged. Model usefulness, longer training, the app and all covering families remain '
        'unqualified/incomplete; cleanup is not a quality milestone.\n\n'
        f'Cleanup plan SHA-256: `{t.sha(out / "plan.json")}`.\n\n'
        f'Final inventory UTC: `{guest["UTC"]}`.\n\n'
        'Evidence: `outputs/cctv_dgp_multiscale_archive_cleanup_v1/plan.json`, '
        '`remote_receipts/cleanup_receipt.json`, `independent_audit.json`, '
        '`inventory_after.json` and saved transport/runtime records.\n'
    )
    (t.ROOT / 'CCTV_DGP_MULTISCALE_STORAGE_CLEANUP_V1.md').write_text(report, encoding='utf-8', newline='\n')
    for name, digest in protected.items():
        assert t.sha(t.ROOT / name) == digest
    t.write(out / 'closure.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
          before_sha256=before_sha, after_sha256={name: t.sha(t.ROOT/name) for name in docs+[guide_name]},
          protected_sha256=protected, closure_script_sha256=t.sha(Path(__file__)),
          documentation_plan_sha256=t.sha(out / 'documentation_plan.json'),
          report_sha256=t.sha(t.ROOT / 'CCTV_DGP_MULTISCALE_STORAGE_CLEANUP_V1.md'),
          cleanup_plan_sha256=t.sha(out / 'plan.json'), independent_audit_sha256=t.sha(out / 'independent_audit.json'),
          final_inventory_sha256=t.sha(out / 'inventory_after.json'), model_gradient_or_training_calls=0,
          manual_VM_run_pending=True, model_qualification=False, goal_complete=False))
    print(dict(complete=True, updated_documents=4, new_cleanup_report=True, manual_VM_run_pending=True), flush=True)


if __name__ == '__main__':
    main()
