"""Prepare a distinct stopped-V32 loss diagnostic; execute no VM or neural work."""
import ast
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_v32_loss_gradient_v1_vm'
STEM = 'cctv-dgp-v32-loss-gradient-v1'
WORKER = 'scripts/' + NAME + '.py'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_preparation'
V32 = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
CLOSED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
ACTIVE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
OLD = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_vm'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def text_file(path, text):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def write(path, value):
    text_file(path, json.dumps(value, indent=2, allow_nan=False) + '\n')


def replace_once(source, before, after):
    assert source.count(before) == 1, before
    return source.replace(before, after)


BALANCED_SELECTION = '''def balanced_exposed_cases(schedule, all_cases):
    """First five touched references from each source; metadata alone decides."""
    sources = sorted({c['source'] for c in all_cases})
    assert len(sources) == 2
    counts = {source: 0 for source in sources}
    selected, references = [], set()
    for indices in schedule[:50]:
        group = [all_cases[i] for i in indices]
        assert len(group) == 5 and len({c['source_person_or_reference'] for c in group}) == 1
        assert {c['profile'] for c in group} == {'clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24'}
        assert len({c['source'] for c in group}) == 1 and all(c['role'] == 'train' for c in group)
        source, reference = group[0]['source'], group[0]['source_person_or_reference']
        if counts[source] < 5 and reference not in references:
            selected.extend(group); references.add(reference); counts[source] += 1
    assert len(selected) == 50 and len(references) == 10 and all(n == 5 for n in counts.values())
    return selected


'''


def common_replacements(source):
    source = source.replace('cctv_dgp_feature_fusion_gradient_v1', 'cctv_dgp_v32_loss_gradient_v1')
    source = source.replace('cctv-dgp-feature-fusion-gradient-v1', STEM)
    source = source.replace('cctv_dgp_profile_batches_vm_v31', 'cctv_dgp_feature_fusion_vm_v32_r2')
    source = source.replace('cctv_dgp_profile_batches_v31_return', 'cctv_dgp_feature_fusion_v32_r2_return')
    source = source.replace('cctv_dgp_profile_batches_v31_vm.py', 'cctv_dgp_feature_fusion_v32_vm.py')
    return source.replace('V31', 'V32').replace('v31', 'v32')


def worker_source():
    old_p = read(OLD / 'protocol.json')
    old_worker = OLD / 'scripts/cctv_dgp_feature_fusion_gradient_v1_vm.py'
    assert sha(old_worker) == old_p['assets_sha256']['scripts/cctv_dgp_feature_fusion_gradient_v1_vm.py'] == '8160d3db275601f371387cc3a2bc70501e2edcb57e882dd6722b7622018f4173'
    source = common_replacements(old_worker.read_text(encoding='utf-8'))
    source = replace_once(source, 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e', sha(V32 / 'protocol.json'))
    source = replace_once(source, 'own-DGP-feature-fusion-zero-update-gradient-diagnostic-v1', 'own-DGP-V32-stopped-loss-gradient-diagnostic-v1')
    source = replace_once(source, 'def matched_unexposed_cases(exposed, all_cases, touched):', 'def matched_unexposed_cases(exposed, all_cases, touched, preview_case_ids):')
    source = replace_once(source, "c['source_person_or_reference'] not in touched and c['id'] not in chosen]", "c['source_person_or_reference'] not in touched and c['id'] not in chosen and c['id'] in preview_case_ids]")
    source = replace_once(source, 'def dependencies(root, p):', BALANCED_SELECTION + 'def dependencies(root, p):')
    source = replace_once(source, "actual_seen = [v32['case_rows'][i] for batch in schedule[:10] for i in batch]", "actual_seen = balanced_exposed_cases(schedule, v32['case_rows'])")
    source = replace_once(source, "matched_unexposed_cases(actual_seen, v32['case_rows'], touched)", "matched_unexposed_cases(actual_seen, v32['case_rows'], touched, v32['preview_case_ids'])")
    source = replace_once(source, "v32['terms'] == p['terms'] and [r['name'] for r in v32['parameter_layout']] == p['decoder_parameter_names']", "v32['terms'] == p['terms'] and v32['parameter_layout'] == p['parameter_layout']")
    source = replace_once(source, "if name not in set(p['decoder_parameter_names']): assert torch.equal(value, initial_state[name]), name", "if name not in {r['name'] for r in p['parameter_layout']}: assert torch.equal(value, initial_state[name]), name")
    source = source.replace('Feature-fusion diagnostic600s cap', 'Stopped-V32 loss diagnostic600s cap')
    assert 'torch.optim' not in source and 'dgp_candidate_v31' not in source and 'cctv_dgp_profile_batches_vm_v31' not in source
    return source


def audit_source():
    original = ROOT / 'scripts/audit_cctv_dgp_feature_fusion_gradient_v1_return.py'
    assert sha(original) == 'cefd32a14d5b5453b8acf656d9be6daa14096d91fb36956eb954632794b1a5e5'
    source = common_replacements(original.read_text(encoding='utf-8'))
    return replace_once(source, "PIN = '5926dea8c477a9d51ff97186e6fec2b0d8eab8e67a7cf7a0ed1a6c716756e0af'", "PIN = 'PROTOCOL_PIN_TO_FILL'")


def guide(pin, digest):
    return f'''# Stopped V32 loss diagnostic: five manual steps

The audited V32 r2 pilot stopped at50/800 with0.970672% structure gain against
the unchanged1% requirement. This separate diagnostic inspects loss gradients at
the original and stopped states. **0 optimizer updates / 0 epochs / no new checkpoint.**
It does not resume V32 or start follow-on training.

Two50-case TRAIN cohorts have five references from each photographic source,
each with its clear control and four degradations. The exposed cohort uses the
first five touched references per source in the frozen schedule. The other uses
all50 fixed preflight/normalization previews, which were not optimized by50.
These are development TRAIN observations, not unseen or native CCTV evidence.
All visible facial features and the existing1%/10% preservation gates remain.

Require an idle existing NVIDIA L4/g2-standard-4 and **6GiB free**. Estimate
**1-5 minutes diagnostic plus1-3 minutes export**, based on earlier280-query runs.
Enforced limits: worker600s; external900s plus30s grace; export300s;
external export330s plus30s grace; allocated VRAM20GiB; raw return1.5GiB.
State/source mismatch or nonfinite output/loss/gradient stops and retains evidence.
No automatic historical launch, deletion, optimizer or application promotion.

Protocol SHA256: {pin}
Execution archive SHA256: {digest}

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_feature_fusion_vm_v32_r2/outputs/update50/dgp_candidate_v32.pth &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_v32_loss
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_loss_gradient.sh {pin}
```

Leave tmux with **Ctrl+B**, release, then **D**. Reconnect with the same step3
command. A completed diagnostic reports gradient_queries:280 and optimizer_updates:0.
Export complete:true means packaging only. If it stops, download the failure too;
keep all files, report the traceback and do not rerun or alter its assertions.

5. Download in **Windows Google Cloud SDK Shell**, after export finishes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Three separate remote downloads are required by Windows/PuTTY. Returned hashes,
all saved gradients, immutable state/source boundaries and CPU inference parity
need independent local audit before choosing a new finite training recipe.
Infinitesimal gradients alone do not reconstruct unsaved AdamW history or qualify
whole-face CCTV quality. Reserved final pixels remain unopened. Goal active.
'''


def main():
    started = time.monotonic()
    assert not BUNDLE.exists() and not PREP.exists()
    decision = read(ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1/user_decision.json')
    assert decision['discussion_satisfied'] and decision['selected_direction'] == 'Diagnose the current DGP learning design first'
    independent = read(ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1_r1/independent_audit.json')
    assert independent['complete'] and independent['cases_verified'] == 50
    closure = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_milestone_v1/independent_closure_audit.json')
    assert closure['complete'] and closure['early_one_percent_failure_preserved'] and closure['returned_files_verified'] == 27625
    old = read(V32 / 'protocol.json')
    assert sha(V32 / 'protocol.json') == '87582314eb1313a6914b4940496eeeb246c3023b7b6bf4f84da43e3f5e739d6c'
    worker, template = worker_source(), audit_source()
    ast.parse(worker, feature_version=(3, 10)); ast.parse(template, feature_version=(3, 10))
    local_worker = ROOT / WORKER
    text_file(local_worker, worker)
    template_path = ROOT / 'scripts/cctv_dgp_v32_loss_gradient_v1_return_audit_template.py'
    text_file(template_path, template)
    spec = importlib.util.spec_from_file_location('new_diagnostic_metadata_only', local_worker)
    helpers = importlib.util.module_from_spec(spec); spec.loader.exec_module(helpers)
    schedule = read(V32 / 'schedule.json')['batches']
    exposed = helpers.balanced_exposed_cases(schedule, old['case_rows'])
    touched = {old['case_rows'][i]['source_person_or_reference'] for batch in schedule[:50] for i in batch}
    matched, pairs = helpers.matched_unexposed_cases(exposed, old['case_rows'], touched, old['preview_case_ids'])
    assert {c['id'] for c in matched} == set(old['preview_case_ids']) and len(matched) == 50
    assert len(touched) == 50 and len(pairs) == 10 and len(set(pairs.values())) == 10
    cohorts = [{'name': 'exposed', 'cases': exposed, 'selection': 'First5 touched references per source from frozen first50 paired batches'},
               {'name': 'unexposed', 'cases': matched, 'selection': 'All50 fixed normalization/preflight TRAIN previews; source/profile matched and unoptimized by50'}]
    imported = read(ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_import.json')['files_sha256']
    names = {'protocol.json', *old['assets_sha256'], 'outputs/failure.json', 'outputs/early_structure_stop.json',
             'outputs/cohort_loss_setup.json', 'outputs/execution_receipt.json', 'outputs/update50/dgp_candidate_v32.pth',
             'outputs/update0/metrics.json', 'outputs/update50/metrics.json'}
    for c in exposed + matched:
        names.update(f'outputs/update{n}/{c["id"]}{suffix}' for n in [0, 50] for suffix in ['.png', '_embedding.npy'])
        names.add('outputs/update0/' + c['id'] + '_target_embedding.npy')
    closed = {}
    for name in sorted(names):
        source = V32 / name if name == 'protocol.json' or name in old['assets_sha256'] else CLOSED / name
        digest = sha(source)
        assert digest == imported[name]
        closed[name] = digest
    parent = read(PARENT / 'protocol.json')
    parent_hashes = {'protocol.json': sha(PARENT / 'protocol.json'), **parent['assets_sha256']}
    for name, digest in parent_hashes.items(): assert sha(PARENT / name) == digest
    data_names = {c[key] for group in cohorts for c in group['cases'] for key in ['input', 'target', 'observed']}
    data = {name: sha(MIXED / name) for name in sorted(data_names)}
    assert all(old['mixed_TRAIN_assets_sha256'][name] == digest for name, digest in data.items())
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    text_file(BUNDLE / WORKER, worker)
    shell = (OLD / 'scripts/run_fusion.sh').read_text(encoding='utf-8').replace('cctv_dgp_feature_fusion_gradient_v1_vm', NAME)
    text_file(BUNDLE / 'scripts/run_loss_gradient.sh', shell)
    normal = read(CLOSED / 'outputs/cohort_loss_setup.json')
    basis = [Path(__file__), local_worker, template_path,
             ROOT / 'scripts/verify_cctv_dgp_v32_loss_gradient_v1_packet.py',
             ROOT / 'scripts/audit_cctv_dgp_feature_fusion_gradient_v1_return.py',
             OLD / 'scripts/cctv_dgp_feature_fusion_gradient_v1_vm.py', OLD / 'scripts/run_fusion.sh',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_milestone_v1/milestone.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return_milestone_v1/independent_closure_audit.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_independent_audit.json',
             ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_failure_review_v1/visual_review.json',
             ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1/user_decision.json',
             ROOT / 'outputs/cctv_dgp_post_v32_architecture_review_v1/primary_research.json',
             ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1_r1/analysis.json',
             ROOT / 'outputs/cctv_dgp_post_v32_saved_losses_v1_r1/independent_audit.json']
    p = {'format': 'own-DGP-V32-stopped-loss-gradient-diagnostic-v1', 'date': '2026-10-07',
         'purpose': 'Measure existing restoration/preservation gradient tradeoffs at audited V32 r2 endpoints before choosing any new training recipe.',
         'hypotheses': ['Active per-case preservation gradients may oppose regional structure improvement, despite all17 delivered group gates passing.',
                        'Raw-gradient magnitudes/directions and batch coherence may differ across optimized and fixed-preview TRAIN cohorts and original/stopped states.'],
         'design_decision': decision['answer'], 'selection_independent_of_output_or_gradient': True,
         'fixed_previews_previously_used_in_normalization': True, 'cohorts': cohorts, 'states': [0, 50],
         'matched_reference_pairs': pairs, 'reference_repetition_matched': True,
         'normalizers': [normal['feature_normalizer'], normal['interior_normalizer']],
         'normalizer_policy': 'Exact frozen initial50 V32 normalizers and seven losses; no rescaling or removal of protection',
         'terms': old['terms'], 'parameter_layout': old['parameter_layout'], 'selected_parameters': 978243, 'selected_tensors': 23,
         'fusion_parameter_names': old['fusion_parameter_names'], 'fusion_parameters': 479616,
         'decoder_parameter_names': old['decoder_parameter_names'], 'decoder_parameters': 498627,
         'optimizer_updates': 0, 'backwards': 0, 'epochs': 0, 'gradient_queries': 280,
         'original_checkpoint_sha256': old['original_checkpoint_sha256'], 'original_DGP_state': old['original_DGP_state'],
         'stopped50_DGP_state': read(CLOSED / 'outputs/update50/metrics.json')['candidate_DGP_state'],
         'recognizer_state': old['frozen_recognizer_state'], 'V32_protocol_sha256': sha(V32 / 'protocol.json'),
         'V32_dependencies_sha256': closed, 'parent_dependencies_sha256': parent_hashes,
         'active_decoder_dependencies_sha256': {name: sha(ACTIVE / name) for name in ['cctv_dgp_active_original_decoder_v28.py', 'cctv_dgp_original_decoder_candidate_v1.py']},
         'TRAIN_assets_sha256': data, 'local_basis_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in basis},
         'assets_sha256': {f.relative_to(BUNDLE).as_posix(): sha(f) for f in sorted(BUNDLE.rglob('*')) if f.is_file()},
         'budgets': {'worker_seconds': 600, 'external_seconds': 900, 'kill_grace_seconds': 30, 'export_seconds': 300,
                     'external_export_seconds': 330, 'peak_vram_bytes': 20 * 1024**3, 'minimum_free_disk_bytes': 6 * 1024**3,
                     'export_uncompressed_bytes': 1536 * 1024**2, 'local_forward_audit_seconds': 1800},
         'CPU_raw_absolute_tolerance': 1e-5, 'CPU_PNG_byte_tolerance': 1, 'CPU_vector_absolute_tolerance': 5e-5,
         'CPU_batch_component_value_tolerance': 1e-4,
         'limits': 'TRAIN only, including prior exposed normalization previews. Infinitesimal gradients do not reconstruct AdamW or prove finite-step/native usefulness. Keep every visible feature.',
         'retained_gates': old['retained_capacity_gates'], 'V32_failure_must_remain': True, 'no_resume_or_new_checkpoint': True,
         'difference_from_prior_diagnostic': 'Both fusion and decoder actually changed in stopped V32; fixed100 source-balanced cases include all50 previews with measured raw regressions, unlike the previous V31 endpoint diagnostic.',
         'new_training_recipe_created': False, 'human_manual_VM_execution_required': True,
         'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False}
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    auditor = ROOT / 'scripts/audit_cctv_dgp_v32_loss_gradient_v1_return.py'
    text_file(auditor, template.replace('PROTOCOL_PIN_TO_FILL', pin))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file(): tar.add(path, arcname=NAME + '/' + path.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive)
    text_file(Path(str(archive) + '.sha256'), digest + '  ' + archive.name + '\n')
    text_file(ROOT / 'CCTV_DGP_V32_LOSS_GRADIENT_V1_VM.md', guide(pin, digest))
    receipt = {'complete': True, 'created_UTC': datetime.now(timezone.utc).isoformat(), 'protocol_sha256': pin,
               'archive_sha256': digest, 'archive_bytes': archive.stat().st_size, 'packet_files': len(p['assets_sha256']) + 1,
               'cases': 100, 'states': 2, 'gradient_queries_bound': 280, 'optimizer_updates': 0,
               'TRAIN_assets_verified': len(data), 'V32_dependency_hashes': len(closed),
               'prospective_return_auditor_sha256': sha(auditor), 'original_data_or_weights_uploaded': False,
               'local_neural_or_gradient_calls': 0, 'actual_VM_run_started': False, 'app_promotion': False,
               'goal_complete': False, 'seconds': time.monotonic() - started}
    write(PREP / 'preparation.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
