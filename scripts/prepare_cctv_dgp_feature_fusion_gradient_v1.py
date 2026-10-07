"""Prepare a new zero-update architecture diagnostic; never execute VM work."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

from cctv_dgp_v30_sampling_gradient_v1_vm import matched_unexposed_cases
from prepare_cctv_dgp_v30_sampling_gradient_v1 import SHELL

ROOT = Path(__file__).resolve().parents[1]
V31 = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
CLOSED = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
ACTIVE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
NAME = 'cctv_dgp_feature_fusion_gradient_v1_vm'
STEM = 'cctv-dgp-feature-fusion-gradient-v1'
WORKER = 'scripts/' + NAME + '.py'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_preparation'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def text_file(path, text):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(text)


def replace_once(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)


def worker_source():
    text = (ROOT / 'scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py').read_text(encoding='utf-8')
    text = text.replace('cctv_dgp_v30_sampling_gradient_v1', 'cctv_dgp_feature_fusion_gradient_v1')
    text = text.replace('cctv-dgp-v30-sampling-gradient-v1', STEM)
    text = text.replace('cctv_dgp_broader_mean_vm_v30', 'cctv_dgp_profile_batches_vm_v31')
    text = text.replace('cctv_dgp_broader_mean_v30_vm.py', 'cctv_dgp_profile_batches_v31_vm.py')
    text = text.replace('V30', 'V31').replace('v30', 'v31')
    text = text.replace('b7b6e26bef102b5ca873d4ff01cd4c4df6bcabfb898e56bd899757ac99b65aa1',
                        'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e')
    text = text.replace('V31-sampling-matched-zero-update-gradient-diagnostic-v1',
                        'own-DGP-feature-fusion-zero-update-gradient-diagnostic-v1')
    text = text.replace('498627', '978243').replace('len(parameters) == 12', 'len(parameters) == 23')
    text = text.replace('len(pieces) == 12', 'len(pieces) == 23').replace("p['selected_tensors'] == 12", "p['selected_tensors'] == 23")
    text = replace_once(text, "v31['terms'] == p['terms'] and v31['parameter_layout'] == p['parameter_layout']",
                        "v31['terms'] == p['terms'] and [r['name'] for r in v31['parameter_layout']] == p['decoder_parameter_names']")
    text = replace_once(text, "len(all_cases) == 3905 and len(touched) == 218", "len(all_cases) == 3905 and len(touched) == 50")
    text = replace_once(text, "helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)",
                        "sys.path.insert(0, str(closed))\n    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)")
    text = replace_once(text, "candidate = MeanCenteredOriginalDecoderV29(original.net)",
                        "candidate = MeanCenteredOriginalDecoderV29(original.net)\n        # New partition: original feature fusion, with the existing decoder for comparison.\n        for name, value in candidate.net.named_parameters():\n            if name in p['fusion_parameter_names']: value.requires_grad_(True)\n        assert not any(value.requires_grad for value in candidate.net.fpn.features.parameters())\n        assert not any(value.requires_grad for value in candidate.net.head4.parameters())")
    text = replace_once(text, "if name not in {r['name'] for r in layout}: assert torch.equal(value, initial_state[name]), name",
                        "if name not in set(p['decoder_parameter_names']): assert torch.equal(value, initial_state[name]), name")
    text = text.replace('Need4GiB', 'Need6GiB').replace('Sampling diagnostic600s cap', 'Feature-fusion diagnostic600s cap')
    text = replace_once(text, "'no_new_training_recipe': True, 'native_or_reserved_used': False,",
                        "'no_new_training_recipe': True, 'selected_tensors': 23, 'selected_parameters': 978243,\n                                     'fusion_parameter_names': p['fusion_parameter_names'], 'native_or_reserved_used': False,")
    assert 'torch.optim' not in text and '218' not in text
    return text


def audit_source():
    text = (ROOT / 'scripts/cctv_dgp_v30_sampling_gradient_v1_return_audit_template.py').read_text(encoding='utf-8')
    text = text.replace('cctv_dgp_v30_sampling_gradient_v1', 'cctv_dgp_feature_fusion_gradient_v1')
    text = text.replace('cctv-dgp-v30-sampling-gradient-v1', STEM)
    text = text.replace('cctv_dgp_broader_mean_v30_return', 'cctv_dgp_profile_batches_v31_return')
    text = text.replace('cctv_dgp_broader_mean_vm_v30', 'cctv_dgp_profile_batches_vm_v31')
    text = text.replace('cctv_dgp_broader_mean_v30_vm.py', 'cctv_dgp_profile_batches_v31_vm.py')
    text = text.replace('V30', 'V31').replace('v30', 'v31').replace('498627', '978243')
    text = text.replace('selected12_improvement_norms', 'selected23_improvement_norms')
    text = replace_once(text, '    return saved, report', '''    partitions = {}
    for label, names in [('feature_fusion', set(p['fusion_parameter_names'])),
                         ('decoder', set(p['decoder_parameter_names']))]:
        indices = np.concatenate([np.arange(r['start'], r['end']) for r in p['parameter_layout'] if r['name'] in names])
        block = total[:, indices]; ni = np.linalg.norm(block[:3].sum(0)); npres = np.linalg.norm(block[3:].sum(0))
        partitions[label] = {'parameters': len(indices), 'component_norms': np.linalg.norm(block, axis=1).tolist(),
                             'improvement_norm': float(ni), 'preservation_norm': float(npres),
                             'improvement_preservation_cosine': float(block[:3].sum(0) @ block[3:].sum(0)) / max(float(ni * npres), 1e-300),
                             'component_directional_derivatives': (-(block @ block.sum(0))).tolist(),
                             'all_improvement_tensors_nonzero': all(report['selected23_improvement_norms'][n] > 0 for n in names)}
    report['partition_analysis'] = partitions
    return saved, report''')
    return text


def guide(pin, digest):
    return f'''# Original DGP feature-fusion diagnostic: five manual steps

V31 stopped at50 with0.694525% against the1% early requirement. It is closed.
This new diagnostic measures gradients through11 original FPN fusion tensors and
compares them with12 decoder tensors. It performs **0 optimizer updates / 0 epochs**.
Original and stopped V31 weights are observations only; neither is resumed.
Two50-case TRAIN cohorts contain ten references each with their clear and four
degraded views, matched by source/profile. No native or reserved final cases are used.
All eyes, nose, mouth, outline and visible appearance remain in scope.

Use the existing NVIDIA L4/g2-standard-4 at ~/forensic-dgp, **6GiB free**, and an idle
GPU. Estimate **3-10 minutes plus1-5 minutes export**; runtime is not a success claim.
The worker stops at600s, the external supervisor at900s plus30s grace, export at300s
(external330s plus30s grace), allocated VRAM20GiB and uncompressed return1.5GiB.
Source/state mismatch or nonfinite output/loss/gradient stops and retains evidence.
No optimizer, .backward(), checkpoint writer, gate relaxation or app change is present.
Zero fusion gradients are recorded as a diagnostic finding, never a reason to train.

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
test -d ~/forensic-dgp/cctv_dgp_profile_batches_vm_v31/outputs/update50 &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_feature_fusion
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_fusion.sh {pin}
```

Expected final measurement:280 queries,0 updates. Detach with Ctrl+B, release, D.
Keep any failed export result. Export complete:true means packaging only.

5. Download from **Windows Google Cloud SDK Shell**, one remote file per call:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

The separate calls avoid PuTTY's multiple-remote-source error. The new prospective
local checker audits returned arrays/arithmetic and replays200 outputs with CPU
inference only. It never executes returned code or recomputes gradients locally.
A completed diagnostic alone does not select a new training recipe. Original
checkpoints, splits, research caches and every previous failure remain preserved.
Useful native restoration, all covering families and independent final review
are still required. The full goal remains active/incomplete.
'''


def main():
    start = time.monotonic()
    assert not BUNDLE.exists() and not PREP.exists()
    closure = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return_milestone/independent_closure_audit.json'
    assert read(closure)['complete'] and read(closure)['one_percent_structure_failure_and_stopped_checkpoint_preserved']
    reviewed = ROOT / 'outputs/cctv_dgp_post_v31_feature_path_review_v1/review.json'
    review = read(reviewed)
    assert review['complete'] and review['feature_fusion_parameters'] == 479616
    old = read(V31 / 'protocol.json')
    schedule = read(V31 / 'schedule.json')['batches']
    exposed = [old['case_rows'][i] for batch in schedule[:10] for i in batch]
    touched = {old['case_rows'][i]['source_person_or_reference'] for batch in schedule[:50] for i in batch}
    assert len(touched) == 50 and len({c['id'] for c in exposed}) == 50
    matched, pairs = matched_unexposed_cases(exposed, old['case_rows'], touched)
    assert len(pairs) == len(set(pairs.values())) == 10
    cohorts = [{'name': 'exposed', 'cases': exposed, 'selection': 'V31 first10 paired batches in frozen order'},
               {'name': 'unexposed', 'cases': matched, 'selection': 'First metadata-only distinct source/profile matches excluding all50 touched references'}]
    for cohort in cohorts:
        for begin in range(0, 50, 5):
            batch = cohort['cases'][begin:begin+5]
            assert len({c['source_person_or_reference'] for c in batch}) == 1
            assert {c['profile'] for c in batch} == {'clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24'}
    decoder_names = [r['name'] for r in old['parameter_layout']]
    fusion_names = [r['name'] for r in review['feature_fusion_tensors']]
    # Read the audited state ordering without constructing a model or graph.
    import torch
    state = torch.load(PARENT / 'weights/dgp_v2.pth', map_location='cpu', weights_only=True)
    layout, offset = [], 0
    for name, value in state.items():
        if name in set(fusion_names + decoder_names):
            layout.append({'name': name, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()})
            offset += value.numel()
    assert len(layout) == 23 and offset == 978243
    assert {r['name'] for r in layout} == set(fusion_names + decoder_names)
    dependency_names = ['protocol.json', *old['assets_sha256'], 'outputs/failure.json', 'outputs/early_structure_stop.json',
                        'outputs/cohort_loss_setup.json', 'outputs/execution_receipt.json', 'outputs/update50/dgp_candidate_v31.pth',
                        'outputs/update0/metrics.json', 'outputs/update50/metrics.json']
    for c in exposed + matched:
        dependency_names += [f'outputs/update{n}/{c["id"]}{s}' for n in [0, 50] for s in ['.png', '_embedding.npy']]
        dependency_names += ['outputs/update0/' + c['id'] + '_target_embedding.npy']
    closed = {name: sha(V31 / name if name == 'protocol.json' or name in old['assets_sha256'] else CLOSED / name)
              for name in sorted(set(dependency_names))}
    parent = read(PARENT / 'protocol.json')
    parent_hashes = {'protocol.json': sha(PARENT / 'protocol.json'), **parent['assets_sha256']}
    for name, digest in parent_hashes.items():
        assert sha(PARENT / name) == digest
    data_names = {c[key] for group in cohorts for c in group['cases'] for key in ['input', 'target', 'observed']}
    data = {name: sha(MIXED / name) for name in sorted(data_names)}
    assert all(old['mixed_TRAIN_assets_sha256'][name] == digest for name, digest in data.items())
    assert all(c['role'] == 'train' for group in cohorts for c in group['cases'])
    worker, template = worker_source(), audit_source()
    ast.parse(worker, feature_version=(3, 10)); ast.parse(template, feature_version=(3, 10))
    text_file(ROOT / WORKER, worker)
    template_path = ROOT / 'scripts/cctv_dgp_feature_fusion_gradient_v1_return_audit_template.py'
    text_file(template_path, template)
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    text_file(BUNDLE / WORKER, worker)
    shell = SHELL.replace('cctv_dgp_v30_sampling_gradient_v1_vm', NAME).replace('run_sampling.sh', 'run_fusion.sh')
    text_file(BUNDLE / 'scripts/run_fusion.sh', shell)
    normalizers = read(CLOSED / 'outputs/cohort_loss_setup.json')
    basis = [Path(__file__), ROOT / WORKER, template_path, reviewed, closure,
             ROOT / 'outputs/cctv_dgp_profile_batches_v31_return_milestone/milestone.json',
             ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json',
             ROOT / 'outputs/cctv_dgp_profile_batches_v31_failure_review_v1/visual_review.json',
             ROOT / 'scripts/cctv_dgp_v30_sampling_gradient_v1_vm.py',
             ROOT / 'scripts/cctv_dgp_v30_sampling_gradient_v1_return_audit_template.py',
             ROOT / 'scripts/prepare_cctv_dgp_v30_sampling_gradient_v1.py',
             ROOT / 'scripts/verify_cctv_dgp_feature_fusion_gradient_v1_packet.py',
             ROOT / 'tests/test_cctv_dgp_feature_fusion_gradient_v1.py']
    p = {'format': 'own-DGP-feature-fusion-zero-update-gradient-diagnostic-v1', 'date': '2026-10-07',
         'purpose': 'Measure original spatial feature-fusion gradients and preservation tradeoffs after audited V31 structure failure; choose no training recipe in advance.',
         'hypotheses': ['Frozen multiscale fusion is a distinct untested trainable partition.',
                        'Its component-gradient tradeoffs may differ from the active decoder; healthy gradients alone cannot prove useful finite updates.'],
         'selection_independent_of_output_or_gradient': True, 'cohorts': cohorts, 'states': [0, 50],
         'matched_reference_pairs': pairs, 'reference_repetition_matched': True,
         'normalizers': [normalizers['feature_normalizer'], normalizers['interior_normalizer']],
         'normalizer_policy': 'Exact frozen original50 V31 normalizers and seven losses; no rescaling',
         'terms': old['terms'], 'parameter_layout': layout, 'selected_parameters': 978243, 'selected_tensors': 23,
         'fusion_parameter_names': fusion_names, 'fusion_parameters': 479616,
         'decoder_parameter_names': decoder_names, 'decoder_parameters': 498627,
         'optimizer_updates': 0, 'backwards': 0, 'epochs': 0, 'gradient_queries': 280,
         'original_checkpoint_sha256': old['original_checkpoint_sha256'], 'original_DGP_state': old['original_DGP_state'],
         'stopped50_DGP_state': read(CLOSED / 'outputs/update50/metrics.json')['candidate_DGP_state'],
         'recognizer_state': old['frozen_recognizer_state'], 'V31_protocol_sha256': sha(V31 / 'protocol.json'),
         'V31_dependencies_sha256': closed, 'parent_dependencies_sha256': parent_hashes,
         'active_decoder_dependencies_sha256': {name: sha(ACTIVE / name) for name in
                                              ['cctv_dgp_active_original_decoder_v28.py', 'cctv_dgp_original_decoder_candidate_v1.py']},
         'TRAIN_assets_sha256': data, 'local_basis_sha256': {path.relative_to(ROOT).as_posix(): sha(path) for path in basis},
         'assets_sha256': {f.relative_to(BUNDLE).as_posix(): sha(f) for f in sorted(BUNDLE.rglob('*')) if f.is_file()},
         'budgets': {'worker_seconds': 600, 'external_seconds': 900, 'kill_grace_seconds': 30, 'export_seconds': 300,
                     'external_export_seconds': 330, 'peak_vram_bytes': 20 * 1024**3, 'minimum_free_disk_bytes': 6 * 1024**3,
                     'export_uncompressed_bytes': 1536 * 1024**2, 'local_forward_audit_seconds': 1800},
         'CPU_raw_absolute_tolerance': 1e-5, 'CPU_PNG_byte_tolerance': 1, 'CPU_vector_absolute_tolerance': 5e-5,
         'CPU_batch_component_value_tolerance': 1e-4,
         'limits': 'TRAIN only. Infinitesimal gradient directions do not reconstruct AdamW or prove finite-step/native usefulness. No individual feature may be dropped.',
         'retained_gates': old['retained_capacity_gates'], 'V31_failure_must_remain': True, 'no_resume_or_new_checkpoint': True,
         'new_training_recipe_created': False, 'human_manual_VM_execution_required': True,
         'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False}
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    prospective = ROOT / 'scripts/audit_cctv_dgp_feature_fusion_gradient_v1_return.py'
    text_file(prospective, template.replace('PROTOCOL_PIN_TO_FILL', pin))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file():
                tar.add(path, arcname=NAME + '/' + path.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive)
    text_file(Path(str(archive) + '.sha256'), digest + '  ' + archive.name + '\n')
    text_file(ROOT / 'CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_VM.md', guide(pin, digest))
    write(PREP / 'preparation.json', {'complete': True, 'created_utc': datetime.now(timezone.utc).isoformat(),
          'protocol_sha256': pin, 'archive_sha256': digest, 'archive_bytes': archive.stat().st_size,
          'packet_files': len(p['assets_sha256']) + 1, 'cases': 100, 'states': 2, 'gradient_queries_bound': 280,
          'optimizer_updates': 0, 'TRAIN_assets_verified': len(data), 'V31_dependency_hashes': len(closed),
          'prospective_return_auditor_sha256': sha(prospective), 'original_data_or_weights_uploaded': False,
          'local_neural_or_gradient_calls': 0, 'actual_VM_run_started': False, 'app_promotion': False,
          'goal_complete': False, 'seconds': time.monotonic() - start})
    print(json.dumps(read(PREP / 'preparation.json'), indent=2))


if __name__ == '__main__':
    main()
