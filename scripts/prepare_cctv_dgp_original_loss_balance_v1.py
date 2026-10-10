"""Prepare a distinct, manual saved-direction VM diagnostic; never launch it."""
import ast
import copy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
import time
from cctv_dgp_original_loss_balance_v1_contract import NAME, STEM, TERMS, SCOPES, FRACTIONS, BUDGETS, sha, read, write, validate

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm'
RETURN = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_vm_return'
REVIEW = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_return_review_v1'
OUT = ROOT/'outputs'/NAME
PREP = ROOT/'outputs/cctv_dgp_original_loss_balance_v1_preparation'


def text(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as f: f.write(value)


def replace_once(source, old, new):
    assert source.count(old) == 1, old[:100]
    return source.replace(old, new)


def worker_source():
    source = (OLD/'scripts/cctv_dgp_original_feature_probe_v1_vm.py').read_text()
    source = source.replace('original_feature_probe_v1', 'original_loss_balance_v1').replace('OriginalFeatureProbe', 'OriginalLossBalanceProbe')
    source = replace_once(source, 'NAME, STEM, BUDGETS, TERMS, SCOPES, FRACTIONS, STATE, read, write, sha, verified_assets)',
                          'NAME, STEM, BUDGETS, TERMS, SCOPES, FRACTIONS, STATE, read, write, sha, verified_assets, directions)')
    source = replace_once(source, "initial = candidate.vector(); np.save", "initial = candidate.vector(); assert np.array_equal(initial, np.load(root/'evidence/initial_parameters.npy', allow_pickle=False)); np.save")
    begin = source.index('        candidate.enable_diagnostic_gradients(root)')
    end = source.index('        trial_seconds = time.monotonic()-trials_start', begin)
    new = '''        with np.load(root/'evidence/aggregate_gradients.npz', allow_pickle=False) as saved:
            assert set(saved.files) == set(TERMS)
            aggregate = {k: saved[k].copy() for k in TERMS}
        assert all(v.dtype == np.float64 and v.shape == (1996035,) and np.isfinite(v).all() for v in aggregate.values())
        computed_directions, computed_ratio, masks = directions(p['parameter_layout'], aggregate)
        ratio = p['measured_balance_ratio']
        assert abs(computed_ratio-ratio) <= 1e-14
        with np.load(root/'evidence/canonical_directions.npz', allow_pickle=False) as saved:
            assert set(saved.files) == set(SCOPES)
            trial_directions = {k: saved[k].copy() for k in SCOPES}
        for key in SCOPES:
            assert trial_directions[key].dtype == np.float64 and trial_directions[key].shape == (1996035,)
            assert np.isfinite(trial_directions[key]).all()
            assert np.allclose(trial_directions[key], computed_directions[key], rtol=0, atol=1e-14)
        candidate.enable_finite_trials(root)
        assert all(not v.requires_grad and v.grad is None for v in candidate.net.parameters())
        frozen(True); trials_start = time.monotonic(); trial_summaries = []
        computed_weight_norm = float(np.linalg.norm(initial.astype(np.float64)[masks['decoder_control']]))
        weight_norm = p['initial_decoder_weight_L2']
        assert abs(computed_weight_norm-weight_norm) <= 1e-10
        for partition in SCOPES:
            direction = trial_directions[partition]
            allowed = masks['feature_only'] if partition == 'feature_identity' else masks['feature_only'] | masks['decoder_control']
            for fraction in FRACTIONS:
                clock(); assert time.monotonic()-trials_start <= BUDGETS['trial_seconds']
                label = partition+'_'+format(fraction, '.0e').replace('-', 'm')
                planned = fraction*weight_norm*direction
                trial = (initial.astype(np.float64)+planned).astype(np.float32)
                assert np.isfinite(trial).all(); actual = trial.astype(np.float64)-initial.astype(np.float64)
                assert np.array_equal(trial[~allowed], initial[~allowed])
                candidate.assign_trial(trial)
                np.save(out/(label+'_parameters.npy'), trial, allow_pickle=False)
                receipt = evaluate(label)
                comparisons = {co['name']: {stage: capacity(baseline['groups'][co['name']][stage], receipt['groups'][co['name']][stage], .01)
                    for stage in ['raw', 'png']} for co in p['cohorts']}
                trial_summaries.append({'variant': label, 'scope': partition, 'relative_fraction': fraction,
                    'actual_displacement_L2': float(np.linalg.norm(actual)), 'selected_weight_L2': weight_norm,
                    'scale_weight_partition': 'decoder_control', 'balance_ratio': ratio,
                    'partition_displacement_L2': {k: float(np.linalg.norm(actual[v])) for k, v in masks.items()},
                    'relative_displacement_actual': float(np.linalg.norm(actual))/weight_norm,
                    'gradient_dot_actual_displacement': {k: float(v@actual) for k, v in aggregate.items()},
                    'candidate_state': receipt['candidate_state'], 'comparisons': comparisons,
                    'model_qualification': False})
                progress['trials_completed'].append(label)
                candidate.assign_trial(initial); frozen(True)
                print({'trial': label, 'of': 9, 'new_gradient_queries': 0, 'optimizer_updates': 0}, flush=True)
'''
    source = source[:begin]+new+source[end:]
    source = replace_once(source, "assert counts == {'original': 20, 'candidate': 210, 'recognizer': 230}",
                          "assert counts == {'original': 20, 'candidate': 200, 'recognizer': 220}")
    source = replace_once(source, "'trial_summaries': trial_summaries, 'seconds':", "'trial_summaries': trial_summaries, 'saved_parent_gradient_queries': 30, 'seconds':")
    source = source.replace('before any gradient query', 'before finite trials')
    return source


def audit_source():
    source = (ROOT/'scripts/audit_cctv_dgp_original_feature_probe_v1_return_r2.py').read_text()
    source = source.replace('original_feature_probe_v1', 'original_loss_balance_v1').replace('OriginalFeatureProbe', 'OriginalLossBalanceProbe')
    source = replace_once(source, 'import math\n', 'import math\nimport shutil\n')
    source = replace_once(source, "    assert OUT.is_dir(), 'R2 reuses the retained R1 import after full archive/manifest checks'",
                          "    assert not OUT.exists(), 'Preserve any previous return import'")
    source = replace_once(source, '        retained = set()', '''        OUT.mkdir()
        for m in members:
            destination = OUT/'/'.join(m.name.split('/')[1:])
            assert destination.resolve().is_relative_to(OUT.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as incoming, destination.open('xb') as target:
                shutil.copyfileobj(incoming, target)
        retained = set()''')
    source = source.replace("result['gradient_queries'] == 30", "result['gradient_queries'] == 0 and result['saved_parent_gradient_queries'] == 30")
    source = source.replace("{'original': 20, 'candidate': 210, 'recognizer': 230}", "{'original': 20, 'candidate': 200, 'recognizer': 220}")
    begin = source.index("    folder = OUT/'outputs'; gradients =")
    end = source.index("    labels = ['baseline']", begin)
    source = source[:begin]+'''    folder = OUT/'outputs'
    with np.load(BUNDLE/'evidence/aggregate_gradients.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(TERMS)
        aggregate = {k: saved[k].copy() for k in TERMS}
    parent_audit = read(BUNDLE/'evidence/parent_independent_audit_r2.json')
    parent_analysis = read(BUNDLE/'evidence/parent_partition_analysis.json')
    assert parent_audit['complete'] and parent_audit['all30_gradient_vectors_and_nine_displacements_arithmetic_verified']
    assert parent_analysis['aggregate_gradients_sha256'] == sha(BUNDLE/'evidence/aggregate_gradients.npz')
    assert parent_analysis['initial_parameters_sha256'] == sha(BUNDLE/'evidence/initial_parameters.npy')
    initial = np.load(folder/'initial_parameters.npy', allow_pickle=False)
    assert np.array_equal(initial, np.load(BUNDLE/'evidence/initial_parameters.npy', allow_pickle=False))
    assert initial.dtype == np.float32 and initial.shape == (1996035,) and np.isfinite(initial).all()
    decoder_mask = np.zeros(1996035, bool); feature_mask = decoder_mask.copy()
    for desc in p['parameter_layout']:
        (decoder_mask if desc['partition'] == 'decoder_control' else feature_mask)[desc['start']:desc['end']] = True
    d = np.zeros(1996035, np.float64); f = d.copy()
    d[decoder_mask] = -aggregate[TERMS[0]][decoder_mask]; f[feature_mask] = -aggregate[TERMS[2]][feature_mask]
    d /= np.linalg.norm(d); f /= np.linalg.norm(f)
    ratio = float(aggregate[TERMS[2]]@d)/(-float(aggregate[TERMS[2]]@f))
    assert abs(ratio-p['measured_balance_ratio']) <= 1e-14
    ratio = p['measured_balance_ratio']
    computed_directions = {'feature_identity': f, 'balanced_1': d+ratio*f, 'balanced_2': d+2*ratio*f}
    with np.load(BUNDLE/'evidence/canonical_directions.npz', allow_pickle=False) as saved:
        assert set(saved.files) == set(SCOPES)
        independent_directions = {k: saved[k].copy() for k in SCOPES}
    for key in SCOPES: assert np.array_equal(computed_directions[key], independent_directions[key])
    weight_norm = p['initial_decoder_weight_L2']
    assert abs(float(np.linalg.norm(initial.astype(np.float64)[decoder_mask]))-weight_norm) <= 1e-10
'''+source[end:]
    begin = source.index('            allowed = np.zeros(1996035, bool)')
    end = source.index('            actual = np.load', begin)
    source = source[:begin]+'''            assert summary['scope'] in SCOPES and summary['balance_ratio'] == ratio
            assert summary['scale_weight_partition'] == 'decoder_control'
            assert abs(summary['selected_weight_L2']-weight_norm) <= 1e-12
            assert summary['relative_fraction'] in FRACTIONS
            direction = independent_directions[summary['scope']]
            expected = (initial.astype(np.float64)+summary['relative_fraction']*weight_norm*direction).astype(np.float32)
'''+source[end:]
    source = source.replace("'all30_gradient_vectors_and_nine_displacements_arithmetic_verified': True,",
                          "'audited_saved_gradients_and_all_nine_balanced_displacements_verified': True,")
    source = source.replace("'checker_R1_source_sha256': sha(ROOT/'scripts/audit_cctv_dgp_original_loss_balance_v1_return.py'),",
                          "'parent_audit_sha256': sha(BUNDLE/'evidence/parent_independent_audit_r2.json'),")
    source = source.replace("        'checker_R1_failure_log_sha256': sha(ROOT/'outputs/cctv_dgp_original_loss_balance_v1_audit_supervision_r1/audit.log'),\n", '')
    source = source.replace("        'receipt_diagnostic_sha256': sha(ROOT/'outputs/cctv_dgp_original_loss_balance_v1_return_review_v1/audit_r1_derived_receipt_diagnostic.json'),\n", '')
    source = source.replace('independent_audit_r2.json', 'independent_audit.json')
    # Keep parent file name R2; only the new return checker has no R1/R2 lineage.
    source = source.replace('evidence/parent_independent_audit.json', 'evidence/parent_independent_audit_r2.json')
    source = source.replace('"""R2: preserve R1; bound brightness-only arithmetic propagation, keep every gate."""',
                          '"""Independent prospective saved-direction return audit; all scientific gates unchanged."""')
    return source


def guide(pin, digest, size):
    return f'''# Coordinated original-DGP directions: manual VM commands

Prepared and verified before transfer; not launched. This tests a different
direction from the failed pure-structure probe. It reuses audited saved
derivatives: **zero new gradient queries, zero optimizer updates, zero epochs**.
Nine reset trials combine original decoder structure descent with original
feature identity descent. No failed pilot is resumed. No checkpoint qualifies
until structure, preservation and useful development outputs pass review.

Estimated diagnostic **5–10 minutes**, export **1–5 minutes**, based on the prior
run. Require **4 GiB free after installation**. Worker1200s/external1230s,
trials600s, cache120s, export300s/external330s,30s kill grace,20GiB allocated
VRAM,1.75GiB uncompressed return and512MiB reserve are enforced. No cleanup
is bundled. All inputs/code/weights/gradient evidence are packaged separately
from the prior immutable folders. Keep the research-cache migration backup.

Archive: **{size:,} bytes**.
Protocol SHA256: `{pin}`
Execution SHA256: `{digest}`

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
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp
```

3. Open tmux:

```bash
tmux new-session -A -s dgp_original_loss_balance_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_original_loss_balance_v1_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_loss_balance.sh {pin}
```

Detach with Ctrl+B, release, D. Reattach with
`tmux attach-session -t dgp_original_loss_balance_v1`.
The script checks idle L4/g2-standard-4 and stops on timing/storage/VRAM or
integrity problems. A diagnostic can finish while every trial fails quality;
the archive retains every failure. It starts no follow-on training.

5. Download from **Windows Google Cloud SDK Shell**, one source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Return all three files for independent audit and whole-face visual review.
The same1% structure and appearance requirements remain in force. Both cohorts
are photographic TRAIN diagnostics, not final identities or native CCTV quality
evidence. All five restoration milestones and seven covering families remain
required; the full goal is active/incomplete.
'''


def main():
    start = time.monotonic(); assert not OUT.exists() and not PREP.exists()
    parent = read(OLD/'protocol.json')
    assert sha(OLD/'protocol.json') == 'ccc30870959c8f4f4f448c1d1e06bb0ae0403d450b81143c6bcb9032d122e94b'
    for n, digest in parent['assets_sha256'].items(): assert sha(OLD/n) == digest, n
    for n, digest in parent['local_sources_sha256'].items(): assert sha(ROOT/n) == digest, n
    audit_path = ROOT/'outputs/cctv_dgp_original_feature_probe_v1_independent_audit_r2.json'
    audit = read(audit_path); analysis = read(REVIEW/'partition_analysis.json'); visual = read(REVIEW/'visual_review.json')
    assert audit['complete'] and visual['complete'] and analysis['complete'] and visual['unique_model_outputs_reviewed'] == 1000
    assert analysis['independent_evidence_audit_sha256'] == sha(audit_path) and analysis['visual_review_sha256'] == sha(REVIEW/'visual_review.json')
    assert sha(RETURN/'outputs/aggregate_gradients.npz') == analysis['aggregate_gradients_sha256']
    assert sha(RETURN/'outputs/initial_parameters.npy') == analysis['initial_parameters_sha256']
    candidate = (OLD/'cctv_dgp_original_feature_probe_v1_candidate.py').read_text()
    candidate = candidate.replace('original_feature_probe_v1', 'original_loss_balance_v1').replace('OriginalFeatureProbe', 'OriginalLossBalanceProbe')
    candidate = candidate.replace('enable_diagnostic_gradients', 'enable_finite_trials').replace('v.requires_grad_(True)', 'v.requires_grad_(False)')
    source = worker_source(); checker = audit_source()
    for code in [candidate, source, checker]: ast.parse(code, feature_version=(3, 10))
    # The new namespaces and transformed paths are all frozen before packing.
    text(ROOT/'scripts/cctv_dgp_original_loss_balance_v1_candidate.py', candidate)
    text(ROOT/'scripts/cctv_dgp_original_loss_balance_v1_vm.py', source)
    text(ROOT/'scripts/audit_cctv_dgp_original_loss_balance_v1_return.py', checker)
    OUT.mkdir(); PREP.mkdir(); mapping = {}
    for n in sorted(parent['assets_sha256']):
        if 'original_feature_probe_v1_' in n or n == 'scripts/run_feature_probe.sh': continue
        destination = OUT/n; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(OLD/n, destination); mapping[n] = (OLD/n).relative_to(ROOT).as_posix()
    evidence = {'aggregate_gradients.npz': RETURN/'outputs/aggregate_gradients.npz',
        'initial_parameters.npy': RETURN/'outputs/initial_parameters.npy', 'parent_protocol.json': OLD/'protocol.json',
        'parent_results.json': RETURN/'outputs/results.json', 'parent_gradient_receipt.json': RETURN/'outputs/gradient_receipt.json',
        'parent_independent_audit_r2.json': audit_path, 'parent_partition_analysis.json': REVIEW/'partition_analysis.json',
        'parent_visual_review.json': REVIEW/'visual_review.json'}
    for n, origin in evidence.items():
        destination = OUT/'evidence'/n; destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(origin, destination); mapping['evidence/'+n] = origin.relative_to(ROOT).as_posix()
    import numpy as np
    from cctv_dgp_original_loss_balance_v1_contract import directions
    with np.load(OUT/'evidence/aggregate_gradients.npz', allow_pickle=False) as saved:
        derivative_values = {k: saved[k].copy() for k in TERMS}
    canonical, ratio, masks = directions(parent['parameter_layout'], derivative_values)
    assert ratio == analysis['feature_to_decoder_displacement_ratio_for_mean_identity_neutrality']
    np.savez_compressed(OUT/'evidence/canonical_directions.npz', **canonical)
    for part in ['contract', 'candidate', 'vm']:
        filename = 'cctv_dgp_original_loss_balance_v1_'+part+'.py'; destination = OUT/('scripts' if part == 'vm' else '')/filename
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(ROOT/'scripts'/filename, destination)
        mapping[destination.relative_to(OUT).as_posix()] = 'scripts/'+filename
    shell = (OLD/'scripts/run_feature_probe.sh').read_text().replace('original_feature_probe_v1', 'original_loss_balance_v1')
    shell = shell.replace('original-feature diagnostic', 'original loss-balance diagnostic')
    text(OUT/'scripts/run_loss_balance.sh', shell)
    p = copy.deepcopy(parent)
    p.update({'format': 'own-DGP-original-loss-balance-finite-saved-direction-v1', 'UTC': datetime.now(timezone.utc).isoformat(),
        'parent_protocol_sha256': sha(OLD/'protocol.json'), 'parent_archive_sha256': audit['archive_sha256'],
        'parent_result_sha256': sha(RETURN/'outputs/results.json'), 'parent_audit_sha256': sha(audit_path),
        'hypothesis': 'Coordinate decoder structure descent with feature identity descent using measured disjoint gradient alignment; finite group preservation is unproven.',
        'architecture_direction': 'Same isolated original DGP, unchanged graph; different predeclared loss directions based on audited prior evidence.',
        'scopes': SCOPES, 'gradient_queries': 0, 'saved_parent_gradient_queries': 30,
        'budgets': BUDGETS, 'scale_weight_partition': 'decoder_control',
        'measured_balance_ratio': analysis['feature_to_decoder_displacement_ratio_for_mean_identity_neutrality'],
        'initial_decoder_weight_L2': analysis['partition_statistics']['decoder_control']['selected_weight_L2'],
        'canonical_direction_arithmetic_atol': 1e-14, 'weight_norm_arithmetic_atol': 1e-10,
        'proposal_rule': 'alpha=initial decoder weight L2*fraction; directions feature identity unit descent, decoder structure unit descent plus r or 2r times feature identity unit descent; reset/float32 once',
        'raw_storage': 'All1000 raw float32, PNG/mean-only/embedding outputs and nine trial vectors; saved derivatives packaged as audited input evidence',
        'assets_sha256': {q.relative_to(OUT).as_posix(): sha(q) for q in sorted(OUT.rglob('*')) if q.is_file()},
        'copied_source_mapping': mapping})
    local_names = ['CCTV_DGP_ORIGINAL_FEATURE_PROBE_V1_RESULTS.md', 'CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_DESIGN.md',
        'scripts/prepare_cctv_dgp_original_loss_balance_v1.py', 'scripts/verify_cctv_dgp_original_loss_balance_v1_packet.py',
        'scripts/audit_cctv_dgp_original_loss_balance_v1_return.py', 'scripts/cctv_dgp_original_loss_balance_v1_contract.py',
        'scripts/close_cctv_dgp_original_feature_probe_v1_review.py',
        'outputs/cctv_dgp_original_feature_probe_v1_independent_audit_r2.json',
        'outputs/cctv_dgp_original_feature_probe_v1_return_review_v1/partition_analysis.json',
        'outputs/cctv_dgp_original_feature_probe_v1_return_review_v1/visual_review.json',
        'outputs/cctv_dgp_original_feature_probe_v1_return_review_v1/gallery/gallery_manifest.json']
    p['local_sources_sha256'] = {**parent['local_sources_sha256'], **{n: sha(ROOT/n) for n in local_names}}
    validate(p); write(OUT/'protocol.json', p); pin = sha(OUT/'protocol.json')
    for q in OUT.rglob('*.py'): ast.parse(q.read_text(encoding='utf-8'), feature_version=(3, 10))
    archive = ROOT/'outputs'/(STEM+'-execution.tar.gz'); assert not archive.exists()
    with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
        for q in sorted(OUT.rglob('*')):
            if not q.is_file(): continue
            info = tar.gettarinfo(str(q), arcname=NAME+'/'+q.relative_to(OUT).as_posix())
            info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0; info.mode = 0o644
            with q.open('rb') as f: tar.addfile(info, f)
    digest = sha(archive); text(Path(str(archive)+'.sha256'), digest+'  '+archive.name+'\n')
    text(ROOT/'CCTV_DGP_ORIGINAL_LOSS_BALANCE_V1_VM.md', guide(pin, digest, archive.stat().st_size))
    write(PREP/'prepared.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
        'archive_bytes': archive.stat().st_size, 'assets': len(p['assets_sha256']), 'cases': 100,
        'new_gradient_queries': 0, 'optimizer_updates': 0, 'neural_calls': 0, 'VM_connections': 0,
        'VM_diagnostic_launched': False, 'independent_packet_audit_pending': True, 'seconds': time.monotonic()-start})
    print({'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest, 'bytes': archive.stat().st_size})


if __name__ == '__main__': main()
