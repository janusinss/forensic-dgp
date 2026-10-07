"""Prepare a fresh original-DGP fusion-partition pilot after its audited diagnostic."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
CLOSED = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
DIAGNOSTIC = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return'
NAME = 'cctv_dgp_feature_fusion_vm_v32'
STEM = 'cctv-dgp-feature-fusion-v32'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_preparation'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def renamed(text):
    for before, after in [
        ('own-DGP-profile-matched-mean-centered-original-decoder-v31', 'own-DGP-original-feature-fusion-and-decoder-v32'),
        ('cctv_dgp_profile_batches_vm_v31', NAME),
        ('cctv_dgp_profile_batches_v31_return', 'cctv_dgp_feature_fusion_v32_return'),
        ('cctv-dgp-profile-batches-v31', STEM),
        ('cctv_dgp_profile_batches_v31', 'cctv_dgp_feature_fusion_v32'),
        ('dgp_candidate_v31', 'dgp_candidate_v32'), ('V31', 'V32'),
    ]:
        text = text.replace(before, after)
    return text.replace('cctv_dgp_feature_fusion_v32_schedule', 'cctv_dgp_profile_batches_v31_schedule')


ADDED_CHECK = '''
def closed_V31_and_fusion_diagnostic_check(root, p):
    for folder, key in [(root.parent / 'cctv_dgp_profile_batches_vm_v31', 'closed_V31_evidence_sha256'),
                        (root.parent / 'cctv_dgp_feature_fusion_gradient_v1_vm', 'closed_fusion_diagnostic_evidence_sha256')]:
        for name, digest in p[key].items():
            path = (folder / name).resolve()
            assert path.is_relative_to(folder) and path.is_file() and not path.is_symlink() and sha(path) == digest, name
    old = root.parent / 'cctv_dgp_profile_batches_vm_v31'
    failure = read(old / 'outputs/failure.json'); early = read(old / 'outputs/early_structure_stop.json')
    assert failure['optimizer_updates'] == 50 and not failure['resume_permitted']
    assert early['minimum'] == .01 and not early['pass'] and not (old / 'outputs/results.json').exists()
    diagnostic = root.parent / 'cctv_dgp_feature_fusion_gradient_v1_vm'
    result = read(diagnostic / 'outputs/results.json')
    assert result['complete'] and result['component_gradient_calls'] == 280
    assert result['optimizer_updates'] == result['backwards'] == result['epochs'] == 0
    assert result['DGP_and_recognizer_unmodified'] and not result['new_checkpoint_created']
    assert result['selected_tensors'] == 23 and result['selected_parameters'] == 978243
    assert result['fusion_parameter_names'] == p['fusion_parameter_names']
    from cctv_dgp_feature_fusion_v32_policy import validate_layout
    validate_layout(p['parameter_layout'])

'''


def worker():
    text = renamed((OLD / 'scripts/cctv_dgp_profile_batches_v31_vm.py').read_text())
    text = once(text, '    assert p[\'decoder_parameters\'] == 498627 and p[\'decoder_parameter_tensors\'] == 12',
                '    assert p[\'decoder_parameters\'] == 498627 and p[\'decoder_parameter_tensors\'] == 12\n    assert p[\'fusion_parameters\'] == 479616 and p[\'selected_parameters\'] == 978243 and p[\'selected_tensors\'] == 23')
    text = once(text, '\ndef run(root, parent, p, pin):', ADDED_CHECK + '\ndef run(root, parent, p, pin):')
    text = once(text, '        paired_batch_and_closed_check(root, p)\n        base = parent_check(parent, p)',
                '        paired_batch_and_closed_check(root, p)\n        closed_V31_and_fusion_diagnostic_check(root, p)\n        base = parent_check(parent, p)')
    text = once(text, '        paired_batch_and_closed_check(root, p)\n        base = parent_check(parent,p)',
                '        paired_batch_and_closed_check(root, p)\n        closed_V31_and_fusion_diagnostic_check(root, p)\n        base = parent_check(parent,p)')
    text = once(text, 'lambda:(parent_check(parent,p),closed_v28_and_diagnostic_check(parent,p),broader_check(root,p),paired_batch_and_closed_check(root,p))',
                'lambda:(parent_check(parent,p),closed_v28_and_diagnostic_check(parent,p),broader_check(root,p),paired_batch_and_closed_check(root,p),closed_V31_and_fusion_diagnostic_check(root,p))')
    text = once(text, 'from cctv_dgp_mean_centered_decoder_v29 import MeanCenteredOriginalDecoderV29 as OriginalDecoderCandidate',
                'from cctv_dgp_feature_fusion_v32 import FeatureFusionCandidateV32 as OriginalDecoderCandidate')
    text = once(text, 'offset == 498627 and len(parameters) == 12', 'offset == 978243 and len(parameters) == 23')
    text = text.replace('(7,498627)', '(7,978243)').replace('len(pieces) == 12', 'len(pieces) == 23')
    text = text.replace('all12', 'all23').replace('selected12', 'selected23').replace('All12 selected original decoder tensors', 'All23 selected original fusion/decoder tensors')
    text = text.replace("'decoder_parameters':498627,'decoder_parameter_tensors':12,'parameter_layout':layout",
                        "'selected_parameters':978243,'selected_tensors':23,'decoder_parameters':498627,'decoder_parameter_tensors':12,'fusion_parameters':479616,'parameter_layout':layout")
    text = once(text, 'for v in candidate.net.fpn.parameters()', 'for v in candidate.net.fpn.features.parameters()')
    text = text.replace('Need6GiB free', 'Need8GiB free')
    # The former V31 launch import failure is fixed in the new immutable source,
    # before any packet schedule/policy helper can be imported in either mode.
    text = once(text, 'a = parser.parse_args(); root = a.root.resolve(); parent = a.parent.resolve(); p = verify(root,a.protocol_sha)',
                'a = parser.parse_args(); root = a.root.resolve(); sys.path.insert(0, str(root)); parent = a.parent.resolve(); p = verify(root,a.protocol_sha)')
    return text


def training():
    text = renamed((OLD / 'cctv_dgp_profile_batches_v31_training.py').read_text())
    text = once(text, 'assert len(selected)==12 and sum(value.numel() for _,value in selected)==498627',
                'assert len(selected)==23 and sum(value.numel() for _,value in selected)==978243\n    assert [name for name,_ in selected]==[row[\'name\'] for row in p[\'parameter_layout\']]')
    text = text.replace('selected12', 'selected23').replace('selected12 original decoder tensors', 'selected23 original fusion/decoder tensors')
    text = once(text, "'optimizer':'AdamW selected23 original decoder tensors'", "'optimizer':'AdamW selected23 original fusion/decoder tensors'")
    text = once(text, "'trained_parameters':498627,'trained_tensors':12", "'trained_parameters':978243,'trained_tensors':23,'fusion_parameters':479616,'decoder_parameters':498627")
    text = once(text, "'batch_formation_changed_only':True", "'parameter_partition_changed_only':True")
    return text


def return_auditor_template():
    text = renamed((ROOT / 'scripts/audit_cctv_dgp_profile_batches_v31_return.py').read_text())
    text = once(text, "PIN = 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'", "PIN = 'FUSION_V32_PROTOCOL_PIN'")
    text = once(text, '    module.BUNDLE = BUNDLE; module.OUT = OUT; module.PIN = PIN',
                '    module.BUNDLE = BUNDLE; module.OUT = OUT; module.PIN = PIN\n    old_matrix = module.matrix\n    module.matrix = lambda a, shape=(7,978243): old_matrix(a,shape)')
    text = text.replace('all12', 'all23').replace('selected12', 'selected23')
    text = once(text, "assert q['decoder_parameters'] == 498627 and q['decoder_parameter_tensors'] == 12",
                "assert q['selected_parameters'] == 978243 and q['selected_tensors'] == 23 and q['fusion_parameters'] == 479616\n        assert q['decoder_parameters'] == 498627 and q['decoder_parameter_tensors'] == 12")
    text = text.replace("terminal['batch_formation_changed_only']", "terminal['parameter_partition_changed_only']")
    addition = '''    closed31 = ROOT/'outputs/cctv_dgp_profile_batches_vm_v31'
    return31 = ROOT/'outputs/cctv_dgp_profile_batches_v31_return'
    old31p = read(closed31/'protocol.json')
    for name,digest in p['closed_V31_evidence_sha256'].items():
        path = closed31/name if name=='protocol.json' or name in old31p['assets_sha256'] else return31/name
        assert sha(path)==digest
    fusion_return = ROOT/'outputs/cctv_dgp_feature_fusion_gradient_v1_return'
    for name,digest in p['closed_fusion_diagnostic_evidence_sha256'].items(): assert sha(fusion_return/name)==digest
    spec = importlib.util.spec_from_file_location('pinned_V32_parameter_policy', BUNDLE/'cctv_dgp_feature_fusion_v32_policy.py')
    policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
    policy.validate_layout(p['parameter_layout'])
    assert p['selected_parameters']==978243 and p['selected_tensors']==23
'''
    text = once(text, '    exported, imported = import_return(expected_sha,expected_bytes,p)', addition + '    exported, imported = import_return(expected_sha,expected_bytes,p)')
    return text


def guide(pin, digest):
    return f'''# V32: original DGP feature fusion — five manual steps

The independently checked 280-query diagnostic connects all 11 fusion and 12
active decoder tensors. V32 tests one parameter-partition change: train those
23 original tensors on a fresh original checkpoint copy. No stopped V31 state
is resumed. The eight original fusion convolutions retain their architecture.
Backbone, inactive head4 and all evaluation-normalization buffers stay frozen.
The paired 781-reference/3,905-case TRAIN data, frozen schedule, seven losses,
initial normalizers, AdamW and every structure/preservation gate remain fixed.

Maximum **800 updates**: 1 complete 781-batch epoch plus 19 reference batches.
Stop at update50 unless structure gain reaches **1%**. Final TRAIN gates require
10% structure gain, all17 preservation groups, both source gains >=0 and mean-only
fraction <=20%. A training pass still requires useful DEV/native output review.
Original models, splits and every failed run remain retained. No automatic app
promotion, native/reserved pixels or local training is included.

Use the existing NVIDIA L4/g2-standard-4 at ~/forensic-dgp, an idle GPU and
**8GiB free**. Estimate **15–35 minutes plus export**. Storage allowance protects
up to3GiB uncompressed return plus its archive and margin. Finite limits:
preflight300s, cache900s, fit3600s, worker4500s, external4800s +30s grace,
export900s/external930s +30s grace, allocated VRAM20GiB and return3GiB.
Initial raw/PNG equality and all23 finite nonzero improvement gradients are
required before any optimizer. Retain any source/nonfinite/time/gate failure.

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
test -f ~/forensic-dgp/cctv_dgp_feature_fusion_gradient_v1_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_feature_fusion_v32
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_feature_fusion_v32_vm.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v32.sh {pin}
```

Detach with Ctrl+B, release, D. Export complete:true means packaging only;
the independent return checker and visible-structure review determine results.
Do not resume a stopped pilot or relax the1% gate.

5. Download from **Windows Google Cloud SDK Shell**, one remote file per call:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

The separate download calls avoid PuTTY's multiple-remote-source error.
All actual new training remains this human/manual existing-L4 workflow.
Useful native restoration, seven covering families with separate automatic and
assisted reviews, independent final review and the app goal remain incomplete.
'''


def main():
    start = time.monotonic()
    assert not BUNDLE.exists() and not PREP.exists()
    audit_path = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_independent_audit.json'
    analysis_path = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_analysis/analysis.json'
    audit, analysis = read(audit_path), read(analysis_path)
    assert audit['complete'] and audit['diagnostic_complete'] and audit['CPU_replay']['cases_at_both_states'] == 200
    assert audit['optimizer_updates'] == audit['local_gradient_calls'] == 0
    assert analysis['complete'] and analysis['new_finite_fusion_partition_pilot_justified_for_preparation']
    diagnostic_p = read(DIAGNOSTIC / 'protocol.json')
    old = read(OLD / 'protocol.json')
    assert sha(OLD / 'protocol.json') == 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'
    failure, early = read(CLOSED / 'outputs/failure.json'), read(CLOSED / 'outputs/early_structure_stop.json')
    assert failure['optimizer_updates'] == 50 and not failure['resume_permitted'] and not early['pass'] and early['minimum'] == .01
    for name, digest in old['assets_sha256'].items(): assert sha(OLD / name) == digest
    for name, digest in old['mixed_TRAIN_assets_sha256'].items(): assert sha(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2' / name) == digest
    BUNDLE.mkdir(); (BUNDLE / 'scripts').mkdir(); PREP.mkdir()
    for name in ['cctv_dgp_mean_centered_decoder_v29.py', 'cctv_dgp_app_input_v28.py',
                 'cctv_dgp_profile_batches_v31_schedule.py', 'schedule.json']:
        (BUNDLE / name).write_bytes((OLD / name).read_bytes())
    for name in ['cctv_dgp_feature_fusion_v32.py', 'cctv_dgp_feature_fusion_v32_policy.py']:
        (BUNDLE / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
    (BUNDLE / 'cctv_dgp_feature_fusion_v32_cache.py').write_text(renamed((OLD / 'cctv_dgp_profile_batches_v31_cache.py').read_text()), encoding='utf-8', newline='\n')
    (BUNDLE / 'cctv_dgp_feature_fusion_v32_training.py').write_text(training(), encoding='utf-8', newline='\n')
    (BUNDLE / 'scripts/cctv_dgp_feature_fusion_v32_vm.py').write_text(worker(), encoding='utf-8', newline='\n')
    shell = renamed((OLD / 'scripts/run_v31.sh').read_text()).replace('scripts/run_v31.sh', 'scripts/run_v32.sh')
    (BUNDLE / 'scripts/run_v32.sh').write_text(shell, encoding='ascii', newline='\n')
    p = copy.deepcopy(old)
    p.update({'format': 'own-DGP-original-feature-fusion-and-decoder-v32',
              'purpose': 'Test previously frozen original fusion weights after audited finite connectivity and first-order descent measurements.',
              'design': 'Trainable parameter partition changes only; fresh original state, same forward, paired data/schedule, objective, normalizers, optimizer and numerical gates.',
              'difference_from_failed_recipes': 'V31 held every original fusion tensor fixed. V32 enables exactly the11 measured fusion tensors alongside the12 active decoder tensors. It never resumes V31.',
              'hypothesis': 'Adaptation of the original spatial fusion path may improve visible structure under unchanged preservation constraints. Connectivity is not proof of useful native outputs.',
              'parameter_partition_changed_only': True, 'batch_formation_changed_only': False,
              'losses_optimizer_architecture_initialization_and_numeric_gates_unchanged': True,
              'selected_tensors': 23, 'selected_parameters': 978243, 'fusion_parameters': 479616,
              'fusion_parameter_names': diagnostic_p['fusion_parameter_names'],
              'decoder_parameter_names': diagnostic_p['decoder_parameter_names'],
              'parameter_layout': diagnostic_p['parameter_layout'],
              'frozen_partition_policy': 'Original backbone, inactive head4, all buffers and evaluation normalization stay unchanged; only measured11 fusion+12 decoder tensors may change.',
              'manual_VM_required': True, 'human_manual_VM_execution_required': True,
              'next': 'Independently audit return and all50 previews; only a capacity pass justifies fixed520 pairedDEV/24 unpaired nativeDEV review. Keep reserved final pixels unopened.'})
    p['budgets']['minimum_free_disk_bytes'] = 8 * 1024**3
    p['storage_budget_reason'] = 'Up to3GiB return plus compressed archive and2GiB margin; larger initial23-tensor proof arrays, no deletion of historical evidence.'
    names31 = ['protocol.json', *old['assets_sha256'], 'outputs/failure.json', 'outputs/early_structure_stop.json',
               'outputs/execution_receipt.json', 'outputs/gradient_preflight.json', 'outputs/cohort_loss_setup.json',
               'outputs/update50/metrics.json', 'outputs/update50/dgp_candidate_v31.pth']
    p['closed_V31_evidence_sha256'] = {n: sha(OLD / n if n == 'protocol.json' or n in old['assets_sha256'] else CLOSED / n) for n in names31}
    names_d = ['protocol.json', *diagnostic_p['assets_sha256'], 'outputs/results.json', 'supervisor_receipt.json', 'diagnostic_exit_code.txt']
    names_d.extend(f'outputs/state{s}_{c}/{n}' for s in [0, 50] for c in ['exposed', 'unexposed'] for n in ['receipt.json', 'gradient_components.npy'])
    p['closed_fusion_diagnostic_evidence_sha256'] = {n: sha(DIAGNOSTIC / n) for n in names_d}
    basis = [audit_path, analysis_path, ROOT / 'CCTV_DGP_FEATURE_FUSION_GRADIENT_V1_RESULTS.md',
             ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json', Path(__file__),
             ROOT / 'scripts/analyze_cctv_dgp_feature_fusion_gradient_v1.py',
             ROOT / 'scripts/cctv_dgp_feature_fusion_v32.py', ROOT / 'scripts/cctv_dgp_feature_fusion_v32_policy.py',
             ROOT / 'scripts/verify_cctv_dgp_feature_fusion_v32_packet.py', ROOT / 'tests/test_cctv_dgp_feature_fusion_v32.py',
             ROOT / 'scripts/audit_cctv_dgp_profile_batches_v31_return.py']
    p['local_basis_sha256'] = {f.relative_to(ROOT).as_posix(): sha(f) for f in basis}
    p['assets_sha256'] = {f.relative_to(BUNDLE).as_posix(): sha(f) for f in sorted(BUNDLE.rglob('*')) if f.is_file()}
    for f in BUNDLE.rglob('*.py'): ast.parse(f.read_text(), feature_version=(3, 10))
    write(BUNDLE / 'protocol.json', p); pin = sha(BUNDLE / 'protocol.json')
    template = return_auditor_template(); assert template.count('FUSION_V32_PROTOCOL_PIN') == 1
    template_file = ROOT / 'scripts/cctv_dgp_feature_fusion_v32_return_audit_template.py'
    with template_file.open('x', encoding='utf-8', newline='\n') as stream: stream.write(template)
    checker = ROOT / 'scripts/audit_cctv_dgp_feature_fusion_v32_return.py'
    with checker.open('x', encoding='utf-8', newline='\n') as stream: stream.write(template.replace('FUSION_V32_PROTOCOL_PIN', pin))
    archive = ROOT / 'outputs' / (STEM + '-execution.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=6) as tar:
        for f in sorted(BUNDLE.rglob('*')):
            if f.is_file(): tar.add(f, arcname=NAME + '/' + f.relative_to(BUNDLE).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream: stream.write(digest + '  ' + archive.name + '\n')
    with (ROOT / 'CCTV_DGP_FEATURE_FUSION_V32_VM.md').open('x', encoding='utf-8', newline='\n') as stream: stream.write(guide(pin, digest))
    write(PREP / 'preparation.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
          'archive_bytes': archive.stat().st_size, 'packet_files': len(p['assets_sha256']) + 1,
          'selected_parameters': 978243, 'selected_tensors': 23, 'TRAIN_cases': 3905, 'TRAIN_references': 781,
          'prospective_return_auditor_sha256': sha(checker), 'prospective_template_sha256': sha(template_file),
          'metadata_calls_only': True, 'local_neural_or_gradient_calls': 0, 'actual_VM_training_started': False,
          'manual_VM_required': True, 'native_or_reserved_used': False, 'app_promotion': False, 'goal_complete': False,
          'seconds': time.monotonic() - start})
    print(json.dumps(read(PREP / 'preparation.json'), indent=2))


if __name__ == '__main__':
    main()
