"""Record an independently audited failed pilot without changing its sources."""
from pathlib import Path
import hashlib
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
AUDIT_DIR = OUT / 'cctv_dgp_head4_capacity_v1_return_audit'
RETURN = OUT / 'cctv_dgp_head4_capacity_vm_v1_return'
RECEIPT = OUT / 'cctv_dgp_head4_capacity_v1_independent_audit.json'
REPORT = ROOT / 'CCTV_DGP_HEAD4_CAPACITY_V1_RESULTS.md'
MILESTONE = OUT / 'cctv_dgp_head4_capacity_v1_return_milestone'
DOCS = ['PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_DGP_HEAD4_CAPACITY_V1_VM_CURRENT.md']


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    assert not MILESTONE.exists() and not REPORT.exists()
    a = read(RECEIPT)
    assert a['complete'] and a['training_completed'] and not a['early_capacity_pass']
    assert a['full_raw_PNG_metric_records_verified'] == 7810
    assert a['optimizer_updates'] == 50 and a['completed_epochs'] == 0
    assert a['full_state_and_frozen_partition_verified'] and not a['model_qualification']
    assert a['local_gradient_queries'] == a['local_optimizer_updates'] == 0
    assert int((AUDIT_DIR / 'checker_exit_code.txt').read_text()) == 0
    plan = read(AUDIT_DIR / 'plan.json')
    protocol = read(OUT / 'cctv_dgp_head4_capacity_vm_v1/protocol.json')
    assert not any(n in protocol['local_sources'] for n in DOCS)
    for n, d in plan['before_sha256'].items():
        assert sha(ROOT / n) == d, n
    result = read(RETURN / 'outputs/results.json')
    gate = read(RETURN / 'outputs/early_gate.json')
    failure = read(RETURN / 'outputs/failure.json')
    assert not result['gate_pass'] and not gate['pass']
    assert result['gates'] == gate['comparisons']
    assert failure['progress']['optimizer_updates'] == 50
    assert 'Structure/preservation requirement failed at50' in failure['traceback']
    assert (RETURN / 'outputs/update50/candidate.pth').is_file()
    assert (RETURN / 'outputs/update50/training_state.pt').is_file()
    assert (RETURN / 'outputs/stopped_training_state.pt').is_file()
    gradients = read(RETURN / 'outputs/gradient_preflight.json')
    assert len(gradients['rows']) == gradients['queries'] == 6
    assert all(all(n > 0 for n in r['part_gradient_L2']) for r in gradients['rows'])
    visual_dir = AUDIT_DIR / 'visual_samples'
    visual_plan = read(visual_dir / 'plan.json')
    visual = read(visual_dir / 'visual_review.json')
    assert visual['complete'] and visual['all_planned_pages_inspected']
    assert visual['planned_pages'] == visual['reviewed_pages'] == 2
    assert visual['case_comparisons'] == 10 and not visual['model_qualification']
    assert visual['plan_sha256'] == sha(visual_dir / 'plan.json')
    for page in visual_plan['pages']:
        assert sha(ROOT / page['path']) == page['sha256']
        for n, d in page['input_files_sha256'].items():
            assert sha(ROOT / n) == d

    import torch
    torch.set_num_threads(4)
    weights = torch.load(RETURN / 'outputs/update50/candidate.pth', map_location='cpu', weights_only=True)
    movement = []
    with torch.inference_mode():
        for name, anchor in [
            ('net.head4.block0.weight', 'anchor_head4_0'),
            ('net.head4.block1.weight', 'anchor_head4_1'),
            ('live_fusion4', 'anchor_fusion4'),
        ]:
            delta = weights[name].double() - weights[anchor].double()
            movement.append({'name': name, 'elements': delta.numel(),
                             'changed_elements': int(torch.count_nonzero(delta)),
                             'delta_L2': float(delta.norm()), 'delta_max_abs': float(delta.abs().max())})
    assert sum(r['elements'] for r in movement) == 147456
    assert all(r['changed_elements'] > 0 and r['delta_L2'] > 0 for r in movement)
    movement_evidence = read(AUDIT_DIR / 'weight_movement.json')
    assert movement_evidence['complete'] and movement == movement_evidence['trainable_parts']
    assert movement_evidence['candidate_sha256'] == sha(RETURN / 'outputs/update50/candidate.pth')
    assert movement_evidence['local_gradient_queries'] == movement_evidence['local_optimizer_updates'] == 0
    protected = {
        'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth':
            '646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b',
        'checkpoints/dgp_zamboanga_final.pth':
            'b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c',
    }
    for n, d in protected.items():
        assert sha(ROOT / n) == d, n
    raw, png = gate['comparisons']['raw'], gate['comparisons']['png']
    raw_gain, png_gain = 100 * raw['relative_feature_gain'], 100 * png['relative_feature_gain']
    raw_brightness, png_brightness = 100 * raw['brightness_gain_fraction'], 100 * png['brightness_gain_fraction']
    diagnosis = {
        'complete': True, 'independent_return_audit_passed': True, 'quality_gate_passed': False,
        'protocol_sha256': a['protocol_sha256'], 'archive_sha256': a['archive_sha256'],
        'optimizer_updates': 50, 'completed_epochs': 0, 'epoch_fraction': 50 / 781,
        'raw_structure_gain_percent': raw_gain, 'PNG_structure_gain_percent': png_gain,
        'minimum_structure_gain_percent': 1, 'raw_brightness_gain_fraction_percent': raw_brightness,
        'PNG_brightness_gain_fraction_percent': png_brightness, 'maximum_brightness_gain_fraction_percent': 20,
        'preservation_failures': gate['comparisons'], 'trainable_weight_movement': movement,
        'original_checkpoints_verified': protected, 'all_saved_metric_records_verified': 7810,
        'native_CCTV_or_final_identity_quality_not_measured': True, 'visual_review_pending': True,
        'diagnostic_visual_pages_inspected': 2, 'diagnostic_visual_case_comparisons': 10,
        'diagnostic_visual_review_sha256': sha(visual_dir / 'visual_review.json'),
        'weight_movement_evidence_sha256': sha(AUDIT_DIR / 'weight_movement.json'),
        'extra_epochs_not_run': True, 'app_promotion': False, 'goal_complete': False,
        'underlying_training_limit_not_established': True, 'unchanged_failed_recipe_not_relaunched': True,
        'local_gradient_queries': 0, 'local_optimizer_updates': 0,
    }
    write(AUDIT_DIR / 'diagnosis.json', diagnosis)
    report = f'''# Head4 capacity V1 return — 10 October 2026

The manual L4 pilot completed all 50 planned optimizer updates, then failed its
unchanged structure/preservation requirement. The downloaded archive and the
full independent audit pass; the experimental model's quality requirement fails.
Location: `scripts/cctv_dgp_head4_capacity_vm_v1.py:210` in the returned VM source.
Cause: insufficient measured structural gain, two small blur-group pixel-error
regressions, and excessive brightness-only contribution. The assertion preserves
the failed candidate and optimizer state and prevents continuation or promotion.

| Saved output | Structure gain | Required gain | Brightness share of pixel-error gain | Maximum share |
| --- | ---: | ---: | ---: | ---: |
| Raw float32 | {raw_gain:.8f}% | 1% | {raw_brightness:.4f}% | 20% |
| Delivered PNG | {png_gain:.8f}% | 1% | {png_brightness:.4f}% | 20% |

Structure gain is the relative reduction in landmark-region high-frequency
error on the degraded paired TRAIN cases. It is not identity accuracy, native
CCTV quality or a percentage of recovered hidden information. Brightness share
is the fixed counterfactual test: how much of the measured pixel-error reduction
is explained by shifting the baseline's mean RGB brightness alone.

Both raw and PNG output increase MSE slightly in the `blur_lr24` groups of the
`dataset/asian_faces` TRAIN source and FFHQ-derived TRAIN source. Raw MSE
changes from 0.005974067877 to 0.005974858028 and from 0.009668438246 to 0.009668655779.
These are small numerical regressions; they do not establish conspicuous visual
damage. The existing group-preservation requirement still rejects them. Dataset
directory labels do not establish ethnicity, capture geography or Zamboanga CCTV
performance.

All three repaired trainable pieces receive positive finite gradients in the six
preflight queries and actually change during fitting. Reconnecting a previously
inactive path was successful; useful finite-step image learning was not proved.
Only three tensors/147,456 elements were trained; all other original weights and
stored normalization remained frozen. The pilot uses 50 updates/781 batches, about
6.402% of one epoch, with zero complete additional epochs. It does not determine
whether longer training, a different objective, learning rate or a different
spatial path would resolve the quality failures. More epochs alone are therefore
not justified by this returned evidence, and this failed state must not resume.

The run took {result['seconds'] / 60:.2f} minutes and reached its planned evaluation.
It did not stop from storage, CUDA, an external deadline or a transfer problem.
The export's `complete:true` means the archive was written successfully.

The 3,586,556,555-byte archive has SHA256
`{a['archive_sha256']}`. The unchanged independent checker verifies all archive members,
all 7,810 saved baseline/candidate raw-and-PNG case records across two 3905-case
snapshots, exact gate decisions, six gradient-query receipts, all 50 step receipts, frozen
state, full optimizer/scheduler/RNG state and two fresh CPU inference replays.
Local auditing used zero gradient queries or optimizer updates. Both planned
diagnostic visual sheets were inspected, comparing all five profiles of one
reference from each TRAIN source (10 comparisons). These are the largest blur
MSE increases within the saved previews, not a representative population sample.
Baseline and update50 look almost identical; heavily degraded eyes and mouths
remain poorly defined. There is no convincing additional whole-face clarity in
this small diagnostic set. Comprehensive preview and native development review
remain pending. The failure, candidate, full stopped state and original
checkpoints remain. Native development and final identities were not used in
this TRAIN capacity pilot. The app/model selection is unchanged.

Evidence: `outputs/cctv_dgp_head4_capacity_v1_independent_audit.json`,
`outputs/cctv_dgp_head4_capacity_v1_return_audit/diagnosis.json`,
`outputs/cctv_dgp_head4_capacity_v1_return_audit/visual_samples/visual_review.json`,
and retained `outputs/cctv_dgp_head4_capacity_vm_v1_return/outputs/early_gate.json`.
The full restoration, completion and application goal remains incomplete.
'''
    with REPORT.open('x', encoding='utf-8', newline='\n') as f:
        f.write(report)
    prefix = f'''**Latest verified status — 10 October 2026: Head4 capacity V1 returned and independently audited; 50 updates failed the unchanged quality requirement. Full goal incomplete.**

The manual L4 training pilot completed all 50 planned optimizer updates (zero
complete additional epochs; 6.402% of one 781-batch epoch). All three reconnected
trainable pieces receive positive gradients and their weights change. Raw
structure gain is {raw_gain:.8f}% and delivered PNG gain is {png_gain:.8f}%, below
the retained 1% requirement. Both output stages slightly increase pixel error in
two blurred TRAIN-source groups. Brightness shifts explain {raw_brightness:.4f}%
raw/{png_brightness:.4f}% PNG of the measured pixel-error improvement, exceeding
the unchanged 20% maximum. The stop is an intentional quality rejection after
evaluation; archive completion is not model qualification.

The unchanged independent checker verifies the 3,586,556,555-byte archive
({a['archive_sha256']}), all 7,810 case records across
baseline/candidate snapshots, frozen weights, full optimizer/scheduler/RNG state
and two fresh CPU replays. Original checkpoints, raw/PNG artifacts and the
failure remain. Local auditing uses zero gradients/updates. No failed-state
resume, extra epochs, new training, model/app promotion or gate change follows.
The current V1 command guide is closed: do not rerun or resume this failed pilot.
Its previous ready status, disk values and commands below are historical.
This is paired photographic TRAIN capacity evidence; native development and
reserved final identities were not evaluated. Two planned diagnostic sheets/10
TRAIN comparisons were inspected; comprehensive visual review remains pending.
See CCTV_DGP_HEAD4_CAPACITY_V1_RESULTS.md and
outputs/cctv_dgp_head4_capacity_v1_independent_audit.json. The earlier prepared
status below is historical and superseded by this returned failed pilot.

---

'''
    MILESTONE.mkdir()
    (MILESTONE / 'before').mkdir()
    before, after = {}, {}
    for n in DOCS:
        path = ROOT / n
        data = path.read_bytes()
        before[n] = sha(path)
        shutil.copyfile(path, MILESTONE / 'before' / n)
        path.write_bytes(prefix.encode('utf-8') + data)
        assert path.read_bytes().endswith(data)
        after[n] = sha(path)
    write(MILESTONE / 'closure.json', {
        'complete': True, 'quality_gate_passed': False, 'goal_complete': False,
        'before_sha256': before, 'after_sha256': after, 'report_sha256': sha(REPORT),
        'diagnosis_sha256': sha(AUDIT_DIR / 'diagnosis.json'), 'independent_audit_sha256': sha(RECEIPT),
        'diagnostic_visual_review_sha256': sha(visual_dir / 'visual_review.json'),
        'original_checkpoints_sha256': protected, 'canonical_prefix_utf8': prefix,
        'failure_sha256': sha(RETURN / 'outputs/failure.json'),
        'gate_sha256': sha(RETURN / 'outputs/early_gate.json'),
        'no_training_relaunch': True, 'no_model_or_app_change': True,
    })
    print(json.dumps({'complete': True, 'quality_gate_passed': False, 'report': str(REPORT),
                      'raw_structure_gain_percent': raw_gain, 'PNG_structure_gain_percent': png_gain,
                      'trained_parts_changed': movement}, indent=2))


if __name__ == '__main__':
    main()
