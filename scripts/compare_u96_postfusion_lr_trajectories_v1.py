"""Compare two audited saved TRAIN trajectories; no neural or VM execution."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from compare_u96_postfusion_anchor_trajectories_v1 import curves

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_comparison'


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    old_folder = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_return_peer'
    new_folder = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_peer'
    old_protocol = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_v1_vm/protocol.json'
    new_protocol = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/protocol.json'
    old, new = read(old_protocol), read(new_protocol)
    for key in ('cases', 'guards', 'model', 'limits'):
        assert old[key] == new[key], key
    learning = deepcopy(new['learning'])
    assert old['learning']['optimizer']['lr'] == 5e-5
    assert learning['optimizer']['lr'] == 1e-4
    learning['optimizer']['lr'] = old['learning']['optimizer']['lr']
    learning['format'] = old['learning']['format']
    assert old['learning'] == learning
    screens = []
    for index, folder in enumerate((old_folder, new_folder)):
        review = read(folder / 'independent_return_review_v1.json')
        assert review['complete'] and review['model_qualified'] is False
        if index == 0:
            assert review['independent_return_review_pass'] and review['updates_confirmed'] == 64
            assert review['scientific_trial_pass'] is False
            screens.append(False)
        else:
            assert review['independent_return_metadata_audit_pass'] and review['optimizer_updates'] == 64
            assert review['optimizer_update_attempts'] == 64 and review['optimizer_outcome_unknown'] is False
            assert review['authoritative_status'] == 'FINITE_B64_LEARNING_SCREEN_PASSED_NOT_QUALIFIED'
            assert review['worker_exit'] == 0 and review['timeout'] is False
            screens.append(True)
    a = curves(old_folder, 'outputs/u96_postfusion_smoothanchors_v1')
    b = curves(new_folder, 'outputs/u96_postfusion_smoothanchors_lr_v1')
    OUT.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=160, sharey=True)
    peak = max(row['gain_percent'][role][view] for trajectory in (a, b)
               for row in trajectory for role in row['gain_percent']
               for view in ('raw', 'native_png'))
    for axis, role, title in zip(axes,
            ('global_gaussian_1p2', 'local_gaussian_1p2'),
            ('Global blur: whole image', 'Localized blur: central region')):
        axis.axhline(1, color='#8c2d24', linestyle=':', linewidth=1.7,
                     label='Required endpoint: 1%')
        for trajectory, colour, label in ((a, '#657786', 'Max LR 0.00005'),
                                          (b, '#086e70', 'Max LR 0.0001')):
            for view, style in (('raw', '--'), ('native_png', '-')):
                axis.plot([row['update'] for row in trajectory],
                          [row['gain_percent'][role][view] for row in trajectory],
                          color=colour, linestyle=style, marker='o', markersize=3,
                          label=label + (' RAW' if view == 'raw' else ' PNG'))
        axis.set_title(title, fontsize=11)
        axis.set_xlabel('Committed optimizer updates')
        axis.set_xticks([0, 16, 32, 48, 64])
        axis.grid(axis='y', alpha=.18)
        axis.set_ylim(-.045, max(1.08, peak * 1.12))
    axes[0].set_ylabel('Reduction in paired adjacent-residual error (%)')
    axes[1].legend(loc='upper left', fontsize=8)
    fig.suptitle('Saved TRAIN-monitor comparison: maximum learning rate only', fontsize=13)
    fig.text(.5, .018,
             'Same inputs, fresh complete U96 initialization and 64-update cap. '
             'Adaptive TRAIN evidence; broader DEV, user and FINAL review pending.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .93))
    plot = OUT / 'structure_trajectory_v1.png'
    fig.savefig(plot)
    plt.close(fig)
    sources = [p for folder in (old_folder, new_folder) for p in
               (folder / 'bounded_stream_readback_v1.json',
                folder / 'independent_return_review_v1.json')]
    sources += [old_protocol, new_protocol,
                ROOT / 'scripts/compare_u96_postfusion_anchor_trajectories_v1.py']
    report = {'complete': True, 'source_receipt_comparison_only': True,
              'only_numeric_training_factor_changed': 'maximum Adam learning rate 5e-5 to 1e-4',
              'cases_guards_model_limits_order_loss_and_sampler_match': True,
              'scalar_sums_ordered_to_match_VM_310': True,
              'max_LR_5e_5_curve': a, 'max_LR_1e_4_curve': b,
              'finite_TRAIN_screen_pass': screens,
              'plot': pin(plot), 'sources': [pin(p) for p in sources],
              'script': pin(Path(__file__)), 'model_qualified': False,
              'pixel_recomputation_and_root_visual_review_are_separate': True,
              'new_model_optimizer_training_VM_calls': 0}
    with (OUT / 'trajectory_comparison_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'complete': True, 'plot': str(plot),
                      'TRAIN_screen_pass': report['finite_TRAIN_screen_pass'],
                      'model_qualified': False}))


if __name__ == '__main__':
    main()
