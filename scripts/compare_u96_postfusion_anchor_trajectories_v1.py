"""Plot independently read saved TRAIN metrics, without loading model weights."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_return_comparison'


def read(path):
    return json.loads(path.read_bytes())


def pin(path):
    raw = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def ordered_mean(values):
    total = 0.0
    count = 0
    for value in values:
        total = total + float(value)
        count += 1
    assert count
    return total / count


def curves(folder, run):
    data = read(folder / 'bounded_stream_readback_v1.json')['documents']
    # Historical B1 readback keys contain the stage prefix; new readback keys do not.
    def bank(u):
        suffix = run + '/snapshots/u%04d/bank_and_gates.json' % u
        matches = [value for key, value in data.items() if key == suffix or key.endswith('/' + suffix)]
        assert len(matches) == 1
        return matches[0]['monitor']['groups']
    initial = bank(0)
    rows = []
    for update in [0, 1, 4, 16, 32, 64]:
        groups = bank(update)
        values = {}
        for role, region in [('local_gaussian_1p2', 'central128'), ('global_gaussian_1p2', 'whole')]:
            values[role] = {}
            for view in ['raw', 'native_png']:
                field = region + '/adjacent_rgb_error'
                a = initial[role]['views'][view][field]['current']
                b = groups[role]['views'][view][field]['current']
                assert set(a) == set(b) and len(a) == 8
                before = ordered_mean(a[k] for k in sorted(a))
                after = ordered_mean(b[k] for k in sorted(b))
                values[role][view] = 100 * (before - after) / before
        rows.append({'update': update, 'gain_percent': values})
    return rows


def main():
    b1folder = ROOT / 'outputs/codeformer_u96_postfusion_learning_v1_return_peer'
    b2folder = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_v1_return_peer'
    before = read(ROOT / 'outputs/cctv_codeformer_u96_postfusion_learning_v1_vm/protocol.json')
    after = read(ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_v1_vm/protocol.json')
    assert before['cases'] == after['cases'] and before['guards'] == after['guards']
    assert before['model'] == after['model'] and before['limits'] == after['limits']
    learning = dict(after['learning'])
    learning.pop('anchor_criterion')
    learning['format'] = before['learning']['format']
    assert before['learning'] == learning
    for folder in [b1folder, b2folder]:
        review = read(folder / 'independent_return_review_v1.json')
        assert review['complete'] and review['independent_return_review_pass']
        assert review['scientific_trial_pass'] is False and review['model_qualified'] is False
        assert review['updates_confirmed'] == 64
    a = curves(b1folder, 'outputs/u96_postfusion_learning_v1')
    b = curves(b2folder, 'outputs/u96_postfusion_smoothanchors_v1')
    OUT.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8), dpi=160, sharey=True)
    for axis, role, title in zip(axes,
        ['global_gaussian_1p2', 'local_gaussian_1p2'],
        ['Global blur: whole image', 'Localized blur: central region']):
        axis.axhline(1, color='#8c2d24', linestyle=':', linewidth=1.7, label='Required endpoint: 1%')
        for dataset, colour, name in [(a, '#657786', 'L1 anchors'), (b, '#086e70', 'Smoother anchors')]:
            for view, style in [('raw', '--'), ('native_png', '-')]:
                axis.plot([row['update'] for row in dataset],
                          [row['gain_percent'][role][view] for row in dataset],
                          color=colour, linestyle=style, marker='o', markersize=3,
                          label=name + (' RAW' if view == 'raw' else ' PNG'))
        axis.set_title(title, fontsize=11)
        axis.set_xlabel('Committed optimizer updates')
        axis.set_xticks([0, 16, 32, 48, 64])
        axis.grid(axis='y', alpha=.18)
        axis.set_ylim(-.045, 1.08)
    axes[0].set_ylabel('Reduction in paired adjacent-residual error (%)')
    axes[1].legend(loc='upper left', fontsize=8)
    fig.suptitle('Saved TRAIN-monitor comparison: both experiments stopped below 1%', fontsize=13)
    fig.text(.5, .018, 'Same inputs, U96 initialization, learning rate and 64-update cap. Reused TRAIN evidence; no model qualification.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .045, 1, .93))
    plot = OUT / 'structure_trajectory_v1.png'
    fig.savefig(plot)
    plt.close(fig)
    sources = [b1folder / 'bounded_stream_readback_v1.json', b2folder / 'bounded_stream_readback_v1.json',
               b1folder / 'independent_return_review_v1.json', b2folder / 'independent_return_review_v1.json']
    report = {'complete': True, 'source_receipt_comparison_only': True,
              'case_model_limits_sampler_optimizer_LR_exact_except_common_anchor_criterion': True,
              'scalar_sums_ordered_to_match_VM_310': True,
              'L1_anchor_curve': a, 'SmoothL1_anchor_curve': b,
              'plot': pin(plot), 'sources': [pin(p) for p in sources], 'script': pin(Path(__file__)),
              'scientific_trials_pass': False, 'model_qualified': False,
              'pixel_recomputation_and_root_visual_review_are_separate': True,
              'no_new_model_optimizer_training_VM_calls': True}
    with (OUT / 'trajectory_comparison_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'complete': True, 'plot': str(plot), 'model_qualified': False}))


if __name__ == '__main__':
    main()
