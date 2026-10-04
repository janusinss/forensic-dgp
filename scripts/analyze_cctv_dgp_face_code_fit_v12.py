"""Analyze already audited V12 logits; no model construction or neural calls."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = '06f056ea4b1b04e6d8c631e5e78df129345c78279230c7946f6b5deb2e226846'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def code_statistics(logits, labels, support):
    """Observed token statistics, with target rank and confidence kept separate."""
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels)
    support = np.asarray(support, dtype=bool)
    require(logits.ndim == 3 and labels.shape == support.shape == logits.shape[:2],
            'Logit/label/support geometry differs')
    require(np.isfinite(logits).all() and support.any() and labels.dtype == np.int64
            and labels.min() >= 0 and labels.max() < logits.shape[-1], 'Invalid code probe')
    selected = logits[support]
    truth = labels[support]
    log_prob = selected - logsumexp(selected, axis=1, keepdims=True)
    prediction = selected.argmax(1)
    truth_log_prob = log_prob[np.arange(len(truth)), truth]
    truth_logit = selected[np.arange(len(truth)), truth]
    # Competition rank; ties share rank. Do not invent ordering among tied logits.
    rank = 1 + (selected > truth_logit[:, None]).sum(1)
    prob = np.exp(log_prob)
    return {
        'observed_tokens': len(truth),
        'code_ce': float(-truth_log_prob.mean()),
        'accuracy': float((prediction == truth).mean()),
        'target_rank_mean': float(rank.mean()),
        'target_rank_median': float(np.median(rank)),
        'target_reciprocal_rank': float((1.0 / rank).mean()),
        'target_probability': float(np.exp(truth_log_prob).mean()),
        'top1_probability': float(prob[np.arange(len(truth)), prediction].mean()),
        'entropy': float(-(prob * log_prob).sum(1).mean()),
        'unique_predicted_codes': int(len(np.unique(prediction))),
        'unique_target_codes': int(len(np.unique(truth))),
    }, prediction, truth


def analyze(bundle, returned, destination):
    started = time.monotonic()
    require(not destination.exists(), 'Preserve completed/partial analysis')
    p = read(bundle / 'face_code_fit_protocol_v12.json')
    require(sha(bundle / 'face_code_fit_protocol_v12.json') == PROTOCOL, 'Protocol differs')
    audit = read(returned / 'local_independent_audit.json')
    out = returned / 'outputs/cctv_dgp_face_code_fit_v12'
    results = read(out / 'results.json')
    require(audit['complete'] and audit['protocol_sha256'] == PROTOCOL and
            audit['results_sha256'] == sha(out / 'results.json') and results['complete'],
            'Independent audit is missing or does not bind this return')
    references = {r['id']: r for r in p['references']}
    require(all(r['role'] == 'train' for r in references.values()), 'Training roles differ')
    stages = {u: read(out / f'update{u}/metrics.json') for u in [0, 100, 300]}
    observed = {}
    targets = {}
    used = {}

    def array(name):
        path = out / name
        require(sha(path) == results['artifacts_sha256'][name], 'Changed saved probe: ' + name)
        used[name] = results['artifacts_sha256'][name]
        return np.load(path, allow_pickle=False)

    for rid, ref in references.items():
        mask_path = bundle / ref['observed']
        require(sha(mask_path) == p['assets_sha256'][ref['observed']], 'Support differs')
        support = np.asarray(Image.open(mask_path)) > 0
        observed[rid] = support[np.arange(16) * 16][:, np.arange(16) * 16].reshape(1, 256)
        targets[rid] = array(f'teacher/{rid}_codes.npy')
    require(all(len(stages[u]['rows']) == 100 for u in stages), 'Incomplete metric rows')
    indices = {u: {(r['id'], r['fidelity']): r for r in stages[u]['rows']} for u in stages}
    records = []
    for case in p['cases']:
        rid = case['reference_id']
        stats = {}
        predictions = {}
        features = {}
        for update in stages:
            row = indices[update][(case['id'], 1.0)]
            require(row['reference_id'] == rid, 'Case reference differs')
            logits = array(row['code_logits'])
            stats[update], predictions[update], truth = code_statistics(logits, targets[rid], observed[rid])
            require(np.isclose(stats[update]['code_ce'], row['code_ce'], atol=2e-6), 'CE audit differs')
            require(np.isclose(stats[update]['accuracy'], row['code_accuracy'], atol=1e-7), 'Accuracy audit differs')
            features[update] = array(row['code_features']).transpose(0, 2, 3, 1).reshape(1, 256, 256)[observed[rid]]
        old, new = predictions[0], predictions[300]
        initial, final = indices[0][(case['id'], 1.0)], indices[300][(case['id'], 1.0)]
        delta = features[300].astype(np.float64) - features[0].astype(np.float64)
        records.append({
            'id': case['id'], 'reference_id': rid, 'source': case['source'], 'profile': case['profile'],
            'statistics': {str(k): v for k, v in stats.items()},
            'initial_final_top1_agreement': float((old == new).mean()),
            'correct_codes_gained': int(((old != truth) & (new == truth)).sum()),
            'correct_codes_lost': int(((old == truth) & (new != truth)).sum()),
            'conditioner_delta_rms': float(np.sqrt(np.square(delta).mean())),
            'initial_feature_rms': float(np.sqrt(np.square(features[0].astype(np.float64)).mean())),
            'render_w1_deltas': {k: float(final[k] - initial[k]) for k in
                                ['MSE', 'PSNR', 'SSIM', 'ArcFace_observed_fixed']},
        })
    groups = defaultdict(list)
    for row in records:
        for key in ['all', row['source'] + '/all', row['source'] + '/' + row['profile'],
                    'profile/' + row['profile'], 'degraded' if row['profile'] != 'clear' else 'clear']:
            groups[key].append(row)
    summaries = {}
    for key, rows in groups.items():
        summaries[key] = {
            'cases': len(rows),
            'statistics': {str(u): {k: float(np.mean([r['statistics'][str(u)][k] for r in rows]))
                                       for k in rows[0]['statistics'][str(u)]} for u in [0, 100, 300]},
            'initial_final_top1_agreement': float(np.mean([r['initial_final_top1_agreement'] for r in rows])),
            'correct_codes_gained': sum(r['correct_codes_gained'] for r in rows),
            'correct_codes_lost': sum(r['correct_codes_lost'] for r in rows),
            'conditioner_delta_rms': float(np.mean([r['conditioner_delta_rms'] for r in rows])),
            'render_w1_deltas': {k: float(np.mean([r['render_w1_deltas'][k] for r in rows]))
                                for k in rows[0]['render_w1_deltas']},
            'cases_with_ce_reduction_and_accuracy_loss': sum(
                r['statistics']['300']['code_ce'] < r['statistics']['0']['code_ce'] and
                r['statistics']['300']['accuracy'] < r['statistics']['0']['accuracy'] for r in rows),
        }
    result = {
        'complete': True, 'protocol_sha256': PROTOCOL, 'results_sha256': sha(out / 'results.json'),
        'independent_audit_sha256': sha(returned / 'local_independent_audit.json'),
        'analyzer_sha256': sha(Path(__file__)), 'cases': records, 'summaries': summaries,
        'used_artifacts_sha256': used, 'seconds': time.monotonic() - started,
        'neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'validation_used': False, 'native_used': False, 'production_promoted': False,
        'limitation': 'Post-run analysis of training-only saved probes. Confidence/rank changes do not prove the causal bottleneck, generalization, identity or native CCTV usefulness.',
    }
    destination.mkdir(parents=True)
    (destination / 'analysis.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: summaries[k] for k in ['all', 'degraded', 'clear']}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2')
    parser.add_argument('--return-dir', type=Path, default=ROOT / 'outputs/cctv_dgp_face_code_fit_return_v12_r2_verified')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs/cctv_dgp_face_code_fit_analysis_v12')
    args = parser.parse_args()
    analyze(args.bundle.resolve(), args.return_dir.resolve(), args.output_dir.resolve())
