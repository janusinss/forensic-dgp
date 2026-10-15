"""Compare real saved RAW/PNG loss values; no neural model or autograd queries."""
import argparse
from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import time

import numpy as np
import torch

from import_cctv_dgp_conditioning_learning_v1_r3_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write
from audit_cctv_dgp_conditioning_learning_v1_r3_return import image, support


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw-helper', type=Path, required=True)
    parser.add_argument('--record', type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    audited = read(ANALYSIS / 'failure_audit.json')
    assert audited['complete'] and audited['completed_snapshots_audited'] == ['initial']
    p, old = read(RETURN / 'protocol.json'), read(PARENT / 'protocol.json')
    assert sha(RETURN / 'protocol.json') == PIN
    refs = {r['id']: r for r in old['references']}
    stage = RETURN / 'outputs/conditioning_learning_v1_r3'
    snapshot = read(stage / 'initial/snapshot.json')
    saved = {r['id']: r for r in snapshot['rows']}
    helper_path = args.raw_helper.resolve()
    assert helper_path.is_relative_to(ROOT)
    raw_helper = module(helper_path, 'reviewed_RAW_losses')
    png_helper = module(RETURN / 'conditioning_losses.py', 'sealed_PNG_losses')
    rows, failed = [], []

    def tensor(a):
        assert a.dtype == np.float32
        return torch.from_numpy(a).permute(0, 3, 1, 2).contiguous()

    for block in p['reference_blocks']:
        assert time.monotonic() - started < 150, 'Finite loss-utility check cap'
        cs = old['cases'][block['begin']:block['begin'] + 5]
        raw = np.load(stage / 'initial' / f"b{block['begin']//5:03d}.npy", allow_pickle=False)
        targets = np.stack([image(PARENT / refs[c['source_person_or_reference']]['target']) for c in cs])
        masks = np.stack([image(PARENT / refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in cs])
        patches = np.stack([support(m, c['landmarks5_canvas_xy']) for m, c in zip(masks, cs)])
        png = np.stack([image(stage / 'initial/png' / (c['id'] + '.png')) for c in cs])
        target = tensor(targets.astype(np.float32) / np.float32(255))
        mask = torch.from_numpy(masks.astype(np.float32))[:, None]
        patch = torch.from_numpy(patches.astype(np.float32))[:, None]
        raw_t = tensor(raw)
        raw_terms = raw_helper.pixel_and_structure_losses(raw_t, target, mask, patch)
        rgb, feature = png_helper.delivered_surrogates(raw_t, torch.from_numpy(png).permute(0, 3, 1, 2).contiguous())
        png_terms = png_helper.pixel_and_structure_losses(rgb, target, mask, patch, feature)
        for stage_name, terms in [('raw', raw_terms), ('png', png_terms)]:
            assert set(terms) == {'MSE', 'SSIM_loss', 'landmark_structure'}
            assert all(v.dtype == torch.float64 and not v.requires_grad and torch.isfinite(v).all() for v in terms.values())
            for i, c in enumerate(cs):
                scores = saved[c['id']][stage_name]
                for term, exact, bound in [
                    ('MSE', scores['MSE'], 1e-13),
                    ('SSIM_loss', 1. - scores['SSIM'], 1e-6),
                    ('landmark_structure', scores['landmark_high_frequency_MSE'], 1e-13),
                ]:
                    value = float(terms[term][i])
                    error = abs(value - exact)
                    row = dict(id=c['id'], source=c['source'], profile=c['profile'], reference=block['reference'],
                               begin=block['begin'], stage=stage_name, term=term,
                               utility_value=value, scientific_saved_value=exact, absolute_difference=error,
                               retained_alignment_bound=bound, pass_alignment=error <= bound)
                    rows.append(row)
                    if error > bound:
                        failed.append(row)
    assert len(rows) == 600
    record = dict(complete=not failed, UTC=datetime.now(timezone.utc).isoformat(),
                  checker_sha256=sha(__file__), protocol_sha256=PIN,
                  failure_audit_sha256=sha(ANALYSIS / 'failure_audit.json'),
                  snapshot_sha256=sha(stage / 'initial/snapshot.json'), raw_helper_sha256=sha(helper_path),
                  png_helper_sha256=sha(RETURN / 'conditioning_losses.py'),
                  device='CPU', cases=100, real_saved_loss_checks=600, rows=rows, failures=failed,
                  maximum_RAW_SSIM_difference=max(r['absolute_difference'] for r in rows if r['stage'] == 'raw' and r['term'] == 'SSIM_loss'),
                  CPU_loss_utility_replay_not_GPU_failure_value_or_neural_gradient_replay=True,
                  local_neural_gradient_optimizer_parameter_displacement_calls=0,
                  no_scientific_quality_or_alignment_tolerance_changed=True,
                  seconds=time.monotonic() - started, cap_seconds=150, model_qualified=False, goal_complete=False)
    write(args.record, record)
    print(dict(complete=record['complete'], real_saved_loss_checks=600, failures=len(failed),
               max_RAW_SSIM_difference=record['maximum_RAW_SSIM_difference'], seconds=record['seconds']), flush=True)
    assert not failed, 'Saved scientific and differentiable-forward definitions differ; retain this failed regression'


if __name__ == '__main__':
    main()
