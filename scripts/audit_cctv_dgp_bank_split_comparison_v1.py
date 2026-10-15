"""Bind independently audited A/B returns to a genuinely matched comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def audit(a_root, b_root, a_analysis, b_analysis, receipt):
    started = time.monotonic()
    assert not receipt.exists()
    pins = ['6ca05c6b87a4588bfea2bc5b80307927c3a2fd47fa419567ee35280b37a78d0b',
            '68197fd20bef1eae52c4c232164a027e095d567d4bfb37aea79c4f908853ea42']
    bindings = {}; protocols = []; audits = []
    for route, root, analysis, pin in zip(['A','B'], [a_root,b_root], [a_analysis,b_analysis], pins):
        assert sha(root/'protocol.json') == pin
        p = read(root/'protocol.json'); assert p['selected_routes'] == [route]
        protocols.append(p)
        for name in ['import.json','execution_audit.json','independent_audit.json']:
            path = analysis/name; r = read(path)
            assert r['complete'] and r['protocol_sha256'] == pin
            bindings[str(path.resolve())] = sha(path)
        independent = read(analysis/'independent_audit.json')
        assert independent['optimizer_updates_in_VM'] == 50
        assert independent['saved_paired_raw_PNG_cases_checked'] == (7810 if route == 'A' else 7910)
        assert independent['native_compositions_checked'] == (48 if route == 'A' else 72)
        assert independent['CPU_CUDA_prior_fixtures_rechecked'] == (0 if route == 'A' else 6)
        assert independent['original_thresholds_and_gate_decisions_unchanged_required']
        assert not independent['model_qualified'] and not independent['app_promotion']
        assert independent['local_optimizer_updates'] == independent['local_gradient_queries'] == independent['local_backwards'] == 0
        audits.append(independent)
        manifest = root/'export_manifest.json'
        assert sha(manifest) == independent['manifest_sha256']
        bindings[str(manifest.resolve())] = sha(manifest)
    a, b = protocols
    common = ['cases','references','schedule','native_development','preview_case_ids',
              'canonical_target_RGB_sha256','original_checkpoint_sha256','original_state',
              'recognizer_state','standalone_prior_sha256','mean_style_sha256','scientific_thresholds',
              'normalizers','active_anchor_normalizers','reconstruction_weights',
              'regression_barrier_coefficient','active_clear_blur_anchor_coefficients','optimizer',
              'maximum_updates_per_route','full_epochs_per_route','fitting_reference_exposures_per_route',
              'fitting_image_exposures_per_route','reserved_final_pixels']
    for name in common:
        assert a[name] == b[name], name
    assert a['reserved_final_pixels'] == 0
    for name in a['assets_sha256']:
        if name.startswith(('data/','weights/')):
            assert a['assets_sha256'][name] == b['assets_sha256'][name]
            assert sha(a_root/name) == sha(b_root/name) == a['assets_sha256'][name]
    sys.path.insert(0, str(a_root.resolve()))
    from cctv_dgp_head4_lossless_v1 import unpack
    baseline_packs = 0
    for begin in range(0,3905,5):
        ids = [c['id'] for c in a['cases'][begin:begin+5]]
        paths = [r/'outputs/bank_comparison_v1/baseline/packs'/f'b{begin//5:03d}.npz' for r in [a_root,b_root]]
        saved = [unpack(path,ids) for path in paths]
        assert np.array_equal(saved[0][0],saved[1][0]), 'Baseline RGB changed across routes'
        assert np.array_equal(saved[0][1],saved[1][1]), 'Baseline recognizer embeddings changed across routes'
        baseline_packs += 1
    for c in a['native_development']:
        paths = [r/'outputs/bank_comparison_v1/baseline/native'/(c['id']+'.npy') for r in [a_root,b_root]]
        assert np.array_equal(np.load(paths[0],allow_pickle=False),np.load(paths[1],allow_pickle=False))
        png_paths = [path.with_suffix('.png') for path in paths]
        assert sha(png_paths[0]) == sha(png_paths[1])
    gates = {route: audit['metric_and_gate_decisions'] for route,audit in zip(['A','B'],audits)}
    for path,expected in bindings.items():
        assert sha(path) == expected
    record = dict(complete=True, checker_sha256=sha(__file__), compared_receipt_bindings=bindings,
        protocol_A_sha256=pins[0], protocol_B_sha256=pins[1], matched_protocol_fields=common,
        unchanged_baseline_RGB_embedding_packs=baseline_packs, unchanged_native_baseline_cases=24,
        optimizer_updates_each=50, fitting_references_each=250, source_references_each=125,
        profile_exposures_each=1250, full_epochs_each=0, unchanged_gate_decisions=gates,
        full_3905_neural_replay_claimed=False, bank_ablation_independently_audited_in_B=True,
        combined_visual_review_pending=True, independent_final_review_pending=True,
        native_unpaired_not_synthetic_reference_metrics=True, all_seven_completion_families_pending=True,
        local_neural_gradient_or_optimizer_calls=0, model_qualified=False, app_promotion=False,
        goal_complete=False, seconds=time.monotonic()-started)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    with receipt.open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(dict(complete=True,matched_baseline_packs=baseline_packs,qualification=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['a-root','b-root','a-analysis','b-analysis','receipt']:
        parser.add_argument('--'+name,type=Path,required=True)
    args = parser.parse_args()
    audit(args.a_root,args.b_root,args.a_analysis,args.b_analysis,args.receipt)
