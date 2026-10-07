"""Independent source/layout/baseline-parity receipt readback; no neural calls."""
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_original_decoder_review_v1'
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    start = time.monotonic()
    receipt = OUT / 'independent_readback.json'
    assert not receipt.exists()
    path = OUT / 'review.json'
    r = read(path)
    assert r['complete'] and r['raw_and_PNG_initial_parity_cases'] == 50
    for name, digest in r['source_bindings_sha256'].items():
        target = (ROOT / name).resolve()
        assert target.is_relative_to(ROOT) and sha(target) == digest, name
    assert r['retained_protocol_sha256'] == sha(BUNDLE / 'protocol.json')
    p = read(BUNDLE / 'protocol.json')
    assert [case['id'] for case in p['cases']] == [row['id'] for row in r['initial_cases']]
    assert all(case['role'] == 'train' for case in p['cases'])
    expected = {'head1.block0.weight': [64,128,3,3], 'head1.block1.weight': [64,64,3,3],
                'head2.block0.weight': [64,128,3,3], 'head2.block1.weight': [64,64,3,3],
                'head3.block0.weight': [64,128,3,3], 'head3.block1.weight': [64,64,3,3],
                'head4.block0.weight': [64,128,3,3], 'head4.block1.weight': [64,64,3,3],
                'smooth.0.weight': [64,256,3,3], 'smooth.0.bias': [64],
                'smooth2.0.weight': [32,64,3,3], 'smooth2.0.bias': [32],
                'final.weight': [3,32,3,3], 'final.bias': [3]}
    assert {row['name']: row['shape'] for row in r['parameter_layout']} == expected
    offset = 0
    for row in r['parameter_layout']:
        size = 1
        for value in row['shape']:
            size *= value
        assert row['start'] == offset and row['end'] == offset + size
        offset += size
    assert offset == r['trained_parameters_proposed'] == 609219 and r['selected_parameter_tensors'] == 14
    assert sum(r['parameter_counts_by_module'].values()) == offset
    assert r['parameter_counts_by_module'] == {'head1':110592,'head2':110592,'head3':110592,'head4':110592,
                                               'smooth':147520,'smooth2':18464,'final':867}
    original = ast.parse((BUNDLE / 'models/dgp_synthesizer.py').read_text(encoding='utf-8'))
    model = next(n for n in original.body if isinstance(n, ast.ClassDef) and n.name == 'DGPSynthesizer')
    forward = next(n for n in model.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    assert hashlib.sha256(ast.dump(forward, include_attributes=False).encode()).hexdigest() == r['original_forward_AST_sha256']
    candidate_source = ROOT / 'scripts/cctv_dgp_original_decoder_candidate_v1.py'
    tree = ast.parse(candidate_source.read_text(encoding='utf-8'), feature_version=(3,10))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'OriginalDecoderCandidate')
    candidate_forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    calls = [ast.unparse(n.func) for n in ast.walk(candidate_forward) if isinstance(n, ast.Call)]
    assert 'self.net' in calls and 'require_gradient_vm' in calls and 'torch.where' in calls
    assert not candidate_forward.decorator_list
    assert not any(name.endswith(('backward','grad','step','load_state_dict','save')) for name in calls)
    assert not any(isinstance(n, ast.Attribute) and n.attr in ['Adam','AdamW','SGD'] for n in ast.walk(tree))
    assert r['DGP_state_before_after'] == 'd89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3'
    assert r['original_model_and_candidate_share_no_tensors'] and r['encoder_FPN_and_all_stored_buffers_frozen']
    assert r['normalization_layers'] == 5 and r['forward_graph_uses_original_implementation']
    assert r['neural_forward_counts'] == {'original':50,'candidate':51}
    assert all(row['raw_parity_exact'] and row['PNG_parity_exact'] and 0 <= row['historical_GPU_cache_maximum_difference'] <= 1e-5
               and 0 <= row['preclip_saturated_fraction'] <= 1 for row in r['initial_cases'])
    assert set(r['boundary_failures_verified']) == {'implicit_local_derivative','empty_support','wrong_canvas','nonfinite_input'}
    assert r['partial_observed_support_preserved'] and r['no_target_or_identity_conditioning'] and r['no_new_weights_or_checkpoint_created']
    assert r['local_gradient_calls'] == r['local_backward_calls'] == r['local_optimizer_updates'] == 0
    assert r['VM_gradient_proof_pending'] and not r['new_training_protocol_created']
    assert not r['native_or_reserved_used'] and not r['app_promotion'] and not r['independent_final_review'] and not r['goal_complete']
    result = {'complete': True,'review_sha256': sha(path),'checker_sha256': sha(Path(__file__)),
              'source_bindings_verified': len(r['source_bindings_sha256']), 'decoder_parameters_verified': offset,
              'original_forward_AST_and_exact_14_weight_layout_verified': True,
              'initial_forward_receipts_verified': 50, 'local_derivative_rejection_preserved': True,
              'local_neural_or_gradient_calls': 0,'VM_actions': False,'app_promotion': False,
              'goal_complete': False,'seconds': time.monotonic()-start}
    with receipt.open('x',encoding='utf-8',newline='\n') as f:
        f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
