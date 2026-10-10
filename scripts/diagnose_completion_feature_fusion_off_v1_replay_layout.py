"""Two frozen direct-network calls isolate input layout; no regeneration or learning."""
from pathlib import Path
import sys
import time
import hashlib
import traceback
import numpy as np
import torch
from completion_feature_fusion_off_v1_common import ROOT, OUT, sha, read, write, rgb, binary, verify_bindings
sys.path.insert(0, str(ROOT))
from dgp_face_workflow_v3 import DGPFaceWorkflow, canonical_tensor
from cctv_dgp_pilot import state_hash

CHECKER = ROOT/'scripts/audit_completion_feature_fusion_off_v1.py'
DIAG = OUT/'replay_layout_diagnostic'


def main():
    assert Path(sys.executable).resolve() == (ROOT/'venv/Scripts/python.exe').resolve()
    assert not DIAG.exists()
    p = read(OUT/'protocol.json'); verify_bindings(p)
    assert sha(OUT/'protocol.json') == '53c038ea900c1b8d0c602679435cf794b6770708196183ec6a1773437becf95b'
    assert not (OUT/'independent_saved_output_audit.json').exists()
    DIAG.mkdir()
    write(DIAG/'contract.json', {
        'protocol_sha256': sha(OUT/'protocol.json'), 'source_sha256': sha(Path(__file__)),
        'original_checker_sha256': sha(CHECKER), 'case': p['parity_case'],
        'hypothesis': 'C-contiguous saved input and actual channels-last adapter input have equal values but different convolution arithmetic',
        'inputs': ['saved_C_contiguous', 'actual_adapter_layout'], 'max_network_forwards': 2,
        'cap_seconds': 120, 'optimizer_updates': 0, 'gradient_calls': 0,
        'outputs_regenerated': 0, 'new_tolerance': False, 'app_changes': False})
    # Retain the first exact checker stop before any follow-up network calls.
    write(DIAG/'original_checker_failure.json', {
        'complete': False, 'checker_sha256': sha(CHECKER), 'protocol_sha256': sha(OUT/'protocol.json'),
        'invocation': r'.\venv\Scripts\python.exe -B -u scripts\audit_completion_feature_fusion_off_v1.py',
        'exit_code': 1, 'tool_chunks': ['78bb9a', '26372a'], 'location': 'scripts/audit_completion_feature_fusion_off_v1.py:121',
        'cause_observed': 'Exact fresh replay logits differ; network RGB raw had already passed exact equality',
        'exception': 'AssertionError: Arrays are not equal', 'shape': [1, 256, 512],
        'mismatched_elements': 131004, 'total_elements': 131072, 'maximum_absolute_error': 0.00341415,
        'maximum_relative_error': 4.3972583, 'original_trace_retained_in_tool_history': True,
        'new_tolerance': False, 'checker_modified': False, 'generation_repeated': False,
        'gradients_or_optimizer': False, 'audit_pass_not_implied': True})
    started = time.monotonic(); calls = 0
    try:
        engine = DGPFaceWorkflow(device='cpu'); engine._runtime(); model = engine._generator()
        assert torch.__version__ == p['torch'] and not torch.cuda.is_available() and torch.get_num_threads() == 4
        before = state_hash(model); assert before == p['parent_model_state']
        assert all(not v.requires_grad and v.grad is None for v in model.parameters())
        c = next(c for c in p['cases'] if c['id'] == p['parity_case'])
        with np.load(OUT/'stages'/(c['id']+'.npz'), allow_pickle=False) as archive:
            data = {k: archive[k] for k in ['neural_input512', 'internal512', 'logits', 'lq_feat', 'code_indices']}
        captured = {}
        class StopBeforeNetwork(Exception): pass
        def capture(module, args, kwargs):
            assert kwargs == {'w': 1, 'adain': False} and len(args) == 1
            captured['input'] = args[0].detach(); raise StopBeforeNetwork
        handle = model.net.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            with torch.inference_mode():
                source = canonical_tensor(rgb(ROOT/c['input']), 'cpu')
                mask = torch.from_numpy(binary(ROOT/c['conditioning']).astype(np.float32))[None, None]
                try: model(source, mask)
                except StopBeforeNetwork: pass
                else: raise AssertionError('Expected adapter stop before network')
        finally: handle.remove()
        actual_input = captured['input']; contiguous = torch.from_numpy(data['neural_input512'])
        np.testing.assert_array_equal(actual_input.numpy(), contiguous.numpy())
        assert contiguous.is_contiguous() and not actual_input.is_contiguous()
        assert actual_input.is_contiguous(memory_format=torch.channels_last)
        findings = []
        for label, x in [('saved_C_contiguous', contiguous), ('actual_adapter_layout', actual_input)]:
            assert time.monotonic()-started < 120
            with torch.inference_mode(): result = model.net(x, w=0, adain=False)
            calls += 1
            row = {'layout': label, 'stride': list(x.stride()), 'input_sha256': hashlib.sha256(x.numpy().tobytes()).hexdigest(), 'stages': {}}
            for name, value in zip(['internal512', 'logits', 'lq_feat'], result):
                array = value.numpy(); expected = data[name]
                row['stages'][name] = {'exact': bool(np.array_equal(array, expected)),
                                      'mismatched_elements': int(np.count_nonzero(array != expected)),
                                      'maximum_absolute_error': float(np.max(np.abs(array.astype(np.float64)-expected.astype(np.float64))))}
            indices = torch.topk(torch.softmax(result[1], dim=2), 1, dim=2).indices.numpy()
            row['code_indices_exact'] = bool(np.array_equal(indices, data['code_indices']))
            findings.append(row); print(row, flush=True)
        assert calls == 2 and findings[0]['stages']['internal512']['exact'] and not findings[0]['stages']['logits']['exact']
        assert all(v['exact'] for v in findings[1]['stages'].values()) and all(row['code_indices_exact'] for row in findings)
        assert state_hash(model) == before and all(v.grad is None for v in model.parameters())
        assert engine.restorer is None and engine.detector is None
        verify_bindings(p); assert time.monotonic()-started < 120
        write(DIAG/'results.json', {'complete': True, 'contract_sha256': sha(DIAG/'contract.json'), 'rows': findings,
              'adapter_capture_before_network': 1, 'actual_network_forwards': calls, 'same_input_values_exact': True,
              'source_and_actual_model_state_unchanged': True, 'root_cause_confirmed': 'Replay changed memory layout from actual channels-last to C-contiguous',
              'layout_only_fix_proposed': True, 'tolerances_changed': False, 'outputs_regenerated': 0,
              'gradient_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0, 'app_changes': False,
              'audit_pass_or_quality_not_implied': True, 'seconds': time.monotonic()-started, 'cap_seconds': 120})
        print({'complete': True, 'calls': calls, 'seconds': time.monotonic()-started}, flush=True)
    except BaseException as exc:
        write(DIAG/'failure.json', {'complete': False, 'error': repr(exc), 'traceback': traceback.format_exc(),
              'network_forwards': calls, 'seconds': time.monotonic()-started, 'optimizer_updates': 0})
        raise


if __name__ == '__main__': main()
