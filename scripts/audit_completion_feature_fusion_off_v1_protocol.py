"""Independent input/mask/baseline/intervention audit before frozen generation."""
from collections import Counter
from pathlib import Path
import ast
import sys
import time
import numpy as np
from PIL import Image
from completion_feature_fusion_off_v1_common import ROOT, BASE, OUT, sha, read, write, rgb, binary, overlay, verify_bindings


def main():
    start = time.monotonic(); p = read(OUT/'protocol.json'); parent = read(BASE/'protocol.json'); results = read(BASE/'results.json')
    verify_bindings(p)
    assert p['parent_protocol_sha256'] == sha(BASE/'protocol.json') and p['parent_results_sha256'] == sha(BASE/'results.json')
    assert p['family_case_counts'] == parent['family_case_counts'] and len(p['cases']) == 36
    assert p['intervention'] == {'parameter': 'network w', 'baseline': 1, 'candidate': 0, 'adain': False, 'default_inpainting_policy': 'Official w=1; w=0 is an exploratory nondefault setting'}
    assert p['max_forwards'] == {'completion': 29, 'internal512': 29, 'codebook': 29, 'DGP': 0, 'detector': 0}
    assert p['cap_seconds'] == 600 and p['external_timeout_seconds'] == 630 and p['artifact_cap_bytes'] == 268435456
    assert p['input_review_sha256'] == sha(OUT/'input_review.json') and p['user_decision_sha256'] == sha(OUT/'user_architecture_decision.json')
    assert read(OUT/'input_review.json')['all9_existing_input_pages_actually_viewed_this_turn']
    assert read(OUT/'user_architecture_decision.json')['human_answer'] == 'apply the best approach'
    baseline = {r['id']: r for r in results['rows']}; masks = 0; empty = 0; pairs = {}; excluded = 0
    for c, before in zip(p['cases'], parent['cases']):
        for key in c:
            if not key.startswith('baseline_'): assert c[key] == before[key], (c['id'], key)
        assert c['id'] == before['id'] and c['rejected'] == before['rejected']
        if c['rejected']:
            assert c['input_review'] in ['needs_clearer', 'out_of_scope'] and baseline[c['id']]['rejected_before_neural']; excluded += 1
        else:
            final, union = binary(ROOT/c['masks']['removal']), binary(ROOT/c['conditioning'])
            assert np.array_equal(union, final | binary(ROOT/c['reviewed'])) and not (final & binary(ROOT/c['masks']['protected'])).any()
            assert union.mean() < .85
            for key, folder, suffix in [('baseline_output', 'images', '.png'), ('baseline_stages', 'stages', '.npz'), ('baseline_metadata', 'metadata', '.json')]:
                assert c[key] == (BASE/folder/(c['id']+suffix)).relative_to(ROOT).as_posix()
                assert sha(ROOT/c[key]) == results['artifacts_sha256'][folder+'/'+c['id']+suffix]
            if not final.any(): assert not union.any(); empty += 1
            masks += 1
        pairs.setdefault(c['base_id'], []).append(c)
    assert masks == 32 and empty == excluded == 4 and len(pairs) == 18
    for entries in pairs.values():
        assert len(entries) == 2 and {c['condition'] for c in entries} == {'original_photo', 'synthetic_degraded_photo'}
        if entries[0]['rejected']: assert entries[1]['rejected']; continue
        for key in ['removal', 'protected', 'core', 'face']: np.testing.assert_array_equal(binary(ROOT/entries[0]['masks'][key]), binary(ROOT/entries[1]['masks'][key]))
        np.testing.assert_array_equal(binary(ROOT/entries[0]['conditioning']), binary(ROOT/entries[1]['conditioning']))
    cells = 0
    for page in parent['input_pages']:
        assert sha(BASE/page['path']) == page['sha256']
        with Image.open(BASE/page['path']) as im: values = np.array(im.convert('RGB'))
        for entry in page['entries']:
            c = next(c for c in p['cases'] if c['id'] == entry['id']); source = rgb(ROOT/c['input'])
            if c['rejected']:
                x = 256*entry['column']; np.testing.assert_array_equal(values[24:280, x:x+256], source); cells += 1; continue
            final, union = binary(ROOT/c['masks']['removal']), binary(ROOT/c['conditioning'])
            expected = [source, overlay(source, final, (16, 185, 129)), overlay(source, union, (240, 140, 32)), overlay(source, union & ~final, (210, 64, 180))]
            y = 30+296*entry['row']
            for k, image in enumerate(expected): np.testing.assert_array_equal(values[y:y+256, k*256:(k+1)*256], image); cells += 1
    assert cells == 132
    architecture = ast.parse((ROOT/'third_party/codeformer/codeformer_arch.py').read_text())
    cls = next(n for n in architecture.body if isinstance(n, ast.ClassDef) and n.name == 'CodeFormer')
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'forward')
    branches = [n for n in ast.walk(forward) if isinstance(n, ast.If) and ast.unparse(n.test) == 'w > 0']
    assert len(branches) == 1 and 'fuse_convs_dict' in ast.unparse(branches[0].body[0])
    assert forward.args.defaults[0].value == 0 and 'F.softmax(logits, dim=2)' in ast.unparse(forward)
    worker = (ROOT/'scripts/run_completion_feature_fusion_off_v1.py').read_text()
    assert "return args, {'w': active_w, 'adain': False}" in worker and 'fresh, meta = generate(c, 1)' in worker and 'output, meta = generate(c, 0)' in worker
    assert not any(v in worker for v in ['.backward(', 'autograd.grad(', 'torch.optim.', 'requires_grad_(True)', 'torch.save('])
    assert not (OUT/'results.json').exists() and not (OUT/'execution.json').exists()
    assert not any(n in sys.modules for n in ['torch', 'dgp_face_workflow_v3', 'pretrained_completion'])
    write(OUT/'independent_protocol_audit.json', {'complete': True, 'protocol_sha256': sha(OUT/'protocol.json'), 'checker_sha256': sha(Path(__file__)),
          'sources_verified': len(p['sources_sha256']), 'app_bindings_verified': len(p['app_preservation_sha256']),
          'cases': 36, 'same32_final_and_union_masks': True, 'pairs': 18, 'unchanged_exclusions': 4, 'empty_controls': 4,
          'all132_input_preview_cells_exact': True, 'same32_w1_baseline_images_and_stages_bound': True,
          'one_fixed_parameter_intervention': True, 'source_w0_skips_fusion_branch': True, 'training_code_absent_from_worker': True,
          'neural_calls': 0, 'gradient_calls': 0, 'optimizer_updates': 0, 'VM_calls': 0, 'goal_complete': False, 'seconds': time.monotonic()-start})
    print({'complete': True, 'input_cells': cells, 'cases': 36, 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__': main()
