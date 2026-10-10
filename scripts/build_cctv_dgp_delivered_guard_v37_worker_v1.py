"""Create distinct V37 source from the immutable, successful V34 diagnostic."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / 'scripts/cctv_dgp_group_guard_grad_v34_vm.py'
    text = source.read_text(encoding='utf-8')
    def change(old, new, count=1):
        nonlocal text
        assert text.count(old) == count, (old, text.count(old), count)
        text = text.replace(old, new)
    change('cctv_dgp_group_guard_grad_v34', 'cctv_dgp_delivered_guard_grad_v37', 2)
    change('cctv-dgp-group-guard-grad-v34', 'cctv-dgp-delivered-guard-grad-v37')
    change('V34', 'V37', 5)
    change('Original-state, non-hinged source/profile preservation gradients; VM only.',
           'Original-state delivered-PNG coarse preservation gradients; VM only.')
    change('own-DGP-original-nonhinged-group-preservation-gradient-v34',
           'own-DGP-delivered-PNG-coarse-preservation-gradient-v37')
    change("['raw_MSE', 'one_minus_raw_SSIM', 'one_minus_raw_ArcFace']",
           "['PNG_MSE', 'one_minus_PNG_SSIM', 'one_minus_PNG_ArcFace']")
    change("p['budgets']['worker_seconds'] == 600 and p['budgets']['external_seconds'] == 630",
           "p['budgets']['worker_seconds'] == 900 and p['budgets']['external_seconds'] == 930")
    change("'cctv_dgp_loss_cone_probe_v33_vm'", "'cctv_dgp_finite_clearance_probe_v36_vm'")
    change("p['V33_original_readback_sha256']", "p['V36_original_readback_sha256']")
    change("result['optimizer_updates'] == 8 and result['committed_trajectory_updates'] == 0",
           "result['optimizer_updates'] == 0 and result['candidate_displacement_trials'] == 4 and result['all_trial_states_reset']")
    change('        import torch\n', '        import torch\n        from skimage.metrics import structural_similarity\n')
    change('        from cctv_dgp_batchmatched_identity_v26 import batchmatched_scores\n',
           '        from cctv_dgp_delivered_png_guard_v37 import delivered_png, masked_mse, masked_ssim\n')
    change('time.monotonic() - started <= 600', 'time.monotonic() - started <= 900')
    change('diagnostic600s cap', 'diagnostic900s cap')
    change("        ns = {'torch': torch, 'F': F}\n        exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-raw-mean-SSIM-definitions>', 'exec'), ns)\n", '')
    change("                _, cosine = batchmatched_scores(b, pred, identity)\n                values = torch.stack((ns['mean']((pred - b['target']).square(), b['mask']),\n                    1 - ns['ssim'](pred, b['target'], b['valid7']), 1 - cosine), 1)",
           "                delivered = delivered_png(pred, b['x'], b['mask'])\n                embedding = identity.embedding(delivered, b['mask'], b['grid'])\n                cosine = (embedding * b['truth']).sum(1)\n                values = torch.stack((masked_mse(delivered, b['target'], b['mask']),\n                    1 - masked_ssim(delivered, b['target'], b['valid7']), 1 - cosine), 1)")
    change("                matrix = np.empty((15, 978243), np.float32); saved_rows = []\n", """                matrix = np.empty((15, 978243), np.float32); saved_rows = []
                png_values = delivered.detach().permute(0, 2, 3, 1).cpu().numpy().copy()
                vectors = embedding.detach().cpu().numpy().copy()
                truths = b['truth'].detach().cpu().numpy().copy()
                baseline_receipt = read(probe / f'outputs/state0_{label}/before/receipt.json')
""")
    change("                    np.save(folder / (cid + '.npy'), raw[slot], allow_pickle=False); Image.fromarray(png).save(folder / (cid + '.png'))\n                    saved_rows.append({'id': cid, 'source': item['case']['source'], 'profile': item['case']['profile'],\n                        'raw_guard_values': values[slot].detach().cpu().numpy().astype(np.float64).tolist(), 'raw_parity_maximum_error': parity})", """                    canonical = png.astype(np.float32) / np.float32(255)
                    assert np.array_equal(canonical, png_values[slot]), 'Delivered forward differs from actual PNG encoder'
                    with Image.open(old.with_suffix('.png')) as image: prior_png = np.asarray(image.convert('RGB')).copy()
                    png_error = int(np.abs(png.astype(np.int16) - prior_png.astype(np.int16)).max())
                    assert png_error <= p['same_VM_PNG_byte_tolerance'], 'Original delivered input context changed'
                    truth = np.load(old.with_name(cid + '_target_embedding.npy'), allow_pickle=False)
                    prior_vector = np.load(old.with_name(cid + '_embedding.npy'), allow_pickle=False)
                    vector_error = float(np.abs(vectors[slot] - prior_vector).max())
                    truth_error = float(np.abs(truths[slot] - truth).max())
                    assert max(vector_error, truth_error) <= p['same_VM_embedding_tolerance'], 'Fixed five-case identity context changed'
                    target = item['target'].detach().cpu().numpy()[0].transpose(1, 2, 0).copy()
                    support = item['support']; interior = item['valid7'].cpu().numpy()[0, 0] > 0
                    _, score_map = structural_similarity(canonical, target, channel_axis=2, data_range=1., full=True)
                    measured = [float(np.square((canonical - target)[support]).astype(np.float64).mean()),
                                1 - float(score_map[interior].astype(np.float64).mean()), 1 - float(vectors[slot] @ truths[slot])]
                    numeric = values[slot].detach().cpu().numpy().astype(np.float64)
                    error = np.abs(numeric - measured)
                    assert bool((error <= np.asarray(p['forward_metric_tolerances'])).all()), 'GPU forward metric differs from delivered receipt'
                    baseline = baseline_receipt['rows'][begin + slot]
                    assert baseline['id'] == cid
                    assert abs(measured[0] - baseline['metrics']['MSE']) <= p['forward_metric_tolerances'][0]
                    assert abs(measured[1] - (1 - baseline['metrics']['SSIM'])) <= p['forward_metric_tolerances'][1]
                    assert abs(measured[2] - (1 - baseline['metrics']['ArcFace_observed_fixed'])) <= p['same_VM_embedding_tolerance'] * 2
                    np.save(folder / (cid + '.npy'), raw[slot], allow_pickle=False)
                    Image.fromarray(png).save(folder / (cid + '.png'))
                    np.save(folder / (cid + '_embedding.npy'), vectors[slot], allow_pickle=False)
                    np.save(folder / (cid + '_target_embedding.npy'), truths[slot], allow_pickle=False)
                    saved_rows.append({'id': cid, 'source': item['case']['source'], 'profile': item['case']['profile'],
                        'PNG_guard_values': numeric.tolist(), 'independent_PNG_guard_values': measured,
                        'forward_metric_errors': error.tolist(), 'raw_parity_maximum_error': parity,
                        'PNG_parity_maximum_byte_error': png_error, 'embedding_parity_maximum_error': vector_error,
                        'target_embedding_parity_maximum_error': truth_error})""")
    change('del pred, values, cosine, pieces, matrix, b, items',
           'del pred, delivered, embedding, values, cosine, pieces, matrix, b, items')
    change("'nonhinged_derivatives_not_new_training_losses': True",
           "'nonhinged_surrogate_derivatives_not_new_training_losses': True,\n                'derivative_is_coarse_estimate_not_true_PNG_gradient': True, 'identity_batch_size': 5")
    change("'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(), 'new_checkpoint_created': False,",
           "'peak_allocated_VRAM_bytes': torch.cuda.max_memory_allocated(), 'new_checkpoint_created': False,\n            'true_PNG_derivatives_claimed': False, 'backward_API_calls': 0,\n            'coarse_backward_rule_used_inside_autograd_grad': True, 'finite_preservation_implied': False,")
    change("'cap_seconds': 630, 'kill_grace_seconds': 30, 'within_external_bound': a.elapsed <= 660",
           "'cap_seconds': 930, 'kill_grace_seconds': 30, 'within_external_bound': a.elapsed <= 960")
    change('worker600s deadline', 'worker900s deadline')
    change('signal.alarm(600)', 'signal.alarm(900)')
    import ast
    ast.parse(text, feature_version=(3, 10))
    destination = ROOT / 'scripts/cctv_dgp_delivered_guard_grad_v37_vm.py'
    with destination.open('x', encoding='utf-8', newline='\n') as stream: stream.write(text)
    print('Distinct V37 worker created; no worker executed.')


if __name__ == '__main__': main()
