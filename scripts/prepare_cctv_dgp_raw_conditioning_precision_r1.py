"""Retain the unsealed draft; restore original float64 math for RAW32 sizing."""
import ast
import copy
from datetime import datetime, timezone

from import_cctv_dgp_conditioning_learning_v1_r4_return import ROOT, OUT, RETURN, read, sha, write


def main():
    old = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_source_preparation.json')
    red = read(OUT / 'cctv_dgp_raw_conditioning_learning_v1_precision_red.json')
    assert old['complete'] and not red['complete']
    packet = OUT / 'cctv_dgp_raw_conditioning_learning_v1_vm'
    assert not (packet / 'protocol.json').exists() and not (packet / 'outputs').exists()
    contract = packet / 'conditioning_contract.py'
    alias = ROOT / 'scripts/cctv_dgp_raw_conditioning_learning_v1_contract.py'
    assert sha(contract) == sha(alias) == red['contract_sha256']
    stop = OUT / 'cctv_dgp_raw_conditioning_learning_v1_draft_stop'
    assert not stop.exists()
    stop.mkdir()
    preparer = ROOT / 'scripts/prepare_cctv_dgp_raw_conditioning_learning_v1.py'
    for source, name in [(contract, 'conditioning_contract.py'), (alias, alias.name), (preparer, preparer.name)]:
        with (stop / name).open('xb') as stream:
            stream.write(source.read_bytes())
    original = (RETURN / 'conditioning_contract.py').read_text(encoding='utf-8')
    start, end = original.index('def solve_direction('), original.index('def bounded_rate(')
    solver = original[start:end]
    changes = [
        ('len(vectors) == len(records) == 64', 'len(vectors) == len(records) == 32'),
        ('np.zeros((64,64))', 'np.zeros((32,32))'),
        ('rank_bound = 64*', 'rank_bound = 32*'),
        ('Retain all64 loss dimensions', 'Retain all32 RAW loss dimensions'),
        ('np.zeros(64)', 'np.zeros(32)'),
        ('all64_loss_dimensions_retained', 'all32_RAW_loss_dimensions_retained'),
    ]
    for before, after in changes:
        assert solver.count(before) >= 1, before
        solver = solver.replace(before, after)
    assert solver.count('np.float64') == original[start:end].count('np.float64')
    assert solver.count('np.float32') == original[start:end].count('np.float32')
    text = original[:start] + solver + original[end:]
    assert text[text.index('def bounded_rate('):] == original[end:]
    ast.parse(text, feature_version=(3, 10))
    for path in [contract, alias]:
        assert sha(path) == red['contract_sha256']
        with path.open('w', encoding='utf-8', newline='\n') as stream:
            stream.write(text)
    updated = copy.deepcopy(old)
    for row in updated['sources']:
        if row['target'] == 'conditioning_contract.py':
            row['unsealed_draft_sha256'] = row['target_sha256']
            row['target_sha256'] = sha(contract)
    updated.update(UTC=datetime.now(timezone.utc).isoformat(),
        draft_precision_stop_retained=True, precision_repair_preparer_sha256=sha(__file__),
        old_source_preparation_sha256=sha(OUT / 'cctv_dgp_raw_conditioning_learning_v1_source_preparation.json'),
        precision_red_sha256=sha(OUT / 'cctv_dgp_raw_conditioning_learning_v1_precision_red.json'),
        original_float64_rank_Gram_dot_math_and_float32_direction_dtype_preserved=True,
        original_quality_contract_restored_exactly=True)
    write(OUT / 'cctv_dgp_raw_conditioning_learning_v1_source_preparation_r1.json', updated)
    write(stop / 'stop.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        original_preparer_sha256=sha(stop / preparer.name), original_contract_sha256=red['contract_sha256'],
        cause='Unsealed dimension substitution also changed float64 to float32; synthetic full-rank regression rejected it',
        observed_tool_chunk_id='32c757', corrected_contract_sha256=sha(contract),
        original_R4_protocol_worker_checkpoints_outputs_and_all_gates_unchanged=True,
        packet_not_sealed_uploaded_or_trained=True))
    print(dict(complete=True, unsealed_draft_retained=True, original_float64_math_restored=True,
               actual_models_or_training=0))


if __name__ == '__main__':
    main()
