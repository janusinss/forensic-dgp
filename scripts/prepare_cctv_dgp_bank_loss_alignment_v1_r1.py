"""Seal one demonstrated input-layout repair; retain the original failed packet."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import tarfile

from import_cctv_dgp_bank_loss_alignment_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write

NAME = 'cctv_dgp_bank_loss_alignment_v1_r1_vm'
STEM = 'cctv-dgp-bank-loss-alignment-v1-r1'


def main():
    dest = OUT / NAME
    archive = OUT / (STEM + '-execution.tar.gz')
    receipt = OUT / 'cctv_dgp_bank_loss_alignment_v1_r1_preparation.json'
    assert not any(path.exists() for path in [dest, archive, receipt])
    failure = read(ANALYSIS / 'independent_failure_audit.json')
    probe_path = ANALYSIS / 'layout_reproduction/run_results.json'
    probe = read(probe_path)
    assert failure['complete'] and probe['complete']
    assert failure['VM_counts_recomputed']['gradient_queries'] == 0
    assert probe['differences']['initial_canonical_vs_original_canonical']['exact']
    assert not probe['differences']['initial_legacy_vs_initial_canonical']['exact']
    original = read(RETURN / 'protocol.json')
    assert sha(RETURN / 'protocol.json') == PIN
    for name, expected in original['parent_required_sha256'].items():
        assert sha(PARENT / name) == expected
    source_path = RETURN / 'scripts/cctv_dgp_bank_loss_alignment_v1_vm.py'
    source = source_path.read_text(encoding='utf-8')
    assert sha(source_path) == original['assets_sha256']['scripts/' + source_path.name]
    # Exactly one numerical-path change: preserve the parent's contiguous NCHW layout.
    old_tensor = "return torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0,3,1,2).cuda()"
    fixed_tensor = "return torch.from_numpy(np.asarray(a).astype(np.float32)/np.float32(255)).permute(0,3,1,2).contiguous().cuda()"
    assert source.count(old_tensor) == 1
    fixed = source.replace(old_tensor, fixed_tensor)
    fixed = fixed.replace("NAME='cctv_dgp_bank_loss_alignment_v1_vm'", "NAME='" + NAME + "'")
    fixed = fixed.replace("STEM='cctv-dgp-bank-loss-alignment-v1'", "STEM='" + STEM + "'")
    fixed = fixed.replace("(out/'sampled_gradients').mkdir()", "(out/'sampled_gradients').mkdir(); (out/'forward_replay').mkdir()")
    old_parity = "                    assert np.array_equal(frozen.permute(0,2,3,1).cpu().numpy(),expected),'Same L4 frozen raw control differs'"
    new_parity = """                    actual_raw=frozen.permute(0,2,3,1).cpu().numpy().copy()
                    raw_path=out/'forward_replay'/f'{state_label}_b{begin//5:03d}.npy'
                    with raw_path.open('xb') as stream: np.save(stream,actual_raw,allow_pickle=False)
                    exact=np.array_equal(actual_raw,expected)
                    replay=dict(state=state_label,begin=begin,case_ids=ids,
                        input_stride=list(x.stride()),target_stride=list(target.stride()),
                        input_NCHW_contiguous=x.is_contiguous(),target_NCHW_contiguous=target.is_contiguous(),
                        input_RGB_float32_sha256=hashlib.sha256(x.permute(0,2,3,1).cpu().numpy().tobytes()).hexdigest(),
                        raw_file=raw_path.relative_to(out).as_posix(),raw_file_sha256=sha(raw_path),
                        actual_RGB_float32_sha256=hashlib.sha256(actual_raw.tobytes()).hexdigest(),
                        expected_RGB_float32_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),
                        maximum_absolute_error=float(np.abs(actual_raw.astype(np.float64)-expected.astype(np.float64)).max()),
                        changed_values=int(np.count_nonzero(actual_raw!=expected)),exact_raw_replay=exact,
                        scientific_gates_unchanged=True,optimizer_updates=0,gradient_queries_before_batch=counts['gradient_queries'])
                    write(raw_path.with_suffix('.json'),replay)
                    assert x.is_contiguous() and target.is_contiguous(),'Retained NCHW input layout differs'
                    assert exact,'Same L4 frozen raw control differs'"""
    assert fixed.count(old_parity) == 1
    fixed = fixed.replace(old_parity, new_parity)
    assert fixed.count('paired_TRAIN_cases=100,states_measured=2') == 1
    fixed = fixed.replace('paired_TRAIN_cases=100,states_measured=2',
                          'all40_frozen_raw_controls_retained=True,input_layout_matches_parent=True,paired_TRAIN_cases=100,states_measured=2')
    tree = ast.parse(fixed)
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and n.func.attr in ['step', 'backward', 'Adam', 'AdamW', 'SGD'] for n in ast.walk(tree))
    dest.mkdir()
    (dest / 'scripts').mkdir()
    worker_name = 'cctv_dgp_bank_loss_alignment_v1_r1_vm.py'
    (dest / 'scripts' / worker_name).write_text(fixed, encoding='utf-8', newline='\n')
    helper = 'cctv_dgp_bank_loss_alignment_v1.py'
    assert sha(RETURN / helper) == original['assets_sha256'][helper]
    (dest / helper).write_bytes((RETURN / helper).read_bytes())
    shell = (RETURN / 'scripts/run_alignment.sh').read_text(encoding='utf-8')
    shell = shell.replace('cctv_dgp_bank_loss_alignment_v1_vm.py', worker_name)
    (dest / 'scripts/run_alignment.sh').write_text(shell, encoding='utf-8', newline='\n')
    p = dict(original)
    p.update(name=NAME, format=original['format'] + '-input-layout-r1',
        superseded_failed_protocol_sha256=PIN,
        input_RGB_layout='contiguous NCHW; exactly matches the retained parent tensor constructor',
        frozen_raw_forward_blocks_retained=40,
        frozen_raw_forward_payload_bytes=40 * 5 * 256 * 256 * 3 * 4,
        combined_gradient_and_forward_payload_upper_bound_bytes=original['uncompressed_sampled_gradient_upper_bound_bytes'] + 40 * 5 * 256 * 256 * 3 * 4,
        bit_exact_control_check_retained=True, actual_L4_layout_repair_confirmation_pending=True)
    assert p['combined_gradient_and_forward_payload_upper_bound_bytes'] < p['output_byte_cap']
    p['assets_sha256'] = {f.relative_to(dest).as_posix(): sha(f) for f in dest.rglob('*') if f.is_file()}
    evidence = [ANALYSIS / 'import.json', ANALYSIS / 'independent_failure_audit.json', probe_path]
    p['evidence_sha256'] = {**original['evidence_sha256'], **{f.relative_to(ROOT).as_posix(): sha(f) for f in evidence}}
    write(dest / 'protocol.json', p)
    pin = sha(dest / 'protocol.json')
    with tarfile.open(archive, 'x:gz', compresslevel=1) as tar:
        for file in sorted(dest.rglob('*')):
            if file.is_file():
                tar.add(file, arcname=NAME + '/' + file.relative_to(dest).as_posix(), recursive=False)
    digest = sha(archive)
    with Path(str(archive) + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + archive.name + '\n')
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), preparer_sha256=sha(__file__),
        name=NAME, stem=STEM, protocol_sha256=pin, archive_sha256=digest, archive_bytes=archive.stat().st_size,
        packet_assets=len(p['assets_sha256']), parent_required_assets=len(p['parent_required_sha256']),
        prior_failed_protocol_sha256=PIN, prior_failure_audit_sha256=sha(ANALYSIS / 'independent_failure_audit.json'),
        layout_reproduction_sha256=sha(probe_path), single_input_layout_repair=True,
        objective_weights_and_checkpoints_cases_gates_and_limits_unchanged=True,
        bit_exact_control_replay_not_relaxed=True, added_raw_control_failure_evidence=True,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
        independent_transfer_and_input_audit_pending=True, actual_L4_repair_confirmation_pending=True,
        no_agent_VM_connection_or_launch=True, model_qualified=False, goal_complete=False))
    print(dict(complete=True, protocol_sha256=pin, archive_sha256=digest,
               archive_bytes=archive.stat().st_size, manual_VM_diagnostic_pending=True))


if __name__ == '__main__':
    main()
