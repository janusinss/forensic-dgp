"""Independent V26 gradient/count/timing receipt arithmetic; no neural/VM calls."""
import math
from import_cctv_dgp_batchmatched_identity_v26 import PIN, read, require


def check_execution(returned, preflights, snapshots, result, failure, initial_tensors, head_hash):
    """Check the new receipt fields; explicitly distinguish absent partial evidence."""
    path = returned / 'outputs/execution_receipt.json'
    supervisor_path = returned / 'supervisor_receipt.json'
    execution = read(path) if path.exists() else None
    supervisor = read(supervisor_path) if supervisor_path.exists() else None
    timing_path = returned / 'outputs/timing_update20.json'
    timing = read(timing_path) if timing_path.exists() else None
    gradient_path = returned / 'outputs/one_batch_gradient_preflight.json'
    gradient = read(gradient_path) if gradient_path.exists() else None
    require(supervisor is not None and supervisor.get('complete') is True and supervisor.get('protocol_sha256') == PIN,
            'Supervisor receipt missing or changed')
    require(supervisor.get('cap_seconds') == 2100 and supervisor.get('kill_grace_seconds') == 30 and
            type(supervisor.get('seconds')) in (int, float) and math.isfinite(supervisor['seconds']) and supervisor['seconds'] > 0 and
            supervisor.get('within_external_bound') is (supervisor['seconds'] <= 2130) and
            supervisor.get('training_acceptance_not_implied') is True, 'Supervisor timing/scope differs')
    exit_path = returned / 'trainer_exit_code.txt'
    require(exit_path.exists() and int(exit_path.read_text().strip()) == supervisor.get('trainer_exit_code'), 'Trainer exit receipt differs')
    require((result is None or supervisor['trainer_exit_code'] == 0) and
            (failure is None or supervisor['trainer_exit_code'] != 0), 'Training result/failure and exit disagree')
    require(supervisor['trainer_exit_code'] != 0 or result is not None, 'Successful supervised exit lacks completed800 result')
    if result is not None:
        require(supervisor['within_external_bound'] is True, 'Complete result exceeds external supervisor bound')
    if execution is None:
        require(result is None, 'Complete training lacks execution receipt')
        return {'execution_present': False, 'supervisor_seconds': supervisor['seconds'],
                'limit': 'Hard or early termination lacks execution/step/gradient evidence; no fitting success claim'}
    require(execution.get('complete') is True and execution.get('protocol_sha256') == PIN and
            execution.get('terminal') == ('completed800' if result is not None else 'failed_partial') and
            execution.get('app_promotion') is False and execution.get('goal_complete') is False, 'Execution scope differs')
    progress = failure if failure is not None else result
    require(progress is not None and type(execution.get('updates')) is int and execution['updates'] == progress['updates'] and
            type(execution.get('backwards')) is int and execution['backwards'] == progress['backwards'], 'Execution progress differs')
    require(execution.get('fit_cap_seconds') == 1500 and execution.get('worker_cap_seconds') == 1800 and
            execution.get('VRAM_cap_bytes') == 20 * 1024**3 and type(execution.get('peak_allocated_VRAM_bytes')) is int and
            0 < execution['peak_allocated_VRAM_bytes'] <= execution['VRAM_cap_bytes'], 'Allocated VRAM cap/receipt differs')
    worker, fit = execution.get('worker_seconds'), execution.get('fit_seconds')
    require(type(worker) in (int, float) and math.isfinite(worker) and
            progress['seconds'] <= worker <= supervisor['seconds'] and worker <= 1800,
            'Worker/supervisor duration differs')
    require(execution.get('recognizer_state') == execution.get('initial_recognizer_state') and
            all(execution['recognizer_state'] == r['recognizer_state'] for r in preflights), 'Frozen recognizer state changed')
    require(execution.get('head_state') == head_hash, 'Execution and retained head state differ')
    steps = execution.get('step_times_seconds')
    require(isinstance(steps, list) and len(steps) == execution['updates'] and
            all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in steps), 'Update step samples differ')
    if execution['updates'] > 0:
        require(execution.get('optimizer_constructed') is True and execution.get('optimizer') == 'AdamW new head only' and
                type(fit) in (int, float) and math.isfinite(fit) and sum(steps) <= fit <= 1500,
                'Fitting duration/optimizer receipt differs')
        require(execution['backwards'] == execution['updates'] + 1, 'One pre-optimizer gradient plus update backwards required')
        require(gradient is not None and gradient.get('complete') is True and gradient.get('backwards') == 1 and
                gradient.get('optimizer_constructed') is False and gradient.get('recognizer_has_no_gradients') is True,
                'Pre-optimizer gradient receipt differs')
        per_tensor = gradient.get('per_tensor_gradient_sum_squares')
        require(isinstance(per_tensor, dict) and set(per_tensor) == set(initial_tensors) - {'kernel', 'reflect_indices'} and
                all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in per_tensor.values()) and
                per_tensor['direct.weight'] > 0 and sum(per_tensor.values()) > 0 and
                gradient.get('direct_gradient_sum_squares') == per_tensor['direct.weight'], 'Finite/nonzero direct gradient differs')
        require(gradient.get('neural_forward_counts') == {'detail_head': 111, 'DGP': 50, 'DGP_CPU': 4, 'fixed_recognizer': 171},
                'One-batch neural counts differ')
    projection_path = returned / 'outputs/feature_path_gradient_update2.json'
    projection = read(projection_path) if projection_path.exists() else None
    if execution['updates'] >= 2:
        require(projection is not None and projection.get('complete') is True and projection.get('update_before_optimizer') == 2 and
                projection.get('frozen_feature_tensors_have_no_gradients') is True and
                all(projection.get('DGP_state_unchanged') == r['DGP_state_unchanged'] for r in preflights), 'Actual update2 frozen-feature gradient proof missing or changed')
        values = projection.get('per_projection_gradient_sum_squares')
        require(isinstance(values, dict) and set(values) == {'0','1','2','3','4'} and
                all(type(v) in (int,float) and math.isfinite(v) and v > 0 for v in values.values()), 'All five feature-projection gradients must be positive finite')
    if execution['updates'] >= 20:
        require(timing is not None and timing.get('updates') == 20 and timing.get('remaining_updates') == 780 and
                timing.get('safety_factor') == 1.25 and timing.get('overhead_seconds') == 120 and timing.get('cap_seconds') == 1500,
                'Projection policy differs')
        samples = timing.get('steady_sample_seconds')
        require(samples == steps[1:20] and len(samples) == 19 and sum(steps[:20]) <= timing['seconds'] <= fit,
                'Steady timing samples differ')
        projected = timing['seconds'] + 780 * (sum(samples) / 19) * 1.25 + 120
        require(abs(projected - timing['projected_seconds']) <= 1e-9, 'Timing projection sample arithmetic differs')
        require(execution['updates'] == 20 or projected <= 1500, 'Continued after failed timing projection')
    terminal_complete = result is not None or (failure is not None and failure['cause'] == 'No one-percent early structural gain; retain stop')
    if terminal_complete:
        expected_counts = {'detail_head': 60 + 50 * len(snapshots) + execution['backwards'],
                           'DGP': 50, 'DGP_CPU': 4, 'fixed_recognizer': 120 + 50 * len(snapshots) + execution['backwards']}
        require(execution.get('neural_forward_counts') == expected_counts, 'Terminal neural counts differ')
    return {'execution_present': True, 'updates': execution['updates'], 'backwards': execution['backwards'],
            'worker_seconds': worker, 'fit_seconds': fit, 'supervisor_seconds': supervisor['seconds'],
            'peak_allocated_VRAM_bytes': execution['peak_allocated_VRAM_bytes'], 'step_samples': len(steps),
            'update20_projection_arithmetic_verified': timing is not None, 'one_batch_gradient_receipt_verified': gradient is not None,
            'direct_gradient_sum_squares': gradient['direct_gradient_sum_squares'] if gradient else None,
            'neural_counts_verified': terminal_complete, 'neural_forward_counts': execution['neural_forward_counts'],
            'five_projection_update2_gradients_verified': projection is not None, 'projection_gradient_sum_squares': projection['per_projection_gradient_sum_squares'] if projection else None,
            'limits': 'Receipts and source-bound counters, not a new GPU measurement. Peak is torch allocated memory, not total device use.'}
