"""Read-only audit of returned retention archive; does not load checkpoints."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile


def check(condition, message):
    if not condition:
        raise ValueError(message)


def audit_steps(run, steps, result, schedule):
    factors = [1., .5, .25, .125]
    check(run['factors'] == factors and run['tolerance'] == 1e-6, 'Trial policy changed')
    check(run['max_scheduled_batches'] == 21 and run['stop_rejection_streak'] == 3, 'Budget changed')
    check(run['preflight'] is False and result['complete'] is True, 'Not a completed pilot')
    ceilings = run['ceilings']
    check(set(ceilings) == {'covered', 'clear'}, 'Missing replay class')
    check(all(math.isfinite(v) and v >= 0 for v in ceilings.values()), 'Invalid ceilings')
    check(1 <= len(steps) <= 21, 'Invalid step count')
    accepted = streak = trials = 0
    warnings = []
    for number, step in enumerate(steps, 1):
        check(streak < 3, 'Continued after required stop')
        check(step['step'] == number and step['indices'] == schedule[number-1], 'Schedule changed')
        attempts = step['attempts']
        check(1 <= len(attempts) <= 4, 'Invalid trial count')
        check([a['factor'] for a in attempts] == factors[:len(attempts)], 'Trial factors changed')
        for i, attempt in enumerate(attempts):
            losses = attempt['losses']
            check(set(losses) == set(ceilings), 'Missing trial loss')
            within = all(math.isfinite(v) and 0 <= v <= ceilings[k]+1e-6 for k, v in losses.items())
            check(type(attempt['accepted']) is bool, 'Invalid acceptance flag')
            if attempt['accepted']:
                check(within and i == len(attempts)-1, 'Invalid accepted trial')
            elif within:
                warnings.append(f'Step {number} trial {i+1}: losses pass but rejected; parameter finiteness is not logged')
        check(step['accepted'] == attempts[-1]['accepted'], 'Step acceptance disagrees')
        if step['accepted']:
            check(step['factor'] == attempts[-1]['factor'], 'Selected factor differs')
            accepted += 1
            streak = 0
        else:
            check(len(attempts) == 4 and step['factor'] is None, 'Rejection before exhausting trials')
            streak += 1
        trials += len(attempts)
        check(step['accepted_updates'] == accepted and step['rejection_streak'] == streak, 'Counter mismatch')
    reason = 'three_consecutive_rejected_batches' if streak == 3 else 'scheduled_budget_complete'
    check(streak == 3 or len(steps) == 21, 'Premature completion')
    check(result['stop_reason'] == reason, 'Stop reason differs')
    check(result['scheduled_batches'] == len(steps) and result['accepted_updates'] == accepted and result['trials'] == trials, 'Final counters differ')
    check(set(result['final_replay']) == set(ceilings), 'Missing final loss')
    check(all(math.isfinite(v) and 0 <= v <= ceilings[k]+1e-6 for k, v in result['final_replay'].items()), 'Final constraint violated')
    return {'scheduled_batches': len(steps), 'accepted_updates': accepted, 'trials': trials,
            'stop_reason': reason, 'warnings': warnings,
            'scope': 'Log consistency only; independent replay inference, mask recount and state audit still required'}


def audit_archive(path, bundle, protocol_path):
    protocol_bytes = protocol_path.read_bytes()
    protocol = json.loads(protocol_bytes)
    prefix = 'outputs/retention_training_vm/'
    with tarfile.open(path) as returned, tarfile.open(bundle) as sent:
        names = returned.getnames()
        check(len(names) == len(set(names)), 'Duplicate archive paths')
        check(all(m.isfile() or m.isdir() for m in returned.getmembers()), 'Unexpected archive member type')
        def read(name):
            member = returned.getmember(name)
            check(member.isfile(), 'Expected regular file')
            with returned.extractfile(member) as stream:
                return stream.read()
        run = json.loads(read(prefix+'run.json'))
        result = json.loads(read(prefix+'results.json'))
        complete = json.loads(read(prefix+'complete.json'))
        check(run['protocol_sha256'] == hashlib.sha256(protocol_bytes).hexdigest(), 'Protocol hash differs')
        for name, key in [('scripts/train_retention_vm.py', 'script_sha256'),
                          ('coverage_retention.py', 'helper_sha256'),
                          ('COVERAGE_RETENTION_PROTOCOL.md', 'specification_sha256')]:
            content = read(name)
            with sent.extractfile(name) as stream:
                check(content == stream.read(), f'Executed file differs from bundle: {name}')
            check(hashlib.sha256(content).hexdigest() == run[key], f'Source hash differs: {name}')
        check(complete['complete'] is True, 'Missing terminal marker')
        with returned.extractfile(prefix+'final.pth') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        check(digest == complete['final_sha256'], 'Final checkpoint hash differs')
        check((prefix+'best_detector.pth' in names) == result['selected'], 'Selected artifact disagrees')
        steps = [json.loads(line) for line in read(prefix+'steps.jsonl').splitlines() if line.strip()]
        report = audit_steps(run, steps, result, protocol['schedules']['extended']['batches'][0])
        report['final_checkpoint_sha256'] = digest
        return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('outputs/retention-results.tar.gz'))
    parser.add_argument('--bundle', type=Path, default=Path('outputs/retention-code.tar.gz'))
    parser.add_argument('--protocol', type=Path, default=Path('outputs/coverage_protocol_v1/protocol.json'))
    args = parser.parse_args()
    print(json.dumps(audit_archive(args.archive, args.bundle, args.protocol), indent=2))
