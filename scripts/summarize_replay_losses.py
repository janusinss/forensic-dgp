"""Validate and group the frozen-state audit; no model execution or training."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = Path('outputs/replay_loss_audit/results.json')
    audit = json.loads(source.read_text())
    protocol_path = Path('outputs/coverage_protocol_v1/protocol.json')
    protocol = json.loads(protocol_path.read_text())
    run = json.loads(Path('outputs/downloaded_projection/outputs/projection_training_vm/run.json').read_text())
    assert digest(protocol_path) == run['protocol_sha256']
    manifest = Path('dataset/detector_training_extension_v2/manifest.json')
    assert digest(manifest) == protocol['input_hashes'][manifest.as_posix()]
    rows = [r for r in json.loads(manifest.read_text())['records'] if r['split'] == 'train']
    assert len(rows) == 73
    exposures = Counter(i for epoch in protocol['schedules']['extended']['batches'] for batch in epoch for i in batch)
    assert set(audit['arms']) == {'parent', 'ordinary', 'projected'}, 'Audit is incomplete'
    assert audit['optimizer_updates'] == 0
    report = {'source_sha256': digest(source), 'protocol_sha256': digest(protocol_path),
              'manifest_sha256': digest(manifest), 'optimizer_updates': 0, 'arms': {}}
    for arm, result in audit['arms'].items():
        records = result['records']
        assert len(records) == len(exposures) == 711
        assert {r['index'] for r in records} == set(exposures)
        groups = {}
        for r in records:
            i = r['index']
            assert r['exposures'] == exposures[i]
            domain = 'real' if i < 73 else 'replay'
            assert r['domain'] == domain
            covered = rows[i]['kind'] == 'covered' if i < 73 else (i - 73) % 5 != 0
            for group in (domain, domain + ('/covered' if covered else '/clear')):
                groups.setdefault(group, []).append(r)
        summary = {}
        for group, items in groups.items():
            n = sum(r['exposures'] for r in items)
            keys = ['bce', 'dice', 'weighted_visible', 'supervised']
            if group.startswith('replay'):
                keys.append('teacher_kl')
            summary[group] = {'cases': len(items), 'exposures': n,
                              **{key: sum(r[key] * r['exposures'] for r in items) / n for key in keys}}
        for domain in ('real', 'replay'):
            assert summary[domain]['exposures'] == 840
            for key, value in result['weighted'][domain].items():
                assert math.isclose(value, summary[domain][key], abs_tol=1e-10)
        objective = .5 * summary['real']['supervised'] + .5 * summary['replay']['supervised'] + summary['replay']['teacher_kl']
        assert math.isclose(objective, result['mixed_objective'], abs_tol=1e-10)
        report['arms'][arm] = {'groups': summary, 'mixed_objective': objective}
    output = source.with_name('summary.json')
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
