"""Retain the failed mathematical analysis and make a serialization-only R1."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAILED = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1'
SOURCE = ROOT / 'scripts/analyze_cctv_dgp_group_guard_grad_v34_v1.py'
REVISION = ROOT / 'scripts/analyze_cctv_dgp_group_guard_grad_v34_v1_r1.py'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    import numpy as np
    assert FAILED.exists() and not REVISION.exists()
    assert (FAILED / 'analysis.json').stat().st_size == 0
    text = SOURCE.read_text(encoding='utf-8')
    old = "'positive_subgroup_changes': sum(value > 0 for row,value in zip(cohort_labels,predicted)\n                    if row['group'] != 'all' and row['metric'] == metric)"
    new = "'positive_subgroup_changes': int(sum(value > 0 for row,value in zip(cohort_labels,predicted)\n                    if row['group'] != 'all' and row['metric'] == metric))"
    assert text.count(old) == 1
    revised = text.replace(old, new)
    assert revised.count("OUT = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1'") == 1
    revised = revised.replace("OUT = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1'",
        "OUT = ROOT / 'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1'")
    before = "    with Path(path).open('x', encoding='utf-8', newline='\\n') as stream:\n        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\\n')"
    after = "    encoded = json.dumps(value, indent=2, allow_nan=False) + '\\n'\n    with Path(path).open('x', encoding='utf-8', newline='\\n') as stream:\n        stream.write(encoded)"
    assert revised.count(before) == 1
    revised = revised.replace(before, after)
    values = np.array([-1., 2., 3.])
    count = sum(value > 0 for value in values)
    assert isinstance(count, np.integer)
    try:
        json.dumps({'count': count})
    except TypeError:
        pass
    else:
        raise AssertionError('Original serialization failure must be reproduced')
    assert json.loads(json.dumps({'count': int(count)}))['count'] == 2
    saved = FAILED / 'failed_analysis_source.py'
    with saved.open('xb') as stream:
        stream.write(SOURCE.read_bytes())
    with REVISION.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(revised)
    record = {'complete': True, 'failure': 'TypeError: Object of type int64 is not JSON serializable',
        'failure_output_tool_chunk': 'ab9097', 'failed_source_sha256': sha(saved),
        'unchanged_original_source_sha256': sha(SOURCE), 'revision_sha256': sha(REVISION),
        'retained_files_sha256': {q.name: sha(q) for q in sorted(FAILED.iterdir()) if q.is_file()},
        'diagnosed_invalid_assumption': 'Python sum over NumPy comparisons always yields a builtin integer',
        'changes': ['Explicit builtin-int serialization of subgroup count', 'Distinct R1 output directory',
            'Encode JSON before creating the destination'],
        'serialization_regression_passed': True, 'geometry_or_quality_gate_changes': False,
        'VM_calls': 0, 'neural_or_gradient_calls': 0, 'optimizer_updates': 0}
    with (FAILED / 'failure_preservation.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2) + '\n')
    print(json.dumps({'complete': True, 'serialization_fix_only': True,
        'revision': REVISION.relative_to(ROOT).as_posix()}))


if __name__ == '__main__':
    main()
