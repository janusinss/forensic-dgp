"""Correct only the frozen audit's exclusion-list readback; keep that attempt intact."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_mat_mirror_comparison_v1_milestone'
ORIGINAL = ROOT / 'scripts/verify_completion_mat_mirror_comparison_v1_milestone.py'
PIN = 'e094f60633719f19a16f79264c738f3cdd8348635cc7ebac5b9196aa67b4fedb'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert sha(ORIGINAL) == PIN
    text = ORIGINAL.read_text(encoding='utf-8')
    before = "len(protocol['cases']) == 32 and protocol['excluded_before_generation'] == 4"
    after = "len(protocol['cases']) == 32 and len(protocol['excluded_before_generation']) == 4"
    assert text.count(before) == 1
    receipt = {'complete': True, 'original_failed_checker_sha256': PIN,
               'corrected_checker_sha256': sha(Path(__file__)),
               'original_failure': {'exception': 'AssertionError', 'line': 48,
                                    'cause': 'Four exclusion records are stored as a list, not an integer count.'},
               'source_replacement': {'before': before, 'after': after},
               'only_audit_schema_readback_changed': True,
               'frozen_protocol_outputs_original_checker_and_milestone_unchanged': True,
               'model_or_gradient_calls': 0, 'optimizer_updates': 0, 'goal_complete': False}
    with (OUT / 'closure_audit_r1_correction.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    namespace = {'__name__': 'pinned_completion_closure_audit_r1', '__file__': str(Path(__file__).resolve())}
    exec(compile(text.replace(before, after), str(ORIGINAL), 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
