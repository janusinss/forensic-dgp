"""Allow the publisher's empty package marker; preserve the first partial review."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'scripts/acquire_completion_mat_mirror_review_v1.py'
text = OLD.read_text()
before = 'assert response.status == 200 and 0 < len(data) <= cap'
after = "assert response.status == 200 and len(data) <= cap\n            assert data or name.endswith('/__init__.py'), 'Only an empty Python package marker is permitted'"
assert text.count(before) == 1
failed = ROOT / 'outputs/completion_mat_mirror_review_v1'
assert (failed / 'acquisition_failure.json').exists()
with (failed / 'acquirer_at_empty_marker_failure.py').open('xb') as stream: stream.write(OLD.read_bytes())
namespace = {'__file__':__file__,'__name__':'preserved_static_acquisition_with_empty_marker_fix'}
exec(compile(text.replace(before, after), '<static-acquirer-empty-marker-fix>', 'exec'), namespace)
namespace['OUT'] = ROOT / 'outputs/completion_mat_mirror_review_v1_r1'
namespace['main']()
