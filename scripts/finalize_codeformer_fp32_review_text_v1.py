"""Preserve first review drafts, then make the reviewed publication corrections."""
import hashlib
import json
import re
from pathlib import Path


def sha(data):
    return hashlib.sha256(data).hexdigest()


root = Path(__file__).resolve().parents[1]
out = root / 'outputs/codeformer_fp32_comparison_v1_return_review/publication_drafts_v1'
out.mkdir(exist_ok=False)
rows = []
for name, expected in [
    ('CCTV_CODEFORMER_FP32_COMPARISON_V1_RETURN_REVIEW.md', '9395a97837ee55ec5b83802be2cac923df22a494a1cdad7c5f1cfe661bab286f'),
    ('CCTV_CODEFORMER_POST_FP32_DESIGN_REVIEW_V1.md', '8c4178c4553c52acc996d716bff041ea459532a99f6e6718c3c957fa415c48f8')]:
    path = root / name
    before = path.read_bytes(); assert sha(before) == expected
    (out / name).write_bytes(before)
    text = before.decode('utf-8')
    stems = {'All', 'Only', 'Snapshot', 'The', 'At', 'actual', 'all', 'and', 'at', 'completed',
             'delivered', 'exact', 'exactly', 'first', 'historical', 'mean', 'most', 'not',
             'on', 'or', 'predeclared', 'recomputed', 'rises', 'run', 'same', 'saved',
             'step', 'target', 'the', 'took', 'update', 'was', 'with', 'worst', 'within', 'identical', 'native', 'Native'}
    text = re.sub(r'\b([A-Za-z]+)(?=[0-9])', lambda m: m[1] + (' ' if m[1] in stems else ''), text)
    text = text.replace('V2u50', 'V2 u50').replace('FP32u50', 'FP32 u50').replace('256-to192', '256-to-192')
    text = text.replace('|Lowlight|15.228 /', '|Lowlight|15.229 /').replace('|29.218 /0.900000|', '|29.217 /0.900000|')
    text = text.replace('No gate may consume clean targets, simulator exposure labels, source/reference\nIDs, DEV/FINAL labels or user annotations at inference.',
        'During both training forwards and inference, the gate must use only the observed\ninput. It must never consume clean targets, simulator exposure labels, source/\nreference IDs, DEV/FINAL labels or user annotations. Supervised losses may use\nthe declared training targets; the gate inputs may not.')
    text = text.replace('All 6,501 regular members/3,120,598,370 payload bytes were hash-verified before\nexclusive import, including all 1,925 original packet bindings and 1,399 retained\nparent dependencies.',
        'All 6,501 regular members/3,120,598,370 payload bytes were hash-verified before\nexclusive import. The audit separately verified all 1,925 original packet\nbindings, including the 1,399 retained parent dependencies.')
    after = text.encode('utf-8')
    temporary = path.with_name(path.name + '.publication-stage-v1')
    with temporary.open('xb') as stream:
        stream.write(after)
    temporary.replace(path)
    assert path.read_bytes() == after and (out / name).read_bytes() == before
    rows.append({'path': str(path), 'before_sha256': expected, 'after_sha256': sha(after), 'draft_retained': str(out / name)})
with (out / 'receipt.json').open('x', encoding='utf-8') as stream:
    json.dump({'complete': True, 'rows': rows, 'changes': 'Spacing, two reviewed PSNR table roundings, transport wording, gate input isolation in training and inference; no metric or model changes'}, stream, indent=2)
print(json.dumps(rows, indent=2))
