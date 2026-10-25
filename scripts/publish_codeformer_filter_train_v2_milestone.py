"""Publish reviewed data-preparation milestone; preserve full prior document bytes."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREP = ROOT / 'outputs/codeformer_filter_train_v2_preparation'
FILES = ('PROJECT_HANDOFF.md', 'SYSTEM_WORKFLOW_AND_GOAL.md',
         'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md')
HEADER = '''# Broader filter TRAIN preparation complete; finite pilot next — 18 October 2026

[Actual512-input preparation and limits](CCTV_CODEFORMER_FILTER_TRAIN_V2_PREPARATION.md).
Fresh source refresh covered391 historical TRAIN/53 DEV photographs. A deterministic
64-TRAIN cohort produced512 paired filter inputs:1,256 files/105,926,107 bytes in46.53s,
zero neural or optimizer calls. Independent saved-pixel reconstruction has zero
channel mismatches; all128 clean derivatives, footprints,32 sheets/512 exact cells
and namespace hashes pass.18 producer and15 independent verifier fixtures pass.

Root personally reviewed all32 input-only sheets, nonclear before clear. Initial
labels390 readable/93 borderline/1 insufficient/28 out of scope remain retained.
Five multi-face sources exclude all40 variants. A second assistant observer agrees
on46/64 global-mosaic labels;18 disagreements remain visible. Six disputed readable
global cases and both pose-disputed sources are held without replacement/relabeling.
The conservative prospective list is370 inputs from57 sources:57 clear/313 nonclear,
still covering all9 recorded terms/source-category strata. No optimization is admitted.
These are source photographs, not proven distinct identities or a calibrated threshold.

Prepare the changed fusion-only R2 route:72 fusion tensors,443 frozen; preserve useful
brightening and unchanged original-based RAW/PNG quality gates. Current draft ceiling
is157 new Adam updates/628 microbatches, covering313 filter cases plus1 declared padding
duplicate with157 clear and157 dark slots. The earlier164 draft remains retained.
Dark joins, sampler, executable objective/witnesses, resource/call/disk ledger and verified
manual L4/tmux transfer/return packet still need closure. No upload or launch is ready.

Keep all391/53 memberships, full406 historical DEV/source-separated unpaired native
CCTV, all72 mixed TRAIN/DEV filter comparisons and24 refusals intact. Unknown wild
images remain unpaired stress tests. Reserved FINAL, user acceptance, useful output
review and accepted app/Auto/override/download flow remain required. No new model,
app/default switch or completion work occurred; the overall goal remains active.
Storage offload was already complete, was not repeated and reclaimed0 new bytes here.

Earlier status below remains retained.

---

'''.encode('utf-8')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def pin(path):
    data = path.read_bytes()
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': len(data), 'sha256': sha(data)}


def run(review_sha):
    report = ROOT / 'CCTV_CODEFORMER_FILTER_TRAIN_V2_PREPARATION.md'
    review_path = PREP / 'report_independent_review_v1.json'
    assert sha(review_path.read_bytes()) == review_sha
    review = json.loads(review_path.read_bytes())
    assert review.get('complete') is True and review.get('pass') is True
    assert review['report'] == pin(report), 'Current report differs from reviewed bytes'
    held = json.loads((PREP / 'candidate_holds_v1.json').read_bytes())
    assert (held['candidate_nonclear'], held['candidate_clear'], held['planned_updates_if_later_admitted']) == (313, 57, 157)
    assert not (ROOT / 'outputs/codeformer_filter_train_v2_data/failure.json').exists()
    previous = {name: (ROOT / name).read_bytes() for name in FILES}
    dest = PREP / 'milestone_publication_v1'
    assert not dest.exists()
    dest.mkdir()
    (dest / 'header_utf8.md').write_bytes(HEADER)
    for name, data in previous.items():
        with (dest / (name + '.before')).open('xb') as stream:
            stream.write(data)
        assert (dest / (name + '.before')).read_bytes() == data
    out = {'complete': False, 'UTC': datetime.now(timezone.utc).isoformat(),
           'report': pin(report), 'independent_report_review': pin(review_path),
           'header': pin(dest / 'header_utf8.md'), 'documents': []}
    for name, data in previous.items():
        path = ROOT / name
        assert path.read_bytes() == data
        path.write_bytes(HEADER + data)
        actual = path.read_bytes()
        assert actual == HEADER + data and actual[len(HEADER):] == data
        out['documents'].append({'name': name, 'before': pin(dest / (name + '.before')),
                                 'after': pin(path), 'prior_suffix_offset': len(HEADER),
                                 'exact_prior_bytes_retained': True})
    out.update(complete=True, model_qualified=False, actual_training_launched=False, goal_complete=False,
               new_deleted_or_reclaimed_bytes=0)
    with (dest / 'publication_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(out, stream, indent=2); stream.write('\n')
    assert out['report'] == pin(report) and out['independent_report_review'] == pin(review_path)
    for record in out['documents']:
        assert record['after'] == pin(ROOT / record['name'])
        assert (ROOT / record['name']).read_bytes()[record['prior_suffix_offset']:] == (ROOT / record['before']['path']).read_bytes()
    return {'complete': True, 'publication': pin(dest / 'publication_v1.json'),
            'documents': 4, 'full_prior_bytes_verified': True, 'goal_complete': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-sha', required=True)
    print(json.dumps(run(parser.parse_args().review_sha)))
