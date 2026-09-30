"""Check reviewed eyewear proposals against frozen references; no training."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_real_source_pool import signature


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', default='outputs/eyewear_annotation_proposals_v2')
    out = Path(parser.parse_args().directory)
    proposal = json.loads((out / 'manifest.json').read_text())
    v3 = Path('dataset/detector_glare_review_v3/manifest.json')
    split = Path('outputs/downloaded_phase4/outputs/phase4_with_progress/split.json')
    benchmark = Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm/manifest.json')
    refs = set()
    for r in json.loads(v3.read_text())['records']:
        refs.add(r['source'].replace('\\', '/'))
        refs.add((v3.parent / r['image']).as_posix())
    refs.update(p.replace('\\', '/') for p in json.loads(split.read_text())['validation'])
    refs.update(r['source'].replace('\\', '/') for r in json.loads(benchmark.read_text())['cases'])
    proposals = []
    for index, r in enumerate(proposal['records']):
        source = r.get('path', r.get('source'))
        digest = r.get('sha256', r.get('source_sha256'))
        assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == digest
        assert hashlib.sha256(Path(r['image']).read_bytes()).hexdigest() == r['image_sha256']
        assert hashlib.sha256(Path(r['mask']).read_bytes()).hexdigest() == r['mask_sha256']
        proposals.append({'index': r.get('index', index), 'variants': [signature(Path(source)), signature(Path(r['image']))], 'nearest': []})
    for i, path in enumerate(sorted(refs)):
        sig = signature(Path(path))
        for p in proposals:
            distance = min((s['phash'] ^ sig['phash']).bit_count() for s in p['variants'])
            exact = any(s['sha256'] == sig['sha256'] or s['decoded_sha256'] == sig['decoded_sha256'] for s in p['variants'])
            p['nearest'].append({'path': path, 'distance': distance, 'exact': exact})
        if i % 1000 == 0:
            print(f'Reference signatures {i}/{len(refs)}', flush=True)
    rows = []
    for p in proposals:
        nearest = sorted(p['nearest'], key=lambda r: r['distance'])
        rows.append({'index': p['index'], 'flags': [r for r in nearest if r['exact'] or r['distance'] <= 6],
                     'nearest': nearest[:3]})
    result = {'reference_count': len(refs), 'records': rows, 'training_enabled': False,
              'reference_hashes': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (v3, split, benchmark)},
              'limitation': 'Whole-image exact and DCT screening cannot establish identity separation or exclude all alternate crops.'}
    (out / 'overlap_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print([(r['index'], len(r['flags']), r['nearest'][0]) for r in rows])


if __name__ == '__main__':
    main()
