"""Freeze shared replay sources and schedules, without any model optimization."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coverage_protocol import CoverageBatches


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def main():
    paths = {
        'control': 'dataset/detector_glare_review_v3/manifest.json',
        'extended': 'dataset/detector_training_extension_v2/manifest.json',
        'pool': 'outputs/expanded_feature_data_v1/manifest.json',
        'split': 'outputs/downloaded_phase4/outputs/phase4_with_progress/split.json',
        'benchmark': 'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm/manifest.json',
    }
    data = {k: json.loads(Path(p).read_text()) for k,p in paths.items()}
    assert sha(paths['extended']) == '1f9787ba7bf184f641b298863ae90250b90706e74b6163517a22efe293fcbb2c'
    pool = data['pool']
    assert sha(paths['split']) == pool['split_sha256']
    assert sha(paths['control']) == pool['v3_manifest_sha256']
    assert sha(paths['benchmark']) == pool['benchmark_manifest_sha256']
    excluded = set(pool['excluded_sha256'])
    for arm in ('control', 'extended'):
        excluded.update(r[k] for r in data[arm]['records'] for k in ('source_sha256','image_sha256'))
    train = {p.replace('\\','/') for p in data['split']['train']}
    held = {p.replace('\\','/') for p in data['split']['validation']}
    candidates, excluded_pool = [], []
    for r in pool['sources']:
        assert sha(r['path']) == r['sha256']
        assert r['path'] in train and r['path'] not in held
        (excluded_pool if r['sha256'] in excluded else candidates).append(r)
    selected = []
    for source in ('dataset/asian_faces','dataset/thumbnails128x128'):
        group = sorted([r for r in candidates if r['source']==source], key=lambda r:r['sha256'])
        assert len(group) >= 100
        # Same fixed selection independent of the arm, not a validation-driven choice.
        indices = np.random.default_rng(42).permutation(len(group))[:100]
        selected.extend({k:group[i][k] for k in ('path','sha256','source')} for i in indices)
    assert len({r['sha256'] for r in selected}) == 200
    schedules, exposure = {}, {}
    for arm in ('control','extended'):
        rows = [r for r in data[arm]['records'] if r['split']=='train']
        groups = [[i for i,r in enumerate(rows) if r['kind']==kind] for kind in ('covered','uncovered')]
        groups.extend([[len(rows)+i for i in range(2000) if (i%5!=0)==covered] for covered in (True,False)])
        sampler = CoverageBatches(groups,8,21,42)
        batches, synthetic = [], []
        seen = set()
        for epoch in range(10):
            sampler.epoch = epoch
            epoch_batches = list(sampler)
            batches.append(epoch_batches)
            synthetic.append([[(pos,i-len(rows)) for pos,i in enumerate(batch) if i>=len(rows)] for batch in epoch_batches])
            seen.update(i for batch in epoch_batches for i in batch if i<len(rows))
        assert len(seen)==len(rows)
        schedules[arm] = {'batches': batches, 'synthetic_schedule_sha256': digest(synthetic), 'schedule_sha256': digest(batches)}
        exposure[arm] = {'real_images_seen':len(seen),'updates':210,'real_slots':840,'synthetic_slots':840}
    assert schedules['control']['synthetic_schedule_sha256'] == schedules['extended']['synthetic_schedule_sha256']
    out = Path('outputs/coverage_protocol_v1')
    out.mkdir(exist_ok=False)
    result = {'input_hashes':{p:sha(p) for p in paths.values()}, 'sources':selected,
              'excluded_pool_sources':[r['path'] for r in excluded_pool], 'exposure':exposure,
              'schedules': schedules, 'seed':42, 'batch_size':8, 'epochs':10, 'steps':21,
              'training_performed':False,'vm_runner_ready':False,
              'limitation':'Source cleanliness provisional; exclusion is hash/source based, not identity proof. Recipe and archived synthetic tensors still pending.'}
    (out/'protocol.json').write_text(json.dumps(result,indent=2)+'\n')
    print({'excluded_pool_sources':result['excluded_pool_sources'],'exposure':exposure,
           'synthetic_order_matches':True,'sha256':sha(out/'protocol.json')})


if __name__=='__main__':
    main()
