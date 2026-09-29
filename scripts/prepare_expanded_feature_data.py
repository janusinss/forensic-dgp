"""Build and verify the versioned expanded detector dataset locally; no fitting."""
import hashlib,json,sys
from pathlib import Path
from collections import Counter
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from expanded_feature_data import ExpandedCoveringDataset,expanded_balanced_schedule,FORMAT,SOURCES
from completion_data import CompletionDataset


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    if Path.cwd().resolve()!=root:raise RuntimeError('Run from the repository root')
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    audit=Path('outputs/training_diversity_audit');trial=audit/'landmark_trial_v1'
    out=Path('outputs/expanded_feature_data_v1');out.mkdir(exist_ok=False)
    candidate=json.loads((audit/'candidate_sources.json').read_text())
    landmarks=json.loads((trial/'results.json').read_text())
    clipped=json.loads((trial/'clipping_v2/results.json').read_text())
    review=json.loads((trial/'acceptance_review/visual_review.json').read_text())
    screen=json.loads((audit/'duplicate_screen.json').read_text())
    roles=json.loads((audit/'source_roles_v2.json').read_text())['records']
    flags={r['index'] for r in json.loads((audit/'anatomy_review/results.json').read_text())['new_source_review_flags']}
    expected=[r for r in roles if r['source_role'] in ('provisional_thumbnail_pass','clean_candidate_geometry_pending') and r['index'] not in flags]
    fields=('index','path','sha256','source')
    if landmarks['source_role_manifest_sha256']!=sha(audit/'source_roles_v2.json') or [{k:r[k] for k in fields} for r in expected]!=[{k:r[k] for k in fields} for r in landmarks['records']]:
        raise ValueError('Landmark sources do not match current role review')
    if screen['decoded_matches'] or screen['review_flags']:raise ValueError('Unresolved duplicate screen')
    if screen['candidate_manifest_sha256']!=sha(audit/'candidate_sources.json'):raise ValueError('Stale duplicate screen')
    if clipped['source_trial_sha256']!=sha(trial/'results.json') or clipped['augmentation_sha256']!=sha('anatomical_augmentation.py'):
        raise ValueError('Stale anatomy replay')
    if not clipped['default_v1_events_unchanged'] or review['review_scope']['new_clipped_lower_masks']!=35:
        raise ValueError('Missing crop replay/review evidence')
    split_path=Path('outputs/downloaded_phase4/outputs/phase4_with_progress/split.json')
    real_path=Path('dataset/detector_glare_review_v3/manifest.json')
    benchmark=Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm/manifest.json')
    for p,key in [(split_path,'split_sha256'),(real_path,'v3_manifest_sha256'),(benchmark,'benchmark_manifest_sha256')]:
        if sha(p)!=candidate[key]:raise ValueError('Changed split/reference: '+str(p))
    split=json.loads(split_path.read_text());real=json.loads(real_path.read_text())['records']
    excluded={sha(p.replace('\\','/')) for p in split['validation']}
    excluded.update(c['source_sha256'] for c in json.loads(benchmark.read_text())['cases'])
    excluded.update(r[k] for r in real for k in ('image_sha256','source_sha256') if r.get(k))
    manifest={'format':FORMAT,'partition':'train','seed':42,'split_path':split_path.as_posix(),
              'split_sha256':sha(split_path),'excluded_sha256':sorted(excluded),
              'v3_manifest_sha256':sha(real_path),'benchmark_manifest_sha256':sha(benchmark),
              'evidence_sha256':{str(p):sha(p) for p in [audit/'candidate_sources.json',audit/'source_roles_v2.json',audit/'anatomy_review/results.json',audit/'duplicate_screen.json',trial/'results.json',trial/'clipping_v2/results.json',trial/'acceptance_review/visual_review.json']},
              'implementation_sha256':{p:sha(p) for p in ['expanded_feature_data.py','anatomical_augmentation.py','completion.py']},
              'sources':[{k:r[k] for k in ('index','path','sha256','source','landmarks')} for r in landmarks['records']],
              'scope':'Detector training proposal only; provisional source cleanliness, no verified uncovered reconstruction targets',
              'limitations':['Square-resize distortion retained','Only two reviewed real training glare cases',
                             'Procedural anatomical coverings are partial and not photorealistic','Identity-disjointness is not certified']}
    ds=ExpandedCoveringDataset(manifest,root)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'rejections.json').write_text(json.dumps(ds.rejections,indent=2))
    rows=[r for r in real if r['split']=='train']
    groups=[[i for i,r in enumerate(rows) if r['kind']==kind] for kind in ('covered','uncovered')]
    for source in SOURCES:
        for positive in (True,False):groups.append([len(rows)+i for i,c in enumerate(ds.cases) if c['source']==source and (c['kind']!='none')==positive])
    schedule=list(expanded_balanced_schedule(groups,42))
    counts=Counter(i for epoch in schedule for batch in epoch for i in batch)
    if set(counts)!=set(range(len(rows)+len(ds))):raise ValueError('Schedule omits examples')
    checked=0;generic_checked=0;case_counts=Counter()
    legacy={i:CompletionDataset([r['path']],256,42,validation=True) for i,r in enumerate(ds.sources)}
    for i in range(len(ds)):
        item=ds[i];case_counts[(item['source'],item['kind'],item['degraded'])]+=1
        if not torch.isfinite(item['input']).all() or item['input'].min()<0 or item['input'].max()>1:raise ValueError('Invalid RGB')
        if not set(item['mask'].unique().tolist())<={0.,1.}:raise ValueError('Nonbinary mask')
        if bool(item['mask'].any())!=(item['kind']!='none'):raise ValueError('Presence/target mismatch')
        if item['kind'] in ('none','object','irregular'):
            old=legacy[item['source_index']][item['variant']]
            if any(not torch.equal(item[k],old[k]) for k in ('input','target','mask','geometry')):raise ValueError('Generic recipe changed')
            generic_checked+=1
        checked+=1
        if checked%500==0:print('Checked samples',checked,len(ds),flush=True)
    report={'data_checks_complete':True,'vm_runner_ready':False,'training_run':False,
            'sources':len(ds.sources),'real_train':len(rows),'synthetic_cases':len(ds),
            'rejected_variants':len(ds.rejections),'generic_cases_exactly_matching_legacy':generic_checked,
            'group_sizes':[len(g) for g in groups],'steps':sum(len(e) for e in schedule),'batch_size':12,
            'exposure_per_group':[{'min':min(counts[i] for i in g),'max':max(counts[i] for i in g)} for g in groups],
            'manifest_sha256':sha(out/'manifest.json'),
            'case_counts':[{'source':s,'kind':k,'degraded':d,'count':n} for (s,k,d),n in sorted(case_counts.items())],
            'next':'Package GPU-only fixed-budget experiment; no inference promotion'}
    (out/'verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='case_counts'},indent=2))


if __name__=='__main__':main()
