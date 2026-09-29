"""Saved-count attribution only; no inference, fitting, threshold search or selection."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.evaluate_presence_comparison import aggregate,gated_counts
from detector_replay import retention_passes


def main():
    path=Path('outputs/expanded_feature_validation/results.json');data=json.loads(path.read_text())
    baseline=json.loads(Path('outputs/detector_penalty_comparison/results.json').read_text())['baseline']['synthetic_validation']
    out=Path('outputs/expanded_retention_audit');out.mkdir(exist_ok=False)
    report={'evaluation_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'threshold':.5,'baseline':baseline,'arms':{},
            'scope':'Diagnostic error attribution from previously verified saved counts; oracle uses labels and is never deployable'}
    for arm,a in data['arms'].items():
        cases=a['synthetic_cases'];assert len(cases)==400
        for c in cases:assert gated_counts(c['raw'],c['probability'])==c['gated']
        grouped={}
        for stratum in sorted({c['stratum'] for c in cases}):
            cs=[c for c in cases if c['stratum']==stratum]
            grouped[stratum]={'cases':len(cs),'raw':aggregate([c['raw'] for c in cs]),'gated':aggregate([c['gated'] for c in cs]),
                'raw_fn_pixels':sum(c['raw']['fn'] for c in cs),
                'gate_added_fn_pixels':sum(c['gated']['fn']-c['raw']['fn'] for c in cs),
                'gate_rejected_positive_cases':sum(c['raw']['tp']+c['raw']['fn']>0 and c['probability']<.5 for c in cs)}
        oracle=[gated_counts(c['raw'],1. if c['raw']['tp']+c['raw']['fn']>0 else 0.) for c in cases]
        oracle_metrics=aggregate(oracle)
        report['arms'][arm]={'groups':grouped,'raw':aggregate([c['raw'] for c in cases]),
            'gated':aggregate([c['gated'] for c in cases]),'oracle_presence_diagnostic':oracle_metrics,
            'oracle_still_fails_retention':not retention_passes(oracle_metrics,baseline),
            'raw_fn_pixels':sum(c['raw']['fn'] for c in cases),'gate_added_fn_pixels':sum(c['gated']['fn']-c['raw']['fn'] for c in cases),
            'gate_misses':[{'image':c['image'],'stratum':c['stratum'],'source':c['source']} for c in cases if c['raw']['tp']+c['raw']['fn']>0 and c['probability']<.5]}
    (out/'results.json').write_text(json.dumps(report,indent=2))
    for arm,a in report['arms'].items():
        print(arm,json.dumps({k:v for k,v in a.items() if k not in ('groups','gate_misses')}))
        print('strata',[(k,v['raw_fn_pixels'],v['gate_added_fn_pixels']) for k,v in a['groups'].items()])


if __name__=='__main__':main()
