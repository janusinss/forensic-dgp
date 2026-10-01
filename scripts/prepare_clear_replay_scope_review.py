"""Source-only cohort for a training-clear label-scope review; no model inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import torch
from PIL import Image,ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_face_occlusion_results import require,read_json
from scripts.package_face_occlusion_vm import sha
from scripts.audit_face_occlusion_focus_results import PROTOCOL_SHA,REPLAY_SHA,AUDIT_OUTPUT,ARCHIVE


def main():
    out = ROOT/'outputs/clear_replay_scope_v1'; require(not out.exists(), 'Preserve existing review cohort')
    audit = read_json(AUDIT_OUTPUT/'results.json'); reproduction = read_json(AUDIT_OUTPUT/'reproduction.json')
    require(reproduction['complete'] is True and reproduction['selection_agrees_with_vm'] is True and
            sha(ARCHIVE) == audit['archive_sha256'] == reproduction['archive_sha256'], 'Complete return verification required')
    protocol_path = ROOT/'outputs/coverage_protocol_v1/protocol.json'
    cache_path = ROOT/'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(protocol_path) == PROTOCOL_SHA and sha(cache_path) == REPLAY_SHA, 'Fixed training protocol/cache changed')
    protocol = read_json(protocol_path); cache = torch.load(cache_path,map_location='cpu',weights_only=True)
    schedule = protocol['schedules']['extended']['batches']
    exposure = Counter(i-73 for e in schedule for b in e for i in b if i >= 73)
    require(set(cache) == set(exposure) and len(cache) == 638 and sum(exposure.values()) == 840,
            'Cached training membership differs')
    sources = {}
    for case in sorted(cache):
        x,m = cache[case]
        require(x.dtype == m.dtype == torch.uint8 and x.shape == (3,256,256) and m.shape == (1,256,256) and
                ((m == 0) | (m == 1)).all() and bool(m.any()) == (case%5 != 0), 'Training cache type/target differs')
        if m.any(): continue
        source_id = case//10; source = protocol['sources'][source_id]
        if source_id not in sources:
            relative = source['path'].replace('\\','/'); path = (ROOT/relative).resolve()
            require(path.is_relative_to(ROOT) and sha(path) == source['sha256'], 'Original training source changed')
            sources[source_id] = {'source_id':source_id,'path':relative,'sha256':source['sha256'],
                                  'source_pool':source['source'],'clear_cases':[],'foreground_pixels':0}
        sources[source_id]['clear_cases'].append({'case':case,'degraded':case%10 >= 5,'schedule_exposures':exposure[case],
            'input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest()})
    records = [sources[i] for i in sorted(sources)]; require(records, 'No training clear sources')
    out.mkdir(); sheets = []
    for start in range(0,len(records),16):
        subset = records[start:start+16]; sheet = Image.new('RGB',(512,148*4),'white')
        for n,row in enumerate(subset):
            with Image.open(ROOT/row['path']) as im: tile = im.convert('RGB').resize((128,128),Image.Resampling.BILINEAR)
            xx = n%4*128; yy = n//4*148; sheet.paste(tile,(xx,yy+20))
            ImageDraw.Draw(sheet).text((xx+2,yy+2),f'{row["source_id"]} {Path(row["path"]).name}',fill='black')
        name = f'sources_{start//16:02}.png'; sheet.save(out/name)
        sheets.append({'path':name,'source_ids':[r['source_id'] for r in subset],'sha256':sha(out/name)})
    report = {'prepared':True,'review_complete':False,'script_sha256':sha(__file__),
              'archive_sha256':audit['archive_sha256'],'protocol_sha256':PROTOCOL_SHA,'replay_sha256':REPLAY_SHA,
              'unique_training_sources':len(records),'clear_training_cases':sum(len(r['clear_cases']) for r in records),
              'scope':'All unique sources used for clear cached training replay, sorted by source ID; no candidate/error ranking',
              'excluded':'No validation/test image or prediction and no source outside fixed training cache',
              'review_question':'Do unannotated strong lens reflections or physical coverings conflict with procedural clear targets?',
              'annotation_status':'Source-only screening; no new label, split, training, threshold or gate change',
              'records':records,'sheets':sheets,'optimizer_updates_locally':0,'model_inference_performed':False}
    (out/'cohort.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'sources':len(records),'clear_cases':report['clear_training_cases'],'sheets':len(sheets),
                      'output':str(out)},indent=2),flush=True)


if __name__ == '__main__': main()
