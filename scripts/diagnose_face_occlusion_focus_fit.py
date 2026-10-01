"""Read-only cached training fit for the two verified final detectors."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_face_occlusion_focus_results import ARCHIVE,AUDIT_OUTPUT,EXTRACTION,PREFIX,ARMS,MODEL_SHA,REPLAY_SHA,PROTOCOL_SHA
from scripts.audit_face_occlusion_results import require,read_json
from scripts.package_face_occlusion_vm import sha
from scripts.compare_replay_fit import counts
from scripts.diagnose_face_occlusion_replay import summarize_records,verify_reference
from scripts.diagnose_face_occlusion_continuation_fit import area_summary
from scripts.evaluate_coverage_results import binary
from completion import KINDS


def main():
    import numpy as np
    from PIL import Image,ImageDraw
    import torch
    out = ROOT/'outputs/face_occlusion_focus_fit'; require(not out.exists(), 'Preserve existing fit evidence')
    audit = read_json(AUDIT_OUTPUT/'results.json'); reproduction = read_json(AUDIT_OUTPUT/'reproduction.json')
    require(reproduction['complete'] is True and reproduction['saved_masks_compared'] == 2915 and
            reproduction['selection_agrees_with_vm'] is True and
            sha(ARCHIVE) == audit['archive_sha256'] == reproduction['archive_sha256'], 'Verified return required')
    protocol_path = ROOT/'outputs/coverage_protocol_v1/protocol.json'
    cache_path = ROOT/'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(protocol_path) == PROTOCOL_SHA and sha(cache_path) == REPLAY_SHA, 'Fixed training protocol/cache changed')
    protocol = read_json(protocol_path); cache = torch.load(cache_path,map_location='cpu',weights_only=True)
    ids = sorted(cache); schedule = protocol['schedules']['extended']['batches']
    weights = Counter(i-73 for e in schedule for b in e for i in b if i >= 73)
    require(len(cache) == 638 and set(cache) == set(weights) and sum(weights.values()) == 840,
            'Cached training membership/exposure differs')
    previous_path = ROOT/'outputs/face_occlusion_continuation_fit/results.json'; previous = read_json(previous_path)
    prior_glare = read_json(ROOT/'outputs/face_occlusion_continuation_glare/results.json')
    require(previous['complete'] is True and previous['replay_sha256'] == REPLAY_SHA and
            previous['protocol_sha256'] == PROTOCOL_SHA and sha(previous_path) == prior_glare['fit_report_sha256'],
            'Previously verified source fit changed')
    source = previous['training']['pretrained30']
    require(source['checkpoint_sha256'] == MODEL_SHA and source['state_unchanged'] is True, 'Reused source model differs')
    source_records = source['records']; verify_reference(source_records,cache,protocol)
    require(summarize_records(source_records) == source['unique'] and
            summarize_records(source_records,weights) == source['scheduled_exposure'], 'Reused summary differs')
    for row in source_records:
        i = row['case']; mask = ROOT/f'outputs/face_occlusion_continuation_fit/train_masks/pretrained30/{i:04}.png'
        require(counts(binary(mask),cache[i][1][0].numpy().astype(bool)) == row['counts'], 'Reused source mask/count differs')
    result = {'script_sha256':sha(__file__),'archive_sha256':audit['archive_sha256'],
        'reproduction_sha256':sha(AUDIT_OUTPUT/'reproduction.json'),'protocol_sha256':PROTOCOL_SHA,'replay_sha256':REPLAY_SHA,
        'source_fit_sha256':sha(previous_path),'source_records_reused':True,'evaluation_batch_size':8,
        'optimizer_updates_locally':0,'optimizer_constructed':False,'promoted':False,
        'training':{'source30':{'unique':source['unique'],'scheduled_exposure':source['scheduled_exposure'],
                                'area':area_summary(source_records)}}}
    out.mkdir(); torch.set_num_threads(4); sys.path.insert(0,str(ROOT/'outputs/face_extraction_dependencies'))
    from face_occlusion_adapter import load_adapter
    from scripts.train_face_occlusion_vm import dependency_versions
    result['dependencies'] = dependency_versions(ROOT/'outputs/face_extraction_dependencies')
    for arm in ARMS:
        key = f'{arm}/40'; checkpoint = EXTRACTION/PREFIX/arm/'epoch_40.pth'
        require(sha(checkpoint) == reproduction['checkpoint_checks'][key]['sha256'], 'Audited final state changed')
        model,payload = load_adapter(checkpoint,'cpu'); require(payload['focus_arm'] == arm and payload['epoch'] == 40,
                                                                'Final checkpoint arm/epoch differs')
        model.requires_grad_(False).eval(); before = {k:v.clone() for k,v in model.state_dict().items()}
        records = []; mask_dir = out/'train_masks'/arm; mask_dir.mkdir(parents=True)
        with torch.inference_mode():
            for start in range(0,len(ids),8):
                batch = ids[start:start+8]; x = torch.stack([cache[i][0].float()/255 for i in batch])
                predictions = (model.detect(x).sigmoid()[:,0] >= .5).numpy()
                for i,p in zip(batch,predictions):
                    target = cache[i][1][0].numpy().astype(bool)
                    records.append({'case':i,'kind':KINDS[i%5],'degraded':i%10 >= 5,
                                    'source':protocol['sources'][i//10]['source'],'counts':counts(p,target)})
                    Image.fromarray(p.astype('uint8')*255).save(mask_dir/f'{i:04}.png')
                if start%160 == 0: print(arm,'training replay',start,len(ids),flush=True)
        require(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()), 'Read-only fit inference changed model state')
        verify_reference(records,cache,protocol)
        result['training'][arm] = {'records':records,'unique':summarize_records(records),
            'scheduled_exposure':summarize_records(records,weights),'area':area_summary(records),
            'checkpoint_sha256':sha(checkpoint),'model_state_unchanged':True}
        del model,payload,before
        (out/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
    preview_ids = previous['synthetic_preview']['source10_failure_ranked_cases_held_fixed']
    require(len(preview_ids) == len(set(preview_ids)) == 10 and all(i in cache for i in preview_ids), 'Fixed training preview differs')
    sheet = Image.new('RGB',(640,1480),'white')
    for line,i in enumerate(preview_ids):
        target = cache[i][1][0].numpy().astype(bool); image = cache[i][0].permute(1,2,0).numpy()
        masks = [target,binary(ROOT/f'outputs/face_occlusion_continuation_fit/train_masks/pretrained30/{i:04}.png')]
        masks.extend(binary(out/'train_masks'/arm/f'{i:04}.png') for arm in ARMS)
        tiles = [Image.fromarray(image)]+[Image.fromarray(m.astype('uint8')*255).convert('RGB') for m in masks]
        ImageDraw.Draw(sheet).text((2,line*148+2),f'Train{i}: input | target | source30 | control40 | focus40',fill='black')
        for j,tile in enumerate(tiles): sheet.paste(tile.resize((128,128)),(j*128,line*148+20))
    sheet.save(out/'training_preview.png'); result['preview_training_indices'] = preview_ids; result['complete'] = True
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'training':{k:v['unique']['all'] for k,v in result['training'].items()},
                      'optimizer_updates_locally':0},indent=2),flush=True)


if __name__ == '__main__': main()
