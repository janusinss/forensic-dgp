"""Record source-only screening decisions; no accepted masks or dataset mutation."""
import json
from pathlib import Path
import sys
from PIL import Image,ImageDraw,ImageOps

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.audit_face_occlusion_results import require,read_json
from scripts.package_face_occlusion_vm import sha

ASSESSMENTS = {
    32: ('clear_lens_control', 'Native image shows both eyes through transparent lenses; minor reflections do not establish hidden eye detail.'),
    33: ('not_lens_glare', 'Native image has no eyeglass frame; bright patches are skin/image highlights. The thumbnail suspicion was not confirmed.'),
    67: ('needs_boundary_review', 'Profile spectacles contain reflections, but the visible eye and narrow reflected regions need pixel-level scope review.'),
    78: ('needs_boundary_review', 'Glasses are dark in an underexposed image; lens tint versus poor lighting cannot be resolved confidently.'),
    107: ('needs_boundary_review', 'Lens reflections partially overlap eye regions; some eye detail remains visible. No exact boundary was adjudicated.'),
    121: ('needs_boundary_review', 'A colored lens reflection overlaps part of the eye region. It needs consistent treatment with the accepted partial-glare scope.'),
    160: ('likely_intrinsic_occlusion', 'Dark lenses hide eye detail; this is a candidate eye-occlusion image despite zero procedural clear targets.'),
    177: ('likely_intrinsic_occlusion', 'A scene reflection inside the image-left lens obscures eye detail, analogous to the accepted scene-reflection scope.'),
}


def main():
    out = ROOT/'outputs/clear_replay_scope_v1'; destination = out/'source_screening.json'
    require(not destination.exists(), 'Preserve existing source screening')
    cohort_path = out/'cohort.json'; cohort = read_json(cohort_path)
    require(cohort['unique_training_sources'] == 180 and cohort['clear_training_cases'] == 262, 'Prepared cohort differs')
    for item in cohort['sheets']: require(sha(out/item['path']) == item['sha256'], 'Inspected source sheet changed')
    records = []; dimensions = []; pools = {}
    for source in cohort['records']:
        path = ROOT/source['path']; require(sha(path) == source['sha256'], 'Inspected training source changed')
        with Image.open(path) as im: width,height = im.size
        dimensions.append({'source_id':source['source_id'],'width':width,'height':height,'nonsquare':width != height})
        pool = pools.setdefault(source['source_pool'],{'sources':0,'nonsquare':0})
        pool['sources'] += 1; pool['nonsquare'] += int(width != height)
        if source['source_id'] in ASSESSMENTS:
            category,rationale = ASSESSMENTS[source['source_id']]
            records.append({**source,'category':category,'rationale':rationale,
                            'native_image_viewed':True,'accepted_new_mask':False,'foreground_target_unchanged':True})
    require(len(records) == 8, 'Native review cohort differs')
    conflicts = [r for r in records if r['category'] == 'likely_intrinsic_occlusion']
    sheet = Image.new('RGB',(1024,592),'white')
    for n,row in enumerate(records):
        xx = n%4*256; yy = n//4*296
        with Image.open(ROOT/row['path']) as im: tile = ImageOps.contain(im.convert('RGB'),(256,256),Image.Resampling.BILINEAR)
        sheet.paste(tile,(xx+(256-tile.width)//2,yy+40+(256-tile.height)//2))
        draw = ImageDraw.Draw(sheet); draw.text((xx+2,yy+2),f'{row["source_id"]}: {Path(row["path"]).name}',fill='black')
        draw.text((xx+2,yy+18),row['category'],fill='black')
    sheet.save(out/'native_candidates.png')
    report = {'source_screen_complete':True,'pixel_adjudication_complete':False,'cohort_sha256':sha(cohort_path),
              'script_sha256':sha(__file__),'contact_sheets_inspected':12,'source_photos_screened':180,
              'native_photos_inspected':8,'records':records,'source_dimensions':dimensions,'source_pool_dimensions':pools,
              'likely_conflict_sources':[r['source_id'] for r in conflicts],
              'likely_conflict_clear_cases':[c['case'] for r in conflicts for c in r['clear_cases']],
              'likely_conflict_schedule_exposures':sum(c['schedule_exposures'] for r in conflicts for c in r['clear_cases']),
              'prepared_photo_targets':'All262 cached procedural-clear targets have zero foreground pixels',
              'future_policy':'Separate no added synthetic covering from intrinsically unoccluded source. Use obscured sources as unpaired detector candidates, not uncovered completion ground truth.',
              'geometry_observation':'CompletionDataset currently directly resizes nonsquare sources to a square; new preprocessing needs a versioned geometry-preserving path.',
              'code_reference_sha256':sha(ROOT/'completion_data.py'),
              'scope':'Training sources only; all180 contact-screened in source-ID order, without validation/test/error ranking',
              'changes_applied':'No source, mask, cache, split, teacher, threshold, gate or application change',
              'review_limitation':'Assistant visual screening, approximate and not independently adjudicated. Unflagged sources are not certified clean restoration/completion targets.',
              'optimizer_updates_locally':0,'model_inference_performed':False}
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'likely_conflict_sources':report['likely_conflict_sources'],
                      'likely_conflict_clear_cases':report['likely_conflict_clear_cases'],
                      'negative_schedule_exposures':report['likely_conflict_schedule_exposures'],
                      'source_pool_dimensions':pools,'labels_changed':False},indent=2),flush=True)


if __name__ == '__main__': main()
