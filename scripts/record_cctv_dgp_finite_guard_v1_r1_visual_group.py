"""Record a sheet only after actual original-resolution visual inspection."""
import argparse
from pathlib import Path
from datetime import datetime,timezone
from prepare_cctv_dgp_finite_guard_v1_r1_visual_review import ROOT,OUT,sha,read,write


def record(number,note):
    assert 1 <= number <= 20 and len(note.strip()) >= 30
    manifest_path = OUT/'gallery_manifest.json'; manifest = read(manifest_path)
    page = manifest['pages'][number-1]
    assert page['file'] == f'{number:02d}-finite.png' and sha(OUT/page['file']) == page['sha256']
    folder = OUT.parent/'observations'; folder.mkdir(exist_ok=True)
    write(folder/f'{number:02d}.json',{'complete':True,'group_index':number,
        'reviewer':'Primary assistant development review','independent_final_reviewer':False,
        'viewed_at_original_resolution':True,'all_three_proposals_viewed':True,
        'cohort':page['cohort'],'reference':page['reference'],'case_ids':page['case_ids'],
        'page_sha256':page['sha256'],'gallery_manifest_sha256':sha(manifest_path),
        'note':note,'recorded_UTC':datetime.now(timezone.utc).isoformat(),
        'record_script_sha256':sha(Path(__file__)),'neural_calls':0,'optimizer_updates':0,
        'model_qualification':False})
    print({'recorded_group':number,'actually_viewed_pages':1})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--group',required=True,type=int)
    parser.add_argument('--note',required=True); args = parser.parse_args(); record(args.group,args.note)


if __name__ == '__main__': main()
