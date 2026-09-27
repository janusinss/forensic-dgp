"""Export existing crop candidates and proposal masks for human review, never approve them."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--review',default='outputs/real_occlusion_review')
    p.add_argument('--output',default='dataset/detector_review')
    a=p.parse_args();root=Path(a.review);out=Path(a.output)
    if out.exists():raise ValueError('Choose a new review directory')
    manifest=json.loads((root/'crop_manifest.json').read_text());records=[]
    out.mkdir(parents=True);(out/'images').mkdir();(out/'masks').mkdir()
    for r in manifest['rows']:
        if 'file' not in r:continue
        name=r['file'];dst=out/'images'/name;shutil.copy2(root/'input'/name,dst)
        probability=np.load(root/'probability'/f'{name}.npy')
        Image.fromarray(((probability>=.5)*255).astype('uint8')).save(out/'masks'/name)
        records.append(dict(image='images/'+name,mask='masks/'+name,
            image_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),mask_sha256=None,
            source=r['source'],source_sha256=r['sha256'],crop_xy_side=r['crop_xy_side'],padding=r['padding'],
            reviewed=False,kind='pending',split='pending',group='',annotation='automatic proposal; incorrect regions likely'))
    (out/'manifest.json').write_text(json.dumps({'format':'reviewed-real-masks-v1','records':records},indent=2))
    print(f'{len(records)} pending proposals exported. None are approved training labels.')


if __name__=='__main__':main()
