"""Fixed training-only visible-face compatibility preview; zero fitting."""
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'outputs/face_extraction_dependencies'))
import numpy as np
from PIL import Image,ImageDraw
import torch
import segmentation_models_pytorch as smp
from detector_training import load_manifest,ReviewedMasks
from scripts.train_coverage_vm import sha


def main():
    torch.set_num_threads(4)
    checkpoint=ROOT/'outputs/face_extraction_epoch16.ckpt'
    assert sha(checkpoint)=='01d3c3939c28e47a45acb9a5ea8f8ee460e5ecf046c0caa8404e05915e10901b'
    manifest=ROOT/'dataset/detector_training_extension_v2/manifest.json'
    assert sha(manifest)=='1f9787ba7bf184f641b298863ae90250b90706e74b6163517a22efe293fcbb2c'
    rows=[r for r in load_manifest(manifest) if r['split']=='train']
    # Selection depends only on manifest order/kind, never model predictions.
    ids=[i for i,r in enumerate(rows) if r['kind']=='covered'][:6]+[i for i,r in enumerate(rows) if r['kind']=='uncovered'][:2]
    model=smp.Unet(encoder_name='resnet18',encoder_weights=None,classes=1,activation=None)
    weights=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert all(k.startswith('module.') for k in weights)
    model.load_state_dict({k.removeprefix('module.'):v for k,v in weights.items()},strict=True)
    model.requires_grad_(False).eval()
    data=ReviewedMasks(rows,256)
    out=ROOT/'outputs/face_extraction_probe';out.mkdir(exist_ok=False)
    sheet=Image.new('RGB',(512,148*len(ids)),'white');records=[]
    with torch.inference_mode():
        for j,i in enumerate(ids):
            x,target=data[i];logits=model(x[None])
            assert logits.shape==(1,1,256,256) and torch.isfinite(logits).all()
            p=logits.sigmoid()[0,0].numpy();visible=logits[0,0].numpy()>0
            Image.fromarray(visible.astype('uint8')*255).save(out/f'{i:03}_visible.png')
            np.save(out/f'{i:03}_probability.npy',p)
            rgb=(x.permute(1,2,0).numpy()*255).round().astype('uint8')
            tiles=[Image.fromarray(rgb),Image.fromarray((target[0].numpy()*255).astype('uint8')).convert('RGB'),
                   Image.fromarray(visible.astype('uint8')*255).convert('RGB'),Image.fromarray((rgb*visible[:,:,None]).astype('uint8'))]
            ImageDraw.Draw(sheet).text((2,j*148+2),f'{i}: input | occlusion GT | visible face | extracted',fill='black')
            for col,tile in enumerate(tiles):sheet.paste(tile.resize((128,128)),(col*128,j*148+20))
            records.append({'index':i,'image':rows[i]['image'],'visible_fraction':float(visible.mean())})
    sheet.save(out/'preview.png')
    report={'optimizer_updates':0,'strict_load':True,'smp_version':smp.__version__,
            'torch_version':str(torch.__version__),'checkpoint_sha256':sha(checkpoint),'records':records,
            'scope':'Visible-face compatibility only; no visible-face ground truth or occlusion accuracy claim'}
    (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
