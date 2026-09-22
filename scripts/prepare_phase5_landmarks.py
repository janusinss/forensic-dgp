"""Cache reference landmarks once, using the GPU when available."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
import cv2
import face_alignment
import numpy as np
import torch
from tqdm import tqdm
from dataset import DegradedFacesDataset


def prepare(args):
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    ds = DegradedFacesDataset(args.data_dir,landmark_cache_dir=args.cache_dir)
    # Initialize on first cache miss, not on a cache-only rerun.
    original = ds._get_landmarks
    def detect(image):
        if ds.fa is None:
            ds.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                                flip_input=False,device=args.device)
        return original(image)
    ds._get_landmarks = detect
    counts = {}
    files = ds.image_paths[:args.limit] if args.limit else ds.image_paths
    progress = tqdm(files,desc='Preparing reference landmarks',unit='image',dynamic_ncols=True)
    for path in progress:
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f'Unreadable training image: {path}')
        rgb = cv2.cvtColor(cv2.resize(image,ds.hr_size),cv2.COLOR_BGR2RGB)
        landmarks = ds.cached_landmarks(rgb)
        source = 'asian_faces' if 'asian_faces' in Path(path).parts else 'ffhq'
        group = counts.setdefault(source,{'images':0,'detected':0})
        group['images'] += 1
        group['detected'] += int(np.abs(landmarks).sum()>0)
        progress.set_postfix(detected=sum(c['detected'] for c in counts.values()),refresh=False)
    dest = Path(args.cache_dir)
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'preparation.json').write_text(json.dumps({'device':args.device,'groups':counts,'partial':bool(args.limit)},indent=2))
    print(json.dumps(counts,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_dir',default='dataset/thumbnails128x128,dataset/asian_faces')
    parser.add_argument('--cache_dir',default='outputs/landmark_cache')
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--limit',type=int,default=0,help='Preparation smoke test only')
    prepare(parser.parse_args())
