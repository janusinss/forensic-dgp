"""Download the pinned official CodeFormer inpainting checkpoint atomically."""
import hashlib
import sys
import urllib.request
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pretrained_completion import WEIGHTS_URL, WEIGHTS_SHA256


if __name__=='__main__':
    path=Path(__file__).resolve().parents[1]/'checkpoints/codeformer_inpainting.pth'
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest()!=WEIGHTS_SHA256:
            raise SystemExit('Existing checkpoint has a different SHA256; preserve it and investigate before downloading.')
        print(f'Checkpoint verified: {path}')
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.pth.part')
        # Exclusive creation also prevents two downloads writing the same file.
        digest=hashlib.sha256()
        with temporary.open('xb') as target:
            with urllib.request.urlopen(WEIGHTS_URL,timeout=60) as response:
                while chunk:=response.read(1024*1024):
                    target.write(chunk); digest.update(chunk)
        if digest.hexdigest()!=WEIGHTS_SHA256:
            raise SystemExit(f'Download SHA256 mismatch; incomplete file retained at {temporary}')
        temporary.replace(path)
        print(f'Checkpoint downloaded and verified: {path}')
