"""Acquire pinned publisher metadata and static CPU MAT sources, never execute them."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import ssl
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/completion_mat_mirror_review_v1'
MODEL_REPO = 'spacepxl/MAT-inpainting-fp16'
MODEL_FILE = 'MAT_FFHQ_512_fp16.safetensors'
PUBLISHED_SHA = 'eedb8504aef8a07feda7e89ef34e53344eaf3039cb1543615bf1092439ce3d98'
CODE_REPO = 'chaiNNer-org/spandrel'
CODE_PREFIX = 'libs/spandrel_extra_arches/spandrel_extra_arches/architectures/MAT/'


def main():
    assert not OUT.exists(), 'Retain previous acquisitions'
    OUT.mkdir()
    context = ssl.create_default_context(cafile=str(ROOT/'scratch/gcloud_windows_trust.pem'))
    start = time.monotonic(); rows = []
    def fetch(url, name, cap=300000):
        assert time.monotonic()-start < 240 and len(rows) < 45
        request = urllib.request.Request(url, headers={'User-Agent':'Forensic-DGP-Thesis-Static-Review/1'})
        with urllib.request.urlopen(request, context=context, timeout=20) as response:
            data = response.read(cap+1)
            assert response.status == 200 and 0 < len(data) <= cap
            data.decode('utf-8-sig')
        path = OUT/name; path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream: stream.write(data)
        rows.append({'path':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        print(json.dumps({'acquired':name,'bytes':len(data)}),flush=True)
        return data
    try:
        metadata = json.loads(fetch('https://huggingface.co/api/models/'+MODEL_REPO+'?blobs=true','publisher/model_metadata.json',2000000))
        revision = metadata['sha']; assert len(revision)==40
        selected = next(r for r in metadata['siblings'] if r['rfilename']==MODEL_FILE)
        assert selected['lfs']['sha256']==PUBLISHED_SHA
        assert 0 < selected['size']==selected['lfs']['size'] <= 160000000
        fetch('https://huggingface.co/'+MODEL_REPO+'/raw/'+revision+'/README.md','publisher/README.md')
        commit = json.loads(fetch('https://api.github.com/repos/'+CODE_REPO+'/commits/main','cpu_implementation/commit.json',1000000))['sha']
        assert len(commit)==40
        tree = json.loads(fetch('https://api.github.com/repos/'+CODE_REPO+'/git/trees/'+commit+'?recursive=1','cpu_implementation/tree.json',5000000))
        assert not tree.get('truncated') and tree['sha']==commit
        files = sorted(r['path'] for r in tree['tree'] if r['type']=='blob' and r['path'].startswith(CODE_PREFIX))
        assert 0 < len(files) <= 32 and all(n.endswith(('.py','.md','LICENSE','.txt')) for n in files)
        for name in files:
            fetch('https://raw.githubusercontent.com/'+CODE_REPO+'/'+commit+'/'+name,'cpu_implementation/source/'+name)
        license_names = [r['path'] for r in tree['tree'] if r['type']=='blob' and
                         r['path'] in ['LICENSE','LICENSE.md','LICENSE.txt','libs/spandrel_extra_arches/LICENSE','libs/spandrel_extra_arches/pyproject.toml']]
        for name in license_names:
            fetch('https://raw.githubusercontent.com/'+CODE_REPO+'/'+commit+'/'+name,'cpu_implementation/source/'+name)
        report={'complete':True,'retrieved_UTC':datetime.now(timezone.utc).isoformat(),
                'acquirer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'publisher_repository':MODEL_REPO,'publisher_revision':revision,'selected_file':MODEL_FILE,
                'published_weight_sha256':PUBLISHED_SHA,'published_weight_bytes':selected['size'],
                'CPU_implementation_repository':CODE_REPO,'CPU_implementation_revision':commit,
                'files':rows,'source_only':True,'checkpoint_downloaded':False,'source_executed':False,
                'author_original_equivalence_verified':False,'publisher_FP16_conversion_declared':True,
                'model_or_gradient_calls':0,'optimizer_updates':0,'app_changes':False,
                'usefulness_established':False,'goal_complete':False,'seconds':time.monotonic()-start}
        with (OUT/'acquisition.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2)
        print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2))
    except BaseException as exc:
        with (OUT/'acquisition_failure.json').open('x',encoding='utf-8',newline='\n') as stream:
            json.dump({'complete':False,'cause':repr(exc),'files':rows,'model_or_gradient_calls':0,
                       'checkpoint_downloaded':False,'seconds':time.monotonic()-start},stream,indent=2)
        raise


if __name__=='__main__':
    main()
