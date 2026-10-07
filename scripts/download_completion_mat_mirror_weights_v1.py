"""Bounded tensor-only model acquisition against a pinned publisher LFS digest."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import ssl
import time
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/completion_mat_mirror_review_v1_r1'
OUT=ROOT/'outputs/completion_mat_mirror_assets_v1'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    assert not OUT.exists()
    metadata=json.loads((SOURCE/'acquisition.json').read_text())
    audit=json.loads((SOURCE/'independent_source_audit.json').read_text())
    assert audit['complete'] and audit['acquisition_sha256']==sha(SOURCE/'acquisition.json')
    OUT.mkdir();start=time.monotonic();expected=metadata['published_weight_bytes']
    url='https://huggingface.co/'+metadata['publisher_repository']+'/resolve/'+metadata['publisher_revision']+'/'+metadata['selected_file']
    context=ssl.create_default_context(cafile=str(ROOT/'scratch/gcloud_windows_trust.pem'))
    partial=OUT/(metadata['selected_file']+'.part');digest=hashlib.sha256();size=0
    try:
        request=urllib.request.Request(url,headers={'User-Agent':'Forensic-DGP-Thesis-Weights-Audit/1'})
        with urllib.request.urlopen(request,context=context,timeout=25) as response,partial.open('xb') as stream:
            assert response.status==200
            if response.headers.get('Content-Length'):assert int(response.headers['Content-Length'])==expected
            while True:
                assert time.monotonic()-start<240,'Download cap240s; retain partial'
                chunk=response.read(1024**2)
                if not chunk:break
                size+=len(chunk);assert size<=expected
                stream.write(chunk);digest.update(chunk)
        assert size==expected and digest.hexdigest()==metadata['published_weight_sha256']
        destination=OUT/metadata['selected_file'];partial.rename(destination)
        report={'complete':True,'retrieved_UTC':datetime.now(timezone.utc).isoformat(),
                'downloader_sha256':sha(Path(__file__)),'source_audit_sha256':sha(SOURCE/'independent_source_audit.json'),
                'publisher_repository':metadata['publisher_repository'],'publisher_revision':metadata['publisher_revision'],
                'source_url':url,'file':destination.name,'bytes':size,'sha256':digest.hexdigest(),
                'publisher_converted_FP16_EMA':True,'original_author_equivalence_verified':False,
                'network_pickle_loaded':False,'weights_loaded':False,'model_or_gradient_calls':0,'optimizer_updates':0,
                'app_changes':False,'goal_complete':False,'seconds':time.monotonic()-start}
        with (OUT/'acquisition.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2)
        print(json.dumps(report,indent=2))
    except BaseException as exc:
        with (OUT/'acquisition_failure.json').open('x',encoding='utf-8',newline='\n') as stream:
            json.dump({'complete':False,'cause':repr(exc),'bytes_received':size,
                       'weights_loaded':False,'model_or_gradient_calls':0,'seconds':time.monotonic()-start},stream,indent=2)
        raise


if __name__=='__main__':
    main()
