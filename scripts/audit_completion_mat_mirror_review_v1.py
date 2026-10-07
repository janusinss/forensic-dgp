"""Independent static provenance, terms and runtime-boundary audit for converted MAT."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/completion_mat_mirror_review_v1_r1'
PREFIX='cpu_implementation/source/libs/spandrel_extra_arches/spandrel_extra_arches/architectures/MAT/'


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    receipt=json.loads((OUT/'acquisition.json').read_text())
    assert receipt['complete'] and receipt['source_only'] and not receipt['source_executed']
    assert not receipt['checkpoint_downloaded'] and receipt['model_or_gradient_calls']==0
    assert receipt['publisher_revision']=='bef6e8b7d535abe2d42fdaeafec903abdfc638a3'
    assert receipt['CPU_implementation_revision']=='e1f2ea14b2eb9dc912bdf335803f8a3d481c45b8'
    assert receipt['published_weight_bytes']==125280246
    assert receipt['published_weight_sha256']=='eedb8504aef8a07feda7e89ef34e53344eaf3039cb1543615bf1092439ce3d98'
    assert len(receipt['files'])==11
    for row in receipt['files']:
        path=OUT/row['path'];assert sha(path)==row['sha256'] and path.stat().st_size==row['bytes']
        assert row['url'].startswith(('https://raw.githubusercontent.com/chaiNNer-org/spandrel/',
                                     'https://api.github.com/repos/chaiNNer-org/spandrel/',
                                     'https://huggingface.co/spacepxl/MAT-inpainting-fp16/',
                                     'https://huggingface.co/api/models/spacepxl/MAT-inpainting-fp16?'))
    card=(OUT/'publisher/README.md').read_text()
    assert 'EMA Generator only as fp16 safetensors' in card and 'no guarantee' in card
    original_license=(ROOT/'outputs/completion_mat_source_review_v1_r1/LICENSE').read_text().strip()
    assert (OUT/PREFIX/'__arch/LICENSE').read_text().strip()==original_license
    assert 'Attribution-NonCommercial 4.0 International' in original_license
    assert 'Permission is hereby granted, free of charge' in (OUT/'cpu_implementation/source/LICENSE').read_text()
    imports=[];parsed=0
    for row in receipt['files']:
        if row['path'].endswith('.py'):
            source=(OUT/row['path']).read_text();tree=ast.parse(source,feature_version=(3,10));parsed+=1
            imports.extend({'file':row['path'],'statement':ast.unparse(n)} for n in ast.walk(tree)
                           if isinstance(n,(ast.Import,ast.ImportFrom)))
            if row['path'].endswith(('/MAT.py','/utils.py')):
                for n in ast.walk(tree):
                    if isinstance(n,ast.Call):
                        func=ast.unparse(n.func)
                        assert func not in ('eval','exec','compile','__import__','open','pickle.load','torch.load','torch.save')
                assert not any(isinstance(n,ast.ImportFrom) and n.module in ['subprocess','urllib.request','pickle'] for n in ast.walk(tree))
    module=(OUT/PREFIX/'__arch/MAT.py').read_text()
    assert 'from spandrel.util import store_hyperparameters' in module
    assert 'F.dropout(mul_map, training=True)' in module
    assert 'z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3' in module
    assert 'return output * 0.5 + 0.5' in module
    wrappers=(OUT/PREFIX/'__init__.py').read_text()
    assert 'minimum=512, multiple_of=512, square=True' in wrappers
    failed=ROOT/'outputs/completion_mat_mirror_review_v1'
    assert (failed/'acquisition_failure.json').exists()
    assert sha(failed/'acquirer_at_empty_marker_failure.py')==sha(ROOT/'scripts/acquire_completion_mat_mirror_review_v1.py')
    report={'complete':True,'checker_sha256':sha(Path(__file__)),'acquisition_sha256':sha(OUT/'acquisition.json'),
            'source_files_verified':11,'Python_files_parsed_not_executed':parsed,'imports':imports,
            'weight_source':'Third-party publisher declares stripped EMA and FP16 conversion of MAT FFHQ512.',
            'published_weight_hash_verified_in_metadata':True,'original_author_weight_equivalence_verified':False,
            'CPU_implementation':'Pinned ChaiNNer implementation adapted from lama-cleaner; not demonstrated author CUDA parity.',
            'licenses_retained':['Original MAT CC-BY-NC-4.0 research terms','ChaiNNer repository MIT notice'],
            'publisher_has_no_additional_license_grant_in_card':True,
            'comparison_permitted_scope':'Attributed local noncommercial thesis research; not redistribution or deployment qualification.',
            'requirements':['Native model is512; record any256->512->256 processing.',
                            'Mask convention: application1=remove, generator0=missing/1=retained.',
                            'Fix torch/NumPy randomness; evaluation-mode dropout alone is not deterministic.',
                            'Use strict tensor-only state and preserve source outside reviewed removal support.'],
            'first_empty_package_marker_fetch_failure_preserved':True,'checkpoint_downloaded':False,
            'source_executed':False,'model_or_gradient_calls':0,'optimizer_updates':0,'app_changes':False,
            'automatic_or_assisted_quality_qualification':False,'goal_complete':False}
    with (OUT/'independent_source_audit.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['imports','requirements']},indent=2))


if __name__=='__main__':
    main()
