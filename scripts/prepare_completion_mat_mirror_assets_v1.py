"""Validate all tensor bytes and prepare source routing without constructing a model."""
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mat_mirror_completion_v1 import read_tensor_state,sha

SOURCE=ROOT/'outputs/completion_mat_mirror_review_v1_r1'
ASSETS=ROOT/'outputs/completion_mat_mirror_assets_v1'
PREFIX='cpu_implementation/source/libs/spandrel_extra_arches/spandrel_extra_arches/architectures/MAT/__arch/'


def main():
    assert not (ASSETS/'preparation.json').exists() and not (ASSETS/'vendor').exists()
    acquired=json.loads((ASSETS/'acquisition.json').read_text(encoding='utf-8'))
    source=json.loads((SOURCE/'acquisition.json').read_text(encoding='utf-8'))
    assert acquired['complete'] and acquired['sha256']==sha(ASSETS/acquired['file'])
    state,layout=read_tensor_state(ASSETS/acquired['file'])
    vendor=ASSETS/'vendor';vendor.mkdir()
    text=(SOURCE/PREFIX/'MAT.py').read_text(encoding='utf-8')
    before='from spandrel.util import store_hyperparameters'
    assert text.count(before)==1
    adapted=text.replace(before,'from .metadata_only import store_hyperparameters')
    files={'MAT.py':adapted.encode('utf-8'),'utils.py':(SOURCE/PREFIX/'utils.py').read_bytes(),
           '__init__.py':b'', 'metadata_only.py':b'"""Identity decorator replacing unused spandrel descriptor metadata only."""\ndef store_hyperparameters():\n    return lambda cls: cls\n',
           'LICENSE_MAT':(SOURCE/PREFIX/'LICENSE').read_bytes(),
           'LICENSE_ChaiNNer':(SOURCE/'cpu_implementation/source/LICENSE').read_bytes()}
    for name,data in files.items():
        with (vendor/name).open('xb') as stream:stream.write(data)
        if name.endswith('.py'):ast.parse(data.decode('utf-8'),feature_version=(3,10))
    report={'complete':True,'preparer_sha256':sha(Path(__file__)),
            'publisher_revision':source['publisher_revision'],'CPU_source_revision':source['CPU_implementation_revision'],
            'original_weight_sha256':sha(ASSETS/acquired['file']),'all_tensor_values_finite':True,
            'tensors':len(layout),'values':sum(r['elements'] for r in layout),'weight_layout':layout,
            'vendor_sha256':{'vendor/'+name:sha(vendor/name) for name in files},
            'adapter_sha256':sha(ROOT/'mat_mirror_completion_v1.py'),
            'modifications':['Replace only unused spandrel metadata-decorator import with local identity decorator.',
                             'Instantiate Generator directly; do not use the random-latent MAT wrapper.'],
            'all_generator_and_helper_math_unchanged':True,'tensor_only_reader_no_pickle':True,
            'model_constructed':False,'model_or_gradient_calls':0,'optimizer_updates':0,
            'app_changes':False,'author_original_weight_equivalence_verified':False,'goal_complete':False}
    with (ASSETS/'preparation.json').open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k not in ['vendor_sha256','weight_layout']},indent=2))


if __name__=='__main__':
    main()
