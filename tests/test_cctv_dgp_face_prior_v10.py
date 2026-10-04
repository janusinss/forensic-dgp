"""Role, file and quantization boundaries for the inference-only comparison."""
from contextlib import contextmanager
import shutil
import unittest
from pathlib import Path
import uuid
import numpy as np
import cctv_dgp_face_prior_v10 as v


@contextmanager
def project_temp():
    # Python3.13 Windows TemporaryDirectory uses700, hiding it from restricted tokens.
    parent=(v.ROOT/'scratch').resolve()
    root=parent/('prior-test-'+uuid.uuid4().hex)
    root.mkdir(mode=0o777)
    try:
        yield root
    finally:
        if not root.resolve().is_relative_to(parent):
            raise RuntimeError('Refuse cleanup outside project scratch')
        shutil.rmtree(root)


class Boundaries(unittest.TestCase):
    def test_reference_selection_excludes_training_and_is_order_independent(self):
        refs=[{'id':f'v{i}_{source}','source':source,'role':'validation'}
              for source in ['dataset/asian_faces','dataset/thumbnails128x128'] for i in range(8)]
        training=[{'id':'train','source':'dataset/asian_faces','role':'train'}]
        a=v.choose_references(refs+training);b=v.choose_references(list(reversed(refs))+training)
        self.assertEqual(a,b);self.assertEqual(len(a),10)
        self.assertTrue(all(r['role']=='validation' for r in a))
        with self.assertRaisesRegex(ValueError,'Repeated'):v.choose_references(refs+[refs[0]])

    def test_reserved_native_never_enters_development_cohort(self):
        dev=[{'id':str(i),'role':'development','source_file':f'development/{i}.jpg'} for i in range(24)]
        reserved={'id':'reserved','role':'reserved_evaluation','source_file':'reserved_evaluation/hidden.jpg'}
        self.assertEqual(v.development_cases({'cases':dev+[reserved]}),dev)
        bad=[*dev];bad[0]={**bad[0],'source_file':'reserved_evaluation/hidden.jpg'}
        with self.assertRaisesRegex(ValueError,'wrong role'):v.development_cases({'cases':bad})

    def test_asset_path_rejects_traversal_and_cross_platform_absolute_paths(self):
        with project_temp() as directory:
            root=Path(directory)
            self.assertEqual(v.safe(root,'images/face.png'),(root/'images/face.png').resolve())
            for name in ['../hidden.jpg','/absolute.jpg','C:/hidden.jpg','images\\hidden.jpg']:
                with self.subTest(name=name),self.assertRaises(ValueError):v.safe(root,name)

    def test_quantization_preserves_original_padding_and_rejects_nonfinite(self):
        raw=np.full((256,256,3),.7,np.float32);source=np.full(raw.shape,128,np.uint8)
        support=np.zeros((256,256),bool);support[20:200,30:220]=True
        value=v.png(raw,source,support)
        np.testing.assert_array_equal(value[~support],source[~support])
        self.assertTrue((value[support]==178).all())
        raw[30,40,0]=np.nan
        with self.assertRaisesRegex(ValueError,'raw float'):v.png(raw,source,support)

    def test_changed_plan_stops_before_any_model_import(self):
        from unittest.mock import patch
        with project_temp() as directory:
            root=Path(directory);v.write(root/'frozen_plan.json',{'training':False})
            (root/'frozen_plan.sha256').write_text('0'*64,encoding='ascii')
            with patch.object(v,'OUT',root),self.assertRaisesRegex(ValueError,'fingerprint'):v.verify()


if __name__=='__main__':unittest.main()
