import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np


class DiskCacheTests(unittest.TestCase):
    def api(self):
        self.assertIsNotNone(importlib.util.find_spec('feature_disk_cache'))
        from feature_disk_cache import FeatureDiskCache
        return FeatureDiskCache

    def test_round_trip_batch_order_and_provenance(self):
        Cache=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache';context={'arm':'fixed','manifest_sha256':'a'*64}
            cache=Cache.create(path,3,(2,3,3),(1,6,6),context)
            for i in range(3):cache.append(np.full((2,3,3),i,np.float32),np.full((1,6,6),i%2,np.uint8),{'index':i})
            cache.finish();cache.close()
            reader=Cache.open(path,context)
            features,targets,records=reader.batch([2,0,2])
            self.assertEqual(features.dtype,np.float32)
            self.assertEqual(features[:,0,0,0].tolist(),[2,0,2])
            self.assertEqual([r['index'] for r in records],[2,0,2])
            features[:]=99
            self.assertEqual(reader.batch([2])[0][0,0,0,0],2)
            reader.close()
            with self.assertRaises(ValueError):Cache.open(path,{'arm':'other'})

    def test_partial_cache_is_preserved_and_refused(self):
        Cache=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache';cache=Cache.create(path,2,(1,2,2),(1,2,2),{})
            cache.append(np.ones((1,2,2),np.float32),np.zeros((1,2,2),np.uint8),{})
            with self.assertRaises(ValueError):cache.finish()
            cache.close()
            self.assertEqual(json.loads((path/'state.json').read_text())['written'],1)
            with self.assertRaises(ValueError):Cache.open(path,{})
            with self.assertRaises(FileExistsError):Cache.create(path,2,(1,2,2),(1,2,2),{})

    def test_nonfinite_masks_and_tampering_are_detected(self):
        Cache=self.api()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache';cache=Cache.create(path,1,(1,2,2),(1,2,2),{})
            with self.assertRaises(ValueError):cache.append(np.full((1,2,2),np.nan,np.float32),np.zeros((1,2,2),np.uint8),{})
            with self.assertRaises(ValueError):cache.append(np.ones((1,2,2),np.float32),np.full((1,2,2),2,np.uint8),{})
            cache.append(np.ones((1,2,2),np.float32),np.zeros((1,2,2),np.uint8),{})
            cache.finish();cache.close()
            with (path/'features.bin').open('r+b') as f:f.write(b'\x00\x00\x00\x00')
            reader=Cache.open(path,{})
            with self.assertRaises(ValueError):reader.batch([0])
            reader.close()
            (path/'targets.bin').write_bytes(b'')
            with self.assertRaises(ValueError):Cache.open(path,{})
