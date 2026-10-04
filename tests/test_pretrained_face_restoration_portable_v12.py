"""Portable hash boundary rejects changed bytes before architecture/model loading."""
from contextlib import contextmanager
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch
import uuid

import pretrained_face_restoration_portable_v12 as portable


class PortableLoader(unittest.TestCase):
    def test_changed_bytes_reject_without_file_digest_or_model_loading(self):
        scratch = Path(__file__).resolve().parents[1] / 'scratch'
        root = scratch / ('portable-hash-' + uuid.uuid4().hex); root.mkdir(mode=0o777)
        try:
            weights = root / 'changed.pth'; weights.write_bytes(b'changed weight bytes')
            with patch('hashlib.file_digest', side_effect=AssertionError('Unavailable on Python3.10'), create=True):
                with patch.object(portable.torch, 'load', side_effect=AssertionError('Must reject first')):
                    with self.assertRaisesRegex(ValueError, 'fingerprint'): portable.load_face_restorer(weights)
        finally:
            assert root.resolve().is_relative_to(scratch.resolve())
            shutil.rmtree(root)


if __name__ == '__main__': unittest.main()
