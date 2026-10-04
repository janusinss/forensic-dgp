"""Actual entry boundary: reject local execution before neural construction."""
from pathlib import Path
import unittest
import uuid

from scripts.render_cctv_dgp_face_code_controls_v13 import run, ROOT


class RenderControlsEntryContract(unittest.TestCase):
    def test_actual_local_entry_is_rejected_before_output_creation(self):
        root=ROOT/'outputs'/('never_created_v13_'+uuid.uuid4().hex)
        self.assertFalse(root.exists())
        with self.assertRaisesRegex(RuntimeError,'forensic-dgp-thesis Linux VM'):
            run(root,ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2','not_a_protocol')
        self.assertFalse(root.exists())


if __name__=='__main__':unittest.main()
