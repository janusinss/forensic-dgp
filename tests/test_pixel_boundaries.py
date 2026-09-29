import unittest
import numpy as np
from scripts.audit_pixel_boundaries import boundary_counts

class BoundaryTests(unittest.TestCase):
    def test_errors_partition_into_edge_and_far(self):
        t=np.zeros((20,20),bool);t[5:15,5:15]=True
        p=t.copy();p[5,7]=False;p[10,10]=False;p[4,7]=True;p[0,0]=True
        s=boundary_counts(p,t,1)
        self.assertEqual((s['fn_edge'],s['fn_interior'],s['fp_edge'],s['fp_far']),(1,1,1,1))

    def test_empty_target_false_positive_is_far(self):
        t=np.zeros((20,20),bool);p=t.copy();p[9,9]=True
        s=boundary_counts(p,t,4)
        self.assertEqual(s['fp_far'],1)
        self.assertEqual(s['fp_edge'],0)
