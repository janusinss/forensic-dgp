"""No training: distinguish numeric roundoff from mismatched clean statistics."""
from pathlib import Path
import sys
import unittest

import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cctv_dgp_codebook_roundtrip_v14_r2 import check_codebook_statistics_roundtrip
from third_party.codeformer.codeformer_arch import calc_mean_std


class CodebookRoundtripTests(unittest.TestCase):
    def test_mathematically_identical_statistics_allow_measured_roundoff(self):
        torch.manual_seed(14)
        q=torch.randn(1,256,16,16)
        mean,std=calc_mean_std(q)
        _,proof=check_codebook_statistics_roundtrip(q,mean.flatten(1),std.flatten(1))
        self.assertGreater(proof['maximum_latent_difference'],0)
        self.assertLessEqual(proof['maximum_latent_difference'],proof['latent_float32_bound'])

    def test_wrong_target_statistics_are_rejected(self):
        q=torch.zeros(1,256,16,16)
        mean,std=calc_mean_std(q)
        with self.assertRaisesRegex(ValueError,'exactly match'):
            check_codebook_statistics_roundtrip(q,mean.flatten(1)+.01,std.flatten(1))


if __name__=='__main__':unittest.main()
