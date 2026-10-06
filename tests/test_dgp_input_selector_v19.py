import inspect
import unittest
import numpy as np

from dgp_input_selector_v19 import select_restoration


class InputOnlySelectorTests(unittest.TestCase):
    def setUp(self):
        self.support = np.ones((256, 256), dtype=bool)

    def test_constant_does_not_request_a_learned_face(self):
        for level in [0, 1, 17, 64, 128, 192, 254, 255]:
            result = select_restoration(np.full((256, 256, 3), level, dtype=np.uint8), self.support)
            self.assertTrue(result['flat_input'])
            self.assertEqual(result['branch'], 'retained_dgp_v2')

    def test_padding_cannot_create_false_detail(self):
        camera = np.zeros((256, 256, 3), dtype=np.uint8)
        camera[40:216, 40:216] = 128
        support = np.zeros((256, 256), dtype=bool)
        support[40:216, 40:216] = True
        result = select_restoration(camera, support)
        self.assertEqual(result['laplacian_MSE'], 0)
        self.assertFalse(result['high_detail'])
        self.assertEqual(result['branch'], 'retained_dgp_v2')

    def test_high_detail_is_preserved_and_smooth_gradient_can_be_corrected(self):
        checker = np.indices((256, 256)).sum(0) % 2 * 255
        detail = np.repeat(checker.astype(np.uint8)[..., None], 3, axis=2)
        self.assertEqual(select_restoration(detail, self.support)['branch'], 'retained_dgp_v2')
        ramp = np.repeat(np.arange(256, dtype=np.uint8)[None, :, None], 256, axis=0)
        ramp = np.repeat(ramp, 3, axis=2)
        self.assertEqual(select_restoration(ramp, self.support)['branch'], 'structure_v18_update600')

    def test_labels_and_invalid_canvas_are_rejected(self):
        self.assertEqual(list(inspect.signature(select_restoration).parameters), ['camera', 'support'])
        camera = np.zeros((256, 256, 3), dtype=np.uint8)
        with self.assertRaises(TypeError): select_restoration(camera, self.support, profile='clear')
        for bad_camera, bad_mask in [(camera.astype(np.float32), self.support),
                    (camera[:32], self.support), (camera, self.support.astype(np.uint8)),
                    (camera, np.zeros_like(self.support))]:
            with self.assertRaises(ValueError): select_restoration(bad_camera, bad_mask)


if __name__ == '__main__': unittest.main()
