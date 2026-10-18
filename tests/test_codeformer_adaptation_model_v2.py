"""V2 source/loss contract checks; no model load, forward, backward or optimizer."""
import ast
from pathlib import Path
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'research/codeformer_adaptation_v2/model.py'
TREE = ast.parse(SOURCE.read_text(), feature_version=(3, 10))
FUNCTIONS = {node.name: node for node in TREE.body if isinstance(node, ast.FunctionDef)}


def segment(name):
    return ast.get_source_segment(SOURCE.read_text(), FUNCTIONS[name])


class ModelSourceContracts(unittest.TestCase):
    def test_original_assets_and_groups_remain_immutable(self):
        values = {node.targets[0].id: ast.literal_eval(node.value) for node in TREE.body
                  if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
        self.assertEqual(values['INITIALIZER_SHA256'], '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7')
        self.assertEqual(values['FROZEN_GROUPS'], ('quantize', 'generator'))
        self.assertEqual(len(values['TRAINABLE_GROUPS']), 6)
        self.assertEqual(values['LOSS_WEIGHTS'], {'token_ce': .5, 'latent_mse': 1.,
            'pixel_l1_rgb': 1., 'lpips_vgg': 1., 'coarse_rgb_l1': 4.,
            'behavior_rgb_l1': 2., 'behavior_token_kl': .1})

    def test_original_behavior_teacher_is_separate_fp32_frozen(self):
        text = segment('load_behavior_teacher')
        self.assertIn('load_trainable(path', text)
        self.assertIn('.float().eval().requires_grad_(False)', text)
        target = segment('behavior_targets')
        self.assertTrue(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'no_grad'
                            for node in FUNCTIONS['behavior_targets'].decorator_list))
        self.assertNotIn('inference_mode', target)
        self.assertIn('enabled=False', target)
        self.assertIn('rgb.detach().float().clone()', target)
        self.assertIn('logits.detach().float().clone()', target)

    def test_kl_sums_codes_and_averages_tokens_with_temperature_squared(self):
        text = segment('token_distillation')
        self.assertIn('reduction="none"', text)
        self.assertIn('.sum(dim=-1).mean() * temperature**2', text)
        self.assertIn('F.log_softmax(candidate_logits.float() / temperature, dim=-1)', text)
        self.assertIn('F.softmax(target_logits.float() / temperature, dim=-1)', text)

    def test_coarse_loss_has_unclipped_all_resolution_path(self):
        text = segment('forward_training_terms')
        self.assertIn('rgb = (output.float() + 1) / 2', text)
        self.assertIn('F.avg_pool2d(rgb, 32, 32)', text)
        coarse = text.index('"coarse_rgb_l1"')
        self.assertLess(coarse, text.index('if image_eligible:'))
        self.assertIn('with torch.autocast(device_type=output.device.type, enabled=False)', text)
        self.assertNotIn('rgb.detach()', text)
        self.assertNotIn('rgb.clamp', text)

    def test_teacher_codes_use_restoration_dictionary_at_hq_indices(self):
        text = segment('hq_targets')
        self.assertIn('teacher.encoder(clean512.float() * 2 - 1)', text)
        self.assertIn('target_codebook = teacher.quantize if codebook is None else codebook', text)
        self.assertIn('target_codebook.get_codebook_feat(', text)
        terms = segment('forward_training_terms')
        self.assertIn('codebook=net.quantize', terms)
        self.assertIn('detach_16=True, adain=True', terms)

    def test_delivered_geometry_policy_and_loss_paths_are_unchanged(self):
        delivery = segment('delivered_rgb')
        self.assertLess(delivery.index('.clamp(0, 1)'), delivery.index('size=(256, 256)'))
        self.assertIn('align_corners=False', delivery)
        terms = segment('forward_training_terms')
        self.assertIn('perceptual(output.float(), clean512.float() * 2 - 1)', terms)
        self.assertIn('token_distillation(logits, behavior.logits)', terms)
        self.assertNotIn('.backward(', terms)
        self.assertNotIn('optimizer', terms)


if __name__ == '__main__':
    unittest.main()
