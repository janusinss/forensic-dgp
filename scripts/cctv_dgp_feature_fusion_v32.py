"""Separate original-DGP copy: measured fusion plus active decoder, same forward."""
from cctv_dgp_mean_centered_decoder_v29 import MeanCenteredOriginalDecoderV29
from cctv_dgp_feature_fusion_v32_policy import FUSION_NAMES, SELECTED_NAMES, validate_layout


class FeatureFusionCandidateV32(MeanCenteredOriginalDecoderV29):
    def __init__(self, original):
        super().__init__(original)
        for name, value in self.net.named_parameters():
            if name in FUSION_NAMES:
                value.requires_grad_(True)
        layout, offset = [], 0
        for name, value in self.net.named_parameters():
            assert value.requires_grad == (name in SELECTED_NAMES), name
            if value.requires_grad:
                layout.append({'name': name, 'shape': list(value.shape), 'start': offset, 'end': offset + value.numel()})
                offset += value.numel()
        validate_layout(layout)
        assert all(not value.requires_grad for value in self.net.fpn.features.parameters())
        assert all(not value.requires_grad for value in self.net.head4.parameters())
        assert not self.net.training and all(not layer.training for layer in self.net.modules())
        # Inherited forward, VM differentiation guard, mean centering, copied
        # original values and evaluation statistics are unchanged.
