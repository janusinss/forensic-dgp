"""Separate original decoder copy: retain the demonstrated inactive head4 frozen."""
from cctv_dgp_original_decoder_candidate_v1 import OriginalDecoderCandidate

ACTIVE_DECODER_PARAMETERS = 498627
ACTIVE_DECODER_TENSORS = 12
INACTIVE_PARAMETERS = ('head4.block0.weight', 'head4.block1.weight')


class ActiveOriginalDecoderV28(OriginalDecoderCandidate):
    def __init__(self, original):
        super().__init__(original)
        for name, value in self.net.named_parameters():
            if name in INACTIVE_PARAMETERS:
                value.requires_grad_(False)
        selected = [(name, value) for name, value in self.net.named_parameters() if value.requires_grad]
        assert len(selected) == ACTIVE_DECODER_TENSORS
        assert sum(value.numel() for _, value in selected) == ACTIVE_DECODER_PARAMETERS
        assert not any(name in INACTIVE_PARAMETERS for name, _ in selected)
        assert all(not value.requires_grad for value in self.net.head4.parameters())
        # The original forward, values, aliases, buffers and VM differentiation
        # guard are inherited unchanged. No branch is reinitialized or removed.
