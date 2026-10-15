"""New reconstruction hypothesis. The factory rejects local construction first."""
from cctv_dgp_spatial_readout_v1_contract import require_manual, center_kernel


def build_model(root, parent, provider, readout_path, route):
    require_manual(root)
    assert route in ('A', 'B')
    import copy
    import numpy as np
    import torch
    from torch import nn
    from cctv_dgp_bank_comparison_v1_model import CURRENT_NAMES

    assert torch.cuda.is_available() and torch.cuda.get_device_name(0) == 'NVIDIA L4'
    assert not provider.training and all(not p.requires_grad for p in provider.parameters())

    class SpatialReadoutDGP(nn.Module):
        def __init__(self):
            super().__init__()
            self.root, self.route = root, route
            self.current = copy.deepcopy(provider.current).cuda().eval().requires_grad_(False)
            self.projectors = nn.ModuleDict()
            self.anchor_projectors = nn.ModuleDict()
            if route == 'B':
                dimensions = [(16, 512, 128), (32, 512, 128), (64, 256, 128), (128, 128, 64), (256, 64, 32)]
                with np.load(readout_path, allow_pickle=False) as arrays:
                    assert set(arrays.files) == {f'f{size}_{term}' for size, _, _ in dimensions for term in ['weight', 'bias']}
                    for size, inp, dest in dimensions:
                        # VM-only temporary constructor values are immediately replaced
                        # by the declared reused center kernel and reused bias.
                        layer = nn.Conv2d(inp, dest, 3, padding=1).cuda()
                        weight, bias = arrays[f'f{size}_weight'], arrays[f'f{size}_bias']
                        assert weight.shape == (dest, inp, 1, 1) and bias.shape == (dest,)
                        assert bias.dtype == np.float32 and np.isfinite(bias).all()
                        with torch.no_grad():
                            layer.weight.copy_(torch.from_numpy(center_kernel(weight)).cuda())
                            layer.bias.copy_(torch.from_numpy(bias).cuda())
                        self.projectors[str(size)] = layer
                self.anchor_projectors = copy.deepcopy(self.projectors).eval().requires_grad_(False)
            self.eval().requires_grad_(False)

        def train(self, mode=True):
            return super().train(False)

        def learning_names(self):
            names = ['current.'+name for name in CURRENT_NAMES]
            if self.route == 'B':
                names += [f'projectors.{size}.{term}' for size in [16, 32, 64, 128, 256] for term in ['weight', 'bias']]
            return names

        def learning_parameters(self):
            values = dict(self.named_parameters())
            return [values[name] for name in self.learning_names()]

        def enable_learning(self):
            require_manual(self.root)
            self.current.enable_vm_learning(parent)
            self.requires_grad_(False)
            for value in self.learning_parameters():
                value.requires_grad_(True)
            expected = (15, 609219) if self.route == 'A' else (25, 2176419)
            assert (len(self.learning_names()), sum(v.numel() for v in self.learning_parameters())) == expected
            assert all(not value.requires_grad for name, value in self.named_parameters() if name not in self.learning_names())

        def forward_cached(self, image, maps, features):
            require_manual(self.root)
            assert all(not m.training for m in self.modules())
            assert not image.requires_grad and image.dtype == torch.float32 and image.shape[1:] == (3, 256, 256)
            assert len(maps) == 5 and all(not value.requires_grad for value in maps)
            extras = None
            if self.route == 'B':
                assert set(features) == {'16', '32', '64', '128', '256'}
                assert all(not value.requires_grad and torch.isfinite(value).all() for value in features.values())
                with torch.no_grad():
                    anchor = {key: self.anchor_projectors[key](value) for key, value in features.items()}
                extras = {key: self.projectors[key](value)-anchor[key] for key, value in features.items()}
            return self.current.reconstruct(image, maps, extras)

    return SpatialReadoutDGP()
