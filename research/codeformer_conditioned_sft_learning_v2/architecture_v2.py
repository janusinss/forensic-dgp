"""Conditioned SFT V2 multi-scale dilated pre-SFT pyramid & frequency-selective VQGAN spatial head.

Importing this module loads zero neural frameworks (`torch` / `numpy` passed explicitly).
All 575 retained trained-u64 tensors (515 R2 base + 24 U96 adapters + 36 u64 spatial heads)
stay frozen; all V2 residual output projections are zero-initialized so step 0 reproduces
u64 bitwise across every input.
"""
import hashlib
import json
import math

from . import conditioning_v2 as cond

SITES = ('32', '64', '128', '256')
CHANNELS = {'32': 256, '64': 256, '128': 128, '256': 128}
PYR_WIDTH = 96
PYR_BRANCH = 32  # 3 branches (dilation 1, 2, 4) * 32 = 96
PYR_SE_HIDDEN = 24
DARK_WIDTH = 48
HEAD_V2_WIDTH = 48
INIT_SEED_V2 = 20261026

R2_CANONICAL_SHA256 = 'a3b0986c793cb7c937ca82efde6c9e23e70557d76b93406e0f53e51b938f9e7f'
U96_CANONICAL_SHA256 = 'fbfaa9d01c5ab04d2c4383485ba4e1eebb3d9f43cdab427773447248d14a586a'
TERMINAL_HEADS_CANONICAL_SHA256 = '6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def architecture_v2_contract():
    return {
        'status': 'CONDITIONED_SFT_V2_TOP_DOWN_DILATED_PYRAMID_AND_FREQ_HEAD',
        'frozen_backbone': {
            'R2_base_tensors': 515,
            'R2_canonical_sha256': R2_CANONICAL_SHA256,
            'U96_adapter_tensors': 24,
            'U96_canonical_sha256': U96_CANONICAL_SHA256,
            'u64_head_tensors': 36,
            'u64_head_canonical_sha256': TERMINAL_HEADS_CANONICAL_SHA256,
            'total_frozen_tensors': 575,
        },
        'sites_top_down_order': list(SITES),
        'pyramid_width': PYR_WIDTH,
        'pyramid_dilations': [1, 2, 4],
        'dark_refiner_width': DARK_WIDTH,
        'head_v2_width': HEAD_V2_WIDTH,
        'init_seed_v2': INIT_SEED_V2,
        'step0_exact_u64_bitwise_parity': True,
        'clear_and_clean_dark_exact_zero_neural_perturbation': True,
    }


def tensor_state_sha256(state, *, torch):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        need(isinstance(value, torch.Tensor) and value.dtype == torch.float32
             and torch.isfinite(value).all().item(),
             'Finite FP32 tensor state required: ' + str(name))
        detached = value.detach().cpu().contiguous()
        header = json.dumps([name, 'torch.float32', list(detached.shape)], separators=(',', ':')).encode('utf-8')
        digest.update(len(header).to_bytes(8, 'big'))
        digest.update(header)
        digest.update(detached.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def build_frozen_u96_and_u64_modules(*, torch):
    """Construct the exact retained 24-tensor U96 adapters and 36-tensor u64 spatial heads."""
    nn = torch.nn
    F = torch.nn.functional

    class ResidualSkipAdapter(nn.Module):
        def __init__(self, channels: int, width: int = 32):
            super().__init__()
            self.down = nn.Conv2d(channels, width, kernel_size=1, bias=True)
            self.depthwise = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width, bias=True)
            self.act = nn.SiLU()
            self.up = nn.Conv2d(width, channels, kernel_size=1, bias=True)

        def forward(self, x):
            return x + self.up(self.act(self.depthwise(self.act(self.down(x)))))

    class SpatialHead(nn.Module):
        def __init__(self, channels: int, width: int = 32):
            super().__init__()
            self.encoder = nn.Conv2d(channels, width, kernel_size=1, bias=True)
            self.decoder = nn.Conv2d(channels, width, kernel_size=1, bias=False)
            self.fused = nn.Conv2d(channels, width, kernel_size=1, bias=False)
            self.coarse = nn.Conv2d(256, width, kernel_size=1, bias=False)
            self.act = nn.SiLU()
            self.spatial = nn.Conv2d(width, width, kernel_size=3, padding=1, bias=True)
            self.output = nn.Conv2d(width, channels, kernel_size=1, bias=True)

        def forward(self, used, decoder, fused, coarse):
            coarse_term = self.coarse(coarse)
            if coarse_term.shape[-2:] != fused.shape[-2:]:
                coarse_term = F.interpolate(coarse_term, size=fused.shape[-2:], mode='bilinear', align_corners=False)
            hidden = self.encoder(used) + self.decoder(decoder) + self.fused(fused) + coarse_term
            logits = self.output(self.act(self.spatial(self.act(hidden))))
            return fused + torch.tanh(logits) * (fused - decoder)

    u96_adapters = nn.ModuleDict({s: ResidualSkipAdapter(c, 32) for s, c in CHANNELS.items()})
    u64_heads = nn.ModuleDict({s: SpatialHead(c, 32) for s, c in CHANNELS.items()})
    return u96_adapters, u64_heads


def build_v2_trainable_modules(*, torch, seed: int = INIT_SEED_V2):
    """Build deterministic V2 trainable modules with zero-initialized residual projections."""
    nn = torch.nn
    F = torch.nn.functional

    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() and torch.cuda.is_initialized() else None
    try:
        torch.manual_seed(int(seed))

        class MultiScaleDilatedSEBlock(nn.Module):
            """Parallel dilated depthwise (1, 2, 4) + 1x1 fusion + Squeeze-and-Excitation residual block."""
            def __init__(self, width: int = PYR_WIDTH, branch: int = PYR_BRANCH, se_hidden: int = PYR_SE_HIDDEN):
                super().__init__()
                need(width == 3 * branch, 'Width must equal 3 * branch')
                self.branch = branch
                self.dw1 = nn.Conv2d(branch, branch, kernel_size=3, padding=1, dilation=1, groups=branch, bias=True)
                self.dw2 = nn.Conv2d(branch, branch, kernel_size=3, padding=2, dilation=2, groups=branch, bias=True)
                self.dw4 = nn.Conv2d(branch, branch, kernel_size=3, padding=4, dilation=4, groups=branch, bias=True)
                self.pw = nn.Conv2d(width, width, kernel_size=1, bias=True)
                self.se1 = nn.Conv2d(width, se_hidden, kernel_size=1, bias=True)
                self.se2 = nn.Conv2d(se_hidden, width, kernel_size=1, bias=True)
                self.act = nn.SiLU()

            def forward(self, h):
                b1, b2, b3 = torch.split(h, self.branch, dim=1)
                ms = torch.cat([
                    self.act(self.dw1(b1)),
                    self.act(self.dw2(b2)),
                    self.act(self.dw4(b3)),
                ], dim=1)
                fused = self.pw(ms)
                pooled = fused.mean(dim=(-2, -1), keepdim=True)
                se = torch.sigmoid(self.se2(self.act(self.se1(pooled))))
                return h + se * fused

        class PyramidSiteStage(nn.Module):
            def __init__(self, channels: int, has_top_down: bool):
                super().__init__()
                in_ch = channels + 1 + (PYR_WIDTH if has_top_down else 0)
                self.in_proj = nn.Conv2d(in_ch, PYR_WIDTH, kernel_size=1, bias=True)
                self.act = nn.SiLU()
                self.block1 = MultiScaleDilatedSEBlock()
                self.block2 = MultiScaleDilatedSEBlock()
                self.out_proj = nn.Conv2d(PYR_WIDTH, channels, kernel_size=1, bias=True)
                nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.out_proj.bias)

            def forward(self, feat, site_map, prev_hidden=None):
                parts = [feat, site_map]
                if prev_hidden is not None:
                    up_prev = F.interpolate(prev_hidden, size=feat.shape[-2:], mode='bilinear', align_corners=False)
                    parts.append(up_prev)
                h = self.act(self.in_proj(torch.cat(parts, dim=1)))
                h = self.block2(self.block1(h))
                delta = site_map * self.out_proj(self.act(h))
                return delta, h

        class TopDownMultiScaleDilatedSFTPyramid(nn.Module):
            """Top-down 32 -> 64 -> 128 -> 256 dilated pre-SFT pyramid gated by spatial blur map."""
            def __init__(self):
                super().__init__()
                self.stages = nn.ModuleDict({
                    s: PyramidSiteStage(CHANNELS[s], has_top_down=(s != '32'))
                    for s in SITES
                })

            def step_site(self, site: str, feat, site_map, prev_hidden):
                if float(site_map.detach().max().item()) <= 0.0:
                    delta = feat * 0.0
                    next_h = torch.zeros(
                        (feat.shape[0], PYR_WIDTH, int(site), int(site)),
                        dtype=feat.dtype,
                        device=feat.device,
                    )
                    return delta, next_h
                return self.stages[site](feat, site_map, prev_hidden)

            def forward(self, e_u96_dict, site_maps):
                deltas = {}
                prev_h = None
                for s in SITES:
                    deltas[s], prev_h = self.step_site(s, e_u96_dict[s], site_maps[s], prev_h)
                return deltas

        class DarkNoiseSiteStage(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.in_proj = nn.Conv2d(channels, DARK_WIDTH, kernel_size=1, bias=True)
                self.dw1 = nn.Conv2d(DARK_WIDTH, DARK_WIDTH, kernel_size=3, padding=1, dilation=1, groups=DARK_WIDTH, bias=True)
                self.dw2 = nn.Conv2d(DARK_WIDTH, DARK_WIDTH, kernel_size=3, padding=2, dilation=2, groups=DARK_WIDTH, bias=True)
                self.pw = nn.Conv2d(2 * DARK_WIDTH, DARK_WIDTH, kernel_size=1, bias=True)
                self.act = nn.SiLU()
                self.out_proj = nn.Conv2d(DARK_WIDTH, channels, kernel_size=1, bias=True)
                nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.out_proj.bias)

            def forward(self, feat, g_sensor_noise):
                max_g = float(g_sensor_noise.detach().max().item()) if isinstance(g_sensor_noise, torch.Tensor) else float(g_sensor_noise)
                if max_g <= 0.0:
                    return feat * 0.0
                h = self.act(self.in_proj(feat))
                ms = torch.cat([self.act(self.dw1(h)), self.act(self.dw2(h))], dim=1)
                h = h + self.act(self.pw(ms))
                scale = g_sensor_noise if isinstance(g_sensor_noise, torch.Tensor) else float(g_sensor_noise)
                return scale * self.out_proj(h)

        class DarkNoiseSFTRefiner(nn.Module):
            def __init__(self):
                super().__init__()
                self.stages = nn.ModuleDict({s: DarkNoiseSiteStage(CHANNELS[s]) for s in SITES})

            def forward(self, e_u96_dict, g_sensor_noise):
                return {s: self.stages[s](e_u96_dict[s], g_sensor_noise) for s in SITES}

        class FrequencySelectiveSiteHeadV2(nn.Module):
            """Zero-init high-pass VQGAN decoder edge gate + dilated spatial residual refiner."""
            def __init__(self, channels: int):
                super().__init__()
                self.freq_gate = nn.Conv2d(2 * channels, channels, kernel_size=1, bias=True)
                nn.init.zeros_(self.freq_gate.weight)
                nn.init.zeros_(self.freq_gate.bias)

                self.enc_p = nn.Conv2d(channels, HEAD_V2_WIDTH, kernel_size=1, bias=True)
                self.dec_p = nn.Conv2d(channels, HEAD_V2_WIDTH, kernel_size=1, bias=False)
                self.fus_p = nn.Conv2d(channels, HEAD_V2_WIDTH, kernel_size=1, bias=False)
                self.coa_p = nn.Conv2d(256, HEAD_V2_WIDTH, kernel_size=1, bias=False)
                self.dw1 = nn.Conv2d(HEAD_V2_WIDTH, HEAD_V2_WIDTH, kernel_size=3, padding=1, dilation=1, groups=HEAD_V2_WIDTH, bias=True)
                self.dw2 = nn.Conv2d(HEAD_V2_WIDTH, HEAD_V2_WIDTH, kernel_size=3, padding=2, dilation=2, groups=HEAD_V2_WIDTH, bias=True)
                self.pw = nn.Conv2d(2 * HEAD_V2_WIDTH, HEAD_V2_WIDTH, kernel_size=1, bias=True)
                self.act = nn.SiLU()
                self.out_proj = nn.Conv2d(HEAD_V2_WIDTH, channels, kernel_size=1, bias=True)
                nn.init.zeros_(self.out_proj.weight)
                nn.init.zeros_(self.out_proj.bias)

            @staticmethod
            def _highpass5(z):
                lp = F.avg_pool2d(z, kernel_size=5, stride=1, padding=2, count_include_pad=False)
                return z - lp

            def forward(self, base_post, used, decoder, fused, coarse, gate_map):
                if float(gate_map.detach().max().item()) <= 0.0:
                    return base_post
                hp_diff = self._highpass5(decoder) - self._highpass5(base_post)
                g_freq = torch.tanh(self.freq_gate(torch.cat([decoder, base_post], dim=1)))
                c_term = self.coa_p(coarse)
                if c_term.shape[-2:] != fused.shape[-2:]:
                    c_term = F.interpolate(c_term, size=fused.shape[-2:], mode='bilinear', align_corners=False)
                h = self.act(self.enc_p(used) + self.dec_p(decoder) + self.fus_p(fused) + c_term)
                ms = torch.cat([self.act(self.dw1(h)), self.act(self.dw2(h))], dim=1)
                r_spatial = self.out_proj(self.act(h + self.pw(ms)))
                return base_post + gate_map * (g_freq * hp_diff + r_spatial)

        class V2TrainableBundle(nn.Module):
            def __init__(self):
                super().__init__()
                self.pyramid = TopDownMultiScaleDilatedSFTPyramid()
                self.dark_refiner = DarkNoiseSFTRefiner()
                self.head_v2 = nn.ModuleDict({s: FrequencySelectiveSiteHeadV2(CHANNELS[s]) for s in SITES})

            def refine_site_sequential(self, site: str, e_u96_s, cond_info, prev_hidden):
                d_pyr, next_h = self.pyramid.step_site(site, e_u96_s, cond_info['site_maps'][site], prev_hidden)
                d_dark = self.dark_refiner.stages[site](e_u96_s, cond_info['g_sensor_noise'])
                return e_u96_s + d_pyr + d_dark, next_h

            def refine_sft_features(self, e_u96_dict, cond_info):
                used = {}
                prev_h = None
                for s in SITES:
                    used[s], prev_h = self.refine_site_sequential(s, e_u96_dict[s], cond_info, prev_h)
                return used

            def refine_post_sft(self, site: str, base_post, used, decoder, fused, coarse, cond_info):
                g_sn = cond_info['g_sensor_noise']
                sn_term = 0.50 * g_sn if isinstance(g_sn, torch.Tensor) else 0.50 * float(g_sn)
                m_s = torch.clamp(cond_info['site_maps'][site] + sn_term, 0.0, 1.0)
                return self.head_v2[site](base_post, used, decoder, fused, coarse, m_s)

        bundle = V2TrainableBundle()
        return bundle
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)


def export_v2_trainable_npz_dict(bundle, *, np):
    """Export all trainable V2 tensors as sorted FP32 NumPy arrays."""
    out = {}
    for name, param in sorted(bundle.named_parameters()):
        arr = param.detach().cpu().contiguous().numpy().astype(np.float32)
        need(np.isfinite(arr).all(), 'Nonfinite V2 parameter refused: ' + name)
        out[name] = arr
    return out


def load_v2_trainable_npz_dict(bundle, arrays_dict, *, torch, np):
    """Load exact V2 trainable parameter dictionary into bundle."""
    expected = {name: tuple(p.shape) for name, p in bundle.named_parameters()}
    need(set(arrays_dict.keys()) == set(expected.keys()), 'Exact V2 trainable tensor keyset required')
    with torch.no_grad():
        for name, param in bundle.named_parameters():
            arr = np.asarray(arrays_dict[name], dtype=np.float32)
            need(arr.shape == expected[name] and np.isfinite(arr).all(),
                 'Invalid shape or nonfinite values for V2 tensor: ' + name)
            param.copy_(torch.from_numpy(arr).to(device=param.device, dtype=torch.float32))


def max_linf_displacement_from_init(bundle, init_dict, *, np):
    """Compute max_p ||theta_p - theta_p^(0)||_inf across all V2 trainable tensors."""
    max_disp = 0.0
    for name, param in bundle.named_parameters():
        cur = param.detach().cpu().numpy().astype(np.float64)
        ref = np.asarray(init_dict[name], dtype=np.float64)
        max_disp = max(max_disp, float(np.max(np.abs(cur - ref))))
    return max_disp
