"""Guarded reusable exact batch-five decoder measurement context.

Importing this module constructs no network and imports no neural package.
Every model/gradient consumer must first pass the new manual-L4 guard and use
the retained source working directory, as required by its model factory.
"""
from contextlib import contextmanager
import importlib.util
import os
from pathlib import Path
import sys
import time

from cctv_dgp_decoder_activation_v1_contract import read, write, sha, array_sha, require_manual

SOURCE_NAME = 'cctv_dgp_spatial_readout_v1_vm'
PARENT_NAME = 'cctv_dgp_bank_photometric_guard_v1_vm'
REPLAY_NAME = 'cctv_dgp_spatial_path_replay_v1_vm'
REPAIR_STAGE = 'outputs/decoder_hinge_repair_v1'


@contextmanager
def source_cwd(source):
    previous = Path.cwd()
    assert Path(source).resolve() == (Path.home() / 'forensic-dgp' / SOURCE_NAME).resolve()
    os.chdir(source)
    try:
        yield
    finally:
        os.chdir(previous)


def file_module(path, name, expected):
    assert sha(path) == expected
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == Path(path).resolve()
    return module


class MeasurementContext:
    def __init__(self, root, repair, source, parent, replay, p, repaired, old, data, out, counts, clock):
        require_manual(root)
        self.root, self.repair, self.source, self.parent, self.replay = root, repair, source, parent, replay
        self.p, self.repaired, self.old, self.data = p, repaired, old, data
        self.out, self.counts, self.clock = out, counts, clock
        assert source == (root.parent / SOURCE_NAME).resolve()
        assert parent == (root.parent / PARENT_NAME).resolve()
        assert replay == (root.parent / REPLAY_NAME).resolve()
        with source_cwd(source):
            # Resolve the retained factory/guard from its verified source root,
            # rather than requiring an unbound duplicate in this new packet.
            sys.path.insert(0, str(source))
            loader = file_module(source / 'scripts/cctv_dgp_spatial_readout_v1_vm.py',
                'retained_activation_initial_loader', p['source_required_sha256']['scripts/cctv_dgp_spatial_readout_v1_vm.py'])
            np, torch, original, identity, provider, provenance, loading = loader.load_dependencies(source, parent, old)
            self.np, self.torch = np, torch
            self.original, self.identity, self.provider = original, identity, provider
            self.provenance, self.loading = provenance, loading
            torch.set_grad_enabled(False)
            from PIL import Image
            from cctv_dgp_pilot import grid112, state_hash, buffer_hash
            from cctv_dgp_head4_lossless_v1 import unpack
            from cctv_dgp_spatial_readout_v1_model import build_model
            from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
            from frozen_capacity_contract import feature_support
            from frozen_raw_metrics import deliver
            self.Image, self.state_hash, self.buffer_hash = Image, state_hash, buffer_hash
            assert Path(sys.modules[build_model.__module__].__file__).resolve() == source / 'cctv_dgp_spatial_readout_v1_model.py'
            self.pixel_and_structure_losses, self.deliver = pixel_and_structure_losses, deliver
            self.corrected = file_module(repair / 'cctv_dgp_bank_hinge_boundary_v1_loss.py',
                'activation_exact_corrected_helper', repaired['corrected_helper_sha256'])
            assert all(not v.requires_grad for net in (original.net, identity, provider) for v in net.parameters())
            self.frozen_start = dict(original=state_hash(original.net), recognizer=state_hash(identity), provider=state_hash(provider))
            refs, cases = ({r['id']: r for r in data['references']}, {c['id']: c for c in data['cases']})
            source_cache = read(source / 'outputs/spatial_readout_v1/cache_receipt.json')['rows']
            self.caches, cache_rows = [], []
            began = time.monotonic()
            for index, block in enumerate(p['reference_blocks']):
                cs, ref = [cases[c] for c in block['case_ids']], refs[block['reference']]
                camera = [self.pixels(parent / c['input']) for c in cs]
                targets = [self.pixels(parent / ref['target']) for _ in cs]
                masks = [self.pixels(parent / ref['observed'], 'L') > 0 for _ in cs]
                support = [feature_support(m, c['landmarks5_canvas_xy']) for m, c in zip(masks, cs)]
                values = dict(x=self.tensor(camera), target=self.tensor(targets),
                    mask=torch.from_numpy(np.stack(masks).astype(np.float32))[:, None].cuda(),
                    feature=torch.from_numpy(np.stack(support).astype(np.float32))[:, None].cuda(),
                    grid=torch.from_numpy(np.stack([grid112(ref['matrix112'])] * 5)).cuda())
                counts['original_batch_calls'] += 1
                baseline = torch.where(values['mask'].bool(), original(values['x']), values['x'])
                maps = provider.current.observe(values['x'])
                counts['prior_feature_batch_calls'] += 1
                features = provider.conditioned_features(maps)
                retained, embedding = unpack(parent / 'outputs/bank_comparison_v1/baseline/packs' / f"b{block['begin']//5:03d}.npz", block['case_ids'])
                raw = baseline.cpu().permute(0, 2, 3, 1).numpy().copy()
                assert np.array_equal(raw, retained), 'Fresh original RAW cache differs'
                values.update(baseline=baseline, target_vector=torch.from_numpy(embedding[:, 2].copy()).cuda())
                values['old_arc'] = 1 - (torch.from_numpy(embedding[:, 0].copy()).cuda().double() * values['target_vector'].double()).sum(1)
                b = dict(cases=cs, camera=camera, target8=targets, mask8=masks, support8=support,
                    baseline=raw, retained_embeddings=embedding, target_vectors=embedding[:, 2].copy(),
                    CPU={k: v.detach().cpu().contiguous() for k, v in values.items()},
                    old_terms={k: v.detach().cpu() for k, v in pixel_and_structure_losses(baseline, values['target'], values['mask'], values['feature']).items()},
                    maps=[v.detach().cpu().contiguous() for v in maps],
                    features={k: v.detach().cpu().contiguous() for k, v in features.items()})
                proof = dict(reference=block['reference'], source=block['source'],
                    maps=[array_sha(v.numpy()) for v in b['maps']], features={k: array_sha(v.numpy()) for k, v in b['features'].items()})
                assert all(proof[k] == source_cache[index][k] for k in proof), 'Fresh source feature parity differs'
                self.caches.append(b)
                cache_rows.append(proof)
                del values, baseline, maps, features, retained, embedding, raw
                clock()
                elapsed = time.monotonic() - began
                if index == 1:
                    projected = elapsed * 10 * 1.5
                    write(out / 'cache_projection.json', dict(references=2, seconds=elapsed, projected_seconds=projected, cap_seconds=p['budgets']['cache_seconds']))
                    assert projected <= p['budgets']['cache_seconds']
                assert elapsed <= p['budgets']['cache_seconds'], 'Finite cache stop'
                print({'cached_reference': index + 1, 'of': 20}, flush=True)
            write(out / 'cache_receipt.json', dict(complete=True, rows=cache_rows, exact_original_and_features=True, seconds=time.monotonic()-began))
            self.initial = np.load(repair / REPAIR_STAGE / 'initial_vector.npy', allow_pickle=False)
            self.vectors = {label: np.load(repair / REPAIR_STAGE / (label + '_vector.npy'), allow_pickle=False) for label in p['state_labels']}
            self.direction = self.vectors['fit_u001'].astype(np.float64) - self.initial.astype(np.float64)
            assert self.initial.dtype == np.float32 and self.initial.shape == self.direction.shape == (609219,)
            assert np.isfinite(self.direction).all() and 0 < float(np.linalg.norm(self.direction)) <= p['common_radius_L2']
            self.names = old['trainable_parameter_names']['A']
            self.layout = read(source / 'outputs/spatial_readout_v1/A_parameter_layout.json')
            initial_state = torch.load(source / 'outputs/spatial_readout_v1/state_A0/full_state.pth', map_location='cpu', weights_only=True)
            self.model = build_model(source, parent, provider, source / 'readouts.npz', 'A')
            self.model.load_state_dict(initial_state['model'], strict=True)
            self.model.eval().requires_grad_(False)
            assert self.model.learning_names() == self.names and len(self.names) == 15
            receipt = read(source / 'outputs/spatial_readout_v1/state_A0/receipt.json')
            self.initial_model_hash, self.initial_buffer_hash = state_hash(self.model), buffer_hash(self.model)
            assert self.initial_model_hash == receipt['model_state'] and self.initial_buffer_hash == receipt['buffer_state']
            self.fixed_hash = state_hash({k: v for k, v in self.model.state_dict().items() if k not in self.names})
            assert np.array_equal(self.flat(), self.initial)
            self.initial_model_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
            torch.manual_seed(20261014)
            torch.cuda.manual_seed_all(20261014)
            self.CPU_rng = torch.get_rng_state().clone()
            self.CUDA_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]

    def pixels(self, path, mode='RGB'):
        with self.Image.open(path) as image:
            assert image.size == (256, 256)
            return self.np.asarray(image.convert(mode)).copy()

    def tensor(self, values):
        return self.torch.from_numpy(self.np.stack(values).astype(self.np.float32)/self.np.float32(255)).permute(0, 3, 1, 2).contiguous().cuda()

    def gpu(self, b):
        return {k: v.cuda() for k, v in b['CPU'].items()} | dict(maps=[v.cuda() for v in b['maps']], features={k: v.cuda() for k, v in b['features'].items()})

    def flat(self):
        return self.np.concatenate([v.detach().cpu().numpy().reshape(-1) for v in self.model.learning_parameters()]).astype(self.np.float32)

    def assign(self, label):
        vector = self.vectors[label]
        assert label in self.p['state_labels'] and vector.dtype == self.np.float32 and vector.shape == (609219,)
        assert float(self.np.linalg.norm(vector.astype(self.np.float64)-self.initial.astype(self.np.float64))) <= self.p['common_radius_L2']
        with self.torch.no_grad():
            for row, value in zip(self.layout, self.model.learning_parameters()):
                value.copy_(self.torch.from_numpy(vector[row['begin']:row['end']].reshape(row['shape'])).cuda())
        assert self.np.array_equal(self.flat(), vector)
        assert self.buffer_hash(self.model) == self.initial_buffer_hash
        assert self.state_hash({k: v for k, v in self.model.state_dict().items() if k not in self.names}) == self.fixed_hash

    def forward(self, g):
        self.counts['decoder_batch_calls'] += 1
        return self.torch.where(g['mask'].bool(), self.model.forward_cached(g['x'], g['maps'], g['features']), g['x'])

    def objective(self, pred, g, b):
        torch = self.torch
        terms = self.pixel_and_structure_losses(pred, g['target'], g['mask'], g['feature'])
        self.counts['recognizer_batch_calls'] += 1
        embedding = self.identity.embedding(pred, g['mask'], g['grid'])
        arc = 1 - (embedding.double()*g['target_vector'].double()).sum(1)
        old_terms = {k: v.cuda() for k, v in b['old_terms'].items()}
        ns = self.old['normalizers']
        raw = [terms['MSE'].mean(), terms['SSIM_loss'].mean(), arc.mean(), terms['landmark_structure'][1:].mean()]
        components = [w*v/n for w, v, n in zip(self.old['reconstruction_weights'], raw, ns)]
        barriers = [5*torch.relu(terms['MSE']-old_terms['MSE']-1e-12).mean()/ns[0],
                    5*torch.relu(terms['SSIM_loss']-old_terms['SSIM_loss']-1e-6).mean()/ns[1],
                    5*torch.relu(arc-g['old_arc']-1e-6).mean()/ns[2]]
        anchors = [terms['MSE'][0]/self.old['active_anchor_normalizers'][0], terms['MSE'][1]/self.old['active_anchor_normalizers'][1]]
        photo, photo_detail = self.corrected.brightness_guard(pred, g['baseline'], g['target'], g['mask'], ns[0], 10., .2)
        # Preserve the original summation grouping exactly, including the three
        # relu terms before coefficient5 and the two anchors before final sum.
        barrier = 5*(torch.relu(terms['MSE']-old_terms['MSE']-1e-12).mean()/ns[0]
            + torch.relu(terms['SSIM_loss']-old_terms['SSIM_loss']-1e-6).mean()/ns[1]
            + torch.relu(arc-g['old_arc']-1e-6).mean()/ns[2])
        anchor = anchors[0] + anchors[1]
        value = sum(components) + barrier + anchor + photo
        assert bool(torch.isfinite(value))
        mask = g['mask'].detach().double()
        mean_shift = ((pred-g['baseline']).double()*mask).sum((2, 3))/mask.sum((2, 3))
        current = {'MSE': terms['MSE'], 'SSIM_loss': terms['SSIM_loss'], 'ArcFace_loss': arc, 'landmark_structure': terms['landmark_structure']}
        baseline = {'MSE': old_terms['MSE'], 'SSIM_loss': old_terms['SSIM_loss'], 'ArcFace_loss': g['old_arc']}
        record = dict(objective=float(value.detach()), raw_terms=[float(v.detach()) for v in raw],
            normalized_weighted_RGB_SSIM_recognizer_structure=[float(v.detach()) for v in components],
            regression_barriers_RGB_SSIM_recognizer=[float(v.detach()) for v in barriers],
            clear_blur_RGB_anchors=[float(v.detach()) for v in anchors], barrier=float(barrier.detach()),
            anchors=float(anchor.detach()), photometric_guard=float(photo.detach()), per_image_photometric=photo_detail,
            current={k: v.detach().cpu().tolist() for k, v in current.items()},
            baseline={k: v.detach().cpu().tolist() for k, v in baseline.items()},
            channel_mean_shift=mean_shift.detach().cpu().tolist(), recognizer_batch=5,
            embedding_sha256=array_sha(embedding.detach().cpu().numpy()))
        return value, current, baseline, mean_shift, embedding, record

    def save_state(self, label, pin):
        folder = self.out / ('state_' + label)
        folder.mkdir()
        self.torch.save(dict(model=self.model.state_dict(), optimizer=None,
            CPU_rng=self.torch.get_rng_state(), CUDA_rng=self.torch.cuda.get_rng_state_all(),
            counts=dict(self.counts), protocol_sha256=pin, optimizer_updates=0,
            exact_future_resume_guaranteed=False, failed_recipe_resume_allowed=False), folder / 'full_state.pth')
        write(folder / 'receipt.json', dict(model_state=self.state_hash(self.model), buffer_state=self.buffer_hash(self.model),
            full_state_sha256=sha(folder / 'full_state.pth'), CPU_rng_sha256=array_sha(self.torch.get_rng_state().numpy()),
            CUDA_rng_sha256=[array_sha(v.numpy()) for v in self.torch.cuda.get_rng_state_all()], optimizer_present=False))

    def restore(self):
        torch = self.torch
        cuda_before = torch.cuda.get_rng_state_all()
        rng_before = torch.equal(torch.get_rng_state(), self.CPU_rng) and len(cuda_before) == len(self.CUDA_rng) and all(torch.equal(a, b) for a, b in zip(cuda_before, self.CUDA_rng))
        self.model.load_state_dict(self.initial_model_state, strict=True)
        self.model.eval().requires_grad_(False)
        torch.set_rng_state(self.CPU_rng)
        torch.cuda.set_rng_state_all(self.CUDA_rng)
        assert self.np.array_equal(self.flat(), self.initial)
        assert self.state_hash(self.model) == self.initial_model_hash and self.buffer_hash(self.model) == self.initial_buffer_hash
        assert self.frozen_start == dict(original=self.state_hash(self.original.net), recognizer=self.state_hash(self.identity), provider=self.state_hash(self.provider))
        assert all(v.grad is None for v in self.model.parameters())
        assert torch.equal(torch.get_rng_state(), self.CPU_rng)
        after = torch.cuda.get_rng_state_all()
        assert len(after) == len(self.CUDA_rng) and all(torch.equal(a, b) for a, b in zip(after, self.CUDA_rng))
        return dict(complete=True, initial_model_restored_exact=True, frozen_models_unchanged=True,
                    CPU_CUDA_rng_restored_exact=True, RNG_unchanged_before_restore=rng_before,
                    model_state=self.initial_model_hash, buffer_state=self.initial_buffer_hash,
                    CPU_rng_sha256=array_sha(self.CPU_rng.numpy()), CUDA_rng_sha256=[array_sha(v.numpy()) for v in self.CUDA_rng])
