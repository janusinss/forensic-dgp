"""Audited photographic pairs for the finite CodeFormer adaptation.

Pure CPU image preparation: this module imports no neural framework. The HQ
reference is derived from an approved RGB 1024 FFHQ source, never a native CCTV
crop or a merely upscaled old target. Historical TRAIN/validation roles remain
separate; these roles do not establish CodeFormer pretraining non-overlap.

The blur/lowlight/compound bounds repeat the frozen ranges in
``scripts/prepare_cctv_dgp_vm.py:jitter``. The light/JPEG bounds reuse the
camera-only ranges in ``completion_data.py`` (64--256 pixels, sigma .5--1.5,
RGB noise 1--12/255, JPEG 35--95). Exposure/shot/read behavior follows
``scripts/cctv_camera_stress.py``. These are synthetic stress proxies, not a
calibrated or statistically fitted native CCTV distribution. No crop, geometric
warp, covering, glasses removal, facial normalization or completion is applied.
Scientific acceptance criteria are deliberately outside this data module.
"""
from dataclasses import dataclass
import hashlib
import json
from numbers import Integral
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION


PROFILES = ('clear', 'light', 'blur', 'jpeg', 'lowlight', 'compound')
_BASE = 'outputs/codeformer_training_readiness_v1/data_review/'
_MANIFEST_SPECS = {
    'train': (_BASE + 'hq_train_candidates_v1.json',
              '7af5f313743361ae435629171c9e92e2a4fbcf119bdcac8221338a6fd457a493', 391),
    'all': (_BASE + 'ffhq_sources_with_roles_v1.json',
            '4b77dc04c3da7520caf2e2fca5836727d9183e539d4fdc1817e06df50ab7b73c', 510),
}
_VALIDATION_COUNT = 53
_ALLOWED_SOURCE_ROOTS = (
    'outputs/cctv_dgp_hq_cohort_v2/images1024/',
    'outputs/cctv_dgp_hq_counterparts_v1/images1024/',
    'data/sources/',
)


def file_sha256(path: Path) -> str:
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def pixel_sha256(array: np.ndarray) -> str:
    """SHA256 of contiguous row-major bytes; record dtype/shape alongside it."""
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass(frozen=True)
class Reference:
    id: str
    role: str
    workspace: Path
    source_relative_path: str
    source_sha256: str
    clean512_rgb8_sha256: str
    manifest_sha256: str
    attribution_json: str

    def __post_init__(self) -> None:
        prefix = {'train': 'tr_ffhq_', 'validation': 'va_ffhq_'}.get(self.role)
        if prefix is None or not self.id.startswith(prefix):
            raise ValueError('Only audited FFHQ TRAIN/validation references are permitted')
        if (not self.source_relative_path.startswith(_ALLOWED_SOURCE_ROOTS)
                or '\\' in self.source_relative_path
                or '..' in Path(self.source_relative_path).parts):
            raise ValueError('Not an approved FFHQ 1024 source location')
        for digest in (self.source_sha256, self.clean512_rgb8_sha256, self.manifest_sha256):
            if (not isinstance(digest, str) or len(digest) != 64
                    or any(c not in '0123456789abcdef' for c in digest)):
                raise ValueError('Expected an exact lowercase SHA256 fingerprint')

    @property
    def source_path(self) -> Path:
        root = self.workspace.resolve()
        candidate = root / self.source_relative_path
        if candidate.is_symlink():
            raise ValueError('Source symlinks are not approved inputs')
        path = candidate.resolve()
        if not path.is_relative_to(root):
            raise ValueError('Source escapes the verified workspace')
        return path

    @property
    def attribution(self) -> dict[str, Any]:
        return json.loads(self.attribution_json)


@dataclass(frozen=True)
class PairedExample:
    input256: np.ndarray
    clean512: np.ndarray
    record: dict[str, Any]

    @property
    def target512(self) -> np.ndarray:
        return self.clean512

    @property
    def parameters(self) -> dict[str, Any]:
        return self.record['parameters']


def _read_bound_manifest(workspace: Path, name: str) -> tuple[dict[str, Any], str]:
    relative, expected, _ = _MANIFEST_SPECS[name]
    path = workspace / relative
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != expected:
        raise ValueError('Manifest SHA256 differs: ' + relative)
    return json.loads(payload), actual


def _reference(row: dict[str, Any], workspace: Path, manifest_sha: str) -> Reference:
    if (row.get('decision') != 'accept_clean_restoration'
            or row.get('source') != 'FFHQ official aligned 1024 counterpart'
            or row.get('native_size') != [1024, 1024]):
        raise ValueError('Source is not an accepted genuine 1024 FFHQ reference')
    return Reference(
        id=row['id'], role=row['historical_role'], workspace=workspace,
        source_relative_path=row['source_path'], source_sha256=row['source_sha256'],
        clean512_rgb8_sha256=row['proposed_512_rgb_sha256'],
        manifest_sha256=manifest_sha,
        attribution_json=json.dumps(row['attribution'], sort_keys=True, separators=(',', ':')),
    )


def load_reference_manifest(workspace: Path | str, role: str = 'train',
                            manifest_path: Path | str | None = None, *,
                            expected_manifest_sha256: str | None = None) -> tuple[Reference, ...]:
    """Load exact 391 TRAIN or 53 accepted validation references, fail closed.

    The manifests are immutable audit outputs, not caller-supplied wildcard
    datasets. A portable packet manifest must be explicitly pinned by its frozen
    protocol. Native CCTV and reserved-final namespaces are never accepted as
    GT. Source-file bytes are checked whenever a pair is materialized.
    """
    if role not in ('train', 'validation'):
        raise ValueError('Role must be train or validation; native/final GT is forbidden')
    workspace = Path(workspace).resolve()
    if manifest_path is None:
        if expected_manifest_sha256 is not None:
            raise ValueError('Explicit fingerprint requires an explicit packet manifest')
        train, train_sha = _read_bound_manifest(workspace, 'train')
        ledger, ledger_sha = _read_bound_manifest(workspace, 'all')
        candidates, sources = train['candidates'], ledger['sources']
        if len(candidates) != _MANIFEST_SPECS['train'][2] or len(sources) != _MANIFEST_SPECS['all'][2]:
            raise ValueError('Frozen source counts differ')
        accepted = [row for row in sources if row['decision'] == 'accept_clean_restoration']
        canonical_train = [row for row in accepted if row['historical_role'] == 'train']
        if candidates != canonical_train:
            raise ValueError('TRAIN selection differs from the complete source ledger')
    else:
        path = Path(manifest_path)
        path = (workspace / path).resolve() if not path.is_absolute() else path.resolve()
        if not path.is_relative_to(workspace):
            raise ValueError('Packet reference manifest escapes the packet root')
        if not isinstance(expected_manifest_sha256, str) or len(expected_manifest_sha256) != 64:
            raise ValueError('Packet manifest requires its protocol-bound SHA256')
        payload = path.read_bytes()
        train_sha = ledger_sha = hashlib.sha256(payload).hexdigest()
        if train_sha != expected_manifest_sha256:
            raise ValueError('Packet reference manifest SHA256 differs')
        packet = json.loads(payload)
        if packet.get('format') != 'codeformer-adaptation-reference-manifest-v1':
            raise ValueError('Unknown portable reference manifest format')
        accepted = packet['references']
        if any(row.get('decision') != 'accept_clean_restoration' for row in accepted):
            raise ValueError('Packet contains an unapproved clean reference')
        candidates = [row for row in accepted if row['historical_role'] == 'train']
    validation = [row for row in accepted if row['historical_role'] == 'validation']
    if len(candidates) != _MANIFEST_SPECS['train'][2] or len(validation) != _VALIDATION_COUNT:
        raise ValueError('TRAIN or accepted validation counts differ')
    if any(row['historical_role'] not in ('train', 'validation') for row in accepted):
        raise ValueError('Reserved/native/unknown role present in clean targets')
    if any(row.get('suitable_as_hq_train_candidate') is not True for row in candidates):
        raise ValueError('An unapproved source entered TRAIN')
    all_refs = tuple(_reference(row, workspace, ledger_sha) for row in accepted)
    if len({r.id for r in all_refs}) != len(all_refs):
        raise ValueError('Duplicate reference IDs')
    train_refs = [r for r in all_refs if r.role == 'train']
    validation_refs = [r for r in all_refs if r.role == 'validation']
    for field in ('source_relative_path', 'source_sha256'):
        if ({getattr(r, field) for r in train_refs}
                & {getattr(r, field) for r in validation_refs}):
            raise ValueError('Exact TRAIN/validation source overlap')
    selected, pin = (candidates, train_sha) if role == 'train' else (validation, ledger_sha)
    return tuple(_reference(row, workspace, pin) for row in selected)


def _verified_derivations(reference: Reference) -> tuple[np.ndarray, np.ndarray]:
    path = reference.source_path
    if not path.is_file() or path.is_symlink():
        raise ValueError('Missing/nonregular approved source: ' + str(path))
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != reference.source_sha256:
        raise ValueError('Source SHA256 differs: ' + reference.id)
    # Decode the same verified bytes, avoiding a hash/open time-of-check gap.
    from io import BytesIO
    with Image.open(BytesIO(payload)) as source:
        if source.size != (1024, 1024) or source.mode != 'RGB':
            raise ValueError('Expected an original RGB 1024x1024 source')
        clean512 = np.asarray(source.resize((512, 512), Image.Resampling.LANCZOS)).copy()
        input256 = np.asarray(source.resize((256, 256), Image.Resampling.BILINEAR)).copy()
    if pixel_sha256(clean512) != reference.clean512_rgb8_sha256:
        raise ValueError('Canonical 1024-to-512 target pixels differ: ' + reference.id)
    return input256, clean512


def derive_clean_target(reference: Reference) -> np.ndarray:
    """Verified direct RGB 1024→512 LANCZOS, HWC float32 in [0,1]."""
    _, clean512 = _verified_derivations(reference)
    return np.ascontiguousarray(clean512.astype(np.float32) / np.float32(255))


def profile_parameters(profile: str, seed: int) -> dict[str, Any]:
    """Draw one finite recipe; same seed/profile/version gives the same recipe."""
    if profile not in PROFILES:
        raise ValueError('Unknown degradation profile: ' + str(profile))
    if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= int(seed) < 2**64:
        raise ValueError('Seed must be an integer in [0,2**64)')
    rng = np.random.default_rng(int(seed))
    p = {'profile': profile, 'seed': int(seed), 'gaussian_sigma_256': 0.,
         'motion_kernel_size_256': 0, 'motion_direction': 'horizontal',
         'long_edge': 256, 'exposure_linear': 1., 'shot_coefficient': 0.,
         'read_sigma_linear': 0., 'rgb_noise_sigma_255': 0., 'jpeg_quality': None}
    if profile == 'light':
        p.update(gaussian_sigma_256=float(rng.uniform(.5, 1.5)),
                 long_edge=int(rng.integers(64, 257)),
                 rgb_noise_sigma_255=float(rng.uniform(1, 12)),
                 jpeg_quality=int(rng.integers(35, 96)))
    elif profile == 'blur':
        p.update(long_edge=int(rng.integers(24, 33)), gaussian_sigma_256=float(rng.uniform(.6, 2)),
                 jpeg_quality=int(rng.integers(35, 66)))
    elif profile == 'jpeg':
        p.update(jpeg_quality=int(rng.integers(35, 96)))
    elif profile == 'lowlight':
        p.update(long_edge=int(rng.integers(32, 65)), exposure_linear=float(rng.uniform(.2, .65)),
                 gaussian_sigma_256=float(rng.uniform(.5, 1.4)),
                 shot_coefficient=float(rng.uniform(.0004, .0012)),
                 read_sigma_linear=float(rng.uniform(.001, .003)),
                 jpeg_quality=int(rng.integers(30, 61)))
    elif profile == 'compound':
        p.update(long_edge=int(rng.integers(24, 41)), exposure_linear=float(rng.uniform(.15, .4)),
                 gaussian_sigma_256=float(rng.uniform(.7, 2)),
                 motion_kernel_size_256=int(rng.choice([3, 5, 7])),
                 shot_coefficient=float(rng.uniform(.0006, .0018)),
                 read_sigma_linear=float(rng.uniform(.002, .004)),
                 jpeg_quality=int(rng.integers(20, 51)))
    # A declared loss-eligibility policy, not a quality/acceptance criterion:
    # severe generated 24--31px observations remain available to code/feature
    # supervision, while the worker omits their direct image reconstruction loss.
    p['image_eligible'] = p['long_edge'] >= 32
    p['image_eligibility_policy'] = 'generated observation minimum edge >=32; no other exclusion'
    return p


def _degrade(rgb8: np.ndarray, p: dict[str, Any]) -> np.ndarray:
    if p['profile'] == 'clear':
        return rgb8.copy()
    rgb = rgb8.astype(np.float32) / np.float32(255)
    if p['gaussian_sigma_256']:
        rgb = cv2.GaussianBlur(rgb, (0, 0), p['gaussian_sigma_256'], borderType=cv2.BORDER_REPLICATE)
    if p['motion_kernel_size_256']:
        k = p['motion_kernel_size_256']
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2] = 1 / k
        rgb = cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REPLICATE)
    edge = p['long_edge']
    if edge != 256:
        rgb = cv2.resize(rgb, (edge, edge), interpolation=cv2.INTER_AREA)
    # Independent local RNG: no mutation of global NumPy or worker RNG state.
    rng = np.random.default_rng(p['seed'] ^ 0xC0DEF012512)
    if p['exposure_linear'] != 1 or p['shot_coefficient'] or p['read_sigma_linear']:
        linear = np.where(rgb <= .04045, rgb / 12.92, ((rgb + .055) / 1.055)**2.4)
        linear *= p['exposure_linear']
        deviation = np.sqrt(p['shot_coefficient'] * linear + p['read_sigma_linear']**2)
        linear = np.clip(linear + rng.normal(0, 1, linear.shape) * deviation, 0, 1)
        rgb = np.where(linear <= .0031308, linear * 12.92, 1.055 * linear**(1 / 2.4) - .055)
    if p['rgb_noise_sigma_255']:
        rgb = rgb + rng.normal(0, p['rgb_noise_sigma_255'] / 255, rgb.shape)
    low8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    if p['jpeg_quality'] is not None:
        ok, encoded = cv2.imencode('.jpg', cv2.cvtColor(low8, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, p['jpeg_quality']])
        if not ok:
            raise ValueError('JPEG proxy encoding failed')
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape != low8.shape:
            raise ValueError('JPEG proxy decode failed or changed geometry')
        low8 = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    return cv2.resize(low8, (256, 256), interpolation=cv2.INTER_CUBIC) if edge != 256 else low8


def make_pair(reference: Reference, profile: str, seed: int) -> PairedExample:
    """Return prepared256 input, direct clean512 target and reproducible record.

    Both arrays are HWC RGB float32 [0,1]. The training worker must reproduce the
    benchmark's torch bilinear256→512/normalization operation when feeding the
    network; this module does not silently replace it with a PIL/OpenCV resize.
    Seed variation only alters degradations, never the target, crop or role.
    """
    parameters = profile_parameters(profile, seed)
    base256, clean512_rgb8 = _verified_derivations(reference)
    input256_rgb8 = _degrade(base256, parameters)
    input256 = np.ascontiguousarray(input256_rgb8.astype(np.float32) / np.float32(255))
    clean512 = np.ascontiguousarray(clean512_rgb8.astype(np.float32) / np.float32(255))
    if (input256.shape != (256, 256, 3) or clean512.shape != (512, 512, 3)
            or not np.isfinite(input256).all() or not np.isfinite(clean512).all()
            or input256.min() < 0 or input256.max() > 1):
        raise ValueError('Invalid generated pair; no fallback permitted')
    record = {'format': 'codeformer-adaptation-photographic-pair-v1',
        'reference_id': reference.id, 'role': reference.role,
        'source_path': reference.source_relative_path, 'source_sha256': reference.source_sha256,
        'reference_manifest_sha256': reference.manifest_sha256,
        'profile': profile, 'seed': int(seed), 'parameters': parameters,
        'input256_derivation': 'Verified RGB1024→Pillow BILINEAR256, then recorded camera stress',
        'clean512_derivation': 'Verified RGB1024→Pillow LANCZOS512',
        'input256_rgb8_sha256': pixel_sha256(input256_rgb8),
        'input256_float32_sha256': pixel_sha256(input256),
        'clean512_rgb8_sha256': pixel_sha256(clean512_rgb8),
        'clean512_float32_sha256': pixel_sha256(clean512),
        'array_layout': 'HWC RGB float32[0,1], C-contiguous',
        'geometry': {'source': [1024, 1024], 'input': [256, 256], 'target': [512, 512],
                     'crop': None, 'warp': None, 'covering_simulation': False},
        'runtime': {'numpy': np.__version__, 'pillow': PILLOW_VERSION, 'opencv': cv2.__version__},
        'evidence': 'synthetic paired photographic adaptation; not native CCTV ground truth',
        'acceptance_thresholds_defined_here': False}
    return PairedExample(input256=input256, clean512=clean512, record=record)
