"""Terminal-only checks using the byte-exact inherited gate-aligned helper.

The caller supplies the complete unchanged inherited bank result. This module
adds no tolerance, selects no fallback and performs no model or worker action.
"""
import hashlib
import importlib
from pathlib import Path

ROLES = ('local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35')
VIEWS = ('raw', 'native_png')
FIELDS = ('mae', 'adjacent_rgb_error')
INHERITED_SOURCE = 'research/codeformer_u64_gate_aligned_pilot_v1/quality.py'
INHERITED_BYTES = 5124
INHERITED_SHA256 = '2b4b15e99a28d0f45667a15a48983db61c032aa324ed77fbd28c0afd778c739e'
ROOT = Path(__file__).resolve().parents[2]


def need(value, message):
    if not value:
        raise ValueError(message)


def inherited_quality():
    """Authenticate the inert payload before importing or invoking it."""
    source = ROOT / INHERITED_SOURCE
    need(source.resolve(strict=True) == source.absolute(), 'Exact unlinked inherited quality source required')
    need(source.stat().st_size == INHERITED_BYTES, 'Exact inherited quality source size required')
    raw = source.read_bytes()
    need(len(raw) == INHERITED_BYTES and hashlib.sha256(raw).hexdigest() == INHERITED_SHA256,
         'Exact unchanged inherited quality source bytes required')
    module = importlib.import_module('research.codeformer_u64_gate_aligned_pilot_v1.quality')
    need(Path(module.__file__).resolve() == source, 'Inherited quality import origin differs')
    need(module.ROLES == ROLES and module.VIEWS == VIEWS and module.FIELDS == FIELDS,
         'Exact inherited photographic metric schema required')
    return module


def evaluate_quality(current, original, prior, terminal, inherited_gate_result):
    """Bind every original gate at the sole selected-step final quality bank.

    The finite candidate selector must already have finished using optimization
    cases only. The worker passes original for both original and prior because
    there is only one selected final update. A false result rejects the entire
    test; this function neither selects another scale nor authorizes a retry.
    """
    need(type(terminal) is bool and terminal,
         'The selected-step quality bank must bind all terminal requirements')
    return inherited_quality().evaluate_quality(current, original, prior, True, inherited_gate_result)
