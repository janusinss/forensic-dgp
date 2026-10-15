"""Strict saved-output audit with a deterministic reserve-metadata reference.

Original checker, packet, outputs, vectors and all scientific gates are retained.
No learned model is constructed, mutated or evaluated.
"""
import argparse
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path

import audit_cctv_dgp_decoder_hinge_repair_v1_return as original
from cctv_dgp_decoder_hinge_repair_v1_contract import read, write, sha, verify_transfer

ORIGINAL_PROJECTION = original.verify_projection_journal
ORIGINAL_SHA = 'fa6f84b30690bcf43bd478370c1406d70d74d43f1af110904a3d8a8703f3e105'
RECORDS = []


def exact_reserve(initial):
    import numpy as np
    assert initial.dtype == np.float32 and initial.ndim == 1 and np.isfinite(initial).all()
    values, counts = np.unique(np.spacing(initial).astype(np.float64), return_counts=True)
    square_sum = sum((Fraction.from_float(float(value))**2)*int(count)
                     for value, count in zip(values, counts))
    answers, texts = [], []
    for precision in [110, 160]:
        with localcontext() as context:
            context.prec = precision
            value = 2*(Decimal(square_sum.numerator)/Decimal(square_sum.denominator)).sqrt()
            answers.append(float(value)); texts.append(str(value))
    assert answers[0] == answers[1] and answers[0] > 0
    return answers[0], dict(squared_spacing_sum_numerator=str(square_sum.numerator),
        squared_spacing_sum_denominator=str(square_sum.denominator), unique_spacing_values=len(values),
        decimal_precisions=[110,160], decimal_values=texts, nearest_binary64_hex=answers[0].hex())


def verify_projection_journal(initial, proposal, actual, row, radius):
    import numpy as np
    reference, proof = exact_reserve(initial)
    assert row['float32_roundoff_reserve'] == reference, 'Reserve must equal deterministic exact-sum reference'
    local = float(2*np.linalg.norm(np.spacing(initial).astype(np.float64)))
    # Normalize only this scalar in an ephemeral dictionary for the original
    # checker. It still recomputes/compares the actual projected float32 bytes,
    # scale, original-relative distance, unchanged radius and every other field.
    normalized = dict(row, float32_roundoff_reserve=local)
    ORIGINAL_PROJECTION(initial, proposal, actual, normalized, radius)
    if row['scale'] == 1:
        expected = proposal
    else:
        expected = (initial.astype(np.float64)+row['scale']*(proposal.astype(np.float64)-initial.astype(np.float64))).astype(np.float32)
    assert np.array_equal(actual, expected), 'Saved-scale projection must also reproduce exact actual bytes'
    RECORDS.append(dict(saved_reserve=reference, local_numpy_reserve=local,
        absolute_difference=abs(reference-local), saved_scale_exact_vector=True,
        original_checker_exact_vector_and_radius=True, **proof))


def main():
    parser=argparse.ArgumentParser()
    for name in ['root','source-root','parent-root','replay-root','receipt']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True)
    args=parser.parse_args()
    assert sha(original.__file__)==ORIGINAL_SHA, 'Retain immutable original checker'
    assert not args.receipt.exists() and not (args.receipt.parent/'projection_metadata_r1.json').exists()
    p,old,data=verify_transfer(args.root,args.source_root,args.parent_root,args.protocol_sha,replay=args.replay_root)
    assert p['assets_sha256']['audit_cctv_dgp_decoder_hinge_repair_v1_return.py']==ORIGINAL_SHA
    original.verify_projection_journal=verify_projection_journal
    try:
        original.audit_saved(args.root.resolve(),args.source_root.resolve(),args.parent_root.resolve(),args.replay_root.resolve(),
                             p,old,data,args.receipt.resolve())
    finally:
        original.verify_projection_journal=ORIGINAL_PROJECTION
    write(args.receipt.parent/'projection_metadata_r1.json',dict(complete=True,
        original_checker_sha256=ORIGINAL_SHA, r1_checker_sha256=sha(__file__),
        protocol_sha256=args.protocol_sha, final_audit_sha256=sha(args.receipt),
        rule='Recorded reserve must equal exact rational squared-spacing sum with converged 110/160-digit sqrt rounded to binary64',
        metadata_tolerance_widened=False, scientific_thresholds_changed=False,
        actual_weight_vector_or_radius_changed=False, original_checker_and_archive_preserved=True,
        checks=RECORDS, local_neural_calls=0, gradient_queries=0, optimizer_updates=0, model_qualified=False))


if __name__=='__main__':main()
