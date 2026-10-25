"""Run Stage 1 Saved-Array Feasibility Proof for Conditioned SFT V1."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.codeformer_conditioned_sft_learning_v1 import feasibility


def main():
    output_dir = ROOT / 'outputs' / 'codeformer_conditioned_sft_learning_v1_feasibility'
    receipt = feasibility.run_saved_array_feasibility_proof(output_dir=output_dir)
    review_md = feasibility.render_feasibility_review_markdown(receipt)
    review_path = ROOT / 'CCTV_CODEFORMER_CONDITIONED_SFT_FEASIBILITY_V1_REVIEW.md'
    review_path.write_text(review_md, encoding='utf-8')
    print(f"STATUS={receipt['status']}")
    print(f"RECEIPT={output_dir / 'feasibility_receipt_v1.json'}")
    print(f"REVIEW={review_path}")


if __name__ == '__main__':
    main()
