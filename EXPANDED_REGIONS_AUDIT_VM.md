# Existing-checkpoint region audit — inference only

Upload `C:\xampp\htdocs\YEAR 4\Testing\scripts\audit_expanded_regions_vm.py`
using Google Cloud SSH's Upload File button. Run:

```bash
cd ~/forensic-dgp/expanded_feature_bundle &&
source ../feature_vm_bundle/.venv/bin/activate &&
python -u ~/audit_expanded_regions_vm.py --root "$PWD"
```

Requires the original bundle, both completed caches, final heads and completed
`outputs/expanded_fit_audit/results.json`. No encoder extraction, optimizer,
training, threshold search or validation/test inference occurs. No git pull is
needed for this standalone uploaded script. Preserve the original caches.

The audit checks inventory, checkpoint/protocol hashes, source/image hashes,
regenerated target equality and exact per-case counts from the previous audit.
It preserves the original batch layout. It examines all 704 irregular training
cases per arm, separates original core, added border and outside target, and
exports binary raw/gated/core/target masks plus the first six degraded inputs.
Local region-count and CPU-refusal tests pass (2 tests); VM execution is pending.
The output directory must be new; a mismatch stops for inspection, not retraining.

After DONE, download:

```text
/home/janusdominic0/forensic-dgp/expanded_feature_bundle/expanded-regions-audit-results.tar.gz
```

Save to `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-regions-audit-results.tar.gz`.
Next: independently recount exported masks, inspect matching previews, and use
core-versus-border errors to justify the next bounded VM experiment. This audit
alone cannot qualify a candidate for deployment or justify changing the benchmark.
