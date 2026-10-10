# V42 live storage preflight — 9 October 2026

The existing VM has sufficient free space for the prepared V42 study. The fresh
read-only inventory finds the manual V42 launcher and worker active in tmux.
No deletion is needed or performed. No process was launched, interrupted,
restarted or stopped by this maintenance check.

The API confirms instance `forensic-dgp-thesis`, ID `4410777042005672095`,
project `forensic-dgp-thesis`, zone `us-central1-a`, status RUNNING and machine
`g2-standard-4`. The guest confirms the expected home/repository and NVIDIA L4.
TLS verification and the previously verified host key remain enabled. The
read-only source is saved locally; no inventory script was uploaded to the VM.

At observation Unix time `1791546901.766662`, the repository filesystem has
21,663,109,120 free bytes, **20.1753 GiB**. That exceeds V42's required
8,589,934,592 bytes, **8 GiB after installation**. Future free space is not
guaranteed; the worker's storage projection, reserve and export checks remain
necessary.

The observed tmux session is `dgp_residual_epochs_v42`. Worker PID 1324 is a
live GPU process and is running the expected `--run` command under the finite
timeout. The remote protocol matches the prepared SHA256
`c19ca1790ae9ebe7f99000a81678c8fc756d70ce2114debb6f81691c741a6f2d`.
The trainer log exists; terminal result/failure and supervisor receipts are
absent at this observation. This proves an active process, not completed
optimizer updates, successful gates or useful image quality.

The original checkpoints, split files, caches, failed gates, source code and
active work remain untouched. Inventory counters of zero model/gradient calls
describe this inspection, not the separate manual worker. The maintenance
connection does not authorize automatic training. Keep the run and its finite
stop rules intact; independently audit returned files before deciding on another
recipe or app adoption.

Artifacts are in `outputs/cctv_dgp_v42_vm_storage_preflight_v1/`: API and guest
inventories, transport logs/receipts, the saved read-only source and availability
record. The inspector is `scripts/inspect_cctv_dgp_v42_vm_storage_v1.py`.

The full restoration, native development, final review and seven-family
automatic/assisted completion goal remains active and incomplete.
