# VM availability — 8 October 2026

The live Google Cloud API reports `forensic-dgp-thesis` in `us-central1-a`
as **TERMINATED**, with machine type `g2-standard-4`. This means stopped;
it explains why a guest disk/GPU/tmux inventory cannot complete. The agent
performed no VM start, deletion, diagnostic or training operation.
[Google Cloud instance lifecycle](https://docs.cloud.google.com/compute/docs/instances/instance-lifecycle)

The first maintenance attempt stopped at local TLS certificate validation.
The bounded retry reused the existing Windows CA bundle and pinned historical
SSH host key, with TLS verification enabled. SSH then closed through IAP.
A separate read-only API query succeeded and confirms the stopped state.
Both original failures are retained. Current free disk, GPU workload, venv and
guest research-file integrity remain unverified; no cleanup is claimed.

When ready to use the VM, run this from **Windows Google Cloud SDK Shell**:

```bat
set "CLOUDSDK_CORE_CUSTOM_CA_CERTS_FILE=C:\xampp\htdocs\YEAR 4\Testing\scratch\gcloud_windows_trust.pem"
gcloud compute instances start forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a
```

The CA setting affects this SDK Shell session and uses the same existing trust
file as successful prior maintenance; certificate verification stays enabled.
The start command is provided for manual execution and has not been run here.
[Google Cloud start-command reference](https://docs.cloud.google.com/sdk/gcloud/reference/compute/instances/start)

Confirm its state from the same SDK Shell:

```bat
gcloud compute instances describe forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a --format="value(status)"
```

Expected: `RUNNING`. Open the existing VM SSH terminal, then follow the
[verified five manual diagnostic steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM.md>).
Those steps include single-source gcloud uploads/downloads and a distinct tmux
session. The immutable packet and protocol remain unchanged. Its worker checks
the existing L4, idle GPU and **2GiB free** before any derivative query. The
diagnostic has280 queries, zero optimizer updates, a600s worker limit and finite
export/VRAM/size limits. A pass measures learning signals, not useful restoration.

The packet audit and full history closure already pass. V40's1% structure gate
remains failed, original DGP stays primary, and native restoration/covering-family
quality and independent final review remain outstanding. Goal active/incomplete.

[Live instance-status evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v40_learning_signal_v1_vm_readiness/instance_status.json>)
[Retained maintenance transport failure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v40_learning_signal_v1_vm_readiness/transport.json>)
[Independent prepared-packet/history closure](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v40_learning_signal_v1_milestone/independent_closure_audit.json>)
