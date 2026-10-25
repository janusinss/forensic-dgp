"""Publish completed local offload only after saved closure and live status checks."""
import hashlib,json,shutil
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/u96_inactive_restoration_offload_20261021_v1'

def pin(p):
 b=p.read_bytes();return {'path':p.relative_to(ROOT).as_posix(),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def main():
 plan=read('plan_v1.json');apply=read('apply_v1.json');audit=read('closure_author_reconstruction_v1.json');closure=read('closure_v1.json')
 live=read('runtime_after_v1.json');folder=read('drive_folder_v1.json')
 assert apply['complete'] and closure['complete'] and audit['author_reconstructed_closure_pass']
 assert live['complete'] and live['status']['preferred_development_model']=='spatial_u'
 assert all(live['status']['models'][a]['available'] for a in ('spatial_u','spatial_s','r2','original'))
 assert folder['ID'] and folder['IsDir'] and folder['Name']=='20261021_u96_inactive_v1'
 total=apply['logical_bytes_removed'];observed=apply['free_after']-apply['free_before'];url='https://drive.google.com/drive/folders/'+folder['ID']
 report=f'''# Local inactive restoration offload completed: 21 October 2026

Only exact files inside `C:\\xampp\\htdocs\\YEAR 4\\Testing` were removed. The VM was not contacted. U96 remains the preferred development restorer; no model, training recipe, split, input threshold or application source was changed by this offload.

| Measurement | Value |
|---|---:|
| Old transfer/result archives removed | {len(plan['archives']):,} |
| Historical RAW/checkpoint files removed | {len(plan['members']):,} |
| Logical bytes removed | {total:,} ({total/1024**3:.3f} GiB) |
| Observed free-space increase during removal | {observed:,} bytes ({observed/1024**3:.3f} GiB) |
| Free immediately after removal | {apply['free_after']:,} bytes ({apply['free_after']/1024**3:.3f} GiB) |

The newly uploaded archives were fully downloaded through SHA256 verification and matched to unique Drive objects with matching server SHA256/size. The other {len(plan['members']):,} files are exact members of ten existing 19 October backups: each full compressed archive was streamed and hashed again, and every selected uncompressed member matched its local SHA256 and size. Fresh old-archive object hashes/IDs were checked before apply. No archive was extracted over the workspace. Local removal used exact planned regular files, full final SHA256 and exclusive native handles; no recursive directory deletion was used.

The final research-file namespace readback (excluding volatile Git internals) found {audit['retained_namespace_files']:,} retained files with zero unexpected missing paths or retained metadata changes. All {audit['retained_model_hashes']:,} retained model hashes match. The independent-from-apply saved reconstruction was authored by the same root assistant; no independent agent or human review is claimed.

An initial zero-deletion guard stop detected a concurrent manuscript publication. The whole preceding handoff remains an exact prefix, and112 changed document/render files were hash-bound and retained using a fresh research metadata baseline. Volatile Git capture metadata is recorded separately and excluded from research-file comparison. The storage candidate list, all payload hashes and every model preservation requirement remain unchanged. Original failed-check evidence is retained.

## Kept locally

U96's exact frozen R2 base, adapter, resumable U96 checkpoint, final R2 training checkpoint and required verification records remain. Original/R2/U96/S96 comparison exports and the objective DEV comparison remain available. Reusable datasets, caches, source code, PNG galleries, metrics, provenance, splits and gate/failure records remain local. Mask-removal and completion assets remain untouched. Not every non-U96 file was removed: the user explicitly selected retention of reusable data, code and comparisons.

Earlier historical run checkpoint/RAW payloads now require the recorded restore step before replaying an audit that reads them. Their source, metadata and failure status have not been rewritten as successful. Archive offload does not qualify a model or complete the thesis goal. Inference, gradient, optimizer, training and VM calls by this operation: zero.

## Drive location and recovery

[Open the new Google Drive archive]({url}). New object root: `{plan['destination']}`. Ten older source archives remain under `thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261019_v1/outputs/` with exact archive IDs/SHA256 in `plan_v1.json`. The portable `restore_map_v1.csv` joins every removed local path to its remote archive, exact optional member path, size and SHA256.

To restore a transferred archive, use the existing private rclone profile with `copyto` from the exact `remote_object` in that CSV to its original path; check its SHA256 before use. For an archived member, download its `remote_object` to a separate recovery directory, verify the full compressed archive hash from `plan_v1.json`, extract only the listed member into that recovery directory and verify its member SHA256 before placing it at `local_path`. Do not bulk-extract an old archive over current U96 work. The private OAuth token/config was never copied into evidence or Drive.

Evidence: `plan_v1.json`, `plan_author_readback_v1.json`, `combined_v1.txt`, per-run `old_run_*_verified.json`, fresh remote inventories, `deletion_journal_v1.jsonl`, `apply_v1.json`, `closure_v1.json`, `closure_author_reconstruction_v1.json` and before/after namespace/model inventories. Counts exclude all previous offloads. Other disk activity can change later free-space readings.
'''
 target=OUT/'STORAGE_OFFLOAD_REPORT.md'
 with target.open('x',encoding='utf-8') as f:f.write(report)
 entry=f'''# 21 October 2026: local inactive restoration payloads offloaded; U96 retained

The user selected LOCAL Testing storage only and archival of inactive models/bulky old outputs while keeping reusable data, code and comparisons. Exact hash-bound removal freed {total:,} logical bytes ({total/1024**3:.3f} GiB): {len(plan['archives'])} old transfer/result archives plus {len(plan['members']):,} historical RAW/checkpoint copies. Newly uploaded archives passed full downloaded SHA256 verification; selected files in ten existing Drive backups passed full archive and exact per-member verification. Fresh Drive IDs/hash/size and exclusive local final hashes preceded removal. The VM was not contacted. [Offload report and recovery index](<C:/xampp/htdocs/YEAR 4/Testing/outputs/u96_inactive_restoration_offload_20261021_v1/STORAGE_OFFLOAD_REPORT.md>).

U96 base/adapter/resume state and verification records remain local, along with final R2 state, original/R2/U96/S96 comparisons, reusable datasets/caches/code, PNGs/metrics, provenance/splits/failure records and all mask-removal/completion assets. Historical offloaded checkpoint/RAW paths require their portable restore-map lookup before replay. No current model source, weights, app default, training recipe or quality requirement changed. Retained research namespace/model readbacks found no unexpected missing or changed files against the refreshed baseline. The concurrent112-file manuscript publication and complete handoff prefix were preserved; volatile Git internals were recorded separately. The initial zero-deletion guard stop remains in the evidence. U96 and all four comparison choices remain available after maintenance; no new inference, optimizer, training or VM call ran. Same-author saved reconstruction passed; no independent agent/human review is claimed.

The immediate goal remains improving U96's visible restoration while preserving original U96/R2 and declared benchmarks. Completion and removal-mask development stay deferred until restoration is satisfactory. Actual training remains finite manual L4/tmux work with verified transfers, independently audited returns and no unchanged failed recipes. Storage completion does not establish model usefulness, native identity accuracy, independent FINAL review or full thesis completion. Earlier dated entries remain retained history.

---

'''.encode('utf-8')
 publication=OUT/'publication_v1';publication.mkdir(exist_ok=False)
 documents={}
 for name in ('PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_CODEFORMER_FINETUNING_PLAN_V1.md'):
  p=ROOT/name;old=p.read_bytes();backup=publication/(name+'.before')
  with backup.open('xb') as f:f.write(old)
  p.write_bytes(entry+old);assert p.read_bytes()[len(entry):]==old
  documents[name]={'before':pin(backup),'current':pin(p),'preserved_suffix_offset':len(entry),'full_previous_bytes_preserved':True}
 names=['plan_v1.json','plan_author_readback_v1.json','apply_v1.json','closure_v1.json','closure_author_reconstruction_v1.json',
  'restore_map_v1.csv','STORAGE_OFFLOAD_REPORT.md','runtime_before_v1.json','runtime_after_v1.json','drive_folder_v1.json','old_remote_preapply_verified_v1.json',
  'concurrent_publication_recovery_plan_v1.json','apply_zero_deletion_stop_v1.json','old_remote_reconciled_verified_v1.json']
 evidence={n:pin(OUT/n) for n in names}
 evidence.update({p.name:pin(p) for p in sorted(OUT.glob('old_run_*_verified.json'))})
 result={'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),'documents':documents,'evidence':evidence,'drive_URL':url,
  'logical_bytes_removed':total,'observed_free_bytes_increase':observed,'free_after':apply['free_after'],
  'preferred_development_model':'spatial_u','VM_training_neural_calls':0,'independent_agent_review':False,'overall_goal_complete':False}
 with (publication/'publication_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 for row in documents.values():assert pin(ROOT/row['current']['path'])==row['current']
 for row in evidence.values():assert pin(ROOT/row['path'])==row
 print(json.dumps({'complete':True,'logical_GiB':total/1024**3,'free_GiB':apply['free_after']/1024**3,'drive_URL':url,'publication':pin(publication/'publication_v1.json')}))

if __name__=='__main__':main()
