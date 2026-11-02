"""Additive metadata interface adapter; preserves the completed actual-CLI receipt.
No model/CLI/science test rerun and no archive, executable or prior receipt edit.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
HERE=Path(__file__).resolve().parent
PROTOCOL='cc52d1c5554f7aa0ea3cd61dec9b8348dde37b47cd7ca82640805c4ba658e052'
CHECKS=0

def need(ok,msg):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise ValueError(msg)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def pin(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def main():
 original=HERE/'independent_packet_actual_transfer_CLI_admission_draft003.json';runtime=HERE/'independent_runtime_source_semantics.json'
 need(original.stat().st_size==275697 and sha(original)=='0c69ba7b8b387366cd9eaaf65a055c96fc8d03ea1130371bbbd4f5ea65b839fa','Exact completedactualCLI receipt preserved');s=read(original);r=read(runtime);need(s['complete'] is True and s['independent'] is True and s['protocol']['sha256']==PROTOCOL,'Nested protocol metadata verified before additivejoin');need(s['staged_payloads_all166_exact'] and s['closed_regular_namespace167'] and s['exactold48cases4parity8monitorreferences16labels196controls'],'Actual packet scope remains exact');need(s['actual_unmodified_verify_transfer_CLI']['exit_code']==0 and s['actual_unmodified_verify_transfer_CLI']['printed_receipt']['files']==166 and s['actual_unmodified_verify_transfer_CLI']['printed_receipt']['external_controls_files']==196 and not s['actual_unmodified_verify_transfer_CLI']['forbidden_attempts'],'Actual guardedCLI alreadypassed');need(sha(runtime)=='cb038d0e5746295eba36fe5313ee5a1006bc3634f8a5055523553ccb9df7b354' and r['complete'] and r['independent'],'Exact independent runtimesourceadmission');need(r['source_pins']['cctv_completion_codeformer_spatial_code_intervention_v1_vm.py']['sha256']=='ffe76c1cf2c0329c8b719687be21db4d775081f9df47d4369137a84bd1dab411','Exact finalworker semanticsbinding');need(s['all22_current_R1pins_exact'] and s['original_parent_andcontrols_unchanged'],'Protected incumbent status retained');need(s['checks_passed']==1667 and r['checks_passed']==2988,'Prior actualcheckcounts retained notrerun');need(not s['model_qualified'] and not s['full_goal_complete'],'No qualification or goalcompletion');need(sha(original)=='0c69ba7b8b387366cd9eaaf65a055c96fc8d03ea1130371bbbd4f5ea65b839fa','Priorreceipt bytes still exact atclosure')
 result={'format':'independent-spatial-code-intervention-v1-final-source-packet-admission-interface-v1','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(__file__),'protocol_sha256':PROTOCOL,'protocol':s['protocol'],'actual_CLI_source_packet_admission':pin(original),'independent_runtime_source_semantics':pin(runtime),'metadata_interface_join_checks':CHECKS,'checks':CHECKS,'checks_passed':CHECKS,'underlying_source_packet_checks':1667,'underlying_runtime_semantic_checks':2988,'all166payloads_and_protocol167regular_exact':True,'all196external_controls_48fixedcases_16labels_exact':True,'actual_unmodified_verify_transfer_CLI_exit':0,'prior_receipt_preserved_exact':True,'source_or_CLI_tests_rerun':False,'released_for_root_manual_archive_preparation_only':True,'archive_review_pending':True,'actual_VM_launch_authorized_by_this_metadata':False,'automatic_launch_or_resume':False,'model_qualified':False,'goal_complete':False,'local_actions':{'metadata_reads_and_new_receipt_only':True,'model_loads':0,'neural_calls':0,'gradients':0,'optimizer_updates':0,'VM_calls':0,'old_receipt_or_source_changes':False}}
 target=HERE/'independent_source_packet_admission.json'
 with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(target),'metadata_checks':CHECKS,'no_CLI_rerun':True},indent=2),flush=True)
if __name__=='__main__':main()
