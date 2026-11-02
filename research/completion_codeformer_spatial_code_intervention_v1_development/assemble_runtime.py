from pathlib import Path
import ast
D=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION\research\completion_codeformer_spatial_code_intervention_v1_development')
p=D/'cctv_completion_codeformer_spatial_code_intervention_v1_vm.py';s=p.read_text(encoding='utf-8-sig')
s=s.replace('VERIFY_REPLACEMENT',(D/'verify_block.txt').read_text(encoding='utf-8-sig'))
s=s.replace("p['evidence']['route_design_review']","p['evidence']['root_design']")
e=(D/'evaluation_block.txt').read_text(encoding='utf-8-sig');marker=e.index('        external=external_base')
witnessblock=e[:marker];e=e[marker:]
s=s.replace('EVALUATION_REPLACEMENT',e)
s=s.replace('FINAL_REPLACEMENT',(D/'final_block.txt').read_text(encoding='utf-8-sig'))
s=s.replace("        progress['phase']='four_fresh_owned_vs_original_app_parity'",witnessblock+"        progress['phase']='four_fresh_owned_vs_original_app_parity'")
s=s.replace("        except BaseException as exc:","        except BaseException as exc:")
insert=s.index('    try:\n        for module,relative in SOURCE_MODULES.items():')
s=s[:insert]+(D/'failure_state_helper.txt').read_text(encoding='utf-8-sig')+'\n'+s[insert:]
s=s.replace("        bridge='external_bridge' in path.parts\n",'')
s=s.replace("({'input256','mask256','result256','raw512','predicted_codes','used_codes','grid_mask','arm_code','oracle_unavailable_clean_information'}|({'input512'} if bridge else set()))", "{'input256','mask256','result256','raw512','predicted_codes','used_codes','grid_mask','arm_code','oracle_unavailable_clean_information'}")
s=s.replace('input512=x.cpu().numpy(),result256=', 'result256=')
s=s.replace(";before=None;net=None;original_call_hook=None", ";before=None;net=None;original_call_hook=None;generator_start_hook=None;generator_end_hook=None;original_quantize=None;witness={'quantizer':0,'generator_starts':0,'generator_completions':0}")
start=s.index('    def call(kind,fn):');end=s.index('    def state():',start)
s=s[:start]+'''    def call(kind,fn):
        guard();need(kind in TYPED_CAPS and typed[kind]+1<=TYPED_CAPS[kind],'Typed forward cap before dispatch')
        need(counts['forward_slots']+1<=CONTRACT['forward_slots_max'],'Hard forward cap before dispatch')
        need(not torch.is_grad_enabled() and all(not v.requires_grad and v.grad is None for v in net.parameters()) and all(not m.training for m in net.modules()),'Frozen/eval/no-grad before neural dispatch')
        counts['forward_slots']+=1;typed[kind]+=1
        value=fn()
        need(all(witness[key]==counts['forward_slots'] for key in witness),'Actual quantizer/generator witnesses must join every completed dispatch')
        guard();return value
''' +s[end:]
s=s.replace("        if original_call_hook is not None:original_call_hook.remove()", "        if original_call_hook is not None:original_call_hook.remove()\n        if generator_start_hook is not None:generator_start_hook.remove()\n        if generator_end_hook is not None:generator_end_hook.remove()\n        if original_quantize is not None:net.quantize.get_codebook_feat=original_quantize")
s=s.replace("'diagnostic_outputs_planned':96,'teacher_neural_calls'", "'diagnostic_outputs_planned':96,'total_model_calls_planned':108,'teacher_neural_calls'")
s=s.replace("DESIGN_SHA='UNBOUND_ROOT_DESIGN'", "DESIGN_SHA='1983f9fed9bede2078ffc6a1af0dd6ca08fe6ec532e5c3de9c4ce41d4db2c657'")
s=s.replace('oracles', 'spatial oracle interventions')
ast.parse(s,feature_version=(3,10));p.write_text(s,encoding='utf-8',newline='\n')
b=D/'optimized_budget.py';v=b.read_text();v=v.replace("elif category=='array':need(self.totals['array']+growth<=CONTRACT['array_bytes'],'Array write560MiB cap')", "elif category=='array':\n   need(self.totals['array']+growth<=CONTRACT['array_bytes'],'Array write560MiB cap')\n   individual_cap=16*1024**2 if entry.path.relative_to(self.out).as_posix().startswith('fresh_app_parity/') else 6*1024**2\n   need(end<=individual_cap,'Typed individual numeric archive cap')")
ast.parse(v,feature_version=(3,10));b.write_text(v,encoding='utf-8',newline='\n')
print({'worker_bytes':p.stat().st_size,'budget_bytes':b.stat().st_size,'Python3_10_AST_pass':True})
