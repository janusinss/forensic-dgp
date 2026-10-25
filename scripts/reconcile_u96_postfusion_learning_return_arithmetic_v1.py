"""Close Python3.13-vs3.10 reduction readback using saved pixel audit only.

Never repeats the TAR pass or modifies its immutable first audit. Each gate
uses explicit Python float64 left-to-right accumulation, matching Python3.10.
"""
import importlib.util
import json
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'outputs/codeformer_u96_postfusion_learning_v1_return_audit'
SOURCE = ROOT/'scripts/audit_u96_postfusion_learning_return_v1.py'
spec = importlib.util.spec_from_file_location('inert_saved_pixel_audit', SOURCE)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def ordered_sum(values, start=0.0):
    total = float(start)
    for value in values:
        total = total + float(value)
    return total


audit.sum = ordered_sum
original_path = BASE/'saved_pixel_audit_v1.json'
original = json.loads(original_path.read_bytes())
protocol = json.loads((audit.STAGE/'protocol.json').read_bytes())
index = json.loads((BASE/'archive_index_v1.json').read_bytes())
peerpath = ROOT/'outputs/codeformer_u96_postfusion_learning_v1_return_peer/bounded_stream_readback_v1.json'
peer = json.loads(peerpath.read_bytes())
rows = original['all_saved_pixel_metrics']
baseline = {r['id']:r for r in rows if r['kind']=='reference'}
snapshots = {}
for row in rows:
    if row['kind']=='snapshot': snapshots.setdefault(row['update'],{})[row['id']]=row
guardids = {r['id'] for r in protocol['guards']}
monitorids = {r['id'] for r in protocol['cases'] if r['cohort']=='monitor'}
checks, gates = [], []
def check(name, value):
    checks.append({'name':name, 'pass':bool(value)})
for update in audit.SNAPSHOTS:
    bank=snapshots[update]
    legacy=audit.legacy_gate({k:v for k,v in bank.items() if k in guardids},protocol)
    monitor=audit.monitor_gate({k:v for k,v in bank.items() if k in monitorids},baseline,protocol,update)
    returned=peer['documents'][f'{audit.RUN}/snapshots/u{update:04d}/bank_and_gates.json']
    check(f'u{update}_entire_legacy_dictionary_exact',legacy==returned['legacy'])
    check(f'u{update}_entire_monitor_dictionary_exact',monitor==returned['monitor'])
    gates.append({'update':update,'legacy':legacy,'monitor':monitor})
check('independent_member_index_exactly_matches_peer',index['members']==peer['metadata'])
check('archive_pin_matches_peer',index['archive_SHA256']==peer['archive_sha256']==audit.ARCHIVE_SHA)
check('payload_and_compressed_bytes_exactly_match_peer',index['payload_bytes']==peer['payload_bytes'] and index['archive_bytes']==peer['archive_bytes'])
check('peer_actual_decompressed_stream_within1536MiB_plus16MiB_overhead',peer['decompressed_stream_bytes']<=audit.PAYLOAD_CAP+audit.OVERHEAD_CAP)
check('all_original_failures_only_gate_dictionary_aggregation',len(original['integrity_failures'])==7 and
      all(f['check'] in ('legacy_gate_all_fields_exact_no_epsilon','monitor_gate_all_fields_exact_no_epsilon') for f in original['integrity_failures']))
check('all712_floorPNG_and_regional_metrics_exact',original['all712_RAW_to_floorPNG_exact'] and
      not any(f['check'] not in ('legacy_gate_all_fields_exact_no_epsilon','monitor_gate_all_fields_exact_no_epsilon') for f in original['integrity_failures']))
check('all56_zerohead_RAW_PNG_parity',len(original['initial56_zerohead_parity'])==56 and
      all(r['RAW_exact'] and r['PNG_bytes_exact'] for r in original['initial56_zerohead_parity']))
check('all_first4_training_actual_stock_RAW_PNG_parity',len(original['first4_grad_enabled_saved_stock_and_delivery_parity'])==4 and
      all(r['RAW_exact'] and r['PNG_bytes_exact'] and all(z['arrays_exact'] for z in r['stock'].values()) for r in original['first4_grad_enabled_saved_stock_and_delivery_parity']))

def mean(values):
    values=list(values)
    return ordered_sum(values)/len(values)

role_summary={}
for scope,ids in [('TRAIN_monitor',monitorids),('legacy_guard',guardids)]:
    role_summary[scope]={}
    for role in sorted({r['role'] for r in snapshots[64].values() if r['id'] in ids}):
        current=[r for r in snapshots[64].values() if r['id'] in ids and r['role']==role]
        role_summary[scope][role]={'cases':len(current),
            'U96_RAW_mean_absolute_distance':mean(r['U96_behavior_distance']['raw'] for r in current),
            'U96_RAW_maximum_absolute_component_change':max(r['U96_raw_component_difference']['maximum_absolute'] for r in current),
            'U96_RAW_minimum_signed_component_change':min(r['U96_raw_component_difference']['minimum_signed'] for r in current),
            'U96_RAW_maximum_signed_component_change':max(r['U96_raw_component_difference']['maximum_signed'] for r in current),
            'U96_RAW_changed_components':int(ordered_sum(r['U96_raw_component_difference']['changed_components'] for r in current)),
            'U96_PNG_mean_absolute_distance':mean(r['U96_behavior_distance']['native_png'] for r in current),
            'U96_PNG_changed_components':int(ordered_sum(r['U96_PNG_changed_components'] for r in current)),
            'U96_block16_luminance_absolute_distance_RAW':mean(r['U96_illumination_distance']['raw'] for r in current),
            'U96_block16_luminance_absolute_distance_PNG':mean(r['U96_illumination_distance']['native_png'] for r in current),
            'paired_target_metrics':None if current[0]['regional_metrics']['raw'] is None else {
                view:{region:{field:{'frozen_U96_mean':mean(baseline[r['id']]['regional_metrics'][view][region][field] for r in current),
                                    'u64_mean':mean(r['regional_metrics'][view][region][field] for r in current)}
                              for field in ('mae','adjacent_rgb_error')} for region in ('whole','central128','outside')}
                for view in ('raw','native_png')}}

visual=json.loads((BASE/'visual_manifest_v1.json').read_bytes())
check('30_exact_saved256_visual_sheets',len(visual['sheets'])==30 and
      sum(len(s['cells']) for s in visual['sheets'])==952 and
      sum(c['generated_return_PNG'] for s in visual['sheets'] for c in s['cells'])==712)
for sheet in visual['sheets']:
    check('sheet_current_hash_'+Path(sheet['file']['path']).name,audit.filepin(ROOT/sheet['file']['path'])==sheet['file'])

failures=[c for c in checks if not c['pass']]
end=gates[-1]
report={'format':'complete-U96-B64-saved-pixel-arithmetic-companion-v1','complete':True,
    'independent_saved_pixel_integrity_pass':not failures,'pilot_quality_gates_pass':end['legacy']['pass'] and end['monitor']['pass'],
    'model_qualified':False,'root_visual_review_pending':True,
    'closes_immutable_first_audit':audit.filepin(original_path),'script':audit.filepin(Path(__file__)),
    'inert_independent_metric_and_gate_source':audit.filepin(SOURCE),
    'peer_bounded_stream_readback':audit.filepin(peerpath),
    'interpretation':'The immutable first audit ran onPython3.13 using its changed built-in sum. Seven full gate dictionary comparisons differ only in scalar aggregation at~1e-18. This companion uses explicit ordered float64 addition matching the workerPython3.10. Every entire gate dictionary now compares exactly, without epsilon or threshold changes.',
    'arithmetic':'SortedcaseIDs and explicit float64 total=total+value, left-to-right. No tolerance, rounding, metric quantization or gate rebinding.',
    'original712_pixel_recomputations_and256_objectives_retained':True,'no_second_archive_stream':True,
    'decompressed_stream_bytes_crosschecked_with_peer':peer['decompressed_stream_bytes'],
    'payload_bytes_independently_streamed':index['payload_bytes'],
    'checks':checks,'failures':failures,'snapshot_gates_explicit_ordered_sum':gates,
    'u64_per_role_pixel_change_and_paired_metrics':role_summary,
    'objective_reconstruction_summary':original['objective_reconstruction_summary'],
    'visual_manifest':audit.filepin(BASE/'visual_manifest_v1.json'),
    'qualification_limits':original['material_limits'],
    'Torch_neural_gradient_optimizer_training_VM_HTTP_browser_calls':0}
pin=audit.write_json(BASE/'saved_pixel_arithmetic_companion_v1.json',report)
print(json.dumps({'complete':True,'independent_saved_pixel_integrity_pass':not failures,
    'pilot_quality_gates_pass':report['pilot_quality_gates_pass'],'failures':failures,'companion':pin,
    'u64_TRAIN_monitor_role_summary':role_summary['TRAIN_monitor']}))
