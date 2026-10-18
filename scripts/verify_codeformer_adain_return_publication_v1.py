"""Independently read back five history-preserving publication edits, CPU only."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_adain_gradient_isolation_v1.return_archive import read, sha, require, write_new, PROTOCOL_SHA


def main():
    review = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review'
    pub = review / 'publication_v1'; receiptpath = pub / 'receipt.json'
    output = review / 'independent_publication_readback_v1.json'
    require(not output.exists(), 'Preserve existing independent publication readback')
    receipt = read(receiptpath); chainpath = review / 'closed_chain_v1.json'; chain = read(chainpath)
    expected = {'SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md',
        'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md', 'PROJECT_HANDOFF.md',
        'CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_VM.md'}
    require(receipt['complete'] is True and len(receipt['documents']) == 5 and
        {row['path'] for row in receipt['documents']} == expected, 'Exact five publication documents required')
    require(chain['complete'] is True and chain['protocol_sha256'] == PROTOCOL_SHA and
        chain['root_personally_reviewed_sheets'] == 55 and chain['completed_updates'] == 50 and
        chain['model_qualified'] is chain['overall_goal_complete'] is False,
        'Publication cannot turn a closed audit into model acceptance')
    require(receipt['closed_chain_sha256'] == sha(chainpath) and receipt['report_sha256'] == chain['report_sha256'] and
        receipt['visual_review_sha256'] == chain['visual_review_sha256'] and
        receipt['quality_stop_retained'] is True and receipt['automatic_input_gate_implemented'] is False and
        receipt['numerical_threshold_calibrated'] is False and receipt['new_training_packet_prepared'] is False and
        receipt['goal_complete'] is receipt['model_qualified'] is receipt['app_modified'] is False,
        'Publication status or closure chain differs')
    publisher = ROOT / 'scripts/publish_codeformer_adain_return_v1.py'
    require(sha(publisher) == receipt['source_sha256'], 'Publisher source changed')
    before = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_root_pre_audit'
    snapshot = read(before / 'snapshot.json')
    require(sha(before / 'snapshot.json') == receipt['pre_return_snapshot_sha256'] and snapshot['complete'] is True and
        set(snapshot['documents']) == expected, 'Exact pre-return snapshot required')
    results = []
    core_prefix = None
    for row in receipt['documents']:
        name = row['path']; current = ROOT / name; backup = ROOT / row['backup']; original = before / name
        pinned = snapshot['documents'][name]
        require(backup.resolve() == (pub / 'prior' / name).resolve(), 'Publication backup outside its dedicated namespace')
        require(sha(current) == row['after_sha256'] and sha(backup) == sha(original) == row['before_sha256'] == pinned['sha256'],
            'Publication after/backup/original fingerprint differs')
        oldbytes = backup.read_bytes(); newbytes = current.read_bytes(); offset = row['retained_suffix_offset']
        require(type(offset) is int and 0 < offset < len(newbytes) and
            len(oldbytes) == row['prior_bytes'] == pinned['bytes'] and newbytes[offset:] == oldbytes == original.read_bytes() and
            row['prior_bytes_retained_exactly'] is True, 'Prior instruction bytes not an exact retained suffix')
        prefix = newbytes[:offset].decode('utf-8')
        if name == 'CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_VM.md':
            require('DO NOT RELAUNCH OR RESUME UNCHANGED' in prefix and 'Historical AdaIN V1 commands' in prefix and
                'unimplemented' in prefix, 'Historical guide must supersede old prepared/manual commands')
        else:
            require('50 REAL ADAM UPDATES / PRESERVATION FAILURE / NO MODEL PROMOTION' in prefix and
                '0.055917' in prefix and '0.03' in prefix and 'no calibrated automatic facial-structure gate' in prefix and
                'a covered mouth/nose alone does not reject sufficient upper-face' in prefix and
                'user accepts restoration' in prefix and 'no exact' in prefix and 'overall goal remains incomplete' in prefix,
                'Current instruction header lost qualification/usability/full goal limits')
            if core_prefix is None: core_prefix = prefix
            else: require(prefix == core_prefix, 'Four current core instruction prefixes differ')
        require(not current.with_name(name + '.adain-return-publication-stage').exists(), 'Unclosed publication stage exists')
        results.append({'path': name, 'after_sha256': row['after_sha256'], 'before_sha256': row['before_sha256'],
            'old_bytes_retained_exactly': True, 'retained_suffix_offset': offset})
    # Publication touches core instruction history only; frozen execution packet
    # remains its original snapshot, never rewritten with new status headers.
    packet = ROOT / 'outputs/cctv_codeformer_adain_gradient_isolation_v1_vm'
    require(sha(packet / 'protocol.json') == PROTOCOL_SHA, 'Frozen packet protocol changed during publication')
    p = read(packet / 'protocol.json')
    local_instruction_files = [name for name in p['files_sha256'] if name.startswith('instructions/')]
    transport = read(review / 'transport_v1.json')
    for name in local_instruction_files:
        resolved = transport['immutable_sources_and_assets_freshly_verified'][name]
        require(resolved['sha256'] == p['files_sha256'][name] and
            sha(resolved['resolved_path']) == resolved['sha256'], 'Frozen instruction snapshot changed')
    report = ROOT / 'CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_RETURN_REVIEW.md'
    require(sha(report) == chain['report_sha256'] and sha(review / 'visual_review_v1.json') == chain['visual_review_sha256'],
        'Report or root visual record changed after closure')
    result = {'complete': True, 'publication_receipt_sha256': sha(receiptpath), 'closed_chain_sha256': sha(chainpath),
        'publisher_source_sha256': sha(publisher), 'source_sha256': sha(__file__), 'documents': results,
        'all_five_backups_match_pre_return_snapshot': True, 'all_prior_bytes_retained_exactly': True,
        'four_current_headers_match': True, 'frozen_packet_instruction_snapshots_unchanged': len(local_instruction_files),
        'quality_stop_remains': True, 'new_input_requirement_recorded_not_implemented': True,
        'restoration_first_and_all_seven_later_covering_families_preserved': True,
        'no_neural_gradient_optimizer_VM_or_app_execution': True, 'no_training_packet_or_resume_authorized': True,
        'model_qualified': False, 'overall_goal_complete': False,
        'prior_checker_assumption_failure_sha256': sha(review / 'publication_readback_checker_attempt1_failure.json'),
        'source_derived_resolution': 'Frozen instruction snapshots use closed exact transport overlay/parent dependency paths.'}
    write_new(output, result)
    print({'complete': True, 'documents': 5, 'receipt_sha256': sha(output), 'model_qualified': False})


if __name__ == '__main__': main()
