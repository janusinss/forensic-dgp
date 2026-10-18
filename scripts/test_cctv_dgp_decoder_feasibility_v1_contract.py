"""Small non-neural contract and exact-transfer-boundary fixtures."""
import copy
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import shutil
import tarfile
import unittest
import uuid
from unittest.mock import patch

import cctv_dgp_decoder_feasibility_v1_contract as c

ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def directory():
    parent=ROOT/'outputs/cctv_dgp_decoder_feasibility_v1_preparation/fixtures'
    parent.mkdir(parents=True,exist_ok=True)
    path=parent/str(uuid.uuid4());path.mkdir()
    try:yield path
    finally:
        assert path.resolve().is_relative_to(parent.resolve()) and path.parent==parent
        shutil.rmtree(path)


def fixture():
    return dict(name=c.NAME,activation_protocol_sha256=c.ACTIVATION_PIN,
        checkpoint_sha256=c.CHECKPOINT,route='A',state_labels=c.STATES,
        placements=c.AMPLITUDES,common_radius_L2=c.RADIUS,solver=c.SOLVER,
        budgets=c.BUDGETS,scientific_thresholds=c.THRESHOLDS,
        trainable_tensors=15,trainable_elements=609219,
        derivative_reference_indices=[12,4],derivative_reference_ids=['tr_asian_00176','tr_ffhq_53945'],
        trainable_parameter_names=[str(i) for i in range(15)],
        case_ids=[str(i) for i in range(100)],reference_blocks=[{} for _ in range(20)],
        objective_replay_max_abs=1e-10,new_optimizer_updates=0,new_gradient_queries=0,new_full_corpus_epochs=0,
        automatic_launch=False,failed_recipe_resume_allowed=False,model_qualified=False,new_incumbent_selection=False,
        local_neural_construction_allowed=False,local_parameter_proposals_allowed=False,native_pixels_used=False,
        DEV_pixels_used=False,final_identity_pixels=False,automatic_extra_epochs=False,
        diagnostic_only=True,one_direction_only=True,strict_homogeneous_sufficient_subset=True,
        fixed_placements_comparison_not_optimizer_continuation=True,exact_rounded_displacement_certificate_required=True,
        no_certificate_implies_whole_decoder_infeasibility=False)


class ContractTests(unittest.TestCase):
    def test_frozen_design_passes(self):
        p=fixture();c.validate(p,copy.deepcopy(p))

    def test_frozen_design_cannot_be_replaced(self):
        p=fixture();design=copy.deepcopy(p);p['route']='B'
        with self.assertRaises(AssertionError):c.validate(p,design)

    def test_higher_radius_rejected_even_with_matching_design(self):
        p=fixture();p['common_radius_L2']*=2
        with self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_scientific_threshold_changes_rejected(self):
        for key in c.THRESHOLDS:
            p=copy.deepcopy(fixture());p['scientific_thresholds'][key]*=2
            with self.subTest(key=key),self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_every_forbidden_promotion_or_scope_flag_rejected(self):
        for key in ['automatic_launch','failed_recipe_resume_allowed','model_qualified','new_incumbent_selection',
                    'local_neural_construction_allowed','local_parameter_proposals_allowed','native_pixels_used',
                    'DEV_pixels_used','final_identity_pixels','automatic_extra_epochs']:
            p=fixture();p[key]=True
            with self.subTest(key=key),self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_solver_count_and_resource_limits_frozen(self):
        for key in ['maximum_solver_calls','maximum_candidate_directions','maximum_gradient_queries',
                    'maximum_optimizer_updates','worker_seconds','output_byte_cap','disk_reserve_bytes']:
            p=copy.deepcopy(fixture());p['budgets'][key]+=1
            with self.subTest(key=key),self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_fourth_placement_rejected(self):
        p=fixture();p['placements']=[0.,.125,.25,.5,1.]
        with self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_one_direction_and_exact_certificate_required(self):
        for key in ['one_direction_only','strict_homogeneous_sufficient_subset','exact_rounded_displacement_certificate_required',
                    'fixed_placements_comparison_not_optimizer_continuation']:
            p=fixture();p[key]=False
            with self.subTest(key=key),self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_epoch_or_gradient_work_rejected(self):
        for key in ['new_optimizer_updates','new_gradient_queries','new_full_corpus_epochs']:
            p=fixture();p[key]=1
            with self.subTest(key=key),self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_duplicate_or_missing_case_rejected(self):
        for cases in [[str(i) for i in range(99)],[str(i) for i in range(99)]+['0']]:
            p=fixture();p['case_ids']=cases
            with self.subTest(cases=len(cases)),self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_derivative_references_cannot_be_reselected(self):
        p=fixture();p['derivative_reference_indices']=[0,1]
        with self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_no_candidate_is_not_a_whole_decoder_infeasibility_claim(self):
        p=fixture();p['no_certificate_implies_whole_decoder_infeasibility']=True
        with self.assertRaises(AssertionError):c.validate(p,copy.deepcopy(p))

    def test_current_windows_platform_rejected_before_models(self):
        with patch.object(c.platform,'system',return_value='Windows'):
            with self.assertRaisesRegex(AssertionError,'manual Linux'):c.require_manual(Path('.'))

    def test_hash_files_and_reject_corruption(self):
        with directory() as temporary:
            root=Path(temporary);path=root/'asset.txt';path.write_bytes(b'retained research\n')
            digest=c.sha(path);c.verify_paths(root,{'asset.txt':digest})
            path.write_bytes(b'changed\n')
            with self.assertRaises(AssertionError):c.verify_paths(root,{'asset.txt':digest})

    def test_unsafe_paths_rejected_without_reading_outside(self):
        with directory() as temporary:
            for name in ['../outside','/absolute','C:/absolute','folder\\file','./file','folder//file']:
                with self.subTest(name=name),self.assertRaises(AssertionError):c.verify_paths(temporary,{name:'0'*64})

    def test_write_is_exclusive(self):
        with directory() as temporary:
            path=Path(temporary)/'receipt.json';c.write(path,dict(complete=True))
            with self.assertRaises(FileExistsError):c.write(path,dict(complete=False))
            self.assertEqual(c.read(path),dict(complete=True))

    def test_new_stage_reader_accepts_only_hash_bound_new_namespace(self):
        from import_cctv_dgp_decoder_feasibility_v1_return import reader
        known=reader()
        self.assertEqual(known.STAGE,c.STAGE)
        for stage in [c.STAGE,'outputs/decoder_activation_v1']:
            with self.subTest(stage=stage),directory() as temporary:
                protocol_bytes=b'opaque new protocol\n';pin=hashlib.sha256(protocol_bytes).hexdigest()
                payload={'protocol.json':protocol_bytes,stage+'/payload.bin':b'opaque saved evidence'}
                manifest=dict(complete=True,protocol_sha256=pin,optimizer_updates=0,model_qualified=False,
                    files={name:dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest()) for name,value in payload.items()})
                payload['result_manifest.json']=json.dumps(manifest).encode()
                archive=temporary/'results.tar.gz'
                with tarfile.open(archive,'x:gz') as stream:
                    for name,value in payload.items():
                        member=tarfile.TarInfo(name);member.size=len(value);stream.addfile(member,io.BytesIO(value))
                digest=c.sha(archive);sidecar=temporary/'sidecar';sidecar.write_text(digest+'  '+archive.name+'\n')
                export=temporary/'export.json'
                c.write(export,dict(complete=True,protocol_sha256=pin,archive_sha256=digest,bytes=archive.stat().st_size,
                    optimizer_updates=0,model_qualified=False,training_success_not_implied=True,
                    partial_terminal_without_worker_receipt=False,gradient_queries=0))
                p=dict(budgets=dict(output_byte_cap=1024),assets_sha256={})
                target=temporary/'return'
                if stage==c.STAGE:
                    proof=known.import_verified(archive,sidecar,export,target,p,pin)
                    self.assertTrue(proof['all_hashes_before_extraction'])
                    self.assertEqual((target/stage/'payload.bin').read_bytes(),b'opaque saved evidence')
                else:
                    with self.assertRaises(AssertionError):known.import_verified(archive,sidecar,export,target,p,pin)
                    self.assertFalse(target.exists())


if __name__=='__main__':unittest.main()
