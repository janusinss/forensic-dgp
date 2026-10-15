"""No-neural contract, CLI-guard and opaque export regression fixtures."""
import ast
from contextlib import contextmanager, redirect_stdout
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT/'scripts'
sys.path.insert(0, str(SCRIPTS))
import cctv_dgp_decoder_activation_v1_contract as contract
import cctv_dgp_decoder_activation_v1_vm as worker


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024**2), b''):
            value.update(block)
    return value.hexdigest()


def json_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write('\n')


@contextmanager
def workspace_fixture():
    base = ROOT/'scratch/decoder_activation_contract_fixtures'
    base.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(prefix='bounded-', dir=base)
    target = Path(temporary.name)
    try:
        assert target.resolve().is_relative_to(base.resolve())
        assert target.resolve().is_relative_to(ROOT.resolve())
        yield target
    finally:
        # Check the final absolute cleanup target before any recursive cleanup.
        resolved = target.resolve()
        if not resolved.is_relative_to(base.resolve()) or not resolved.is_relative_to(ROOT.resolve()) or target.is_symlink():
            temporary._finalizer.detach()
            raise AssertionError('Refuse cleanup outside verified workspace fixture')
        temporary.cleanup()


def fixture_design():
    old = json.loads((ROOT/'outputs/cctv_dgp_spatial_readout_v1_return/protocol.json').read_text(encoding='utf-8'))
    value = dict(name=contract.NAME, repair_protocol_sha256=contract.REPAIR_PIN,
        source_protocol_sha256=contract.SOURCE_PIN, parent_protocol_sha256=contract.PARENT_PIN,
        replay_protocol_sha256=contract.REPLAY_PIN, checkpoint_sha256=contract.CHECKPOINT,
        state_labels=copy.deepcopy(contract.STATES), retained_amplitudes=copy.deepcopy(contract.AMPLITUDES),
        budgets=copy.deepcopy(contract.BUDGETS), derivative_reference_indices=[12,4],
        derivative_reference_ids=['tr_asian_00176','tr_ffhq_53945'],
        gradient_layout=[*[f'MSE_{i}' for i in range(5)], *[f'SSIM_loss_{i}' for i in range(5)],
            *[f'ArcFace_loss_{i}' for i in range(5)], *[f'landmark_structure_{i}' for i in range(1,5)],
            *[f'channel_mean_{i}_{c}' for i in range(5) for c in range(3)], 'objective'],
        trainable_elements=609219, trainable_tensors=15, common_radius_L2=.04018470287178282,
        new_optimizer_updates=0, new_parameter_proposals=0, new_full_corpus_epochs=0,
        scientific_thresholds=dict(structure_gain=.01, brightness_gain_fraction=.2, MSE_slack=1e-12, SSIM_ArcFace_slack=1e-6),
        case_ids=old['case_ids'], reference_blocks=old['reference_blocks'], diagnostic_only=True,
        training_objective_batch=5, historical_diagnostic_embedding_batch=15,
        clipped_gradient_replay_bound=dict(max_abs=2e-7, cosine_min=.999999), objective_replay_max_abs=1e-10,
        gradient_norm_metadata_bound=dict(rtol=2e-6, atol=1e-12),
        clip_factor_metadata_max_ULPs=2,
        trainable_parameter_names=old['trainable_parameter_names']['A'])
    for key in ('automatic_launch','failed_recipe_resume_allowed','model_qualified','new_incumbent_selection',
                'local_neural_construction_allowed','direction_search','new_amplitudes','native_pixels_used','DEV_pixels_used','final_identity_pixels'):
        value[key] = False
    return value


class ContractValidation(unittest.TestCase):
    def test_valid_frozen_design(self):
        p = fixture_design()
        contract.validate(p, copy.deepcopy(p))

    def test_no_state_or_amplitude_substitution(self):
        for key, changed in [('state_labels',['initial','new_probe','probe_1_4','fit_u001']),
                             ('retained_amplitudes',[0.,1/32,1/4,1.]), ('derivative_reference_indices',[4,12]),
                             ('gradient_layout',['wrong']*35)]:
            with self.subTest(key=key):
                p = fixture_design(); p[key] = changed
                with self.assertRaises(AssertionError): contract.validate(p, copy.deepcopy(p))

    def test_no_extra_queries_or_neural_calls(self):
        for key in ('maximum_gradient_queries','decoder_batch_calls','recognizer_batch_calls','maximum_optimizer_updates','maximum_backwards'):
            with self.subTest(key=key):
                p = fixture_design(); p['budgets'][key] += 1
                with self.assertRaises(AssertionError): contract.validate(p, copy.deepcopy(p))

    def test_scientific_and_replay_tolerances_are_fixed(self):
        mutations = [('scientific_thresholds','structure_gain',.009),
                     ('scientific_thresholds','brightness_gain_fraction',.3),
                     ('scientific_thresholds','MSE_slack',1e-8),
                     ('scientific_thresholds','SSIM_ArcFace_slack',1e-4),
                     ('clipped_gradient_replay_bound','max_abs',1e-5)]
        for outer, inner, changed in mutations:
            with self.subTest(field=outer+'.'+inner):
                p = fixture_design(); p[outer][inner] = changed
                with self.assertRaises(AssertionError): contract.validate(p, copy.deepcopy(p))
        for changed in (0,1,3,10):
            with self.subTest(clip_factor_metadata_max_ULPs=changed):
                p = fixture_design(); p['clip_factor_metadata_max_ULPs'] = changed
                with self.assertRaises(AssertionError): contract.validate(p, copy.deepcopy(p))

    def test_holdout_no_search_and_zero_epoch_fields(self):
        for key in ('native_pixels_used','DEV_pixels_used','final_identity_pixels','direction_search','new_amplitudes','automatic_launch'):
            with self.subTest(key=key):
                p = fixture_design(); p[key] = True
                with self.assertRaises(AssertionError): contract.validate(p, copy.deepcopy(p))
        p = fixture_design(); p['new_full_corpus_epochs'] = 1
        with self.assertRaises(AssertionError): contract.validate(p, copy.deepcopy(p))

    def test_frozen_design_and_unique_case_ids(self):
        p = fixture_design(); design = copy.deepcopy(p); p['diagnostic_only'] = False
        with self.assertRaises(AssertionError): contract.validate(p, design)
        p = fixture_design(); p['case_ids'][1] = p['case_ids'][0]
        with self.assertRaises(AssertionError): contract.validate(p, copy.deepcopy(p))

    def test_transfer_rejects_non_train_role(self):
        # NumPy arrays are synthetic serialization fixtures, never installed in
        # a neural network or used for a direction/gradient/model calculation.
        import numpy as np
        with workspace_fixture() as temp:
            roots = [temp/x for x in ('new','repair','source','parent','replay')]
            root, repair, source, parent, replay = roots
            p = fixture_design(); design = copy.deepcopy(p)
            p.update(design_sha256='d'*64, assets_sha256={}, repair_required_sha256={},
                     source_required_sha256={}, parent_required_sha256={}, replay_required_sha256={})
            old = dict(case_ids=p['case_ids'], reference_blocks=p['reference_blocks'], trainable_parameter_names={'A':p['trainable_parameter_names']})
            repaired = dict(case_ids=p['case_ids'], reference_blocks=p['reference_blocks'], schedule=[12,4])
            profiles = ['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24']
            data = dict(preview_case_ids=p['case_ids'], cases=[dict(id=cid, role='train', reference_id=b['reference'], profile=profile)
                for b in p['reference_blocks'] for cid, profile in zip(b['case_ids'],profiles)])
            docs = {root/'protocol.json':p, root/'design.json':design, repair/'protocol.json':repaired,
                    source/'protocol.json':old, parent/'protocol.json':data,
                    repair/'outputs/decoder_hinge_repair_v1/run_results.json':dict(complete=True,training_updates=1,stop_reason='quality_stop')}
            hashes = {root/'protocol.json':'f'*64, root/'design.json':'d'*64, repair/'protocol.json':contract.REPAIR_PIN,
                      source/'protocol.json':contract.SOURCE_PIN, parent/'protocol.json':contract.PARENT_PIN,
                      replay/'protocol.json':contract.REPLAY_PIN,
                      repair/'cctv_dgp_bank_hinge_boundary_v1_loss.py':contract.CORRECTED_HELPER_SHA}
            initial = np.zeros(609219,dtype=np.float32)
            changed = np.full(609219,1e-7,dtype=np.float32)
            def load(path, allow_pickle=False):
                self.assertIs(allow_pickle,False)
                return initial.copy() if Path(path).name in ('initial_vector.npy','A_initial_vector.npy') else changed.copy()
            with patch.object(contract,'read',side_effect=lambda path:docs[Path(path)]), patch.object(contract,'sha',side_effect=lambda path:hashes[Path(path)]), \
                 patch.object(contract,'verify_paths'), patch.object(np,'load',side_effect=load):
                contract.verify_transfer(root,repair,source,parent,replay,'f'*64)
                data['cases'][0]['role']='reserved_evaluation'
                with self.assertRaises(AssertionError): contract.verify_transfer(root,repair,source,parent,replay,'f'*64)


class SafePaths(unittest.TestCase):
    def test_regular_file_digest_and_mismatch(self):
        with workspace_fixture() as temp:
            path=temp/'file.bin'; path.write_bytes(bytes(range(256)))
            contract.verify_paths(temp,{'file.bin':digest(path)})
            with self.assertRaises(AssertionError): contract.verify_paths(temp,{'file.bin':'0'*64})

    def test_traversal_absolute_noncanonical_and_colon_rejected(self):
        with workspace_fixture() as temp:
            for rel in ('../outside.bin','/absolute.bin','C:/absolute.bin','a\\b','a//b','./file.bin'):
                with self.subTest(path=rel), self.assertRaises(AssertionError): contract.verify_paths(temp,{rel:'0'*64})

    def test_symlink_policy_rejected_before_hash(self):
        with workspace_fixture() as temp:
            path=temp/'file.bin'; path.write_bytes(bytes(range(256)))
            # Deterministically exercise the filesystem symlink metadata guard
            # without needing Windows CreateSymbolicLink privilege.
            with patch.object(Path,'is_symlink',return_value=True), patch.object(contract,'sha') as hashing:
                with self.assertRaises(AssertionError): contract.verify_paths(temp,{'file.bin':'0'*64})
                hashing.assert_not_called()


class CliAndSourceGuards(unittest.TestCase):
    @unittest.skipUnless(platform.system()=='Windows','Actual Windows guard fixture')
    def test_actual_windows_run_rejected_before_protocol_or_neural_import(self):
        with workspace_fixture() as temp:
            root=temp/'vm';root.mkdir()
            (root/'protocol.json').write_bytes(b'INVALID_PROTOCOL_MUST_NOT_BE_READ')
            sentinel=temp/'sentinel';sentinel.mkdir(); attempts=temp/'forbidden_import_attempts.txt'
            sentinel_source = """import sys,importlib.abc,os
from pathlib import Path
class BlockNeural(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','PIL','cv2','onnx2torch'):
   Path(os.environ['ACTIVATION_IMPORT_SENTINEL']).write_text(fullname)
   raise RuntimeError('FORBIDDEN_NEURAL_IMPORT')
sys.meta_path.insert(0,BlockNeural())
"""
            (sentinel/'sitecustomize.py').write_text(sentinel_source,encoding='utf-8')
            env=dict(os.environ,PYTHONPATH=str(sentinel),PYTHONDONTWRITEBYTECODE='1',ACTIVATION_IMPORT_SENTINEL=str(attempts))
            result=subprocess.run([sys.executable,'-B',str(SCRIPTS/'cctv_dgp_decoder_activation_v1_vm.py'),
                '--root',str(root),'--protocol-sha','0'*64,'--run'],cwd=ROOT,env=env,capture_output=True,text=True,timeout=20)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Human manual Linux L4 tmux only',result.stderr)
            self.assertNotIn('JSONDecodeError',result.stderr)
            self.assertNotIn('FORBIDDEN_NEURAL_IMPORT',result.stderr)
            self.assertFalse(attempts.exists())
            self.assertEqual({p.name for p in root.iterdir()},{'protocol.json'})
            self.assertEqual((root/'protocol.json').read_bytes(),b'INVALID_PROTOCOL_MUST_NOT_BE_READ')

    def test_ast_no_optimizer_backward_or_historical_run(self):
        for name in ('cctv_dgp_decoder_activation_v1_vm.py','cctv_dgp_decoder_activation_v1_context.py'):
            text=(SCRIPTS/name).read_text(encoding='utf-8');tree=ast.parse(text,feature_version=(3,10));compile(text,name,'exec')
            for node in ast.walk(tree):
                if isinstance(node,ast.Call):
                    call=ast.unparse(node.func)
                    self.assertFalse(call.endswith(('.backward','.step')) or '.optim.' in call,call)
                    self.assertNotIn(call,('loader.run','loader.train','old.run','old.train','repaired.run','repaired.train'))
            for node in tree.body:
                if isinstance(node,(ast.Import,ast.ImportFrom)):
                    names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module or '']
                    self.assertFalse(any(n.split('.')[0] in ('torch','PIL','numpy','cv2') for n in names))


class OpaqueExport(unittest.TestCase):
    def exercise(self, kind):
        with workspace_fixture() as temp:
            root=temp/'run';root.mkdir();home=temp/'home';home.mkdir()
            stage=root/worker.STAGE;stage.mkdir(parents=True)
            payload=bytes(range(256));(stage/'partial_payload.bin').write_bytes(payload)
            (root/'asset.txt').write_bytes(b'opaque source fixture\n')
            p=dict(assets_sha256={'asset.txt':digest(root/'asset.txt')},budgets=copy.deepcopy(contract.BUDGETS))
            json_write(root/'protocol.json',p)
            terminal=dict(exit_code=0 if kind=='success' else -9,timeout=kind!='success',kill_sent=kind!='success')
            json_write(root/'run_exit.json',terminal)
            if kind=='success':
                json_write(stage/'run_results.json',dict(complete=True,counts=dict(gradient_queries=70,optimizer_updates=0)))
            elif kind=='caught_failure':
                json_write(stage/'failure.json',dict(complete=False,counts=dict(gradient_queries=13,optimizer_updates=0)))
            if kind in ('partial','malformed'):
                (stage/'progress.jsonl').write_bytes(b'{"counts":{"gradient_queries":35,"optimizer_updates":0},"progress":{"stage":"VJP"}}\n{"counts":')
            if kind=='malformed': (stage/'run_results.json').write_bytes(b'{"complete":true,"counts":')
            before={p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
            with patch.object(worker,'require_manual') as manual, patch.object(Path,'home',return_value=home), redirect_stdout(io.StringIO()):
                worker.export(root,p,'f'*64)
                manual.assert_called_once_with(root)
            archive=home/(worker.STEM+'-results.tar.gz')
            sidecar=Path(str(archive)+'.sha256')
            receipt=json.loads((home/(worker.STEM+'-export.json')).read_text())
            self.assertEqual(sidecar.read_bytes(),(digest(archive)+'  '+archive.name+'\n').encode('utf-8'))
            self.assertNotIn(b'\r',sidecar.read_bytes())
            self.assertFalse(sidecar.read_bytes().startswith(b'\xef\xbb\xbf'))
            self.assertEqual(receipt['archive_sha256'],digest(archive));self.assertEqual(receipt['bytes'],archive.stat().st_size)
            self.assertFalse(receipt['model_qualified']);self.assertEqual(receipt['optimizer_updates'],0)
            expected_queries=70 if kind=='success' else 13 if kind=='caught_failure' else None
            self.assertEqual(receipt['gradient_queries'],expected_queries)
            self.assertEqual(receipt['gradient_query_count_exact'],kind in ('success','caught_failure'))
            if kind in ('partial','malformed'):
                self.assertIsNone(receipt['gradient_queries'])
                self.assertEqual(receipt['last_known_progress']['counts']['gradient_queries'],35)
                self.assertTrue(receipt['partial_terminal_without_worker_receipt'])
                self.assertEqual(receipt['terminal_worker']['exit_code'],-9)
            if kind=='malformed': self.assertEqual(len(receipt['incomplete_worker_receipt_errors']),1)
            with tarfile.open(archive,'r:gz') as stream:
                members=stream.getmembers();names=[m.name for m in members]
                self.assertEqual(set(names),set(before)|{'result_manifest.json'})
                self.assertEqual(len(names),len(set(names)))
                self.assertTrue(all(m.isfile() for m in members))
                manifest=json.loads(stream.extractfile('result_manifest.json').read())
                self.assertEqual(set(manifest['files']),set(before))
                for rel,value in before.items():
                    actual=stream.extractfile(rel).read();self.assertEqual(actual,value)
                    self.assertEqual(manifest['files'][rel],dict(bytes=len(value),sha256=hashlib.sha256(value).hexdigest()))
                self.assertEqual(stream.extractfile(worker.STAGE+'/partial_payload.bin').read(),payload)
                self.assertEqual(manifest['gradient_queries'],expected_queries)
            retained={str(p.relative_to(temp)):digest(p) for p in temp.rglob('*') if p.is_file()}
            with patch.object(worker,'require_manual'),patch.object(Path,'home',return_value=home),self.assertRaises(AssertionError):
                worker.export(root,p,'f'*64)
            self.assertEqual(retained,{str(p.relative_to(temp)):digest(p) for p in temp.rglob('*') if p.is_file()})

    def test_complete_opaque_export(self):self.exercise('success')
    def test_caught_failure_export(self):self.exercise('caught_failure')
    def test_sigkill_partial_export_unknown_exact_count(self):self.exercise('partial')
    def test_malformed_worker_receipt_still_preserves_partial(self):self.exercise('malformed')

    def test_partial_without_terminal_receipt_cannot_claim_export(self):
        with workspace_fixture() as temp:
            root=temp/'run';root.mkdir();home=temp/'home';home.mkdir();stage=root/worker.STAGE;stage.mkdir(parents=True)
            payload=bytes(range(256));(stage/'partial_payload.bin').write_bytes(payload)
            p=dict(assets_sha256={},budgets=copy.deepcopy(contract.BUDGETS));json_write(root/'protocol.json',p)
            with patch.object(worker,'require_manual'),patch.object(Path,'home',return_value=home),self.assertRaises(AssertionError):worker.export(root,p,'f'*64)
            self.assertEqual((stage/'partial_payload.bin').read_bytes(),payload)
            self.assertEqual(list(home.iterdir()),[])
            self.assertFalse((root/'result_manifest.json').exists())


def main():
    started=time.monotonic();output=io.StringIO()
    suite=unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result=unittest.TextTestRunner(stream=output,verbosity=2).run(suite)
    base=ROOT/'outputs/cctv_dgp_decoder_activation_v1_preparation';base.mkdir(parents=True,exist_ok=True)
    number=1
    while (base/f'contract_test_run_{number:03d}').exists():number+=1
    folder=base/f'contract_test_run_{number:03d}';folder.mkdir()
    with (folder/'tests.txt').open('x',encoding='utf-8',newline='\n') as stream:stream.write(output.getvalue())
    files=[Path(__file__),SCRIPTS/'cctv_dgp_decoder_activation_v1_contract.py',SCRIPTS/'cctv_dgp_decoder_activation_v1_vm.py',SCRIPTS/'cctv_dgp_decoder_activation_v1_context.py']
    receipt=dict(complete=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=result.skipped,
        seconds=time.monotonic()-started,python=sys.version,source_sha256={p.relative_to(ROOT).as_posix():digest(p) for p in files},
        log_sha256=digest(folder/'tests.txt'),model_forwards=0,gradient_queries=0,optimizer_updates=0,VM_connections=0,
        fixture_cleanup_scope='fresh resolved workspace-local temporary directories only',
        export_guard_stub_scope='export function only; never run/context/model methods',scientific_quality_or_CUDA_validation=False)
    json_write(folder/'tests.json',receipt)
    print(output.getvalue(),end='');print(json.dumps(dict(evidence=folder.relative_to(ROOT).as_posix(),**receipt)))
    return 0 if result.wasSuccessful() else 1


if __name__=='__main__':raise SystemExit(main())
