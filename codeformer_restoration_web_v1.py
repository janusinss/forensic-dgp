"""Injectable U96 HTTP boundary; importing this module mounts no app or model.

This is a staging seam for the existing application. The caller must supply the
reviewed controller and its readiness check. It does not promote a model or run
the separate covering detector/completion workflow.
"""
import base64
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from codeformer_restoration_workflow_v1 import (
    CoveringWorkflowRequired, RestorationBusy, RestorationSourceMismatch,
    RestorationWorkerStateInvalid, bundle_bytes,
)
from research.protected_surroundings_brush_v1 import (
    MAX_MASK_BYTES, MAX_UPLOAD_BYTES, decode_original, png,
)


def data_url(raw, kind='image/png'):
    return 'data:' + kind + ';base64,' + base64.b64encode(raw).decode('ascii')


def create_router(controller, readiness):
    """Create routes without mounting, loading models or invoking readiness.

    readiness must recheck the exact reviewed source/model manifests. The
    controller independently revalidates source identity before/after work.
    """
    if not callable(readiness):
        raise TypeError('A reviewed readiness check is required.')
    router = APIRouter(prefix='/face/restoration', tags=['Reviewed U96 restoration'])

    def ready():
        try:
            readiness()
        except Exception as error:
            raise HTTPException(503, 'The reviewed restoration sources or model files are unavailable.') from error

    def invoke(function, *args, **kwargs):
        ready()
        try:
            return function(*args, **kwargs)
        except RestorationSourceMismatch as error:
            raise HTTPException(503, 'Reviewed restoration sources changed. Restore the reviewed service before use.') from error
        except RestorationWorkerStateInvalid as error:
            raise HTTPException(409, str(error)) from error
        except CoveringWorkflowRequired as error:
            raise HTTPException(422, str(error)) from error
        except RestorationBusy as error:
            raise HTTPException(409, str(error)) from error
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except (FileNotFoundError, OSError) as error:
            raise HTTPException(503, 'A reviewed restoration file is unavailable.') from error
        except RuntimeError as error:
            # Includes uncertain owned-worker cleanup; never retry in this seam.
            raise HTTPException(409, str(error)) from error
        except Exception as error:
            logging.exception('Reviewed U96 restoration boundary failed')
            raise HTTPException(500, 'Restoration stopped. Review its saved status before retrying.') from error

    @router.get('/status')
    async def status():
        def read_status():
            value = controller.status()
            return {'available': True, 'maximum_restore_forwards': 1,
                    'maximum_off_forwards': 0, 'preparation_neural_calls': 0,
                    'automatic_preparation': False, 'model_qualified': False,
                    'main_app_default_changed': False, 'covering_workflow': 'separate',
                    'training_optimizer_VM_calls': 0, 'controller': value}
        return await run_in_threadpool(invoke, read_status)

    @router.post('/inspect')
    async def inspect(file: UploadFile = File(...)):
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        def read_input():
            native = decode_original(contents)
            return {'width': native.shape[1], 'height': native.shape[0],
                    'source': data_url(png(native)), 'neural_calls': 0,
                    'input_review_required': True}
        return await run_in_threadpool(invoke, read_input)

    @router.post('/preview')
    async def preview(file: UploadFile = File(...), protect_mask: UploadFile = File(...),
                      apply_mask: UploadFile = File(...), radius: float = Form(1.),
                      input_review: str = Form('')):
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        protect = await protect_mask.read(MAX_MASK_BYTES + 1)
        apply = await apply_mask.read(MAX_MASK_BYTES + 1)
        def make_preview():
            response, payloads = controller.preview(contents, protect, apply, radius, input_review)
            return {**response, 'bundle': data_url(bundle_bytes(payloads), 'application/zip'),
                    'neural_calls': 0}
        return await run_in_threadpool(invoke, make_preview)

    @router.post('/restore')
    async def restore(file: UploadFile = File(...), input_review: str = Form(''),
                      restoration: str = Form('auto'), preparation_mode: str = Form('bypass'),
                      protect_mask: UploadFile | None = File(None),
                      apply_mask: UploadFile | None = File(None), radius: float = Form(1.),
                      preview_id: str = Form(''), confirmed: bool = Form(False),
                      include_bundle: bool = Form(True)):
        contents = await file.read(MAX_UPLOAD_BYTES + 1)
        if preparation_mode == 'manual':
            if protect_mask is None or apply_mask is None:
                raise HTTPException(422, 'Manual preparation requires both Protect and Apply masks.')
            option = {'mode': 'manual', 'protect': await protect_mask.read(MAX_MASK_BYTES + 1),
                      'apply': await apply_mask.read(MAX_MASK_BYTES + 1), 'radius': radius,
                      'preview_id': preview_id, 'confirmed': confirmed}
        elif preparation_mode == 'bypass':
            # Reject ambiguous requests rather than silently discarding masks.
            if protect_mask is not None or apply_mask is not None or preview_id or confirmed:
                raise HTTPException(422, 'Bypass uses the original upload; remove preparation fields.')
            option = {'mode': 'bypass'}
        else:
            raise HTTPException(422, 'Choose bypass or reviewed manual preparation.')
        def make_result():
            result = controller.restore(contents, input_review, restoration, option,
                                        include_bundle=include_bundle)
            response = {'original_input': data_url(png(result['original256'])),
                        'prepared_input': data_url(png(result['prepared256'])),
                        'result': data_url(result['png']), 'metadata': result['metadata']}
            if result['bundle'] is not None:
                response['bundle'] = data_url(result['bundle'], 'application/zip')
            return response
        return await run_in_threadpool(invoke, make_result)

    return router
