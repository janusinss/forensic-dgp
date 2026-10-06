"""Main-application endpoints for mask review and one plausible face estimate."""
import base64
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from face_workflow import UPLOAD_LIMIT, png_bytes
from dgp_face_workflow_v3 import DGPFaceWorkflow, decode_crop, decode_mask, make_dgp_bundle

router = APIRouter(prefix="/face", tags=["Reviewed face restoration"])
engine = DGPFaceWorkflow()


def data_url(array):
    return "data:image/png;base64," + base64.b64encode(png_bytes(array)).decode("ascii")


def review(contents):
    result = engine.review_mask(decode_crop(contents))
    return {**{k: v for k, v in result.items() if k not in ("original", "mask", "raw_mask")},
            "original": data_url(result["original"]), "mask": data_url(result["mask"]*255),
            "raw_mask": data_url(result["raw_mask"]*255)}


def generate(contents, mask_contents, restoration, include_bundle, input_review="", mask_reviewed=False):
    rgb = decode_crop(contents)
    mask = decode_mask(mask_contents, rgb.shape[:2])
    processed = engine.generate(rgb, mask, restoration, input_review, mask_reviewed)
    output, metadata = processed["output"], processed["metadata"]
    result = {"original": data_url(processed["original"]), "mask": data_url(processed["mask"]*255),
              "output": data_url(output), "metadata": metadata,
              "message": "256×256 research output. Hidden regions are plausible estimates; native CCTV usefulness remains unverified."}
    if include_bundle:
        bundle = make_dgp_bundle(rgb, mask, processed["original"], processed["mask"], output, metadata, processed["raw"])
        result["bundle"] = "data:application/zip;base64," + base64.b64encode(bundle).decode("ascii")
    return result


@router.get("/status")
def status():
    return engine.configuration()


@router.post("/mask")
async def mask_endpoint(file: UploadFile = File(...)):
    try:
        return await run_in_threadpool(review, await file.read(UPLOAD_LIMIT+1))
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(503, "Region detector unavailable. Paint or import the removal area manually.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logging.exception("Face removal-area prediction failed")
        raise HTTPException(503, "Region detector unavailable. Paint or import the removal area manually.") from exc


@router.post("/generate")
async def generate_endpoint(file: UploadFile = File(...), mask: UploadFile = File(...),
                            restoration: str = Form("auto"), include_bundle: bool = Form(True),
                            input_review: str = Form(""), mask_reviewed: bool = Form(False)):
    try:
        return await run_in_threadpool(generate, await file.read(UPLOAD_LIMIT+1),
                                       await mask.read(UPLOAD_LIMIT+1), restoration, include_bundle, input_review, mask_reviewed)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(503, "A configured DGP or completion model is unavailable. Check the local model files.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logging.exception("Reviewed face generation failed")
        raise HTTPException(500, "Generation failed. Review the crop and removal area before retrying.") from exc
