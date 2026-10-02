"""Main-application endpoints for mask review and one plausible face estimate."""
import base64
import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from face_workflow import (UPLOAD_LIMIT, decode_image,
                           decode_removal_mask, make_review_bundle, png_bytes)
from face_workflow_palette import PaletteFaceWorkflow

router = APIRouter(prefix="/face", tags=["Reviewed face restoration"])
engine = PaletteFaceWorkflow()


def data_url(array):
    return "data:image/png;base64," + base64.b64encode(png_bytes(array)).decode("ascii")


def review(contents):
    result = engine.review_mask(decode_image(contents))
    return {**{k: v for k, v in result.items() if k not in ("original", "mask", "raw_mask")},
            "original": data_url(result["original"]), "mask": data_url(result["mask"]*255),
            "raw_mask": data_url(result["raw_mask"]*255)}


def generate(contents, mask_contents, restoration, include_bundle):
    rgb = decode_image(contents)
    mask = decode_removal_mask(mask_contents, rgb.shape[:2])
    output, metadata = engine.generate(rgb, mask, restoration)
    result = {"original": data_url(rgb), "mask": data_url(mask*255),
              "output": data_url(output), "metadata": metadata,
              "message": "Hidden facial regions are plausible estimates. Review the original and removal area alongside the result."}
    if include_bundle:
        result["bundle"] = "data:application/zip;base64," + base64.b64encode(make_review_bundle(rgb, mask, output, metadata)).decode("ascii")
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
                            restoration: str = Form("auto"), include_bundle: bool = Form(True)):
    try:
        return await run_in_threadpool(generate, await file.read(UPLOAD_LIMIT+1),
                                       await mask.read(UPLOAD_LIMIT+1), restoration, include_bundle)
    except (FileNotFoundError, OSError) as exc:
        raise HTTPException(503, "The configured pretrained model is unavailable. Check the local model files.") from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        logging.exception("Reviewed face generation failed")
        raise HTTPException(500, "Generation failed. Review the crop and removal area before retrying.") from exc
