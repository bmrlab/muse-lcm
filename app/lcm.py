from typing import Optional

from diffusers.utils import load_image
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.pipelines import load_default_pipeline
from app.system import validate_token
from app.utils.image import get_base64_from_image, get_image_from_base64

router = APIRouter()

pipeline, default_params = load_default_pipeline()


class PipelineRequest(BaseModel):
    pipeline: str


class GenerateRequest(BaseModel):
    prompt: str
    image_base64: str


@router.post("/generate", dependencies=[Depends(validate_token)])
async def generate(req: GenerateRequest):
    pil_img = load_image(get_image_from_base64(req.image_base64))
    image = pipeline(req.prompt, image=pil_img, **default_params).images[0]

    base64_string = get_base64_from_image(image)

    return {"result": base64_string}


@router.post("/update_pipeline")
async def update_pipeline(req: PipelineRequest):
    pass
