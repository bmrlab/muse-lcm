import time
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.pipelines import pipeline
from app.utils.image import get_base64_from_image
from app.utils.websockets import ConnectionManager

router = APIRouter()

manager = ConnectionManager()


class PipelineRequest(BaseModel):
    pipeline: str
    build_args: Optional[dict] = {}


class GenerateRequest(BaseModel):
    prompt: str
    call_args: Optional[dict] = {}


def handle_request(prompt: str, call_args: dict):
    start_time = time.time()

    image = pipeline.pipeline[0](
        prompt=prompt,
        num_inference_steps=pipeline.default_params["num_inference_steps"],
        denoising_end=pipeline.default_params["denoising_end"],
        output_type="latent",
        num_images_per_prompt=call_args.get("base_batch_size", 1)
    ).images
    image = pipeline.pipeline[1](
        prompt=prompt,
        num_inference_steps=pipeline.default_params["num_inference_steps"],
        denoising_start=pipeline.default_params["denoising_start"],
        image=image,
        num_images_per_prompt=call_args.get("refiner_batch_size", 1)
    ).images[0]

    end_time = time.time()
    duration = end_time - start_time
    print(f"time: {duration}s, prompt: {prompt}")

    base64_string = get_base64_from_image(image)

    return base64_string


@router.post(
    "/generate",
)
async def generate(req: GenerateRequest):
    base64_string = handle_request(req.prompt, req.call_args)
    return {"result": base64_string}
