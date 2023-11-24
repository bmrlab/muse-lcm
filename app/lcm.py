import time
from typing import Optional

from diffusers.utils import load_image
from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.pipelines import load_default_pipeline
from app.system import validate_token
from app.utils.image import get_base64_from_image, get_image_from_base64

router = APIRouter()

pipeline, default_params = load_default_pipeline()
cached_prompt = ""
cached_prompt_embedding = None


class PipelineRequest(BaseModel):
    pipeline: str


class GenerateRequest(BaseModel):
    prompt: str
    image_base64: str


@router.post("/generate", dependencies=[Depends(validate_token)])
async def generate(req: GenerateRequest):
    global pipeline, default_params, cached_prompt, cached_prompt_embedding

    start_time = time.time()

    pil_img = load_image(get_image_from_base64(req.image_base64))

    if cached_prompt_embedding is None or req.prompt != cached_prompt:
        # tokenize the prompt
        prompt_inputs = pipeline.tokenizer(
            req.prompt, return_tensors="pt", padding="max_length"
        ).to("cuda")
        # create prompt encoding
        prompt_embeds = pipeline.text_encoder(**prompt_inputs)
        # extract CLIP embedding
        prompt_embeds = prompt_embeds["last_hidden_state"]

        cached_prompt_embedding = prompt_embeds
        cached_prompt = req.prompt

    image = pipeline(
        image=pil_img, prompt_embeds=cached_prompt_embedding, **default_params
    ).images[0]

    base64_string = get_base64_from_image(image)

    end_time = time.time()
    duration = end_time - start_time

    print(f"time: {duration}s, prompt: {req.prompt}")

    return {"result": base64_string}


@router.post("/update_pipeline")
async def update_pipeline(req: PipelineRequest):
    pass
