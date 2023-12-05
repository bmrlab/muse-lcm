import importlib
import time

from diffusers.utils import load_image
from fastapi import (
    APIRouter,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
)
from pydantic import BaseModel

from typing import Optional

from app.pipelines import load_default_pipeline
from app.system import validate_token
from app.utils.image import get_base64_from_image, get_image_from_base64
from app.utils.websockets import ConnectionManager

router = APIRouter()

pipeline, default_params = load_default_pipeline()
cached_prompt = ""
cached_prompt_embedding = None

manager = ConnectionManager()


class PipelineRequest(BaseModel):
    pipeline: str
    build_args: Optional[dict] = {}


class GenerateRequest(BaseModel):
    prompt: str
    image_base64: str
    call_args: Optional[dict] = {}


def handle_request(image_base64: str, prompt: str, call_args: dict):
    global pipeline, default_params, cached_prompt, cached_prompt_embedding

    start_time = time.time()

    pil_img = load_image(get_image_from_base64(image_base64))

    if call_args.get("use_prompt_cache", False):
        if cached_prompt_embedding is None or prompt != cached_prompt:
            # tokenize the prompt
            prompt_embedding, _ = pipeline.encode_prompt(
                device="cuda",
                prompt=prompt,
                do_classifier_free_guidance=False,
                num_images_per_prompt=1,
            )

            cached_prompt_embedding = prompt_embedding
            cached_prompt = prompt

        image = pipeline(
            image=pil_img, prompt_embeds=cached_prompt_embedding, **default_params
        ).images[0]
    else:
        image = pipeline(image=pil_img, prompt=prompt, **default_params).images[0]

    base64_string = get_base64_from_image(image)

    end_time = time.time()
    duration = end_time - start_time

    print(f"time: {duration}s, prompt: {prompt}")

    return base64_string


@router.post("/generate", dependencies=[Depends(validate_token)])
async def generate(req: GenerateRequest):
    base64_string = handle_request(req.image_base64, req.prompt, req.call_args)
    return {"result": base64_string}


@router.websocket(
    "/generate/ws",
)
async def websocket_endpoint(websocket: WebSocket, token: str):
    if token is None or len(token) == 0:
        print("no token valid")
        raise WebSocketException(code=401, reason="token should be provided")

    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()

            try:
                validate_token(token=token)
            except Exception:
                print("invalid token")
                await manager.disconnect(websocket)
                break

            result = handle_request(
                data["image_base64"], data["prompt"], data.get("call_args", {})
            )

            await manager.send_personal_json(
                {"result": result},
                websocket,
            )
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.post("/update_pipeline")
async def update_pipeline(req: PipelineRequest):
    global pipeline, default_params

    try:
        pipeline_module = importlib.import_module(f"app.pipelines.{req.pipeline}")
        pipeline, default_params = pipeline_module.build_pipeline(req.build_args)

        return {"status": "ok"}
    except Exception as e:
        print(f"failed to update pipeline{req.pipeline}: {e}")
