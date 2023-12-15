import time
from typing import Optional

from diffusers.utils import load_image
from fastapi import (
    APIRouter,
    Depends,
    WebSocket,
    WebSocketDisconnect,
    WebSocketException,
)
from pydantic import BaseModel

from app.system import validate_token, pipeline, default_params
from app.utils.image import get_base64_from_image, get_image_from_base64
from app.utils.translate import trans
from app.utils.websockets import ConnectionManager

router = APIRouter()

cached_prompt = ""
cached_prompt_embedding = None

manager = ConnectionManager()


class GenerateRequest(BaseModel):
    prompt: str
    image_base64: str
    call_args: Optional[dict] = {}


def handle_request(image_base64: str, prompt: str, call_args: dict):
    global cached_prompt, cached_prompt_embedding

    pil_img = load_image(get_image_from_base64(image_base64))

    start_time = time.time()

    if call_args.get("use_prompt_cache", False):
        if cached_prompt_embedding is None or prompt != cached_prompt:
            original_prompt = prompt
            if call_args.get("use_translate", False):
                try:
                    prompt = trans.translate(prompt)
                except Exception as e:
                    print(f"failed to translate: {e}")

            # tokenize the prompt
            prompt_embedding, _ = pipeline.encode_prompt(
                device="cuda",
                prompt=prompt,
                do_classifier_free_guidance=False,
                num_images_per_prompt=1,
            )

            cached_prompt_embedding = prompt_embedding
            cached_prompt = original_prompt

        image = pipeline(
            image=pil_img, prompt_embeds=cached_prompt_embedding, **default_params
        ).images[0]
    else:
        if call_args.get("use_translate", False):
            try:
                prompt = trans.translate(prompt)
            except Exception as e:
                print(f"failed to translate: {e}")

        image = pipeline(
            image=pil_img,
            prompt=prompt,
            **default_params,
        ).images[0]

    end_time = time.time()
    duration = end_time - start_time
    print(f"time: {duration}s, prompt: {prompt}")

    base64_string = get_base64_from_image(image)

    return base64_string


@router.post(
    "/generate",
    # FIXME disable temporarily
    #  dependencies=[Depends(validate_token)]
)
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
