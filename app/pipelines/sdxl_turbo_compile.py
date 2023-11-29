import torch
from diffusers import AutoPipelineForImage2Image, AutoencoderTiny


def build_pipeline():
    from . import disabled_safety_checker

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )
    pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesdxl", torch_dtype=torch.float16)
    pipe.to("cuda")
    pipe.safety_checker = disabled_safety_checker

    pipe.unet.to(memory_format=torch.channels_last)
    try:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    except Exception as e:
        print(f"failed to compile unet: {e}")

    pipe.set_progress_bar_config(disable=True)

    return pipe, {
        "num_inference_steps": 2,
        "guidance_scale": 0.0,
        "strength": 0.5,
    }
