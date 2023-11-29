import torch
from diffusers import LCMScheduler, AutoPipelineForImage2Image, AutoencoderTiny


def build_pipeline():
    from . import disabled_safety_checker

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "SimianLuo/LCM_Dreamshaper_v7",
        torch_dtype=torch.float16,
    )
    pipe.vae = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd", torch_dtype=torch.float16
    )
    pipe.to("cuda")
    pipe.safety_checker = disabled_safety_checker

    pipe.unet.to(memory_format=torch.channels_last)
    try:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    except Exception as e:
        print(f"failed to compile unet: {e}")

    pipe.set_progress_bar_config(disable=True)

    return pipe, {
        "num_inference_steps": 5,
        "guidance_scale": 1.0,
        "strength": 0.8,
    }
