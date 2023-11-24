import torch
from diffusers import LCMScheduler, AutoPipelineForImage2Image


def build_pipeline():
    from . import disabled_safety_checker

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "Lykon/dreamshaper-7",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda")
    pipe.safety_checker = disabled_safety_checker

    # load and fuse lcm lora
    pipe.load_lora_weights(
        "latent-consistency/lcm-lora-sdv1-5"
    )
    pipe.fuse_lora()

    return pipe, {
        "num_inference_steps": 5,
        "guidance_scale": 1,
        "strength": 0.8,
    }
