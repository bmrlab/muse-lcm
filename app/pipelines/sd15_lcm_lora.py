import torch
from diffusers import LCMScheduler, AutoPipelineForText2Image
import os


def build_pipeline():
    from . import MODEL_BASE_PATH, disabled_safety_checker

    pipe = AutoPipelineForText2Image.from_pretrained(
        # os.path.join(MODEL_BASE_PATH, "Lykon/dreamshaper-7"),
        "Lykon/dreamshaper-7",
        torch_dtype=torch.float32,
        variant="fp32",
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # pipe.to("cuda")
    pipe.safety_checker = disabled_safety_checker

    # load and fuse lcm lora
    pipe.load_lora_weights(
        # os.path.join(MODEL_BASE_PATH, "latent-consistency/lcm-lora-sdv1-5")
        "latent-consistency/lcm-lora-sdv1-5"
    )
    pipe.fuse_lora()

    return pipe, {
        "num_inference_steps": 4,
        "guidance_scale": 1,
        "strength": 0.6,
    }
