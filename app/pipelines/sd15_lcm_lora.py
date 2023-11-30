import torch
from diffusers import LCMScheduler, AutoPipelineForImage2Image, AutoencoderTiny


def build_pipeline(build_args: dict):
    from . import disabled_safety_checker

    if build_args is None:
        build_args = {}

    pipe = AutoPipelineForImage2Image.from_pretrained(
        "Lykon/dreamshaper-7",
        torch_dtype=torch.float16,
        variant="fp16",
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    # load and fuse lcm lora
    pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    pipe.fuse_lora()

    if build_args.get("use_tiny_vae", False):
        pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=torch.float16
        )

    pipe.to("cuda")
    pipe.safety_checker = disabled_safety_checker

    if build_args.get("compile_unet", False):
        pipe.unet.to(memory_format=torch.channels_last)
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            print(f"failed to compile unet: {e}")

    if build_args.get("compile_vae", False):
        try:
            pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            print(f"failed to compile vae: {e}")

    pipe.set_progress_bar_config(disable=True)

    default_params = build_args.get(
        "default_params",
        {
            "num_inference_steps": 5,
            "guidance_scale": 1,
            "strength": 0.8,
        },
    )

    return pipe, default_params
