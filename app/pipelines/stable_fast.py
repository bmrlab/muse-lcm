import time

import torch
from diffusers import AutoencoderTiny, AutoPipelineForImage2Image, LCMScheduler
from sfast.compilers.stable_diffusion_pipeline_compiler import (
    CompilationConfig,
    compile,
)


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

    if build_args.get("safety_checker_none", False):
        pipe.safety_checker = None
    else:
        pipe.safety_checker = disabled_safety_checker

    if build_args.get("use_tiny_vae", False):
        pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesd", torch_dtype=torch.float16
        )

    pipe.to("cuda")

    pipe.set_progress_bar_config(disable=True)

    # start stable-fast model compile
    config = CompilationConfig.Default()

    if build_args.get("use_triton", False):
        try:
            import triton

            config.enable_triton = True
        except ImportError:
            print("Triton not installed, skip")

    config.enable_cuda_graph = True
    print("start compiling...")
    start_time = time.time()
    pipe = compile(pipe)
    print(f"compiling finished in {time.time() - start_time}s")

    default_params = build_args.get(
        "default_params",
        {
            "num_inference_steps": 5,
            "guidance_scale": 1,
            "strength": 0.8,
        },
    )

    return pipe, default_params
