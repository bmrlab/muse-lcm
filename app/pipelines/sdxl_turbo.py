import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderTiny, AutoencoderKL


def build_pipeline(build_args: dict):
    if build_args is None:
        build_args = {}

    torch.set_grad_enabled(False)
    # refer to https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # this will make float32 matmul faster
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16"
    )

    if build_args.get("use_tiny_vae", False):
        pipe.vae = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", torch_dtype=torch.float16
        )

    if build_args.get("use_fp16_vae", False):
        # use this can save time for precision cast
        pipe.vae = AutoencoderKL.from_pretrained(
            "madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16
        )

    pipe.safety_checker = None
    pipe.set_progress_bar_config(disable=True)
    if not build_args.get("use_tiny_vae", False):
        # diffusers suggest enable this to avoid dtype conversion
        pipe.upcast_vae()
    pipe.to("cuda")

    if build_args.get("use_torch_compile", False):
        pipe.unet.to(memory_format=torch.channels_last)
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            print(f"failed to compile unet: {e}")
        try:
            pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)
        except Exception as e:
            print(f"failed to compile vae: {e}")

    if build_args.get("use_stablefast", False):
        from sfast.compilers.stable_diffusion_pipeline_compiler import (
            CompilationConfig,
            compile,
        )

        config = CompilationConfig.Default()

        if build_args.get("use_triton", False):
            try:
                import triton

                config.enable_triton = True
            except ImportError:
                print("Triton not installed, skip")

        if build_args.get("use_xformers", False):
            try:
                import xformers

                config.enable_xformers = True
            except ImportError:
                print("xformers not installed, skip")

        config.enable_cuda_graph = True
        config.trace_scheduler = build_args.get("sfast_trace_scheduler", False)
        config.preserve_parameters = build_args.get("sfast_preserve_parameters", False)

        pipe = compile(pipe, config)

    default_params = build_args.get(
        "default_params",
        {
            "num_inference_steps": 1,
            "guidance_scale": 0.0,
            "strength": 0.8,
        },
    )

    return pipe, default_params
