import torch
from diffusers import DiffusionPipeline


def build_pipeline(build_args: dict):
    if build_args is None:
        build_args = {}

    torch.set_grad_enabled(False)
    # refer to https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    # this will make float32 matmul faster
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    base = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )
    base.enable_model_cpu_offload()

    if build_args.get("use_torch_compile", False):
        base.unet = torch.compile(base.unet, mode="reduce-overhead", fullgraph=True)

    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=base.text_encoder_2,
        vae=base.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    refiner.enable_model_cpu_offload()

    if build_args.get("use_torch_compile", False):
        refiner.unet = torch.compile(
            refiner.unet, mode="reduce-overhead", fullgraph=True
        )

    default_params = build_args.get(
        "default_params",
        {
            "num_inference_steps": 30,
            "denoising_start": 0.8,
            "denoising_end": 0.8,
        },
    )

    return (base, refiner), default_params
