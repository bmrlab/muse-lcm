from . import sd15_lcm_lora
from PIL import Image


def load_default_pipeline():
    return sd15_lcm_lora.build_pipeline(
        {
            "use_stablefast": True,
            "use_triton": True,
            "use_xformers": True,
            "use_tiny_vae": True,
        }
    )


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False
