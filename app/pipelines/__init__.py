from . import sd15_lcm_lora

def load_default_pipeline():
    return sd15_lcm_lora.build_pipeline(
        {
            "use_stablefast": True,
            "use_triton": True,
            "use_xformers": True,
        }
    )


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False] * num_images
    else:
        return images, False


class Pipeline:
    def __init__(self):
        self.pipeline, self.default_params = load_default_pipeline()
    
    def update(self, pipe, params):
        self.pipeline = pipe
        self.default_params = params

pipeline = Pipeline()
