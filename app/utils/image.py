from PIL import Image
import io
import base64


def get_image_from_base64(base64_string):
    base64_bytes = base64_string.encode("ascii")
    image_bytes = base64.b64decode(base64_bytes)

    # Convert to PIL Image object
    image = Image.open(io.BytesIO(image_bytes))

    return image


def get_base64_from_image(image: Image.Image):
    # Convert to base64 string
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return base64_string
