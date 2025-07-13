import torch
import numpy as np
from PIL import Image
from typing import Tuple
import math

def convert_to_rgb(image: Image) -> Image:
    """
    Converts an image to RGB format and ensure any transparent regions are converted to white
    """
    if image.mode == "RGB":
        return image

    image = image.convert("RGBA")

    new_image = Image.new("RGBA", image.size, "WHITE")
    new_image.paste(image, (0, 0), image)
    new_image = new_image.convert("RGB")
    return new_image


def size_to_longest_edge_and_patch_size(
    input_size: tuple,
    max_size: tuple,
    patch_size: tuple,
) -> tuple:
    """
    Compute the output size for resizing an image while maintaining aspect ratio and constraining to a
    maximum bounding box while keeping each dimension a multiple of the corresponding patch dimension.
    """

    assert all(p % d == 0 for p, d in zip(max_size, patch_size)), \
        "max_size must be a multiple of patch_size"

    # Reduce to bounding box

    ratio = max(input_size[0] / max_size[0], input_size[1] / max_size[1])
    if ratio > 1:
        output_size = tuple(max(1, int(np.floor(d / ratio))) for d in input_size)
    else:
        output_size = input_size

    # Align size to patch grid

    output_size = tuple((((d + p - 1) // p) * p) for d, p in zip(output_size, patch_size))
    return output_size

def normalize_image(
    image: np.ndarray,
    mean: tuple,
    std: tuple,
) -> np.ndarray:
    """
    Normalizes RGB image in numpy format using the mean and standard deviation specified by `mean` and `std`:
    image = (image - mean(image)) / std
    """

    assert len(mean) == 3 and len(std) == 3, \
        "mean and std arguments must be 3D"

    # Upcast image to float32 if it's not already a float type

    if not np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32)

    mean = np.array(mean, dtype = image.dtype)
    std = np.array(std, dtype = image.dtype)
    image = (image - mean) / std
    return image


def smart_resize(
    size: tuple,
    factor: int = 28,
    min_pixels: int = 56 * 56,
    max_pixels: int = 14 * 14 * 4 * 1280
):
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.

    Adapted from:
    https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/image_processing_qwen2_vl.py
    """

    height, width = size

    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")

    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar