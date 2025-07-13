from __future__ import annotations

import torch
import numpy as np
from PIL import Image
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.config import ExLlamaV2Config
from exllamav2.vlm.util import (
    convert_to_rgb,
    size_to_longest_edge_and_patch_size,
    normalize_image
)

def preprocess(
    config: ExLlamaV2Config,
    image: Image
) -> (torch.Tensor, tuple):

    assert "longest_edge" in config.vision_size, \
        "preprocessing size must specify longest_edge"

    patch_size = tuple(config.vision_patch_size[d] * 2 for d in ["height", "width"])
    longest_edge = config.vision_size["longest_edge"]
    resample = Image.Resampling(config.vision_resample)
    image_mean = tuple(config.vision_image_mean)
    image_std = tuple(config.vision_image_std)
    rescale_factor = config.vision_rescale_factor

    # Convert to RGB and resize as necessary

    image = convert_to_rgb(image)
    old_size = image.size
    new_size = size_to_longest_edge_and_patch_size(image.size, (longest_edge, longest_edge), patch_size)
    if old_size != new_size:
        image = image.resize(new_size, resample = resample)

    # Convert to numpy array and normalize

    image = np.array(image).astype(np.float32)
    image = image * rescale_factor
    image = normalize_image(image, image_mean, image_std)

    # Convert to tensor, shape (3, resized_height, resized_width)

    image = image.transpose(2, 0, 1)
    image = torch.from_numpy(image).half()
    return image, new_size, None, None

def postprocess(
    model: ExLlamaV2,
    tokenizer: ExLlamaV2Tokenizer,
    embeddings: torch.Tensor,
    features_y: int,
    features_x: int,
):
    """
    Insert [IMG_BREAK] and [IMG_END] tokens in image feature embeddings
    """

    features_x //= model.config.vision_spatial_merge_size
    features_y //= model.config.vision_spatial_merge_size

    assert embeddings.shape[0] == features_y * features_x, \
        "Invalid shape for embeddings"

    id_break = tokenizer.single_id("[IMG_BREAK]")
    id_end = tokenizer.single_id("[IMG_END]")
    img_break = model.modules[0].forward(torch.tensor([id_break], dtype=torch.long)).to(embeddings.device)
    img_end = model.modules[0].forward(torch.tensor([id_end], dtype=torch.long)).to(embeddings.device)

    dim = embeddings.shape[-1]
    embeddings = embeddings.view((features_y, features_x, dim))
    break_col = img_break.expand(features_y, -1, -1)
    embeddings = torch.cat((embeddings, break_col), dim = 1)
    embeddings = embeddings.view((features_y * (features_x + 1)), dim)
    embeddings = torch.cat((embeddings, img_end), dim = 0)

    return embeddings, 0, 0


def position_embeddings(
    config: ExLlamaV2Config,
    height: int,
    width: int,
    max_width: int,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    thw_grid: tuple | None = None,
):
    """
    Create flat position IDs tensor for grid of patches: id(row, col) = row * max_width + col
    """

    assert thw_grid is None, \
        "Video not supported for Pixtral"

    row_indices = torch.arange(height).unsqueeze(1) * max_width
    col_indices = torch.arange(width).unsqueeze(0)
    ids = row_indices + col_indices
    ids = ids.flatten().unsqueeze(0)

    cos = rope_cos[ids]
    sin = rope_sin[ids]
    return sin, cos
