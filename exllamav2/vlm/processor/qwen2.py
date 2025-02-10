from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.config import ExLlamaV2Config
from exllamav2.vlm.util import (
    convert_to_rgb,
    normalize_image,
    smart_resize,
)

def preprocess(
    config: ExLlamaV2Config,
    images: Image | list[Image]
) -> (torch.Tensor, tuple):

    resample = Image.Resampling(config.vision_resample)
    image_mean = tuple(config.vision_image_mean)
    image_std = tuple(config.vision_image_std)
    rescale_factor = config.vision_rescale_factor

    # Make list and truncate to whole number of spatial patches

    if not isinstance(images, list):
        mode = "image"
        images = [images]
    else:
        mode = "video"
        g = config.vision_temporal_patch_size
        frames = len(images)
        if frames > 1:
            frames = frames // g * g
            images = images[:frames]

    # Convert to RGB and resize as necessary

    images = [convert_to_rgb(image) for image in images]

    old_size = images[0].size
    assert all(old_size == frame.size for frame in images), \
        "All frames in video must have same dimensions"

    new_size = smart_resize(
        old_size,
        config.vision_spatial_patch_size * config.vision_spatial_merge_size,
        config.vision_min_pixels,
        config.vision_max_pixels,
    )
    if old_size != new_size:
        images = [image.resize(new_size, resample = resample) for image in images]

    # Convert to numpy array and normalize

    images = [np.array(image).astype(np.float32) for image in images]
    images = [image * rescale_factor for image in images]
    images = [normalize_image(image, image_mean, image_std) for image in images]

    # Reshape and convert to tensor

    patches = np.array(images)
    patches = patches.transpose(0, 3, 1, 2)
    if patches.shape[0] == 1:
        patches = np.tile(patches, (config.vision_temporal_patch_size, 1, 1, 1))
    channels = patches.shape[1]
    grid_t = patches.shape[0] // config.vision_temporal_patch_size
    grid_h = new_size[1] // config.vision_spatial_patch_size
    grid_w = new_size[0] // config.vision_spatial_patch_size
    patches = patches.reshape(
        grid_t,
        config.vision_temporal_patch_size,
        channels,
        grid_h // config.vision_spatial_merge_size,
        config.vision_spatial_merge_size,
        config.vision_spatial_patch_size,
        grid_w // config.vision_spatial_merge_size,
        config.vision_spatial_merge_size,
        config.vision_spatial_patch_size,
    )
    patches = patches.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten_patches = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * config.vision_temporal_patch_size * config.vision_spatial_patch_size ** 2
    )

    if mode == "image":
        image = torch.from_numpy(flatten_patches).half()
        return image, new_size, (grid_t, grid_h, grid_w), config.vision_spatial_patch_size ** 2
    else:
        video = torch.from_numpy(flatten_patches).half()
        return video, new_size, (grid_t, grid_h, grid_w), config.vision_spatial_patch_size ** 2

def postprocess(
    model: ExLlamaV2,
    tokenizer: ExLlamaV2Tokenizer,
    embeddings: torch.Tensor,
    features_y: int,
    features_x: int,
):
    """
    Enclose image tokens in <|vision_start|> and <|vision_end|> control tokens
    """

    id_start = tokenizer.single_id("<|vision_start|>")
    id_end = tokenizer.single_id("<|vision_end|>")
    img_start = model.modules[0].forward(torch.tensor([id_start], dtype = torch.long)).to(embeddings.device)
    img_end = model.modules[0].forward(torch.tensor([id_end], dtype = torch.long)).to(embeddings.device)

    embeddings = torch.cat((img_start, embeddings, img_end), dim = 0)
    return embeddings, 1, 1


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
    Create position IDs for Qwen2 grid
    """

    if thw_grid is not None:
        t, h, w = thw_grid
    else:
        h, w = height, width

    spm = config.vision_spatial_merge_size

    hpos_ids = torch.arange(h).unsqueeze(1).expand(-1, w)
    hpos_ids = hpos_ids.reshape(h // spm, spm, w // spm, spm)
    hpos_ids = hpos_ids.permute(0, 2, 1, 3)
    hpos_ids = hpos_ids.flatten()

    wpos_ids = torch.arange(w).unsqueeze(0).expand(h, -1)
    wpos_ids = wpos_ids.reshape(h // spm, spm, w // spm, spm)
    wpos_ids = wpos_ids.permute(0, 2, 1, 3)
    wpos_ids = wpos_ids.flatten()

    # ids = torch.stack([hpos_ids, wpos_ids], dim = -1).repeat(t, 1)
    ids = torch.stack([hpos_ids, wpos_ids], dim = -1)

    cos = rope_cos[ids].flatten(1)
    sin = rope_sin[ids].flatten(1)
    cos = cos.unsqueeze(1).repeat(1, 1, 2).contiguous()
    sin = sin.unsqueeze(1).repeat(1, 1, 2).contiguous()

    return sin, cos


def get_window_index(grid_thw, config: ExLlamaV2Config):

    window_index: list = []
    cu_window_seqlens: list = [0]
    window_index_id = 0
    vit_merger_window_size = (
        config.vision_window_size //
        config.vision_spatial_merge_size //
        config.vision_patch_size["height"]
    )

    for grid_t, grid_h, grid_w in grid_thw:
        llm_grid_h, llm_grid_w = (
            grid_h // config.vision_spatial_merge_size,
            grid_w // config.vision_spatial_merge_size,
        )
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w).reshape(grid_t, llm_grid_h, llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", -100)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != -100).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != -100]
        window_index.append(index_new + window_index_id)
        cu_seqlens_tmp = seqlens.cumsum(0) * config.vision_spatial_merge_size**2 + cu_window_seqlens[-1]
        cu_window_seqlens.extend(cu_seqlens_tmp.tolist())
        window_index_id += (grid_t * llm_grid_h * llm_grid_w).item()

    window_index = torch.cat(window_index, dim  =0)
    return window_index, cu_window_seqlens