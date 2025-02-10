from __future__ import annotations
import os, sys

import threading

import torch
import torch.nn.functional as F
from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer
from exllamav2.conv import ExLlamaV2Conv
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.attn import ExLlamaV2Attention
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.config import ExLlamaV2Config
from exllamav2.module import ExLlamaV2Module
from exllamav2.compat import safe_move_tensor
from exllamav2.generator import ExLlamaV2MMEmbedding

from exllamav2.vlm.processor import pixtral, qwen2
from exllamav2.vlm.util import convert_to_rgb

from PIL.Image import Image

import math

class ExLlamaV2VisionTower(ExLlamaV2):

    config: ExLlamaV2Config
    modules: list[ExLlamaV2Module]

    # noinspection PyMissingConstructor
    def __init__(
        self,
        config: ExLlamaV2Config
    ):
        self.config = config
        cfg = self.config
        self.archparams = cfg.arch.vt
        km = self.archparams.keys
        self.modules = []

        # Preprocessor

        if cfg.vision_model_type == "pixtral":
            self.preprocess_func = pixtral.preprocess
            self.postprocess_func = pixtral.postprocess
            self.video_preprocess_func = None
            self.video_postprocess_func = None
        elif cfg.vision_model_type in ["qwen2", "qwen2.5"]:
            self.preprocess_func = qwen2.preprocess
            self.postprocess_func = qwen2.postprocess
            self.video_preprocess_func = qwen2.preprocess
            self.video_postprocess_func = qwen2.postprocess

        else:
            raise ValueError(f"Unknown vision model type: {cfg.vision_model_type}")

        # Position embeddings

        if cfg.vision_model_type == "pixtral":
            self.p_maxedge = cfg.vision_size["longest_edge"] // cfg.vision_patch_size["width"]
            freqs = 1.0 / (cfg.vision_rope_theta ** (torch.arange(0, cfg.vision_head_dim, 2).float() / cfg.vision_head_dim))
            h = torch.arange(self.p_maxedge, device = freqs.device)
            w = torch.arange(self.p_maxedge, device = freqs.device)
            freqs_h = torch.outer(h, freqs[::2]).float()
            freqs_w = torch.outer(w, freqs[1::2]).float()
            inv_freq = torch.cat(
                [
                    freqs_h[:, None, :].repeat(1, self.p_maxedge, 1),
                    freqs_w[None, :, :].repeat(self.p_maxedge, 1, 1),
                ],
                dim = -1,
            ).reshape(-1, cfg.vision_head_dim // 2)
            inv_freq = torch.cat((inv_freq, inv_freq), dim = -1)

            self.rope_cos = inv_freq.cos().half()
            self.rope_sin = inv_freq.sin().half()

            self.position_emb_func = pixtral.position_embeddings

        elif cfg.vision_model_type in ["qwen2", "qwen2.5"]:
            self.p_maxedge = cfg.vision_max_size
            dim = cfg.vision_head_dim // 2
            max_seqlen = int(math.ceil(cfg.vision_max_size / cfg.vision_spatial_patch_size))
            inv_freq = 1.0 / (cfg.vision_rope_theta ** (torch.arange(0, dim, 2).float() / dim)).half()
            s = torch.arange(max_seqlen, dtype = inv_freq.dtype).half()
            inv_freq = torch.outer(s, inv_freq)
            # inv_freq = torch.cat((inv_freq, inv_freq), dim = -1)

            self.rope_cos = inv_freq.cos().half()
            self.rope_sin = inv_freq.sin().half()

            self.position_emb_func = qwen2.position_embeddings

        # Patch embeddings

        if cfg.arch.vt.vision_conv3d:
            patch_size = (
                config.vision_temporal_patch_size,
                config.vision_patch_size["height"],
                config.vision_patch_size["width"],
            )
        else:
            patch_size = (
                config.vision_patch_size["height"],
                config.vision_patch_size["width"],
            )

        patch_conv = ExLlamaV2Conv(
            model = self,
            key = cfg.arch.vt_prefix + km["patch_conv"],
            in_channels = self.config.vision_num_channels,
            out_channels = self.config.vision_hidden_size,
            kernel_size = patch_size,
            has_bias = self.archparams.patch_conv_bias,
            archparams = self.archparams,
        )
        self.modules += [patch_conv]

        # Input norm

        if self.archparams.vision_input_norm:
            norm = ExLlamaV2RMSNorm(
                model = self,
                key = cfg.arch.vt_prefix + "ln_pre",
                archparams = self.archparams,
            )
            self.modules += [norm]

        # Decoder layers

        for layer_idx in range(self.config.vision_num_layers):
            layer_key = cfg.arch.vt_prefix + km["layers"] + f".{layer_idx}"
            attn = ExLlamaV2Attention(self, layer_key, layer_idx, archparams = self.archparams)
            mlp = ExLlamaV2MLP(self, layer_key, layer_idx, archparams = self.archparams, pad32 = False)
            self.modules += [attn, mlp]

        # Multimodal projection

        merge = cfg.vision_spatial_merge_size ** 2
        mmp = ExLlamaV2MLP(
            self,
            cfg.arch.mmp_prefix,
            0,
            archparams = cfg.arch.mmp,
            in_features = cfg.vision_hidden_size * merge,
            out_features = cfg.hidden_size,
            interm_features = cfg.vision_merger_intermediate_size,
            has_norm = True,
            has_residual = False,
            merge = merge,
            pad32 = False,
        )
        self.modules += [mmp]


    def forward(self, **kwargs):
        raise NotImplementedError()
    def forward_chunk(self, **kwargs):
        raise NotImplementedError()
    def load_tp(self, **kwargs):
        raise ValueError("load_tp not supported for vision model")
    def load_tp_gen(self, **kwargs):
        raise ValueError("load_tp not supported for vision model")
    def load_autosplit(self, **kwargs):
        raise ValueError("load_autosplit not supported for vision model")
    def load_autosplit_gen(self, **kwargs):
        raise ValueError("load_autosplit not supported for vision model")


    def process(
        self,
        hidden_states: torch.Tensor,
        patches_size = None,
        abort_event: threading.Event | None = None,
        thw_grid: tuple | None = None,
        **kwargs
    ):
        cfg = self.config

        if len(hidden_states.shape) == 2:
            hidden_states = hidden_states.unsqueeze(0)
        if len(hidden_states.shape) == 3:
            hidden_states = hidden_states.unsqueeze(0)

        bsz, channels, height, width = hidden_states.shape

        if patches_size is None:
            p_height = height // cfg.vision_patch_size["height"]
            p_width = width // cfg.vision_patch_size["width"]
        else:
            p_height, p_width = patches_size

        sin, cos = self.position_emb_func(
            self.config,
            p_height,
            p_width,
            self.p_maxedge,
            self.rope_sin,
            self.rope_cos,
            thw_grid
        )

        attn_params = ExLlamaV2Attention.Params(non_causal_attn = True)
        if self.config.vision_window_size:
            attn_params.block_diag_layers = set([
                l for l in range(self.config.vision_num_layers)
                if l not in self.config.vision_fullatt_block_indexes
            ])

            thw_grid_t = torch.tensor(list(thw_grid), dtype = torch.int).unsqueeze(0)
            window_index, csl = qwen2.get_window_index(thw_grid_t, self.config)
            csl = torch.tensor(csl, device = hidden_states.device, dtype = torch.int)
            csl = torch.unique_consecutive(csl)
            attn_params.cu_seqlens = csl
        else:
            window_index = None

        device = self.modules[0].device_idx
        for idx, module in enumerate(self.modules):

            # Respect abort signal

            if abort_event and abort_event.is_set():
                return None, None

            # Onward

            n_device = module.device_idx
            if idx == 0 or (n_device is not None and n_device != device and n_device >= 0):
                hidden_states = safe_move_tensor(hidden_states, n_device, non_blocking = True)

            if cos.device != hidden_states.device:
                cos = safe_move_tensor(cos, hidden_states.device)
                sin = safe_move_tensor(sin, hidden_states.device)

            if thw_grid is not None and isinstance(module, ExLlamaV2Attention):
                pa_shape = hidden_states.shape
                hidden_states = hidden_states.view(
                    thw_grid[0],
                    hidden_states.shape[1] // thw_grid[0],
                    hidden_states.shape[2]
                )

            hidden_states = module.forward(
                hidden_states,
                attn_params = attn_params,
                **kwargs | {
                    "alt_rope_embedding": (cos, sin),
                }
            )

            if thw_grid is not None and isinstance(module, ExLlamaV2Attention):
                hidden_states = hidden_states.view(pa_shape)

            if  window_index is not None and idx == 0:
                sh = hidden_states.shape
                unit = self.config.vision_spatial_merge_size ** 2
                seq_len = hidden_states.shape[0] * hidden_states.shape[1]
                hidden_states = hidden_states.reshape(seq_len // unit, unit, -1)
                hidden_states = hidden_states[window_index, :, :]
                hidden_states = hidden_states.reshape(sh)
                sh = sin.shape
                seq_len = sin.shape[0]
                sin = sin.reshape(seq_len // unit, unit, -1)
                sin = sin[window_index[:seq_len // unit], :, :]
                sin = sin.reshape(sh)
                cos = cos.reshape(seq_len // unit, unit, -1)
                cos = cos[window_index[:seq_len // unit], :, :]
                cos = cos.reshape(sh)

        if window_index is not None:
            reverse_indices = torch.argsort(window_index)
            hidden_states = hidden_states[:, reverse_indices]

        return hidden_states


    def get_image_embeddings(
        self,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
        image: Image,
        text_alias: str | None = None,
        embeddings_cpu: bool = True
    ) -> ExLlamaV2MMEmbedding:
        """
        :param model:
            Text model for which to produce embeddings

        :param tokenizer:
            Tokenizer

        :param image:
            Input PIL image

        :param text_alias:
            Text string to represent this embedding for tokenizing

        :param embeddings_cpu:
            Move embeddings to CPU. This can be skipped for simple jobs, but ideally embeddings should be cached
            when used with the dynamic generator, and it is not ideal to keep some large cache of data in VRAM. The
            overhead of copying them back to VRAM is relatively low. If this argument is False, embeddings will
            reside on whatever device the vision tower is loaded on.

        :return:
            ExLlamaV2MMEmbedding
        """

        width, height = image.size
        original_size = (height, width)

        maxsize = self.config.vision_max_size
        assert all(s <= maxsize for s in original_size), \
            f"Input image exceeds maximum size of {maxsize} x {maxsize}"

        image_tensor, prep_image_size, grid_thw, _ = self.preprocess_func(self.config, image)
        features_x = prep_image_size[0] // self.config.vision_patch_size["width"]
        features_y = prep_image_size[1] // self.config.vision_patch_size["height"]

        embedding_tensor = self.process(
            image_tensor,
            (features_y, features_x),
            thw_grid = grid_thw
        )

        if embeddings_cpu:
            embedding_tensor = embedding_tensor.cpu()

        embedding_tensor, pre_tokens, post_tokens = self.postprocess_func(
            model,
            tokenizer,
            embedding_tensor[0],
            features_y,
            features_x,
        )

        mme = ExLlamaV2MMEmbedding(
            model = model,
            embeddings = embedding_tensor,
            text_alias = text_alias,
            thw_grid = (1, features_y, features_x),
            pre_tokens = pre_tokens,
            post_tokens = post_tokens
        )

        mme.metadata.update({
            "original_size": original_size,
            "preprocessed_size": prep_image_size,
            "patches_size": (features_y, features_x),
        })

        return mme


    def get_video_embeddings(
        self,
        model: ExLlamaV2,
        tokenizer: ExLlamaV2Tokenizer,
        video: list[Image],
        text_alias: str | None = None,
        embeddings_cpu: bool = True
    ) -> ExLlamaV2MMEmbedding:
        """
        :param model:
            Text model for which to produce embeddings

        :param tokenizer:
            Tokenizer

        :param video:
            Video as list of PIL images, one per frame

        :param text_alias:
            Text string to represent this embedding for tokenizing

        :param embeddings_cpu:
            Move embeddings to CPU. This can be skipped for simple jobs, but ideally embeddings should be cached
            when used with the dynamic generator, and it is not ideal to keep some large cache of data in VRAM. The
            overhead of copying them back to VRAM is relatively low. If this argument is False, embeddings will
            reside on whatever device the vision tower is loaded on.

        :return:
            ExLlamaV2MMEmbedding
        """

        width, height = video[0].size
        assert all((width, height) == frame.size for frame in video), \
            "All video frames must have same dimensions"

        original_size = (height, width)

        video_tensor, prep_image_size, video_grid_thw, merge = self.preprocess_func(self.config, video)
        features_x = prep_image_size[0] // self.config.vision_patch_size["width"]
        features_y = prep_image_size[1] // self.config.vision_patch_size["height"]

        embedding_tensor = self.process(
            video_tensor,
            (features_y, features_x),
            thw_grid = video_grid_thw,
        )

        if embeddings_cpu:
            embedding_tensor = embedding_tensor.cpu()

        embedding_tensor, pre_tokens, post_tokens = self.postprocess_func(
            model,
            tokenizer,
            embedding_tensor[0],
            features_y,
            features_x,
        )

        mme = ExLlamaV2MMEmbedding(
            model = model,
            embeddings = embedding_tensor,
            text_alias = text_alias,
            thw_grid = video_grid_thw,
            pre_tokens = pre_tokens,
            post_tokens = post_tokens
        )

        mme.metadata.update({
            "original_size": original_size,
            "preprocessed_size": prep_image_size,
            "patches_size": (features_y, features_x),
        })

        return mme