from __future__ import annotations

import torch
from torch import nn
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.headnorm import ExLlamaV2HeadNorm
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.architecture import RopeStyle
from exllamav2.tensor_p import BROADCAST_KV, BROADCAST_Q
import math
import torch.nn.functional as F
import inspect
import os

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2

# Detect available options for attention

has_flash_attn = False
has_flash_attn_with_paged = False
has_flash_attn_with_window = False
has_flash_attn_with_softcap = False
has_xformers = False
has_lower_right_sdpa = False

if 'EXLLAMA_NO_FLASH_ATTN' not in os.environ:

    try:
        import flash_attn
        flash_attn_ver = [int(t) for t in flash_attn.__version__.split(".") if t.isdigit()]
        is_ampere_or_newer_gpu = any(torch.cuda.get_device_properties(i).major >= 8 for i in range(torch.cuda.device_count()))

        if not is_ampere_or_newer_gpu:
            print(" ## Warning: Flash Attention is installed but unsupported GPUs were detected.")

        if [2, 2, 1] <= flash_attn_ver < [2, 5, 7]:
            from flash_attn import flash_attn_func, flash_attn_varlen_func
            has_flash_attn = True

        if [2, 5, 7] <= flash_attn_ver:
            from flash_attn import flash_attn_func, flash_attn_varlen_func, flash_attn_with_kvcache
            # import flash_attn_2_cuda as flash_attn_cuda

            signature = list(inspect.signature(flash_attn_func).parameters)
            has_flash_attn_with_window = "window_size" in signature
            has_flash_attn_with_softcap = "softcap" in signature

            import flash_attn_2_cuda as flash_attn_cuda
            # ext_c.set_flash_attn_func()

            has_flash_attn = True
            has_flash_attn_with_paged = True

    except ModuleNotFoundError:
        pass
    except NameError:
        pass

if 'EXLLAMA_NO_XFORMERS' not in os.environ:

    try:
        import xformers.ops as xops
        # LowerTriangularFromBottomRightMask was added in xformers version 2.4
        from xformers.ops.fmha.attn_bias import LowerTriangularFromBottomRightMask
        has_xformers = True
    except ModuleNotFoundError:
        pass

if 'EXLLAMA_NO_SDPA' not in os.environ:
    try:
        from torch.nn.attention.bias import causal_lower_right
        has_lower_right_sdpa = True
    except ImportError:
        pass


def assert_paged_attn():
    """
    Raise an exception if paged attention is not available.
    """
    global has_flash_attn_with_paged
    assert has_flash_attn_with_paged, \
        "Paged attention required Flash Attention 2.5.7 or later"


class ExLlamaV2Attention(ExLlamaV2Module):

    name: str = "Attention"

    layer_idx: int
    pre_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm | None
    post_layernorm: ExLlamaV2RMSNorm | ExLlamaV2LayerNorm | None
    q_proj: ExLlamaV2Linear | None
    k_proj: ExLlamaV2Linear | None
    v_proj: ExLlamaV2Linear | None
    o_proj: ExLlamaV2Linear | None
    q_norm: ExLlamaV2HeadNorm | None
    k_norm: ExLlamaV2HeadNorm | None

    q_handle: int | None

    temp_state: torch.tensor
    temp_q: torch.tensor
    temp_k: torch.tensor
    temp_v: torch.tensor
    temp_o: torch.tensor
    temp_dq: torch.tensor
    # temp_kv: torch.tensor

    temp_lora_size: int

    has_norm: bool
    has_residual: bool
    scaling: float
    sliding_window: int

    is_tp: bool
    tp_dq_size: list[int] | None

    from exllamav2.attn_params import Params
    from exllamav2.attn_params import PagedParams

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        layer_idx: int,
        has_norm: bool = True,
        has_residual: bool = True,
        sliding_window: int = 0,
        archparams = None
    ):
        super().__init__(model, key, archparams)

        cfg = self.model.config
        ap = self.archparams
        km = self.archparams.keys

        self.is_tp = False
        self.tp_dq_size = None

        self.layer_idx = layer_idx
        self.has_norm = has_norm
        self.has_residual = has_residual

        self.q_handle = None
        self.temp_lora_size = 0

        if ap.is_vision:
            self.num_attention_heads = cfg.vision_num_attention_heads
            self.num_key_value_heads = cfg.vision_num_key_value_heads
            self.num_key_value_groups = cfg.vision_num_key_value_groups
            self.head_dim = cfg.vision_head_dim
            self.hidden_size = cfg.vision_hidden_size
        else:
            self.num_attention_heads = cfg.num_attention_heads
            if type(cfg.num_key_value_heads) is list:
                self.num_key_value_heads = cfg.num_key_value_heads[layer_idx]
            else:
                self.num_key_value_heads = cfg.num_key_value_heads
            if type(cfg.num_key_value_groups) is list:
                self.num_key_value_groups = cfg.num_key_value_groups[layer_idx]
            else:
                self.num_key_value_groups = cfg.num_key_value_groups
            self.head_dim = cfg.head_dim
            self.hidden_size = cfg.hidden_size

        hidden_size = self.hidden_size

        if self.has_norm and (km["norm_1"] or km["norm_1_post"]):
            if ap.norm == "layernorm":
                self.pre_layernorm = ExLlamaV2LayerNorm(model, key + km["norm_1"], archparams)
                self.post_layernorm = ExLlamaV2LayerNorm(model, key + km["norm_1_post"], archparams) if km["norm_1_post"] else None
            elif ap.norm == "rmsnorm":
                self.pre_layernorm = ExLlamaV2RMSNorm(model, key + km["norm_1"], archparams)
                self.post_layernorm = ExLlamaV2RMSNorm(model, key + km["norm_1_post"], archparams) if km["norm_1_post"] else None
        else:
            self.pre_layernorm = None
            self.post_layernorm = None
            self.has_norm = False

        f_a = 0
        f_b = self.num_attention_heads * self.head_dim
        f_c = f_b + self.num_key_value_heads * self.head_dim
        f_d = f_c + self.num_key_value_heads * self.head_dim
        f_key = (key + km["fused_qkv"]) if km["fused_qkv"] else None

        if type(cfg.num_key_value_heads) is list and ["self_attn.linear_attn"] in cfg.arch.lm.layer_keys[layer_idx]:
            self.q_proj = None
            self.k_proj = None
            self.v_proj = None
            self.o_proj = ExLlamaV2Linear(model, key + km["linear_attn"], self.num_attention_heads * self.head_dim, hidden_size, ap.attention_bias_o, prescale = cfg.scale_depth)
        else:
            self.q_proj = ExLlamaV2Linear(model, key + km["attn_q"], hidden_size, self.num_attention_heads * self.head_dim, ap.attention_bias_qkv, f_key = f_key, f_beg = f_a, f_end = f_b, altpack_qkv = ap.fused_qkv_altpack)
            self.k_proj = ExLlamaV2Linear(model, key + km["attn_k"], hidden_size, self.num_key_value_heads * self.head_dim, ap.attention_bias_qkv, f_key = f_key, f_beg = f_b, f_end = f_c, altpack_qkv = ap.fused_qkv_altpack)
            self.v_proj = ExLlamaV2Linear(model, key + km["attn_v"], hidden_size, self.num_key_value_heads * self.head_dim, ap.attention_bias_qkv, f_key = f_key, f_beg = f_c, f_end = f_d, altpack_qkv = ap.fused_qkv_altpack)
            self.o_proj = ExLlamaV2Linear(model, key + km["attn_o"], self.num_attention_heads * self.head_dim, hidden_size, ap.attention_bias_o, prescale = cfg.scale_depth)

        if cfg.use_qk_norm:
            self.q_norm = ExLlamaV2HeadNorm(model, key + ".self_attn.q_norm", self.num_attention_heads, self.head_dim)
            self.k_norm = ExLlamaV2HeadNorm(model, key + ".self_attn.k_norm", self.num_key_value_heads, self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if type(cfg.num_key_value_heads) is list and ["self_attn.linear_attn"] in cfg.arch.lm.layer_keys[layer_idx]:
            self.submodules = [
                self.o_proj
            ]
        else:
            self.submodules = [
                self.q_proj,
                self.k_proj,
                self.v_proj,
                self.o_proj
            ]

        if self.pre_layernorm:
            self.submodules += [self.pre_layernorm]
        if self.post_layernorm:
            self.submodules += [self.post_layernorm]
        if cfg.use_qk_norm:
            self.submodules += [self.q_norm, self.k_norm]

        if cfg.attention_multiplier:
            self.scaling = cfg.attention_multiplier
        elif cfg.query_pre_attn_scalar:
            self.scaling = cfg.query_pre_attn_scalar ** (-0.5)
        else:
            self.scaling = 1 / math.sqrt(self.head_dim)

        self.sliding_window = sliding_window


    def numel(self) -> int:

        numel = self.o_proj.numel()
        if self.q_proj is not None: numel += self.q_proj.numel()
        if self.k_proj is not None: numel += self.k_proj.numel()
        if self.v_proj is not None: numel += self.v_proj.numel()

        if self.pre_layernorm is not None: numel += self.pre_layernorm.numel()
        if self.post_layernorm is not None: numel += self.post_layernorm.numel()
        if self.q_norm is not None: numel += self.q_norm.numel()
        if self.k_norm is not None: numel += self.k_norm.numel()

        return numel


    @torch.inference_mode
    def load(self, device_context: bool = True):

        cfg = self.model.config

        if self.pre_layernorm is not None: self.pre_layernorm.load()
        if self.post_layernorm is not None: self.post_layernorm.load()
        self.o_proj.load(device_context = device_context)
        if self.q_proj is None and self.k_proj is None and self.v_proj is None:
            return
        self.q_proj.load(device_context = device_context)
        self.k_proj.load(device_context = device_context)
        self.v_proj.load(device_context = device_context)
        if self.q_norm is not None: self.q_norm.load()
        if self.k_norm is not None: self.k_norm.load()

        if self.q_proj.is_quant():

            assert self.k_proj.is_quant() and self.v_proj.is_quant() and self.o_proj.is_quant(), "Partially quantized attention layer"

            if device_context:
                device_context = self.model.get_device_context(self.device_idx)
                device_context.begin_scratch_alloc()
                self.temp_state = device_context.get_scratch_slice(self.temp_state_size())
                # self.temp_q = device_context.get_scratch_slice(self.temp_q_size())
                # self.temp_k = device_context.get_scratch_slice(self.temp_k_size())
                # self.temp_v = device_context.get_scratch_slice(self.temp_v_size())
                self.temp_dq = device_context.get_scratch_slice(self.temp_dq_size())
                # self.temp_kv = device_context.get_scratch_slice(self.temp_kv_size()) if self.num_attention_heads != self.num_key_value_heads else None
            else:
                self.temp_state = none_tensor
                self.temp_dq = none_tensor

            if self.has_norm:
                norm_weight = self.pre_layernorm.weight if self.pre_layernorm.weight is not None else none_tensor
                norm_bias = self.pre_layernorm.bias if self.pre_layernorm.bias is not None else none_tensor
                is_rms = isinstance(self.pre_layernorm, ExLlamaV2RMSNorm)
                eps = self.pre_layernorm.variance_epsilon
            else:
                norm_weight = none_tensor
                norm_bias = none_tensor
                is_rms = False
                eps = 0

            if self.post_layernorm is not None:
                post_norm_weight = self.post_layernorm.weight if self.post_layernorm.weight is not None else none_tensor
                post_norm_bias = self.post_layernorm.bias if self.post_layernorm.bias is not None else none_tensor
            else:
                post_norm_weight = none_tensor
                post_norm_bias = none_tensor

            if self.q_norm is None:
                q_norm = none_tensor
            else:
                q_norm = self.q_norm.weight

            if self.k_norm is None:
                k_norm = none_tensor
            else:
                k_norm = self.k_norm.weight

            self.q_handle = ext_c.make_q_attn(
                norm_weight,
                norm_bias,
                is_rms,
                eps,
                self.q_proj.q_handle,
                self.k_proj.q_handle,
                self.v_proj.q_handle,
                self.o_proj.q_handle,
                self.temp_state,
                # self.temp_q,
                # self.temp_k,
                # self.temp_v,
                self.temp_dq,
                cfg.max_input_len * cfg.max_batch_size,
                self.hidden_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.head_dim,
                cfg.max_seq_len,
                self.has_residual,
                self.archparams.rope_style.value,
                q_norm,
                k_norm,
                post_norm_weight,
                post_norm_bias,
                self.archparams.residual_stream_fp32,
                not cfg.no_graphs
            )


    def unload(self):
        if self.q_handle is not None:
            ext_c.free_q_attn(self.q_handle)
            self.q_handle = None

        if self.pre_layernorm is not None: self.pre_layernorm.unload()
        if self.post_layernorm is not None: self.post_layernorm.unload()
        if self.q_proj is not None: self.q_proj.unload()
        if self.k_proj is not None: self.k_proj.unload()
        if self.v_proj is not None: self.v_proj.unload()
        self.o_proj.unload()

        self.temp_state = None
        self.temp_dq = None

        if self.q_norm is not None: self.q_norm.unload()
        if self.k_norm is not None: self.k_norm.unload()


    def weight_footprint(self):

        fp = self.o_proj.weight_footprint()
        if self.q_proj is not None: fp += self.q_proj.weight_footprint()
        if self.k_proj is not None: fp += self.k_proj.weight_footprint()
        if self.v_proj is not None: fp += self.v_proj.weight_footprint()
        if self.pre_layernorm is not None:
            fp += self.pre_layernorm.weight_footprint()
        if self.post_layernorm is not None:
            fp += self.post_layernorm.weight_footprint()
        if self.q_norm is not None:
            fp += self.q_norm.weight_footprint()
        if self.k_norm is not None:
            fp += self.k_norm.weight_footprint()

        return fp


    def scratch_space_fixed(self):

        return self.temp_state_size() + \
               self.temp_dq_size()


    def scratch_space(self):

        return self.temp_state_size() + \
               self.temp_q_size() + \
               self.temp_k_size() + \
               self.temp_v_size() + \
               self.temp_dq_size() + \
               self.temp_kv_size()
               # self.temp_attn_size() +  # Accounted for separately in model.set_device_map()


    def temp_state_size(self):

        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * max(self.num_attention_heads * self.head_dim, self.hidden_size) * 2 + 128


    def temp_q_size(self):

        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * self.num_attention_heads * self.head_dim * 2 + 128


    def temp_k_size(self):

        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * self.num_key_value_heads * self.head_dim * 2 + 128


    def temp_v_size(self):

        cfg = self.model.config
        return cfg.max_input_len * cfg.max_batch_size * self.num_key_value_heads * self.head_dim * 2 + 128


    def temp_dq_size(self):

        if self.q_proj is None and self.k_proj is None and self.v_proj is None:
            return self.o_proj.temp_dq_size()
        else:
            return max(self.q_proj.temp_dq_size(),
                       self.k_proj.temp_dq_size(),
                       self.v_proj.temp_dq_size(),
                       self.o_proj.temp_dq_size())


    def temp_kv_size(self):

        cfg = self.model.config
        if self.num_key_value_heads == self.num_attention_heads: return 0
        return 2 * cfg.max_seq_len * cfg.max_batch_size * self.num_attention_heads * self.head_dim * 2 + 128


    def temp_attn_size(self):
        global has_flash_attn
        global has_xformers

        cfg = self.model.config
        att_max = min(cfg.max_attention_size, cfg.max_seq_len ** 2)

        if (has_flash_attn and not cfg.no_flash_attn) or (has_xformers and not cfg.no_xformers) :
            #in sm>=80 devices, xformers uses the same memory as flash_attn
            #todo: due to the different implementions. in sm<80 devices, xformers uses less memory than it in sm>=80. There may still be room for optimization.
            eff = cfg.max_attention_size ** 0.5 / 190  # based on supposed memory savings listed in flash-attn repo + some fudging
            att_max //= eff

        return 2 * att_max * self.num_attention_heads * 2 + 128


    def set_device_idx(self, idx: int | None):
        super().set_device_idx(idx)

        if self.pre_layernorm is not None: self.pre_layernorm.set_device_idx(idx)
        if self.post_layernorm is not None: self.post_layernorm.set_device_idx(idx)
        if self.q_proj is not None: self.q_proj.set_device_idx(idx)
        if self.k_proj is not None: self.k_proj.set_device_idx(idx)
        if self.v_proj is not None: self.v_proj.set_device_idx(idx)
        self.o_proj.set_device_idx(idx)
        if self.q_norm is not None: self.q_norm.set_device_idx(idx)
        if self.k_norm is not None: self.k_norm.set_device_idx(idx)


    def repeat_kv(self, hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:

        if n_rep == 1: return hidden_states

        batch, num_key_value_heads, slen, head_dim = hidden_states.shape
        hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        hidden_states = hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        return hidden_states


    # @profile
    def forward_paged(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.PagedParams | None = None,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:

        if self.q_proj is None and self.k_proj is None and self.v_proj is None:
            return self.forward_paged_linear(
                hidden_states,
                cache,
                attn_params,
                loras,
                **kwargs,
            )
        if self.is_tp:
            return self.forward_paged_tp(
                hidden_states,
                cache,
                attn_params,
                loras,
                **kwargs,
            )

        is_q = self.q_handle is not None
        cfg = self.model.config
        constants = self.model.get_device_context(self.device_idx, scratch = is_q)
        page_size = attn_params.page_size
        batch_size, q_len, _ = hidden_states.shape
        cache_seqlens = attn_params.get_cache_seqlens(self.device_idx)
        block_table = attn_params.get_block_index(self.device_idx)

        # TODO: We only need keys/values when preprocess_only == True, so we could skip q projection and attention
        #   on the last layer. Would need custom kernel to update paged cache if not calling flash_attn_with_kvcache
        # skip_attn = kwargs.get("kv_only")

        # TODO: Potentially we could emulate paged cache when in Q4 mode, since that requires copying the active part
        #   of the current cache layer anyway. Test if block diagonal masking works with lower-right aligned mask.

        if cache.q_block > 1:
            k_cache_f, v_cache_f = cache.get_kv_state(self.layer_idx, batch_size, 0, attn_params.max_cache_seqlen, page_size, cache_seqlens, block_table)
        else:
            k_cache_f, v_cache_f = cache.get_kv_state(self.layer_idx, batch_size, 0, 0, page_size, cache_seqlens, block_table)

        k_cache = k_cache_f.view(k_cache_f.shape[1] // page_size, page_size, k_cache_f.shape[2], k_cache_f.shape[3])
        v_cache = v_cache_f.view(v_cache_f.shape[1] // page_size, page_size, v_cache_f.shape[2], v_cache_f.shape[3])

        sc = attn_params.get_alt_rope_embed(self.device_idx)
        if not sc:
            sin, cos = constants.sin, constants.cos
        else:
            sin, cos = sc

        cache_seqlens_rope = cache_seqlens
        offsets = attn_params.get_rope_offsets(self.device_idx)
        if offsets is not None:
            cache_seqlens_rope = cache_seqlens_rope + offsets

        if is_q:
            q = torch.empty((batch_size, q_len, self.num_attention_heads, self.head_dim), device = hidden_states.device, dtype = torch.half)
            if attn_params.is_sequential:
                assert batch_size == 1
                k = k_cache_f[:, attn_params.first_index : attn_params.first_index + q_len, :, :]
                v = v_cache_f[:, attn_params.first_index : attn_params.first_index + q_len, :, :]
            else:
                k = torch.empty((batch_size, q_len, self.num_key_value_heads, self.head_dim), device = hidden_states.device, dtype = torch.half)
                v = torch.empty((batch_size, q_len, self.num_key_value_heads, self.head_dim), device = hidden_states.device, dtype = torch.half)

            if loras is None or self.temp_lora_size == 0:
                pass_loras = []
                pass_lora_temp = none_tensor
            else:
                pass_loras = [id(x) for x in loras]
                pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

            ext_c.q_attn_forward_1(
                self.q_handle,
                hidden_states,
                batch_size,
                q_len,
                0,
                cache_seqlens_rope,
                q,
                k,
                v,
                sin,
                cos,
                pass_loras,
                pass_lora_temp
            )
        else:
            residual = hidden_states
            hidden_states = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states
            q = self.q_proj.forward(hidden_states, loras = loras)
            k = self.k_proj.forward(hidden_states, loras = loras)
            v = self.v_proj.forward(hidden_states, loras = loras)
            q = q.view(batch_size, q_len, self.num_attention_heads, self.head_dim)
            k = k.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
            v = v.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
            if cfg.use_qk_norm:
                q = self.q_norm.forward(q)
                k = self.k_norm.forward(k)
            if self.archparams.rope_style != RopeStyle.NONE:
                for t, heads in [(q, self.num_attention_heads), (k, self.num_key_value_heads)]:
                    ext_c.rope_(
                        t,
                        sin,
                        cos,
                        0,
                        heads,
                        self.head_dim,
                        cache_seqlens_rope,
                        self.archparams.rope_style == RopeStyle.NEOX
                    )
            if attn_params.is_sequential:
                k_ = k_cache_f[:, attn_params.first_index : attn_params.first_index + q_len, :, :]
                v_ = v_cache_f[:, attn_params.first_index : attn_params.first_index + q_len, :, :]
                k_.copy_(k)
                v_.copy_(v)

        if attn_params.is_sequential:
            k = None
            v = None
            cache_seqlens_a = attn_params.get_cache_seqlens_after(self.device_idx)
        else:
            cache_seqlens_a = cache_seqlens

        if cache.q_block == 1:
            cache.get_kv_state(self.layer_idx, batch_size, 0, attn_params.max_cache_seqlen, page_size, cache_seqlens, block_table)

        flash_kwargs = {}
        if self.sliding_window:
            # assert has_flash_attn_with_window, \
            #     "Installed version of flash-attn does not support sliding window"
            if has_flash_attn_with_window:
                flash_kwargs["window_size"] = (self.sliding_window, self.sliding_window)
        if cfg.attn_logit_softcapping:
            # assert has_flash_attn_with_softcap, \
            #     "Installed version of flash-attn does not support softcapping"
            if has_flash_attn_with_softcap:
                flash_kwargs["softcap"] = cfg.attn_logit_softcapping

        attn_output = flash_attn_with_kvcache(
            q = q,
            k = k,
            v = v,
            k_cache = k_cache,
            v_cache = v_cache,
            cache_seqlens = cache_seqlens_a,
            block_table = block_table,
            causal = True,
            softmax_scale = self.scaling,
            **flash_kwargs
        )

        attn_output = attn_output.view((batch_size, q_len, self.num_attention_heads * self.head_dim))

        cache.store_kv_state(self.layer_idx, batch_size, 0, q_len, page_size, cache_seqlens, block_table)

        # Output projection

        if is_q:
            ext_c.q_attn_forward_2(
                self.q_handle,
                hidden_states,
                attn_output,
                batch_size,
                q_len,
                pass_loras,
                pass_lora_temp
            )
        else:
            hidden_states = self.o_proj.forward(attn_output, loras = loras)
            if self.post_layernorm:
                hidden_states = self.post_layernorm.forward(hidden_states)
            if self.has_residual:
                hidden_states += residual

        return hidden_states

    # @profile
    def forward_paged_linear(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.PagedParams | None = None,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:

        is_q = self.q_handle is not None
        cfg = self.model.config
        constants = self.model.get_device_context(self.device_idx, scratch = is_q)
        page_size = attn_params.page_size
        batch_size, q_len, _ = hidden_states.shape
        cache_seqlens = attn_params.get_cache_seqlens(self.device_idx)
        block_table = attn_params.get_block_index(self.device_idx)

        sc = attn_params.get_alt_rope_embed(self.device_idx)
        if not sc:
            sin, cos = constants.sin, constants.cos
        else:
            sin, cos = sc

        cache_seqlens_rope = cache_seqlens
        offsets = attn_params.get_rope_offsets(self.device_idx)
        if offsets is not None:
            cache_seqlens_rope = cache_seqlens_rope + offsets

        residual = hidden_states
        hidden_states = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states

        attn_output = hidden_states.view((batch_size, q_len, self.num_attention_heads * self.head_dim))

        cache.store_kv_state(self.layer_idx, batch_size, 0, q_len, page_size, cache_seqlens, block_table)

        hidden_states = self.o_proj.forward(attn_output, loras = loras)
        if self.post_layernorm:
            hidden_states = self.post_layernorm.forward(hidden_states)
        if self.has_residual:
            hidden_states += residual

        return hidden_states


    # @profile
    def forward_paged_tp(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.PagedParams | None = None,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:

        cfg = self.model.config
        ctx = self.model.tp_context

        assert not self.sliding_window, \
            "Sliding window not supported in TP mode"

        attn_params.prep_tp(self.model)
        page_size = attn_params.page_size

        batch_size, q_len, _ = hidden_states.shape
        rows = batch_size * q_len
        hidden_states = hidden_states.view(-1, self.hidden_size)
        dtype = hidden_states.dtype

        k_cache_f, v_cache_f = cache.get_kv_state(
            self.layer_idx,
            batch_size,
            0,
            attn_params.max_cache_seqlen,
            page_size,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp
        )

        k_cache = [x.view(x.shape[1] // page_size, page_size, x.shape[2], x.shape[3]) for x in k_cache_f]
        v_cache = [x.view(x.shape[1] // page_size, page_size, x.shape[2], x.shape[3]) for x in v_cache_f]

        sin, cos = ctx.get_sin_cos()

        ext_c.tp_attn_forward_paged_(
            self.model.tp_context.ext_tp_context,
            hidden_states,
            self.temp_bc0,
            self.temp_bc1,
            self.temp_bc2,
            self.temp_q,
            self.temp_k,
            self.temp_v,
            self.temp_o,
            k_cache,
            v_cache,
            self.pre_layernorm.weight if self.pre_layernorm is not None else [],
            self.pre_layernorm.variance_epsilon if self.pre_layernorm is not None else 0.0,
            self.q_proj.q_handle,
            self.k_proj.q_handle,
            self.v_proj.q_handle,
            self.o_proj.q_handle,
            self.head_dim,
            int(self.archparams.rope_style),
            batch_size,
            q_len,
            sin,
            cos,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp,
            self.scaling
        )

        cache.store_kv_state(
            self.layer_idx,
            batch_size,
            0,
            q_len,
            page_size,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp
        )

        return ctx.get_pinned(0, batch_size, q_len, self.hidden_size)


    # @profile
    def forward_paged_tp_old(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.PagedParams | None = None,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:

        assert self.q_handle is not None
        cfg = self.model.config
        split = self.model.tp_context.get_split(BROADCAST_KV)
        batch_size, q_len, _ = hidden_states.shape
        attn_params.prep_tp(self.model)
        page_size = attn_params.page_size
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])

        k_cache_f, v_cache_f = cache.get_kv_state(
            self.layer_idx,
            batch_size,
            0,
            attn_params.max_cache_seqlen,
            page_size,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp
        )

        k_cache = [x.view(x.shape[1] // page_size, page_size, x.shape[2], x.shape[3]) for x in k_cache_f]
        v_cache = [x.view(x.shape[1] // page_size, page_size, x.shape[2], x.shape[3]) for x in v_cache_f]

        hidden_states = self.model.tp_context.broadcast(0, hidden_states, BROADCAST_KV, dim = self.head_dim)

        residual = hidden_states

        post_norm = self.pre_layernorm.forward_tp(hidden_states, output_split = True) if self.has_norm else hidden_states
        q = self.q_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim)
        k = self.k_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim)
        v = self.v_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim)
        q = [q_.view(batch_size, q_len, q_.shape[1] // self.head_dim, self.head_dim) for q_ in q]
        k = [k_.view(batch_size, q_len, k_.shape[1] // self.head_dim, self.head_dim) for k_ in k]
        v = [v_.view(batch_size, q_len, v_.shape[1] // self.head_dim, self.head_dim) for v_ in v]
        if cfg.use_qk_norm:
            assert False, "TP not implemented for QK norm"  # TODO: ...
            # q = self.q_norm.forward(q)
            # k = self.k_norm.forward(k)
        if self.archparams.rope_style != RopeStyle.NONE:
            for idx, (dev, a, b) in enumerate(split):
                context = self.model.get_device_context(dev)
                torch.cuda.set_stream(context.stream)
                for t, heads in [(q[idx], self.num_key_value_groups), (k[idx], 1)]:
                    ext_c.rope_(
                        t,
                        context.sin,
                        context.cos,
                        0,
                        (b - a) * heads,
                        self.head_dim,
                        attn_params.cache_seqlens_tp[idx],
                        self.archparams.rope_style == RopeStyle.NEOX
                    )
        if attn_params.is_sequential:
            k_ = [x[:, attn_params.first_index: attn_params.first_index + q_len, :, :] for x in k_cache_f]
            v_ = [x[:, attn_params.first_index: attn_params.first_index + q_len, :, :] for x in v_cache_f]
            for (dev, a, b), x_, x, y_, y in zip(split, k_, k, v_, v):
                context = self.model.get_device_context(dev)
                torch.cuda.set_stream(context.stream)
                x_.copy_(x)
                y_.copy_(y)
            k = None
            v = None
            cache_seqlens_a = attn_params.cache_seqlens_after_tp
        else:
            cache_seqlens_a = attn_params.cache_seqlens_tp

        # if cache.q_block == 1:
        #     cache.get_kv_state(
        #         self.layer_idx,
        #         batch_size,
        #         0,
        #         attn_params.max_cache_seqlen,
        #         page_size,
        #         attn_params.cache_seqlens_tp,
        #         attn_params.block_index_tp
        #     )

        flash_kwargs = {}
        if self.sliding_window:
            # assert has_flash_attn_with_window, \
            #     "Installed version of flash-attn does not support sliding window"
            if has_flash_attn_with_window:
                flash_kwargs["window_size"] = (self.sliding_window, self.sliding_window)
        if cfg.attn_logit_softcapping:
            # assert has_flash_attn_with_softcap, \
            #     "Installed version of flash-attn does not support softcapping"
            if has_flash_attn_with_softcap:
                flash_kwargs["softcap"] = cfg.attn_logit_softcapping

        attn_outputs = []
        for idx in range(len(split)):
            dev, a, b = split[idx]
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)

            attn_output = flash_attn_with_kvcache(
                q = q[idx],
                k = k[idx] if k is not None else None,
                v = v[idx] if v is not None else None,
                k_cache = k_cache[idx],
                v_cache = v_cache[idx],
                cache_seqlens = cache_seqlens_a[idx],
                block_table = attn_params.block_index_tp[idx],
                causal = True,
                softmax_scale = self.scaling,
                **flash_kwargs
            )
            attn_output = attn_output.view(batch_size * q_len, (b - a) * self.head_dim * self.num_key_value_groups)
            attn_outputs.append(attn_output)

        cache.store_kv_state(
            self.layer_idx,
            batch_size,
            0,
            q_len,
            page_size,
            attn_params.cache_seqlens_tp,
            attn_params.block_index_tp
        )

        # Output projection

        attn_outputs = self.model.tp_context.allgather(1, attn_outputs, BROADCAST_Q, BROADCAST_Q, dim = self.head_dim)

        hidden_states = self.o_proj.forward_tp(attn_outputs, loras = loras, dim = self.head_dim, output_split = True)

        if self.has_residual:
            self.model.tp_context.add_residual(hidden_states, residual, BROADCAST_Q, dim = self.head_dim)

        hidden_states = self.model.tp_context.gather(0, hidden_states, BROADCAST_Q, dim = self.head_dim)

        # if self.post_layernorm:  # TODO: ...
        #     hidden_states = self.post_layernorm.forward(hidden_states)

        hidden_states = hidden_states.view(batch_size, q_len, hidden_states.shape[-1])
        return hidden_states


    def _attn_torch(self, batch_size, q_len, q_states, k_states, v_states, attn_params, cfg, causal = True):

        num_attn_heads = q_states.shape[2]
        head_dim = q_states.shape[3]

        q_states = q_states.transpose(1, 2)
        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        # SDPA

        if has_lower_right_sdpa and not cfg.no_sdpa and not cfg.attn_logit_softcapping:

            k_states = self.repeat_kv(k_states, self.num_key_value_groups)
            v_states = self.repeat_kv(v_states, self.num_key_value_groups)

            if self.sliding_window and k_states.shape[2] >= self.sliding_window:
                k_states = k_states[:, :, -self.sliding_window:, :]
                v_states = v_states[:, :, -self.sliding_window:, :]

            if self.layer_idx in attn_params.block_diag_layers:
                attn_mask_lr = attn_params.get_block_diag_mask(q_states.device)
            elif attn_params.is_causal():
                attn_mask_lr = causal_lower_right(q_len, k_states.shape[2])
            else:
                attn_mask_lr = attn_params.get_attn_mask(q_states.device)
            attn_output = F.scaled_dot_product_attention(
                q_states,
                k_states,
                v_states,
                attn_mask_lr if causal else None,
                scale = self.scaling
            )

        # Matmul attn

        else:

            k_states = self.repeat_kv(k_states, self.num_key_value_groups)
            k_states = k_states.transpose(-1, -2)

            attn_weights = torch.matmul(q_states, k_states)

            attn_weights *= self.scaling
            if self.layer_idx in attn_params.block_diag_layers:
                attn_mask = attn_params.get_block_diag_mask(attn_weights.device)
            elif causal:
                attn_mask = attn_params.get_attn_mask(attn_weights.device)

            if cfg.attn_logit_softcapping:
                ext_c.softcap_(attn_weights, cfg.attn_logit_softcapping)
            if causal and attn_mask is not None:
                attn_weights = attn_weights + attn_mask
            if self.sliding_window and k_states.shape[-1] >= self.sliding_window:
                attn_weights = attn_weights[:, :, :, -self.sliding_window:]
                v_states = v_states[:, :, -self.sliding_window:, :]

            attn_weights = nn.functional.softmax(attn_weights, dim = -1, dtype = torch.float16)

            v_states = self.repeat_kv(v_states, self.num_key_value_groups)
            attn_output = torch.matmul(attn_weights, v_states)

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape((batch_size, q_len, num_attn_heads * head_dim))
        return attn_output


    def _attn_flash(self, batch_size, q_len, q_states, k_states, v_states, attn_params, cfg, causal = True):

        flash_kwargs = {}
        if self.sliding_window:
            # assert has_flash_attn_with_window, \
            #     "Installed version of flash-attn does not support sliding window"
            if has_flash_attn_with_window:
                flash_kwargs["window_size"] = (self.sliding_window, self.sliding_window)
        if cfg.attn_logit_softcapping:
            # assert has_flash_attn_with_softcap, \
            #     "Installed version of flash-attn does not support softcapping"
            if has_flash_attn_with_softcap:
                flash_kwargs["softcap"] = cfg.attn_logit_softcapping

        if self.layer_idx in attn_params.block_diag_layers:
            q_states = q_states.flatten(start_dim = 0, end_dim = 1)
            k_states = k_states.flatten(start_dim = 0, end_dim = 1)
            v_states = v_states.flatten(start_dim = 0, end_dim = 1)
            max_seqlen = attn_params.get_cu_seqlens_max()
            cu_seqlens = attn_params.get_cu_seqlens(self.device_idx)
            attn_output = flash_attn_varlen_func(
                q_states,
                k_states,
                v_states,
                cu_seqlens,
                cu_seqlens,
                max_seqlen,
                max_seqlen
            )
        else:
            attn_output = flash_attn_func(
                q_states,
                k_states,
                v_states,
                causal = causal,
                softmax_scale = self.scaling,
                **flash_kwargs
            )
        attn_output = attn_output.reshape((batch_size, q_len, self.num_attention_heads * self.head_dim))
        return attn_output


    def _attn_xformers(self, batch_size, q_len, q_states, k_states, v_states, attn_params, cfg, causal = True):

        # assert not self.sliding_window, \
        #     "Sliding window not currently supported for xformers"

        # assert not cfg.attn_logit_softcapping, \
        #     "Softcap not yet supported for xformers"

        # xformers memory_efficient_attention, could be beneficial if your device's architecture is less than <sm_80
        # xformer does not expand the kv automatically, we need to do it manually. The efficiency between
        # xformers.memory_efficient_attention and flash_attn in >sm_80 are almost the same. But the martix operation
        # make this implemention much slower.

        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        k_states = self.repeat_kv(k_states, self.num_key_value_groups)
        v_states = self.repeat_kv(v_states, self.num_key_value_groups)

        k_states = k_states.transpose(1, 2)
        v_states = v_states.transpose(1, 2)

        attn_output = xops.memory_efficient_attention(
            q_states,
            k_states,
            v_states,
            attn_bias = LowerTriangularFromBottomRightMask() if causal else None,
            scale = self.scaling
        )
        attn_output = attn_output.reshape((batch_size, q_len, self.num_attention_heads * self.head_dim))

        return attn_output


    # @profile
    def forward(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.Params | None = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict[str: torch.Tensor]:

        cfg = self.model.config
        global has_flash_attn
        global has_xformers
        use_flash_attn = has_flash_attn and not cfg.no_flash_attn

        if isinstance(attn_params, ExLlamaV2Attention.PagedParams):
            return self.forward_paged(
                hidden_states,
                cache,
                attn_params,
                loras = loras,
                **kwargs
            )

        if self.is_tp:
            if cache is not None and use_flash_attn:
                return self.forward_tp(
                    hidden_states,
                    cache,
                    attn_params,
                    past_len,
                    intermediates,
                    loras,
                    **kwargs,
                )
            else:
                # TODO: Can't use the optimized forward function because it writes directly to a fixed output
                #   tensor, and flash-attn currently has a bug that prevents that from working when q_len == 1
                return self.forward_tp_old(
                    hidden_states,
                    cache,
                    attn_params,
                    past_len,
                    intermediates,
                    loras,
                    **kwargs,
                )

        if self.q_handle is None or intermediates:
            return self.forward_torch(
                hidden_states,
                cache,
                attn_params,
                past_len,
                intermediates,
                loras = loras,
                **kwargs
            )

        constants = self.model.get_device_context(self.device_idx)

        batch_size, q_len, _ = hidden_states.shape
        direct = (batch_size == 1 and cache is not None and isinstance(cache, ExLlamaV2CacheBase))

        q_shape = hidden_states.shape[:-1] + (self.q_proj.out_features,)
        k_shape = hidden_states.shape[:-1] + (self.k_proj.out_features,)
        v_shape = hidden_states.shape[:-1] + (self.v_proj.out_features,)
        q_states = torch.empty(q_shape, device = hidden_states.device, dtype = torch.half)

        # If conditions are right we can write the K/V projections directly into the cache

        if direct:
            batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
            k_states = batch_keys[:batch_size, past_len : past_len + q_len, :]
            v_states = batch_values[:batch_size, past_len : past_len + q_len, :]
        else:
            k_states = torch.empty(k_shape, device = hidden_states.device, dtype = torch.half)
            v_states = torch.empty(v_shape, device = hidden_states.device, dtype = torch.half)

        # RMS norm, Q/K/V projections, position embeddings

        if loras is None or self.temp_lora_size == 0:
            pass_loras = []
            pass_lora_temp = none_tensor
        else:
            pass_loras = [id(x) for x in loras]
            pass_lora_temp = torch.empty((self.temp_lora_size,), dtype = torch.half, device = hidden_states.device)

        if attn_params.position_offsets is not None:
            pass_past_len_1 = past_len
            pass_past_len_2 = attn_params.get_position_offsets(hidden_states.device)
            if pass_past_len_1 == 0:
                offsets = attn_params.get_rope_offsets(self.device_idx)
                if offsets is not None:
                    pass_past_len_2 = pass_past_len_2 + offsets
        else:
            pass_past_len_1 = past_len
            pass_past_len_2 = none_tensor
            if attn_params.rope_offsets is not None:
                offset = attn_params.rope_offsets.cpu().item()
                pass_past_len_1 += offset

        ext_c.q_attn_forward_1(
            self.q_handle,
            hidden_states,
            batch_size,
            q_len,
            pass_past_len_1,
            pass_past_len_2,
            q_states,
            k_states,
            v_states,
            constants.sin,
            constants.cos,
            pass_loras,
            pass_lora_temp
        )

        # Select attention function

        if (has_flash_attn and not cfg.no_flash_attn) and attn_params.is_causal():
            attn_func = self._attn_flash
        elif (has_xformers and not cfg.no_xformers) and attn_params.is_causal():
            attn_func = self._attn_xformers
        else:
            attn_func = self._attn_torch

        # Straight attention without cache

        if cache is None:

            q_states = q_states.view(batch_size, q_len, self.num_attention_heads, self.head_dim)
            k_states = k_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
            v_states = v_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)

            attn_output = attn_func(batch_size, q_len, q_states, k_states, v_states, attn_params, cfg)

        # Regular cache (FP16, FP8, Q4)

        elif isinstance(cache, ExLlamaV2CacheBase):

            q_states = q_states.view(batch_size, q_len, self.num_attention_heads, self.head_dim)
            k_states = k_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)
            v_states = v_states.view(batch_size, q_len, self.num_key_value_heads, self.head_dim)

            if not direct:
                batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
                batch_keys[:batch_size, past_len:past_len + q_len, :].copy_(k_states)
                batch_values[:batch_size, past_len:past_len + q_len, :].copy_(v_states)

            k_states = batch_keys[:batch_size, :past_len + q_len, :]
            v_states = batch_values[:batch_size, :past_len + q_len, :]

            cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

            attn_output = attn_func(batch_size, q_len, q_states, k_states, v_states, attn_params, cfg)

        # Output projection

        ext_c.q_attn_forward_2(
            self.q_handle,
            hidden_states,
            attn_output,
            batch_size,
            q_len,
            pass_loras,
            pass_lora_temp
        )

        if self.archparams.clamp_hidden_states:
            hidden_states.clamp_(-65504, 65504)

        return hidden_states

    def forward_tp(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.Params | None = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor:

        cfg = self.model.config
        ctx = self.model.tp_context

        assert not cache or cache.q_block != 1, \
            "Models with odd key/value dims not supported in TP mode with quantized cache"
        assert not self.sliding_window, \
            "Sliding window not supported in TP mode"

        attn_params.prep_tp(self.model)

        batch_size, q_len, _ = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)
        past_len = 0 if cache is None else cache.current_seq_len

        k_cache, v_cache = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len) if cache else ([], [])

        sin, cos = ctx.get_sin_cos()

        ext_c.tp_attn_forward_(
            self.model.tp_context.ext_tp_context,
            hidden_states,
            self.temp_bc0,
            self.temp_bc1,
            self.temp_bc2,
            self.temp_q,
            self.temp_k,
            self.temp_v,
            self.temp_o,
            k_cache,
            v_cache,
            self.pre_layernorm.weight if self.pre_layernorm is not None else [],
            self.pre_layernorm.variance_epsilon if self.pre_layernorm is not None else 0.0,
            self.q_proj.q_handle,
            self.k_proj.q_handle,
            self.v_proj.q_handle,
            self.o_proj.q_handle,
            self.head_dim,
            int(self.archparams.rope_style),
            batch_size,
            q_len,
            sin,
            cos,
            attn_params.past_len_tp,
            self.scaling
        )

        if cache is not None:
            cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

        return ctx.get_pinned(0, batch_size, q_len, self.hidden_size)


    def forward_tp_old(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.Params | None = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
   ):
        cfg = self.model.config
        split = self.model.tp_context.get_split(BROADCAST_KV)
        batch_size, q_len, _ = hidden_states.shape
        attn_params.prep_tp(self.model)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        past_len = 0 if cache is None else cache.current_seq_len

        assert self.q_handle is not None
        use_flash_attn = has_flash_attn and not cfg.no_flash_attn
        if not use_flash_attn:
            assert has_lower_right_sdpa and attn_params.is_causal() and not cfg.no_sdpa and not cfg.attn_logit_softcapping, \
                "TP attention without flash-attn must use Torch SDPA with lower-right attention mask " \
                "(use PyTorch 2.4.0+) and does not support logit softcapping."

        hidden_states = self.model.tp_context.broadcast(0, hidden_states, BROADCAST_KV, dim = self.head_dim)

        residual = hidden_states

        post_norm = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states
        q = self.q_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim)
        k = self.k_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim)
        v = self.v_proj.forward_tp(post_norm, loras = loras, output_split = True, dim = self.head_dim)

        q = [q_.view(batch_size, q_len, q_.shape[1] // self.head_dim, self.head_dim) for q_ in q]
        k = [k_.view(batch_size, q_len, k_.shape[1] // self.head_dim, self.head_dim) for k_ in k]
        v = [v_.view(batch_size, q_len, v_.shape[1] // self.head_dim, self.head_dim) for v_ in v]

        if cache:
            k_cache, v_cache = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
        else:
            k_cache, v_cache = None, None

        if self.archparams.rope_style != RopeStyle.NONE:
            for idx, (dev, a, b) in enumerate(split):
                context = self.model.get_device_context(dev)
                torch.cuda.set_stream(context.stream)
                for t, heads in [(q[idx], self.num_key_value_groups), (k[idx], 1)]:
                    ext_c.rope_(
                        t,
                        context.sin,
                        context.cos,
                        past_len,
                        (b - a) * heads,
                        self.head_dim,
                        attn_params.position_offsets_tp[idx] if attn_params.position_offsets is not None else none_tensor,
                        self.archparams.rope_style == RopeStyle.NEOX
                    )

        attn_outputs = []
        for idx in range(len(split)):
            dev, a, b = split[idx]
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)

            if k_cache is not None:
                if use_flash_attn:
                    attn_output = flash_attn_with_kvcache(
                        q = q[idx],
                        k = k[idx],
                        v = v[idx],
                        k_cache = k_cache[idx],
                        v_cache = v_cache[idx],
                        causal = True,
                        softmax_scale = self.scaling,
                        cache_seqlens = attn_params.past_len_tp[idx]
                    )
                else:
                    cache_a = attn_params.past_len
                    cache_b = attn_params.past_len + q_len
                    k_cache[idx][:batch_size, cache_a:cache_b, :, :].copy_(k[idx])
                    v_cache[idx][:batch_size, cache_a:cache_b, :, :].copy_(v[idx])
                    attn_output = self._attn_torch(
                        batch_size,
                        q_len,
                        q[idx],
                        k_cache[idx][:batch_size, :cache_b, :, :],
                        v_cache[idx][:batch_size, :cache_b, :, :],
                        attn_params,
                        cfg
                    )
            else:
                if use_flash_attn:
                    attn_output = flash_attn_func(
                        q[idx],
                        k[idx],
                        v[idx],
                        causal = True,
                        softmax_scale = self.scaling,
                    )
                else:
                    attn_output = self._attn_torch(
                        batch_size,
                        q_len,
                        q[idx],
                        k[idx],
                        v[idx],
                        attn_params,
                        cfg
                    )

            attn_output = attn_output.view(batch_size * q_len, (b - a) * self.head_dim * self.num_key_value_groups)
            attn_outputs.append(attn_output)

        if cache is not None:
            cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

        # Output projection

        attn_outputs = self.model.tp_context.allgather(1, attn_outputs, BROADCAST_Q, BROADCAST_Q, dim = self.head_dim)

        hidden_states = self.o_proj.forward_tp(attn_outputs, loras = loras, dim = self.head_dim, output_split = True)

        if self.has_residual:
            self.model.tp_context.add_residual(hidden_states, residual, BROADCAST_Q, dim = self.head_dim)

        hidden_states = self.model.tp_context.gather(0, hidden_states, BROADCAST_Q, dim = self.head_dim)

        # if self.post_layernorm:  # TODO: ...
        #     hidden_states = self.post_layernorm.forward(hidden_states)

        hidden_states = hidden_states.view(batch_size, q_len, hidden_states.shape[-1])
        return hidden_states


    def forward_torch(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.Params | None = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict:

        if self.q_proj is None and self.k_proj is None and self.v_proj is None:
            return self.forward_torch_linear(
                hidden_states,
                cache,
                attn_params,
                past_len,
                intermediates,
                loras,
                **kwargs,
            )
    
        global has_flash_attn
        global has_xformers

        cfg = self.model.config
        num_attention_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        head_dim = self.head_dim

        batch_size, q_len, _ = hidden_states.size()

        past_len = 0 if cache is None else cache.current_seq_len

        # Project q, k, v

        residual = hidden_states
        post_norm = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states

        query_states = self.q_proj.forward(post_norm, loras = loras)
        key_states = self.k_proj.forward(post_norm, loras = loras)
        value_states = self.v_proj.forward(post_norm, loras = loras)

        # Shape for attention

        query_states = query_states.view(batch_size, q_len, num_attention_heads, head_dim)
        key_states = key_states.view(batch_size, q_len, num_key_value_heads, head_dim)
        value_states = value_states.view(batch_size, q_len, num_key_value_heads, head_dim)

        # Apply Q/K norms

        if cfg.use_qk_norm:
            query_states = self.q_norm.forward(query_states)
            key_states = self.k_norm.forward(key_states)

        # Apply position embeddings

        constants = self.model.get_device_context(self.device_idx, scratch = False)

        if attn_params.position_offsets is not None:
            position_offsets = attn_params.get_position_offsets(hidden_states.device)
        else:
            position_offsets = none_tensor

        if self.archparams.rope_style != RopeStyle.NONE:
            alt_cs = kwargs.get("alt_rope_embedding")
            cos, sin = alt_cs if alt_cs else (constants.cos, constants.sin)

            sc = attn_params.get_alt_rope_embed(self.device_idx)
            if sc: sin, cos = sc

            past_len_rope = past_len
            if attn_params.position_offsets is None:
                if attn_params.rope_offsets is not None:
                    offset = attn_params.rope_offsets.cpu().item()
                    past_len_rope += offset
            else:
                offsets = attn_params.get_rope_offsets(self.device_idx)
                if offsets is not None:
                    position_offsets = position_offsets + offsets

            ext_c.rope_(query_states, sin, cos, past_len_rope, num_attention_heads, head_dim, position_offsets, self.archparams.rope_style == RopeStyle.NEOX)
            ext_c.rope_(key_states, sin, cos, past_len_rope, num_key_value_heads, head_dim, position_offsets, self.archparams.rope_style == RopeStyle.NEOX)

        # Add keys and values to cache

        if cache is not None:

            batch_keys, batch_values = cache.get_kv_state(self.layer_idx, batch_size, 0, past_len)
            new_keys = batch_keys.narrow(1, past_len, q_len).narrow(0, 0, batch_size)
            new_values = batch_values.narrow(1, past_len, q_len).narrow(0, 0, batch_size)
            new_keys.copy_(key_states)
            new_values.copy_(value_states)

            # Key/value tensors with past

            key_states = batch_keys.narrow(1, 0, past_len + q_len).narrow(0, 0, batch_size)
            value_states = batch_values.narrow(1, 0, past_len + q_len).narrow(0, 0, batch_size)

        use_flash_attn = has_flash_attn and not cfg.no_flash_attn
        use_xformers = has_xformers and not cfg.no_xformers

        # Select attention function

        if not (use_flash_attn or use_xformers) or not attn_params.is_causal():
            attn_func = self._attn_torch
        elif use_flash_attn:
            attn_func = self._attn_flash
        else:
            attn_func = self._attn_xformers

        # Attention

        attn_output = attn_func(
            batch_size,
            q_len,
            query_states,
            key_states,
            value_states,
            attn_params,
            cfg,
            not attn_params.non_causal_attn
        )

        # Update 8-bit/Q4 cache

        if cache is not None:
            cache.store_kv_state(self.layer_idx, batch_size, past_len, q_len)

        # Output projection

        attn_proj = self.o_proj.forward(attn_output, loras = loras)

        # Post layernorm

        if self.post_layernorm:
            attn_proj = self.post_layernorm.forward(attn_proj, output_fp32 = self.archparams.residual_stream_fp32)

        # Add residual connection

        hidden_states = (attn_proj + residual) if self.has_residual else attn_proj

        if self.archparams.residual_stream_fp32:
            hidden_states = hidden_states.float()
        elif self.archparams.clamp_hidden_states:
            hidden_states.clamp_(-65504, 65504)

        if intermediates:
            return {"post_norm": post_norm,
                    "attn_output": attn_output,
                    "hidden_states": hidden_states}
        else:
            return hidden_states

    def forward_torch_linear(
        self,
        hidden_states: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        attn_params: ExLlamaV2Attention.Params | None = None,
        past_len: int | None = None,
        intermediates: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        **kwargs
    ) -> torch.Tensor | dict:

        cfg = self.model.config
        num_attention_heads = self.num_attention_heads
        num_key_value_heads = self.num_key_value_heads
        head_dim = self.head_dim

        batch_size, q_len, _ = hidden_states.size()

        past_len = 0 if cache is None else cache.current_seq_len

        # Project q, k, v

        residual = hidden_states
        post_norm = self.pre_layernorm.forward(hidden_states) if self.has_norm else hidden_states

        # Output projection

        attn_proj = self.o_proj.forward(post_norm, loras = loras)

        # Post layernorm

        if self.post_layernorm:
            attn_proj = self.post_layernorm.forward(attn_proj, output_fp32 = self.archparams.residual_stream_fp32)

        # Add residual connection

        hidden_states = (attn_proj + residual) if self.has_residual else attn_proj

        if self.archparams.residual_stream_fp32:
            hidden_states = hidden_states.float()
        elif self.archparams.clamp_hidden_states:
            hidden_states.clamp_(-65504, 65504)

        if intermediates:
            return {"post_norm": post_norm,
                    "hidden_states": hidden_states}
        else:
            return hidden_states

    def update_loras(self):

        if self.q_handle is None: return

        cfg = self.model.config

        q_proj_lora_a = { id(k): v for k, v in self.q_proj.lora_a_tensors.items() }
        q_proj_lora_b = { id(k): v for k, v in self.q_proj.lora_b_tensors.items() }
        k_proj_lora_a = { id(k): v for k, v in self.k_proj.lora_a_tensors.items() }
        k_proj_lora_b = { id(k): v for k, v in self.k_proj.lora_b_tensors.items() }
        v_proj_lora_a = { id(k): v for k, v in self.v_proj.lora_a_tensors.items() }
        v_proj_lora_b = { id(k): v for k, v in self.v_proj.lora_b_tensors.items() }
        o_proj_lora_a = { id(k): v for k, v in self.o_proj.lora_a_tensors.items() }
        o_proj_lora_b = { id(k): v for k, v in self.o_proj.lora_b_tensors.items() }

        temp_lora_size = ext_c.q_attn_set_loras(
            self.q_handle,
            q_proj_lora_a,
            q_proj_lora_b,
            k_proj_lora_a,
            k_proj_lora_b,
            v_proj_lora_a,
            v_proj_lora_b,
            o_proj_lora_a,
            o_proj_lora_b
        )

        self.temp_lora_size = temp_lora_size * cfg.max_batch_size * cfg.max_input_len


    def is_quant(self):
        return self.q_handle is not None


    def tp_split(self):

        cfg = self.model.config
        ctx = self.model.tp_context

        if self.pre_layernorm is not None:
            self.pre_layernorm.tp_split(BROADCAST_KV)
        if self.post_layernorm is not None:
            self.post_layernorm.tp_split(BROADCAST_KV)

        self.q_proj.tp_split(BROADCAST_Q, dim = self.head_dim)
        self.k_proj.tp_split(BROADCAST_KV, dim = self.head_dim)
        self.v_proj.tp_split(BROADCAST_KV, dim = self.head_dim)
        self.o_proj.tp_split(BROADCAST_Q, dim = self.head_dim)

        maxrows = cfg.max_batch_size * cfg.max_input_len
        dtype = torch.half

        ctx.begin_scratch_alloc_tp()
        ctx.reserve_scratch(self.tp_dq_size)
        self.temp_bc0 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_bc1 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_bc2 = ctx.get_scratch_slice_tp_bc(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_q = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)
        self.temp_k = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_KV, dim = self.head_dim)
        self.temp_v = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_KV, dim = self.head_dim)
        self.temp_o = ctx.get_scratch_slice_tp(maxrows, dtype, BROADCAST_Q, dim = self.head_dim)

        self.is_tp = True
        self.set_device_idx(None)


    def scratch_space_tp(self):

        cfg = self.model.config
        ctx = self.model.tp_context
        devs = ctx.num_devices
        scratch = [0] * devs

        def add(res: list[int]):
            for i, s in enumerate(res):
                scratch[i] += s

        def amax(res: list[int]):
            for i, s in enumerate(res):
                scratch[i] = max(scratch[i], s)

        amax(self.q_proj.scratch_space_tp(BROADCAST_Q, self.head_dim))
        amax(self.k_proj.scratch_space_tp(BROADCAST_KV, self.head_dim))
        amax(self.v_proj.scratch_space_tp(BROADCAST_KV, self.head_dim))
        amax(self.o_proj.scratch_space_tp(BROADCAST_Q, self.head_dim))
        self.tp_dq_size = [s for s in scratch]

        maxrows = cfg.max_batch_size * cfg.max_input_len

        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))
        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))
        add(ctx.get_temp_tensors_bc_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_KV, dim = self.head_dim))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_KV, dim = self.head_dim))
        add(ctx.get_temp_tensors_s(maxrows, 2, BROADCAST_Q, dim = self.head_dim))

        return scratch
