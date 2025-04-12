from __future__ import annotations
import os, sys

min_version = (3, 8)
if sys.version_info < min_version:
    print("")
    print(f" ## Warning: this project requires Python {min_version[0]}.{min_version[1]} or higher.")
    print("")

# Set CUDA context to lazy loading since we won't need 95% of the modules in Torch
os.environ["CUDA_MODULE_LOADING"] = "LAZY"

# # Set cudaMallocAsync allocator by default as it appears slightly more memory efficient, unless Torch is already
# # imported in which case changing the allocator would cause it to crash
# if not "PYTORCH_CUDA_ALLOC_CONF" in os.environ:
#     try:
#         x = torch.__version__
#     except NameError:
#         os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:cudaMallocAsync"

import torch
if not (torch.version.cuda or torch.version.hip):
    print("")
    print(f" ## Warning: The installed version of PyTorch is {torch.__version__} and does not support CUDA or ROCm.")
    print("")

# PyTorch, especially v2.3.1, gets confused when working with small CPU tensors and likes to use way too many worker
# threads for small operations, adding considerable overhead. Limit it to a single thread to avoid that (globally
# because that seems to be the only way)
torch.set_num_threads(1)

import math
from exllamav2.config import ExLlamaV2Config
from exllamav2.cache import ExLlamaV2CacheBase
from exllamav2.linear import ExLlamaV2Linear
from exllamav2.module import ExLlamaV2Module
from exllamav2.rmsnorm import ExLlamaV2RMSNorm
from exllamav2.layernorm import ExLlamaV2LayerNorm
from exllamav2.attn import ExLlamaV2Attention, has_flash_attn, has_xformers
from exllamav2.lora import ExLlamaV2Lora
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.moe_mlp import ExLlamaV2MoEMLP
from exllamav2.parallel_decoder import ExLlamaV2ParallelDecoder
from exllamav2.embedding import ExLlamaV2Embedding
from exllamav2.pos_embedding import ExLlamaV2PosEmbedding
from exllamav2.compat import safe_move_tensor
from exllamav2.stloader import cleanup_stfiles
from exllamav2.device import ExLlamaV2DeviceContext, set_device_streams
from exllamav2.tensor_p import TPContext, BROADCAST_VC
from exllamav2.architecture import RopeStyle
import gc
import threading
from typing import Callable
# from exllamav2.util import list_live_tensors, print_vram_usage, set_snapshot, diff_snapshot, print_vram_usage_peak
from exllamav2.util import get_basic_progress, timed
# from line_profiler import profile
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor


def _torch_device(idx):
    if idx == -1: return "cpu"
    return f"cuda:{idx}"


class ExLlamaV2:

    config: ExLlamaV2Config
    modules: list[ExLlamaV2Module]
    modules_dict: dict[str: ExLlamaV2Module]
    device_context: list[ExLlamaV2DeviceContext | None]
    cache_map: dict[int: str]
    last_kv_layer_idx: int
    head_layer_idx: int
    loaded: bool

    tp_context: TPContext | None

    def __init__(
        self,
        config: ExLlamaV2Config,
        lazy_load = False,
        archparams = None
    ):

        self.config = config
        self.modules = []
        self.modules_dict = {}
        self.device_context = []
        self.cache_map = {}
        self.loaded = False
        self.tp_context = None

        # Build model

        cfg = self.config
        if archparams is None: archparams = cfg.arch.lm
        self.archparams = archparams

        emb = ExLlamaV2Embedding(self, cfg.arch.lm_prefix + "model.embed_tokens")
        self.modules += [emb]

        if archparams.keys["learned_pos_emb"]:
            pos_emb = ExLlamaV2PosEmbedding(self, archparams.keys["learned_pos_emb"])
            self.modules += [pos_emb]

        for layer_idx in range(cfg.num_hidden_layers):

            layer_key = cfg.arch.lm_prefix + f"model.layers.{layer_idx}"

            if cfg.arch.lm.alternating_swa:
                swa = cfg.sliding_window if (layer_idx + 1) % cfg.sliding_window_pattern != 0 else 0
            elif cfg.arch.lm.swa:
                swa = cfg.sliding_window
            else:
                swa = 0

            if cfg.arch.lm.parallel_decoder_blocks:
                pd = ExLlamaV2ParallelDecoder(self, layer_key, layer_idx, sliding_window = swa)
                self.modules += [pd]
            else:
                attn = ExLlamaV2Attention(self, layer_key, layer_idx, sliding_window = swa)
                if cfg.arch.lm.is_moe: mlp = ExLlamaV2MoEMLP(self, layer_key, layer_idx)
                else: mlp = ExLlamaV2MLP(self, layer_key, layer_idx)
                self.modules += [attn, mlp]

        if cfg.arch.lm.norm == "layernorm":
            norm = ExLlamaV2LayerNorm(self, cfg.arch.lm_prefix + "model.norm")
        elif cfg.arch.lm.norm == "rmsnorm":
            norm = ExLlamaV2RMSNorm(self, cfg.arch.lm_prefix + "model.norm")
        else:
            raise ValueError("unknown norm type")
        self.modules += [norm]

        self.head_layer_idx = len(self.modules)
        head = ExLlamaV2Linear(
            self,
            cfg.arch.lm_prefix + "lm_head",
            cfg.hidden_size,
            cfg.vocab_size,
            False,
            max_out_len = cfg.max_output_len,
            prescale = cfg.logit_scale,
            is_sub_module = False,
            normalize_unq = bool(cfg.norm_head)
        )
        if archparams.keys["lm_head"] != "lm_head":
            head.alt_key = archparams.keys["lm_head"]
        self.modules += [head]

        # Compile dictionary of modules

        for module in self.modules:
            if len(module.submodules) > 0:
                for m in module.submodules: self.modules_dict[m.key] = m
            else:
                self.modules_dict[module.key] = module

        # Find last layer that affects k/v cache

        layer_idx = len(self.modules)
        while True:
            layer_idx -= 1
            if isinstance(self.modules[layer_idx], ExLlamaV2Attention) or \
               isinstance(self.modules[layer_idx], ExLlamaV2ParallelDecoder):
                break

        self.last_kv_layer_idx = layer_idx


    def set_device_map(
        self,
        allocation: list[float],
        embed_cpu: bool = True
    ) -> list[float]:

        cfg = self.config
        self.cache_map = {}

        # Constant shared between layers

        if self.archparams.rope_style != RopeStyle.NONE:
            sincos_size = cfg.head_dim * cfg.max_seq_len * 2
            constant_size = sincos_size * 2
        else:
            constant_size = 0

        # Max size of hidden state

        state_size = cfg.hidden_size * cfg.max_input_len * cfg.max_batch_size * 2
        mask_size = cfg.max_input_len ** 2 * cfg.max_batch_size * 2

        # Bytes remaining per device

        allocation_bytes =  [a * 1024**3 - (constant_size + state_size + mask_size) for a in allocation]

        # Scratch space required per device

        reserve_bytes = [0 for a in allocation]
        reserve_bytes_attn = [0 for a in allocation]
        fixed_bytes = [0 for a in allocation]

        # Start from the last device index
        current_idx = len(allocation_bytes) - 1
        for idx, module in reversed(list(enumerate(self.modules))):

            # Special case for token embeddings on CPU
            if isinstance(module, ExLlamaV2Embedding) and embed_cpu:
                module.set_device_idx(-1)
                continue

            # Special case for attention
            attn_bytes_current = 0
            if isinstance(module, ExLlamaV2Attention):
                attn_bytes_current = module.temp_attn_size()

            # Move current_idx backward until module fits in allocation
            footprint = module.weight_footprint()   # Footprint, in bytes
            scratch = module.scratch_space()        # Scratch space required by module

            while True:
                assert current_idx >= 0, "Insufficient space in device allocation"
                dev_scratch = max(scratch, reserve_bytes[current_idx])
                dev_scratch_attn = max(attn_bytes_current, reserve_bytes_attn[current_idx])
                if footprint + dev_scratch + dev_scratch_attn <= allocation_bytes[current_idx]:
                    break
                current_idx -= 1

            # Size for fixed tensors
            scratch_fixed = module.scratch_space_fixed()
            fixed_bytes[current_idx] = max(scratch_fixed, fixed_bytes[current_idx])

            # Subtract module size from allocation
            reserve_bytes[current_idx] = dev_scratch
            reserve_bytes_attn[current_idx] = dev_scratch_attn
            allocation_bytes[current_idx] -= footprint

            module.set_device_idx(current_idx)

        # Prepare to prepare device tensors

        self.device_context = []
        for idx, scratch_bytes in enumerate(fixed_bytes):
            if scratch_bytes > 0:
                self.device_context.append(ExLlamaV2DeviceContext(self, idx, scratch_bytes))
            else:
                self.device_context.append(None)

        # Create map for cache

        self.set_cache_map()

        # Return unused space, in GB

        return [(ab - rb - rba) / 1024**3 for (ab, rb, rba) in zip(allocation_bytes, reserve_bytes, reserve_bytes_attn)]


    def load(
        self,
        gpu_split: list[float] | None = None,
        lazy: bool = False,
        stats: bool = False,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None,
        progress: bool = False
    ):
        """
        Load model, regular manual split mode.

        :param gpu_split:
            List of VRAM allocations for weights and fixed buffers per GPU. Does not account for the size of the cache
            which must be allocated with reference to the model subsequently and whose split across GPUs will depend
            on which devices end up receiving which attention layers.

            If None, only the first GPU is used.

        :param lazy:
            Only set the device map according to the split, but don't actually load any of the modules. Modules can
            subsequently be loaded and unloaded one by one for layer-streaming mode.

        :param stats:
            Legacy, unused

        :param callback:
            Callable function that triggers after each layer has loaded, for progress update etc.

        :param callback_gen:
            Same as callback, but for use by async functions

        :param progress:
            If True, create a rich progress bar in the console while loading. Cannot be used with callbacks
        """

        if progress:
            progressbar = get_basic_progress()
            progressbar.start()
            task_id = progressbar.add_task("Loading: " + self.config.model_dir, total = len(self.modules))
            module = 0
            def callback_pb(a, b):
                progressbar.update(task_id, advance = 1)
            assert callback is None, \
                "Cannot use callback function and console progress bar at the same time."
            callback = callback_pb
        f = self.load_gen(gpu_split, lazy, stats, callback, callback_gen)
        for item in f:
            pass
        if progress:
            progressbar.stop()


    def load_gen(
        self,
        gpu_split: list[float] | None = None,
        lazy: bool = False,
        stats: bool = False,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None
    ):
        with torch.inference_mode():
            set_device_streams()

            stats_ = self.set_device_map(gpu_split or [99999])

            # Load module weights

            if not lazy:

                for idx, module in enumerate(self.modules):

                    if callback is not None: callback(idx, len(self.modules))
                    if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                    module.load()

                if callback is not None: callback(len(self.modules), len(self.modules))
                if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            # Cache map

            self.set_cache_map()

            self.loaded = True
            cleanup_stfiles()


    def load_tp(
        self,
        gpu_split: list[float] | None = None,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None,
        progress: bool = False,
        expect_cache_tokens: int = 0,
        expect_cache_base: type = None
    ):
        """
        Load model, tensor-parallel mode.

        :param gpu_split:
            List of VRAM allocations per GPU. The loader attempts to balance tensor splits to stay within these
            allocations, accounting for an uneven distribution of attention heads and the expected size of the cache.

            If None, the loader attempts to use all available GPUs and creates a split based on the currently available
            VRAM according to nvidia-smi etc.

        :param callback:
            Callable function that triggers after each layer has loaded, for progress update etc.

        :param callback_gen:
            Same as callback, but for use by async functions

        :param progress:
            If True, create a rich progress bar in the console while loading. Cannot be used with callbacks

        :param expect_cache_tokens:
            Expected size of the cache, in tokens (i.e. max_seq_len * max_batch_size, or just the cache size for use
            with the dynamic generator) to inform the automatic tensor split. If not provided, the configured
            max_seq_len for the model is assumed.

        :param expect_cache_base:
            Cache type to expect, e.g. ExLlamaV2Cache_Q6. Also informs the tensor split. If not provided, FP16 cache
            is assumed.
        """
        if progress:
            progressbar = get_basic_progress()
            progressbar.start()
            task_id = progressbar.add_task("Loading: " + self.config.model_dir, total = len(self.modules))
            module = 0
            def callback_pb(a, b):
                progressbar.update(task_id, advance = 1)
            assert callback is None, \
                "Cannot use callback function and console progress bar at the same time."
            callback = callback_pb
        f = self.load_tp_gen(gpu_split, callback, callback_gen, expect_cache_tokens, expect_cache_base)
        for item in f:
            pass
        if progress:
            progressbar.stop()


    def load_tp_gen(
        self,
        gpu_split: list[float] | None = None,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None,
        expect_cache_tokens: int = 0,
        expect_cache_base: type = None
    ):
        self.config.no_graphs = True
        self.tp_context = TPContext(self, gpu_split, expect_cache_tokens, expect_cache_base)

        # Create device tensors

        devs = self.tp_context.num_devices
        scratch = [0] * devs
        for module in self.modules[1:]:
            ms = module.scratch_space_tp()
            for i, m in enumerate(ms):
                scratch[i] = max(scratch[i], m)

        self.device_context = [None] * devs
        for idx in range(devs):
            if scratch[idx] > 0:
                self.device_context[idx] = ExLlamaV2DeviceContext(self, idx, scratch[idx])

        # Load module weights

        with torch.inference_mode():
            set_device_streams()

            for idx in range(len(self.modules)):
                module = self.modules[idx]

                if callback is not None: callback(idx, len(self.modules))
                if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                if isinstance(module, ExLlamaV2Embedding):
                    module.set_device_idx(-1)
                else:
                    module.set_device_idx(self.tp_context.device)

                module.load(device_context = False)

                if isinstance(module, ExLlamaV2Embedding):
                    module.tp_split()
                if isinstance(module, ExLlamaV2Attention):
                    module.tp_split()
                if isinstance(module, ExLlamaV2MLP):
                    module.tp_split()
                if isinstance(module, ExLlamaV2RMSNorm):
                    module.tp_split(BROADCAST_VC)
                elif isinstance(module, ExLlamaV2Linear):
                    module.tp_split(BROADCAST_VC)

                module.set_device_idx(None)

            torch.cuda.empty_cache()
            gc.collect()

            if callback is not None: callback(len(self.modules), len(self.modules))
            if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            self.loaded = True
            cleanup_stfiles()

            self.tp_context.finalize()


    def load_autosplit(
        self,
        cache: ExLlamaV2CacheBase,
        reserve_vram: int | list[int] | None = None,
        last_id_only: bool = False,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None,
        progress: bool = False
    ):
        """
        Load model, auto-split mode. This mode loads the model and builds the cache in parallel, using available
        devices in turn and moving on to the next device whenever the previous one is full.

        :param cache:
            Cache constructed with lazy = True. Actual tensor allocation for the cache will happen while loading the
            model.

        :param reserve_vram:
            Number of bytes to reserve on each device, either for all devices (as an int) or per-device (as a list).

        :param last_id_only:
            If True, model will be loaded in a mode that does can only output one set of logits (i.e. one token
            position) per forward pass. This conserves memory if the model is only to be used for generating text and
            not e.g. perplexity measurement.

        :param callback:
            Callable function that triggers after each layer has loaded, for progress update etc.

        :param callback_gen:
            Same as callback, but for use by async functions

        :param progress:
            If True, create a rich progress bar in the console while loading. Cannot be used with callbacks
        """
        if progress:
            progressbar = get_basic_progress()
            progressbar.start()
            task_id = progressbar.add_task("Loading: " + self.config.model_dir, total = len(self.modules))
            module = 0
            def callback_pb(a, b):
                progressbar.update(task_id, advance = 1)
            assert callback is None, \
                "Cannot use callback function and console progress bar at the same time."
            callback = callback_pb
        f = self.load_autosplit_gen(cache, reserve_vram, last_id_only, callback, callback_gen)
        for item in f:
            pass
        if progress:
            progressbar.stop()

    def load_autosplit_gen(
        self,
        cache: ExLlamaV2CacheBase,
        reserve_vram: int | list[int] | None = None,
        last_id_only: bool = False,
        callback: Callable[[int, int], None] | None = None,
        callback_gen: Callable[[int, int], None] | None = None
    ):

        # Limit model's max_input_len to max_seq_len if necessary
        self.config.max_input_len = min(self.config.max_input_len, self.config.max_seq_len)

        minimum_reserve_vram = 256 * 1024**2
        last_touched_device = -1
        current_device = 0
        num_devices = torch.torch.cuda.device_count()
        loras = None  # TODO: Autosplit load with LoRAs

        with torch.inference_mode():
            set_device_streams()

            self.device_context = []

            # Reserved space

            if reserve_vram is None:
                reserve_vram = [192 * 1024**2] + [64 * 1024**2] * (num_devices - 1)
            elif isinstance(reserve_vram, int):
                reserve_vram = [reserve_vram] * num_devices

            reserved_vram_tensors = []
            minimum_reserve_tensor = None

            # Largest hidden state to ever forward through model

            hidden_state = torch.zeros((1, self.config.max_input_len), dtype = torch.long)
            batch_size, seq_len = hidden_state.shape
            past_len = 0
            attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, past_len, None, None)

            # Size of fixed scratch space

            scratch_fixed = max(module.scratch_space_fixed() for module in self.modules)

            # Load modules and create cache tensors sequentially

            self.cache_map = {}
            for idx, module in enumerate(self.modules):

                if callback is not None: callback(idx, len(self.modules))
                if callback_gen is not None: yield from callback_gen(idx, len(self.modules))

                # Embedding layer on CPU

                if idx == 0:

                    module.set_device_idx(-1)
                    module.load()
                    hidden_state = module.forward(hidden_state)
                    continue

                while True:

                    # If we've reached a new device, allocate fixed tensors

                    if current_device > last_touched_device:

                        self.device_context.append(ExLlamaV2DeviceContext(self, current_device, scratch_fixed))
                        # if attn_mask is not None:
                        #     reserved_vram_tensors.append(attn_mask)
                        #     attn_mask = safe_move_tensor(attn_mask, _torch_device(current_device))
                        # else:
                        #     attn_mask = self.build_attn_mask(batch_size, seq_len, past_len, None, _torch_device(current_device))

                        b = reserve_vram[current_device]
                        reserved_vram_tensors.append(torch.empty((b,), dtype = torch.int8, device = _torch_device(current_device)))
                        minimum_reserve_tensor = torch.empty((minimum_reserve_vram,), dtype = torch.int8, device = _torch_device(current_device))

                        last_touched_device = current_device

                    # Attempt to load module and forward state

                    module.set_device_idx(current_device)

                    hidden_state_backup = hidden_state
                    # if hidden_state.device == "cpu":
                    #     hidden_state_backup = hidden_state.clone()
                    # else:
                    #     hidden_state_backup = safe_move_tensor(hidden_state, "cpu").clone()

                    try:
                        if isinstance(module, ExLlamaV2Attention) or \
                           isinstance(module, ExLlamaV2ParallelDecoder):
                            self.cache_map[module.layer_idx] = module.device()
                            cache.update_cache_tensors()

                        module.load()

                        if idx == self.head_layer_idx:
                            if last_id_only:
                                hidden_state = hidden_state.narrow(-2, -1, 1)
                            elif self.config.max_output_len is not None:
                                hidden_state = hidden_state.narrow(-2, -self.config.max_output_len, self.config.max_output_len)

                        hidden_state = safe_move_tensor(hidden_state, _torch_device(current_device))
                        hidden_state = module.forward(hidden_state, cache = cache, attn_params = attn_params, past_len = past_len, loras = loras)
                        fail = False

                    except Exception as e:

                        test = 0
                        if e.__class__.__name__ == "OutOfMemoryError" or \
                            "CUDA out of memory" in str(e) or \
                            "HIP out of memory" in str(e):
                            fail = True  # Exception object will hold references to tensors so we can't free them here
                        else:
                            raise

                    # If we failed, roll back and advance to next device

                    if fail:

                        module.unload()
                        hidden_state = None

                        if minimum_reserve_tensor is not None: del minimum_reserve_tensor
                        minimum_reserve_tensor = None

                        gc.collect()
                        torch.cuda.empty_cache()
                        hidden_state = hidden_state_backup.clone()

                        current_device += 1
                        if current_device >= num_devices:
                            raise RuntimeError("Insufficient VRAM for model and cache")

                        continue

                    break

            if callback is not None: callback(len(self.modules), len(self.modules))
            if callback_gen is not None: yield from callback_gen(len(self.modules), len(self.modules))

            hidden_state = None
            attn_params = None
            reserved_vram_tensors = None

            gc.collect()
            torch.cuda.empty_cache()
            self.loaded = True
            cleanup_stfiles()

        if 'yield' in locals():
            yield


    def unload(self):
        """
        Unloads the model and frees all unmanaged resources.
        """

        for module in self.modules:
            module.unload()

        self.modules = []
        self.modules_dict = {}
        self.device_context = []

        if self.tp_context:
            self.tp_context.unload()
            self.tp_context = None


    def set_cache_map(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention) or \
               isinstance(module, ExLlamaV2ParallelDecoder):
                self.cache_map[module.layer_idx] = module.device()


    def get_cache_devices(self) -> list[str]:

        return list(set(self.cache_map.values()))


    def create_device_context(self, scratch_bytes):

        for idx, b in enumerate(scratch_bytes):

            tensors = ExLlamaV2DeviceContext(self, idx, b)
            self.device_context.append(tensors)


    def drop_device_context(self):

        for dc in self.device_context:
            dc.drop()


    def free_device_context(self):

        for dc in self.device_context:
            dc.free()


    def get_device_context(self, device_idx, scratch = True):

        if device_idx == -1:
            return None

        context = self.device_context[device_idx]
        if not context.ready: context.prepare(scratch)
        return context


    def get_modules(self) -> list[ExLlamaV2Module]:

        return [module for module in self.modules]  #?


    def update_loras(self):

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention): module.update_loras()
            if isinstance(module, ExLlamaV2MLP): module.update_loras()
            if isinstance(module, ExLlamaV2MoEMLP): module.update_loras()


    def is_quant(self) -> bool:

        for module in self.modules:
            if isinstance(module, ExLlamaV2Attention):
                if module.is_quant(): return True

        return False


    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        cache: ExLlamaV2CacheBase | list[ExLlamaV2CacheBase] | None = None,
        input_mask: torch.Tensor | None = None,
        preprocess_only: bool = False,
        last_id_only: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        return_last_state: bool = False,
        position_offsets: torch.Tensor | None = None,
        abort_event: threading.Event | None = None,
        cpu_logits: bool = False,
        **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None:
        """
        Runs a forward pass through the model. If a cache is used, also appends keys/values to the cache
        and advances it.

        :param input_ids:
            LongTensor of input token IDs, shape (batch_size, q_len)

        :param cache:
            Optional ExLlamaV2Cache. If not provided, q_len must be less than config.max_input_len

        :param input_mask:
            Additive attention bias, shape (batch_size, past_len + q_len, q_len)

        :param preprocess_only:
            Only forward up to the last layer that affects the K/V cache. Does not return logits. Used
            to prefill the cache.

        :param last_id_only:
            Process the entire input sequence but only pass the last token through the head layer and
            only return logits for the last token.

        :param loras:
            List of ExLlamaV2Lora objects to apply during the forward pass

        :param return_last_state:
            Also return the hidden state right before the head layer

        :param position_offsets:
            Tensor of position offsets, shape (batch_size, 1). Offset is applied to position IDs during
            RoPE application.

        :param abort_event:
            Optional event that, if set, will abort the forward pass. Function will return None if aborted.

        :param cpu_logits:
            If True, logits are collected and returned in system RAM. This is somewhat slower but can prevent
            out-of-memory errors when computing logits for all positions in a long sequence, such as during a
            perplexity test.

        :return:
            FP16 logits tensor, shape (batch_size, q_len, vocab_size)
            (optional) state tensor, shape (batch_size, q_len, hidden_size)

        :indexed_embeddings:
            Tensor of embeddings, shape (batch_size, q_len, hidden_size), indexed by input token IDs >=
            ExLlamaV2.EMBEDDING_INDEX
        """

        bsz, q_len = input_ids.shape
        remaining_q_len = q_len

        # Attn and MLP layers have preallocated buffers for temp states, sized by the model config. Effective max input
        # length depends on the current batch size

        effective_max_input_len = self.config.max_input_len * self.config.max_batch_size // bsz

        # Without a cache we can't process the sequence in chunks, so forward the whole thing and assume the input length
        # is less than config.max_input_len

        if cache is None or not isinstance(cache, ExLlamaV2CacheBase):

            assert q_len <= effective_max_input_len, "Maximum input length exceeded in model.forward"

            result = self.forward_chunk(
                input_ids = input_ids,
                cache = cache,
                input_mask = input_mask,
                preprocess_only = preprocess_only,
                last_id_only = last_id_only,
                loras = loras,
                return_last_state = return_last_state,
                position_offsets = position_offsets,
                abort_event = abort_event,
                **kwargs
            )

            if abort_event and abort_event.is_set(): return

            if "last_state" in result:
                return result.get("logits"), result["last_state"]
            else:
                return result.get("logits")

        # Confirm that the input fits within the allocated cache space

        past_len = cache.current_seq_len
        assert past_len + q_len <= cache.max_seq_len, "Total sequence length exceeds cache size in model.forward"

        # Split sequence

        result = None
        last_state = None

        chunk_begin = 0
        while chunk_begin < q_len:

            # Limit chunk_size to max_input_len

            chunk_size = min(remaining_q_len, effective_max_input_len)

            # Limit chunk_size to keep size of attention operation <= max_attention_size

            if (has_flash_attn and not self.config.no_flash_attn) \
                or (has_xformers and not self.config.no_xformers):

                # Can't measure increase in VRAM usage with longer k_len, assume usage is constant
                # for given chunk_size
                pass

            else:

                past_len = cache.current_seq_len
                attn_size = (past_len + remaining_q_len) * remaining_q_len
                max_a = self.config.max_attention_size
                if attn_size > max_a:
                    cs = (math.sqrt(past_len ** 2 + 4 * max_a) - past_len) / 2
                    chunk_size = min(chunk_size, math.floor(cs))

            # Process chunk

            chunk_end = min(chunk_begin + chunk_size, q_len)

            # print(f"Forward chunk length: {chunk_end - chunk_begin}")

            _last_id_only = last_id_only
            _preprocess_only = preprocess_only or (chunk_end < q_len and last_id_only)

            r = self.forward_chunk(
                input_ids = input_ids[:, chunk_begin : chunk_end],
                cache = cache,
                input_mask = input_mask,
                preprocess_only = _preprocess_only,
                last_id_only = _last_id_only,
                loras = loras,
                return_last_state = return_last_state and remaining_q_len <= chunk_size,
                position_offsets = position_offsets,
                abort_event = abort_event,
                **kwargs
            )

            if abort_event and abort_event.is_set(): return

            if not _preprocess_only:
                if cpu_logits:
                    r["logits"] = r["logits"].cpu()
                result = r["logits"] if result is None else torch.cat((result, r["logits"]), dim = 1)

            chunk_begin = chunk_end
            remaining_q_len -= chunk_size
            last_state = r.get("last_state")

        if last_state is None:
            return result
        else:
            return result, last_state


    @torch.inference_mode()
    # @profile
    def forward_chunk(
        self,
        input_ids: torch.Tensor,
        cache: ExLlamaV2CacheBase | None = None,
        input_mask: torch.Tensor | None = None,
        preprocess_only: bool = False,
        last_id_only: bool = False,
        loras: list[ExLlamaV2Lora] | None = None,
        return_last_state: bool = False,
        position_offsets: torch.Tensor | None = None,
        abort_event: threading.Event | None = None,
        attn_params: ExLlamaV2Attention.Params | ExLlamaV2Attention.PagedParams | None = None,
        extract_state_indices: list[int] | None = None,
        **kwargs
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len = input_ids.shape
        past_len = 0
        if cache is not None:
            past_len = cache.current_seq_len

        assert self.config.max_output_len is None or \
            preprocess_only or \
            last_id_only or \
            seq_len <= self.config.max_output_len, \
            "seq_len exceeds max_output_len"

        if self.config.arch.lm.mrope and kwargs.get("indexed_embeddings"):
            assert attn_params is not None and (
                attn_params.rope_offsets is not None or
                attn_params.get_alt_rope_embed("cpu") is not None
            ), "MRoPE model requires precomputed RoPE when using indexed embeddings."

        # Ensure streams are always set in the current thread

        set_device_streams()

        # Output

        r = {}
        extract_state_indices = set(extract_state_indices or [])
        if return_last_state:
            extract_state_indices.add(self.head_layer_idx - 1)
        if extract_state_indices:
            r["states"] = {}

        # assert cache is None or isinstance(cache, list) or batch_size <= cache.batch_size

        x = input_ids

        if not attn_params:
            attn_params = ExLlamaV2Attention.Params(batch_size, seq_len, past_len, input_mask, position_offsets)
        else:
            if not isinstance(attn_params, ExLlamaV2Attention.PagedParams):
                past_len = attn_params.past_len
                cache.current_seq_len = past_len

        device = self.modules[0].device_idx

        for idx, module in enumerate(self.modules):

            if idx == self.head_layer_idx and last_id_only:
                x = x.narrow(-2, -1, 1)

            if idx in extract_state_indices:
                r["states"][idx] = x.clone()
                if idx == self.head_layer_idx - 1:
                    r["last_state"] = r["states"][idx]

            # Respect abort signal

            if abort_event and abort_event.is_set():
                return None, None

            # Onward

            n_device = module.device_idx
            if n_device is not None and n_device != device and n_device >= 0:
                x = safe_move_tensor(x, n_device, non_blocking = True)

            x = module.forward(x, cache = cache, attn_params = attn_params, past_len = past_len, loras = loras, **kwargs)

            if preprocess_only and idx == self.last_kv_layer_idx:
                x = None
                break

        # Advance cache

        if cache is not None:
            cache.current_seq_len += seq_len

        # Final sync if gathering logits

        if self.tp_context:
            self.tp_context.wait_streams()

        if x is not None and x.is_cuda:
            context = self.get_device_context(x.device.index)
            torch.cuda.set_stream(context.stream)

        # Apply logit scale

        # if x is not None and self.config.logit_scale != 1:
        #     x.mul_(self.config.logit_scale)

        if x is not None and self.config.final_logit_softcapping:
            ext_c.softcap_(x, self.config.final_logit_softcapping)

        # Set padding logits to -inf

        if x is not None:
            head_padding = self.modules[-1].padding
            if head_padding > 0:
                x[:, :, -head_padding:] = -65504.
            r["logits"] = x

        return r
