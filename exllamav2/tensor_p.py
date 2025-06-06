from __future__ import annotations
import torch
from exllamav2.util import get_all_gpu_memory, integer_split
from exllamav2.device import global_streams
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
# from line_profiler import profile
import sys

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2
    from exllamav2.attn import ExLlamaV2Attention

BROADCAST_KV = 0
BROADCAST_ID = 1
BROADCAST_VC = 2
BROADCAST_RS = 3
BROADCAST_Q = 4


class TPContext:

    model: ExLlamaV2

    kv_split: list[tuple[int, int, int]] | None
    kv_split_devs: list[int] | None
    id_split: list[tuple[int, int, int]] | None
    id_split_devs: list[int] | None
    vc_split: list[tuple[int, int, int]] | None
    vc_split_devs: list[int] | None
    rs_split: list[tuple[int, int, int]] | None
    rs_split_devs: list[int] | None
    q_split: list[tuple[int, int, int]] | None
    q_split_devs: list[int] | None

    pinned_temp: list[torch.Tensor] | None

    device: int | None
    all_devs: list[int | None] | None
    num_devices: int | None
    streams: list[int | None] | None

    ext_tp_context: int | None

    sin: list[torch.Tensor | None] | None
    cos: list[torch.Tensor | None] | None


    def __init__(
        self,
        model: ExLlamaV2,
        gpu_split: list[float] | None,
        expect_cache_tokens: int = 0,
        expect_cache_base: type = None
    ):
        self.model = model
        cfg = self.model.config

        assert cfg.arch.lm.supports_tp, \
            f"Tensor-parallel is not supported for {cfg.arch.arch_string}"
        assert cfg.intermediate_size % 128 == 0, \
            "Model intermediate size must be divisible by 128"

        self.kv_split = None
        self.kv_split_devs = None
        self.id_split = None
        self.id_split_devs = None
        self.vc_split = None
        self.vc_split_devs = None
        self.rs_split = None
        self.rs_split_devs = None
        self.q_split = None
        self.q_split_devs = None
        self.device = None
        self.all_devs = None
        self.num_devices = None
        self.streams = None
        self.pinned_temp = None
        self.ext_tp_context = None

        self.sin = None
        self.cos = None

        self.define_split(gpu_split, expect_cache_tokens, expect_cache_base)


    def unload(self):

        if self.ext_tp_context is not None:
            ext_c.free_tp_context(self.ext_tp_context)
            self.ext_tp_context = None


    def all_devices(self) -> list[int]:

        devs = set([d for d, _, _ in self.kv_split])
        devs |= set([d for d, _, _ in self.id_split])
        devs |= set([d for d, _, _ in self.vc_split])
        return sorted(devs)


    def define_split(
        self,
        gpu_split: list[float] | None,
        expect_cache_tokens: int,
        expect_cache_base: type
    ):
        cfg = self.model.config

        if gpu_split is None:
            gpu_memory = get_all_gpu_memory()
            gpu_split = [0] * (max(gpu_memory.keys()) + 1)
            for k, v in gpu_memory.items():
                gpu_split[k] = v["free"]
        else:
            gpu_split = [gs * 1024 for gs in gpu_split]

        # Q and KV splits

        kv_split = integer_split(cfg.num_key_value_heads, gpu_split)
        q_split = [s * cfg.num_key_value_groups for s in kv_split]
        attn_ratio = [s / cfg.num_key_value_heads for s in kv_split]

        # Subtract size of cache according to KV split

        if not expect_cache_tokens:
            expect_cache_tokens = cfg.max_seq_len * cfg.max_batch_size
        if expect_cache_base == sys.modules["exllamav2.cache"].ExLlamaV2Cache_8bit:
            bytes_per_element = 1
        elif expect_cache_base == sys.modules["exllamav2.cache"].ExLlamaV2Cache_Q8:
            bytes_per_element = 8.5/8
        elif expect_cache_base == sys.modules["exllamav2.cache"].ExLlamaV2Cache_Q6:
            bytes_per_element = 6.5/8
        elif expect_cache_base == sys.modules["exllamav2.cache"].ExLlamaV2Cache_Q4:
            bytes_per_element = 4.5/8
        else:
            bytes_per_element = 2

        cache_size = 2 * bytes_per_element * cfg.num_key_value_heads * cfg.head_dim * cfg.num_hidden_layers * expect_cache_tokens
        gpu_split = [max(0, gs - int(cache_size * r / 1024**2)) for gs, r in zip(gpu_split, attn_ratio)]

        # Subtract size of attn layers

        ExLlamaV2Attention_ = sys.modules["exllamav2.attn"].ExLlamaV2Attention
        for module in self.model.modules:
            if not isinstance(module, ExLlamaV2Attention_):
                continue
            wfp = module.weight_footprint()
            gpu_split = [max(0, gs - int(wfp * r / 1024**2)) for gs, r in zip(gpu_split, attn_ratio)]

        # MLP split

        id_split = [s * 128 for s in integer_split(cfg.intermediate_size // 128, gpu_split, 2)]
        rs_split = [s * 32 for s in integer_split(cfg.hidden_size // 32, gpu_split, 8)]

        # Vocab split

        vc_split = [s * 32 for s in integer_split((cfg.vocab_size + 31) // 32, gpu_split, 16)]

        def set_split(raw_split):
            b = 0
            split = []
            devs = []
            for d, s in enumerate(raw_split):
                a = b
                b = a + s
                if s:
                    split.append((d, a, b))
                    devs.append(d)
            return split, devs

        self.kv_split, self.kv_split_devs = set_split(kv_split)
        self.id_split, self.id_split_devs = set_split(id_split)
        self.vc_split, self.vc_split_devs = set_split(vc_split)
        self.rs_split, self.rs_split_devs = set_split(rs_split)
        self.q_split, self.q_split_devs = set_split(q_split)

        self.all_devs = self.all_devices()
        self.device = self.all_devs[0]
        self.num_devices = max(self.all_devs) + 1


    def finalize(self):
        cfg = self.model.config

        size = (cfg.max_output_len if cfg.max_output_len is not None else cfg.max_input_len) * cfg.vocab_size
        size = max(size, cfg.max_input_len * cfg.intermediate_size)

        self.pinned_temp = [
            torch.empty((size,), dtype = torch.half, pin_memory = True)
            for _ in range(2)
        ]

        devices = self.all_devices()
        max_device = max(devices)

        self.streams = [0] * (max_device + 1)
        for idx in devices:
            self.streams[idx] = global_streams[idx].cuda_stream

        self.ext_tp_context = ext_c.make_tp_context(
            self.kv_split,
            self.id_split,
            self.vc_split,
            self.rs_split,
            self.q_split,
            self.pinned_temp,
            self.streams
        )


    def get_split(self, broadcast_type: int):

        return [
            self.kv_split,
            self.id_split,
            self.vc_split,
            self.rs_split,
            self.q_split
        ][broadcast_type]


    def get_devs(self, broadcast_type: int):

        return [
            self.kv_split_devs,
            self.id_split_devs,
            self.vc_split_devs,
            self.rs_split_devs,
            self.q_split_devs
        ][broadcast_type]


    def get_temp_tensors_bc(self, rows: int, dtype: torch.dtype, broadcast_type: int, dim: int = 1):

        split = self.get_split(broadcast_type)
        dim = split[-1][2] * dim
        return [torch.empty((rows, dim), device = dev, dtype = dtype) for dev, _, _ in split]


    def get_temp_tensors_bc_s(self, rows: int, esize: int, broadcast_type: int, dim: int = 1):

        scratch = [0] * self.num_devices
        split = self.get_split(broadcast_type)
        dim = split[-1][2] * dim
        for dev, _, _ in split:
            scratch[dev] = rows * dim * esize
        return scratch


    def get_temp_tensors(self, rows: int, dtype: torch.dtype, broadcast_type: int, dim: int = 1):

        split = self.get_split(broadcast_type)
        return [torch.empty((rows, (b - a) * dim), device = dev, dtype = dtype) for dev, a, b in split]


    def get_temp_tensors_s(self, rows: int, esize: int, broadcast_type: int, dim: int = 1):

        scratch = [0] * self.num_devices
        split = self.get_split(broadcast_type)
        for dev, a, b in split:
            scratch[dev] = rows * (b - a) * dim * esize
        return scratch


    def get_pinned(self, buffer: int, batch_size: int, q_len: int, dim: int):

        pt = self.pinned_temp[buffer][:batch_size * q_len * dim]
        pt = pt.view(batch_size, q_len, dim)
        return pt


    def broadcast(
        self,
        buffer: int,
        input_tensor: torch.Tensor,
        broadcast_type: int,
        dim: int = 1
    ):
        split = self.get_split(broadcast_type)

        bc_tensors = []
        for idx, _, _ in split:
            bc_tensors.append(torch.empty_like(input_tensor, device = idx))

        ext_c.tp_broadcast(
            self.ext_tp_context,
            buffer,
            input_tensor,
            broadcast_type,
            bc_tensors,
            dim,
            -1
        )

        return bc_tensors


    def gather(
        self,
        buffer: int,
        inputs: list[torch.Tensor],
        broadcast_type: int,
        dim: int = 1

    ):
        split = self.get_split(broadcast_type)

        ext_c.tp_gather(
            self.ext_tp_context,
            buffer,
            inputs,
            broadcast_type,
            [],
            -1,
            dim,
            -1
        )

        pt = self.pinned_temp[buffer][:split[-1][2] * dim * inputs[0].shape[0]]
        pt = pt.view(inputs[0].shape[0], split[-1][2] * dim)
        return pt

    # @profile
    def allgather(
        self,
        buffer,
        inputs: list[torch.Tensor],
        broadcast_type_g: int,
        broadcast_type_b: int,
        dim: int = 1
    ):
        # split_g = self.get_split(broadcast_type_g)
        split_b = self.get_split(broadcast_type_b)
        sh = (inputs[0].shape[0], split_b[-1][-1] * dim)
        dtype = inputs[0].dtype

        bc_tensors = [torch.empty(sh, device = dev, dtype = dtype) for dev, _, _ in split_b]

        ext_c.tp_gather(
            self.ext_tp_context,
            buffer,
            inputs,
            broadcast_type_g,
            bc_tensors,
            broadcast_type_b,
            dim,
            -1
        )

        return bc_tensors


#    def copy_pinned(
#        self,
#        buffer: int,
#        inputs: torch.Tensor
#    ):
#        pt = self.pinned_temp[buffer][:inputs.numel()]
#        pt = pt.view(inputs.shape)
#        pt.copy_(inputs)
#        return pt
        
    def copy_pinned(self, buffer: int, inputs: torch.Tensor):
        pt = self.pinned_temp[buffer][:inputs.numel()]
        pt = pt.view(inputs.shape)

        # Bypass PyTorch entirely - direct memory copy
        import ctypes
        ctypes.memmove(pt.data_ptr(), inputs.data_ptr(), inputs.numel() * inputs.element_size())
        return pt        

    def add_residual(
        self,
        target: list[torch.Tensor],
        source: list[torch.Tensor],
        broadcast_type: int,
        dim: int = 1
    ):
        split = self.get_split(broadcast_type)

        for idx, (dev, a, b) in enumerate(split):
            context = self.model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)
            target[idx].add_(source[idx][:, a * dim : b * dim])


    def wait_streams(self):
        for dev in self.all_devs:
            s = global_streams[dev]
            s.synchronize()
        torch.cuda.synchronize()


    def get_sin_cos(self):
        if self.sin is not None:
            return self.sin, self.cos
        self.sin = []
        self.cos = []
        for dev in range(self.num_devices):
            if dev in self.all_devs:
                devctx = self.model.get_device_context(dev)
                self.sin.append(torch.cat(devctx.sin, 0))
                self.cos.append(torch.cat(devctx.cos, 0))
            else:
                self.sin.append(none_tensor)
                self.cos.append(none_tensor)
        return self.sin, self.cos


    def begin_scratch_alloc_tp(self):

        for devctx in self.model.device_context:
            if devctx:
                devctx.begin_scratch_alloc()


    def get_scratch_slice_tp_bc(self, rows: int, dtype: torch.dtype, broadcast_type: int, dim: int = 1):

        split = self.get_split(broadcast_type)
        dim = split[-1][2] * dim
        if dtype == torch.half: esize = 2
        if dtype == torch.float: esize = 4

        tensors = []
        for dev, _, _ in split:
            devctx = self.model.get_device_context(dev)
            size_bytes = rows * dim * esize
            tensor = devctx.get_scratch_slice(size_bytes)
            tensor = tensor.view(rows, dim)
            tensors.append(tensor)

        return tensors


    def get_scratch_slice_tp(self, rows: int, dtype: torch.dtype, broadcast_type: int, dim: int = 1):

        split = self.get_split(broadcast_type)
        if dtype == torch.half: esize = 2
        if dtype == torch.float: esize = 4

        tensors = []
        for dev, a, b in split:
            devctx = self.model.get_device_context(dev)
            size_bytes = rows * (b - a) * dim * esize
            tensor = devctx.get_scratch_slice(size_bytes)
            tensor = tensor.view(rows, (b - a) * dim)
            tensors.append(tensor)

        return tensors


    def reserve_scratch(self, scratch: list[int]):

        for dev, s in enumerate(scratch):
            if s == 0: continue
            devctx = self.model.get_device_context(dev)
            devctx.get_scratch_slice(s)
