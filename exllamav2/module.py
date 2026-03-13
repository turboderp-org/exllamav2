from __future__ import annotations
import torch
import torch.nn as nn
from exllamav2.config import ExLlamaV2Config
from exllamav2.stloader import STFile
from exllamav2.compat import safe_move_tensor

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from exllamav2.model import ExLlamaV2
    from exllamav2.lora import ExLlamaV2Lora

def _torch_device(idx: int) -> str:
    if idx == -1: return "cpu"
    return f"cuda:{idx}"


class ExLlamaV2Module:

    config: ExLlamaV2Config
    key: str
    alt_key: str | None
    device_idx: int | list
    footprint: int
    submodules: list[ExLlamaV2Module]
    assumed_footprint: int

    def __init__(
        self,
        model: ExLlamaV2,
        key: str,
        archparams = None,
    ):

        if archparams is None:
            archparams = model.config.arch.lm
        self.archparams = archparams

        self.model = model
        self.key = key
        self.alt_key = None
        self.footprint = -1
        self.submodules = []


    def numel(self): raise(NotImplementedError())
    def load(self, device_context: bool): raise(NotImplementedError())
    def unload(self): raise(NotImplementedError())
    def scratch_space_fixed(self): raise(NotImplementedError())
    def scratch_space_tp(self): raise(NotImplementedError())
    def scratch_space(self): raise(NotImplementedError())

    def forward(self,
                hidden_states,
                cache = None,
                attn_params = None,
                past_len = None,
                intermediates = None,
                loras = None):
        raise(NotImplementedError())


    def device(self) -> str:
        return _torch_device(self.device_idx)


    def load_multi(self,
                   key: str,
                   keys: list[str],
                   measure: bool = False,
                   cpu: bool = False) -> int | dict[str: torch.Tensor]:

        tensors = {}
        submap = {}
        submap_i = {}
        size = 0

        # key = self.key if override_key is None else override_key

        for k in keys:
            ck = key + "." + k
            if ck in self.model.config.tensor_file_map:
                submap[k] = self.model.config.tensor_file_map[ck]

        for k, v in submap.items():
            if v not in submap_i:
                submap_i[v] = []
            submap_i[v].append(k)

        for v, ks in submap_i.items():
            stfile = STFile.open(v, keymap = self.model.config.arch.keymap)
            if measure:
                for k in ks:
                    size += stfile.measure(key + "." + k)
            else:
                loaded = stfile.get_tensors([key + "." + k for k in ks], device = self.device() if not cpu else "cpu")
                for k, tensor in zip(ks, loaded.values()):
                    tensors[k] = tensor

        return size if measure else tensors


    def load_weight(
        self,
        override_key: str | None = None,
        cpu: bool = False
    ):

        if override_key is not None:
            keys = [override_key]
        else:
            keys = [self.key]
            if self.alt_key is not None:
                keys += [self.alt_key]

        for key in keys:

            # EXL2

            if key + ".q_weight" in self.model.config.tensor_file_map:
                qtensors = self.load_multi(key, ["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm", "bias"], cpu = cpu)
                qtensors["q_perm"] = torch.argsort(qtensors["q_invperm"]).to(torch.int)
                return qtensors

            # GPTQ

            if key + ".qweight" in self.model.config.tensor_file_map:
                qtensors = self.load_multi(key, ["qweight", "qzeros", "scales", "g_idx", "bias"], cpu = cpu)
                if "bias" in qtensors and torch.all(qtensors["bias"].eq(0)):
                    del qtensors["bias"]
                qtensors["scales"] = qtensors["scales"].half()
                return qtensors

            # Torch

            if key + ".weight" in self.model.config.tensor_file_map:
                if key + ".bias" in self.model.config.tensor_file_map:
                    tensors = self.load_multi(key, ["weight", "bias"], cpu = cpu)
                    tensor = tensors["weight"].half()
                    bias = tensors["bias"].half()
                    if self.model.config.arch.orig_weights_transposed and len(tensor.shape) == 2:
                        tensor = tensor.T
                    return nn.Parameter(tensor, requires_grad = False), nn.Parameter(bias, requires_grad = False)
                else:
                    tensors = self.load_multi(key, ["weight"], cpu = cpu)
                    tensor = tensors["weight"].half()
                    # if self.model.config.arch.orig_weights_transposed:
                    #     tensor = tensor.T
                    return nn.Parameter(tensor, requires_grad = False)

            # No weights found for key

        return None


    def load_weight_fused(
        self,
        f_key: str,
        f_beg: int,
        f_end: int,
        in_feat: int,
        out_feat: int,
        altpack_qkv: bool
    ):
        res = []
        for key in [f_key, f_key + ".weight", f_key + ".bias"]:

            cfg = self.model.config
            filename = cfg.tensor_file_map.get(key)
            if not filename: continue

            stfile = STFile.open(filename, keymap = cfg.arch.keymap)
            tensor = stfile.get_tensor(key, device = "cpu", out_dtype = torch.half)

            if cfg.arch.orig_weights_transposed and len(tensor.shape) == 2:
                tensor = tensor.T

            if altpack_qkv:
                ts = tensor.shape
                h, gs, d = cfg.num_key_value_heads, cfg.num_key_value_groups + 2, cfg.head_dim
                tensor = tensor.view(h, gs, d, -1).transpose(0, 1).reshape(ts)

            tensor = tensor[f_beg:f_end]

            if altpack_qkv:
                ts = tensor.shape
                h, gs, d = cfg.num_key_value_heads, (f_end - f_beg) // cfg.num_key_value_heads // cfg.head_dim, cfg.head_dim
                tensor = tensor.view(gs, h, d, -1).transpose(0, 1).reshape(ts)

            if not key.endswith(".bias"):
                if in_feat != out_feat and \
                    tensor.shape[1] == out_feat and \
                    tensor.shape[0] == in_feat:
                    tensor = tensor.T

            tensor = safe_move_tensor(tensor.contiguous(), self.device())
            res.append(nn.Parameter(tensor, requires_grad = False))

        if len(res) == 2: return res[0], res[1]
        if len(res) == 1: return res[0]
        return None


    def weight_footprint(self) -> int:

        if self.footprint == -1:

            keys = [self.key]
            if self.alt_key is not None:
                keys += [self.alt_key]

            for key in keys:

                # EXL2

                if key + ".q_weight" in self.model.config.tensor_file_map:
                    self.footprint = self.load_multi(key, ["q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_perm", "q_perm", "bias"], measure = True)

                # GPTQ

                elif key + ".qweight" in self.model.config.tensor_file_map:
                    self.footprint = self.load_multi(key, ["qweight", "qzeros", "scales", "g_idx", "bias"], measure = True)

                # Torch

                elif key + ".weight" in self.model.config.tensor_file_map:
                    self.footprint = self.load_multi(key, ["weight", "bias"], measure = True)

                if self.footprint != -1: break

            # Error

            if self.footprint == -1:
                # raise ValueError("Unknown tensor type: " + self.key)
                return self.assumed_footprint

        return self.footprint


    def set_device_idx(self, idx: int | None):
        self.device_idx = idx


    def is_quant(self) -> bool:
        return False


    def reload(self):
        self.unload()
        self.load()


class Intervention(ExLlamaV2Module):

    inner: ExLlamaV2Module

    def __init__(self, inner: ExLlamaV2Module, pre_forward = None, post_forward = None):
        super().__init__(inner.model, inner.key)

        self.inner = inner
        self.pre_forward = pre_forward
        self.post_forward = post_forward

        self.device_idx = self.inner.device_idx
        if hasattr(self.inner, "padding"): self.padding = self.inner.padding

    def numel(self): return self.inner.numel()
    def load(self): return self.inner.load()
    def unload(self): return self.inner.unload()
    def scratch_space_fixed(self): return self.inner.scratch_space_fixed()
    def scratch_space_tp(self): return self.inner.scratch_space_fixed()
    def scratch_space(self): return self.inner.scratch_space()
    def device(self): return self.inner.device()
    def set_device_idx(self, idx: int): raise NotImplementedError()
    def weight_footprint(self): return self.inner.weight_footprint()
    def reload(self): return self.inner.reload()
    def is_quant(self): return self.inner.is_quant()

    def forward(self, hidden_states, *args, **kwargs):

        if self.pre_forward:
            dev = hidden_states.device
            hidden_states = self.pre_forward(hidden_states, *args, **kwargs)
            hidden_states = safe_move_tensor(hidden_states, dev)

        hidden_states = self.inner.forward(hidden_states, *args, **kwargs)

        if self.post_forward:
            dev = hidden_states.device
            hidden_states = self.post_forward(hidden_states, *args, **kwargs)
            hidden_states = safe_move_tensor(hidden_states, dev)

        return hidden_states
