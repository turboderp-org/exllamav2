from __future__ import annotations
import torch
from exllamav2.compat import safe_move_tensor
from exllamav2.tensor_p import BROADCAST_KV

class Params:
    batch_size: int
    seq_len: int
    past_len: int | None
    past_len_tp: list[torch.Tensor | None] | None
    past_lens: list[int] | None
    input_mask: torch.Tensor | None
    multi_cache: bool
    attn_mask: torch.Tensor | None
    attn_masks: torch.Tensor | None
    position_offsets: torch.Tensor | None
    position_offsets_tp: list[torch.Tensor | None] | None
    past_lens_tensor: torch.Tensor | None
    paged: bool
    alt_rope_embed: tuple[torch.Tensor, torch.Tensor] | None
    alt_rope_embed_dict: dict | None
    rope_offsets: torch.Tensor | None
    non_causal_attn: bool
    block_diag_layers: set
    block_diag_mask: torch.Tensor | None
    cu_seqlens: torch.Tensor | None
    cu_seqlens_max: int | None

    def __init__(
        self,
        batch_size: int = 1,
        seq_len: int | None = None,
        past_len: int | list[int] | None = None,
        input_mask: torch.Tensor | None = None,
        position_offsets: torch.Tensor | None = None,
        paged = False,
        alt_rope_embed: tuple[torch.Tensor, torch.Tensor] | dict | None = None,
        non_causal_attn: bool = False,
        rope_offsets: torch.Tensor | None = None
    ):

        self.batch_size = batch_size
        self.paged = paged

        if paged: return

        self.alt_rope_embed = alt_rope_embed if isinstance(alt_rope_embed, tuple) else None
        self.alt_rope_embed_dict = alt_rope_embed if isinstance(alt_rope_embed, dict) else None
        self.rope_offsets = rope_offsets

        self.non_causal_attn = non_causal_attn

        self.seq_len = seq_len
        if isinstance(past_len, list):
            self.past_len = None
            self.past_lens = past_len
            self.multi_cache = True
        else:
            self.past_len = past_len
            self.past_lens = None
            self.multi_cache = False
        self.input_mask = input_mask

        self.attn_mask = None
        self.attn_masks = None

        self.position_offsets = position_offsets
        self.position_offsets_tp = None
        self.past_lens_tensor = None
        self.past_len_tp = None
        self.paged = paged

        self.block_diag_layers = set()
        self.block_diag_mask = None
        self.cu_seqlens = None
        self.cu_seqlens_max = None

    def is_causal(self) -> bool:
        return self.input_mask is None

    def get_position_offsets(self, device) -> torch.Tensor | None:
        assert self.position_offsets is not None
        if self.position_offsets.device != device:
            self.position_offsets = safe_move_tensor(self.position_offsets, device)
        return self.position_offsets

    def get_past_lens(self, device) -> torch.Tensor | None:
        assert self.past_lens is not None
        if self.past_lens_tensor is None:
            self.past_lens_tensor = torch.tensor(self.past_lens, dtype=torch.int, device=device)
        elif self.past_lens_tensor.device != device:
            self.past_lens_tensor = safe_move_tensor(self.past_lens_tensor, device)
        return self.past_lens_tensor

    def get_attn_mask(self, device, force: bool = False) -> torch.Tensor | None:
        if self.attn_mask is None:
            self.attn_mask = self.build_attn_mask(device, force)
        elif self.attn_mask.device != device:
            self.attn_mask = safe_move_tensor(self.attn_mask, device)
        return self.attn_mask

    def get_attn_masks(self, device) -> torch.Tensor | None:
        if self.attn_masks is None:
            self.attn_masks = self.build_attn_masks(device)
        elif self.attn_masks[0] is not None and self.attn_masks[0].device != device:
            self.attn_masks = [(safe_move_tensor(m, device) if m is not None else None) for m in self.attn_masks]
        return self.attn_masks

    def build_single_attn_mask(self, batch_size, seq_len, past_len, device, input_mask):
        attn_mask = torch.zeros((batch_size, 1, seq_len, past_len + seq_len), dtype=torch.float16, device=device)
        attn_mask_triu = torch.triu(torch.full((seq_len - 1, seq_len - 1), -65504.0))
        attn_mask[:, :, : seq_len - 1, past_len + 1: past_len + seq_len] = attn_mask_triu
        if input_mask is not None:
            min_mask_width = min(input_mask.shape[-1], seq_len + past_len)
            input_mask_part = safe_move_tensor(input_mask[:, :min_mask_width], attn_mask.device)
            input_mask_part = input_mask_part.unsqueeze(1).unsqueeze(2)
            attn_mask[:, :, :, :min_mask_width] = torch.minimum(attn_mask[:, :, :, :min_mask_width], input_mask_part)
        return attn_mask

    def build_attn_mask(self, device, force: bool = False) -> torch.Tensor | None:
        assert not self.multi_cache, "Building single mask for multiple caches"
        if self.input_mask is None and self.seq_len == 1 and not force: return None
        return self.build_single_attn_mask(self.batch_size, self.seq_len, self.past_len, device, self.input_mask)

    def build_attn_masks(self, device) -> torch.Tensor | None:
        assert self.multi_cache, "Building multiple masks for single cache"
        attn_masks = []
        for i, past_len in enumerate(self.past_lens):
            if self.input_mask is None and self.seq_len == 1:
                attn_masks.append(None)
            else:
                attn_masks.append(self.build_single_attn_mask(1, self.seq_len, past_len, device, self.input_mask[i]))
        return attn_masks

    def prep_tp(self, model):
        if self.position_offsets_tp is not None:
            return
        split = model.tp_context.get_split(BROADCAST_KV)
        self.position_offsets_tp = []
        self.past_len_tp = []
        pl = torch.tensor([self.past_len] * self.batch_size, dtype = torch.int)
        for dev, a, b in split:
            context = model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)
            if self.position_offsets is None:
                self.position_offsets_tp.append(None)
            else:
                self.position_offsets_tp.append(safe_move_tensor(self.position_offsets, dev, non_blocking = True))
            if self.past_len is None:
                self.past_len_tp.append(None)
            else:
                self.past_len_tp.append(safe_move_tensor(pl, dev, non_blocking = True))

    def get_alt_rope_embed(self, device):
        if not self.alt_rope_embed_dict:
            return None
        emb = self.alt_rope_embed_dict.get(device)
        if not emb:
            emb_cpu = self.alt_rope_embed_dict.get("cpu")
            if not emb_cpu:
                return None
            emb = (
                safe_move_tensor(emb_cpu[0], device, non_blocking = True),
                safe_move_tensor(emb_cpu[1], device, non_blocking = True)
            )
            self.alt_rope_embed_dict[device] = emb
        return emb

    def get_rope_offsets(self, device_idx: int) -> torch.Tensor | None:
        if self.rope_offsets is None:
            return None
        if self.rope_offsets.device.index != device_idx:
            self.rope_offsets = safe_move_tensor(self.rope_offsets, device_idx, non_blocking = True)
        return self.rope_offsets

    def get_cu_seqlens(self, device: int) -> torch.Tensor | None:
        if self.cu_seqlens is None:
            return None
        if self.cu_seqlens.device.index != device:
            self.cu_seqlens = safe_move_tensor(self.cu_seqlens, device, non_blocking = True)
        return self.cu_seqlens

    def get_cu_seqlens_max(self) -> torch.Tensor | None:
        assert self.cu_seqlens is not None
        if self.cu_seqlens_max is not None:
            return self.cu_seqlens_max
        self.cu_seqlens_max = (self.cu_seqlens[1:] - self.cu_seqlens[:-1]).max().item()
        return self.cu_seqlens_max

    def get_block_diag_mask(self, device: int) -> torch.Tensor | None:
        if self.block_diag_mask is None:
            csl = self.get_cu_seqlens(device)
            if csl is None:
                return None
            positions = torch.arange(csl[-1], device = csl.device)
            labels = torch.searchsorted(csl[1:], positions, right = True)
            self.block_diag_mask = labels.unsqueeze(0) == labels.unsqueeze(1).repeat(self.batch_size)
        if self.block_diag_mask.device.index != device:
            self.block_diag_mask = safe_move_tensor(self.block_diag_mask, device, non_blocking = True)
        return self.block_diag_mask


class PagedParams(Params):

    block_index: torch.Tensor
    block_index_tp: torch.Tensor | None
    cache_seqlens: torch.Tensor
    cache_seqlens_tp: torch.Tensor | None
    cache_seqlens_after: torch.Tensor
    cache_seqlens_after_tp: torch.Tensor | None
    alt_rope_embed: dict | None
    rope_offsets: torch.Tensor | None
    max_cache_seqlen: int
    page_size: int
    is_sequential: bool
    first_index: int

    def __init__(
        self,
        batch_size: int,
        block_index: torch.Tensor,
        cache_seqlens: torch.Tensor,
        max_cache_seqlen: int,
        page_size: int,
        q_len: int = 0,
        alt_rope_embed: dict | None = None,
        rope_offsets: torch.Tensor | None = None
    ):
        super().__init__(
            batch_size = batch_size,
            paged = True
        )

        self.block_index = block_index
        self.cache_seqlens = cache_seqlens
        self.max_cache_seqlen = max_cache_seqlen
        self.page_size = page_size
        self.alt_rope_embed = alt_rope_embed or {}
        self.rope_offsets = rope_offsets

        self.is_sequential = False
        assert self.block_index.device.type == "cpu"
        assert self.cache_seqlens.device.type == "cpu"
        assert self.rope_offsets is None or self.rope_offsets.device.type == "cpu"
        assert q_len > 0
        if self.block_index.shape[0] == 1:
            vi0 = self.cache_seqlens[0].item()
            vi1 = vi0 + q_len
            vp0 = vi0 // page_size
            vp1 = (vi1 - 1) // page_size
            for i in range(vp0 + 1, vp1 + 1):
                if self.block_index[0, i].item() != self.block_index[0, i - 1].item() + 1:
                    break
            else:
                self.is_sequential = True
                self.first_index = self.block_index[0, vp0].item() * page_size + vi0 - vp0 * page_size
                self.cache_seqlens_after = self.cache_seqlens + q_len

        self.block_index_tp = None
        self.cache_seqlens_tp = None
        self.cache_seqlens_after_tp = None

    def get_attn_mask(self, device, force: bool = False):
        raise NotImplementedError()

    def get_block_index(self, device) -> torch.Tensor:
        if self.block_index.device != device:
            self.block_index = safe_move_tensor(self.block_index, device)
        return self.block_index

    def get_cache_seqlens(self, device_idx: int) -> torch.Tensor:
        if self.cache_seqlens.device.index != device_idx:
            self.cache_seqlens = safe_move_tensor(self.cache_seqlens, device_idx, non_blocking = True)
        return self.cache_seqlens

    def get_cache_seqlens_after(self, device_idx: int) -> torch.Tensor:
        if self.cache_seqlens_after.device.index != device_idx:
            self.cache_seqlens_after = safe_move_tensor(self.cache_seqlens_after, device_idx, non_blocking = True)
        return self.cache_seqlens_after

    def get_rope_offsets(self, device_idx: int) -> torch.Tensor | None:
        if self.rope_offsets is None:
            return None
        if self.rope_offsets.device.index != device_idx:
            self.rope_offsets = safe_move_tensor(self.rope_offsets, device_idx, non_blocking = True)
        return self.rope_offsets

    def prep_tp(self, model):
        if self.block_index_tp is not None:
            return
        self.block_index_tp = []
        self.cache_seqlens_tp = []
        if self.is_sequential:
            self.cache_seqlens_after_tp = []
        split = model.tp_context.get_split(BROADCAST_KV)
        for dev, a, b in split:
            context = model.get_device_context(dev)
            torch.cuda.set_stream(context.stream)
            self.block_index_tp.append(safe_move_tensor(self.block_index, dev, non_blocking = True))
            self.cache_seqlens_tp.append(safe_move_tensor(self.cache_seqlens, dev, non_blocking = True))
            if self.is_sequential:
                self.cache_seqlens_after_tp.append(safe_move_tensor(self.cache_seqlens_after, dev, non_blocking = True))

    def get_alt_rope_embed(self, device):
        if not self.alt_rope_embed:
            return None
        emb = self.alt_rope_embed.get(device)
        if not emb:
            emb_cpu = self.alt_rope_embed.get("cpu")
            if not emb_cpu:
                return None
            emb = (
                safe_move_tensor(emb_cpu[0], device, non_blocking = True),
                safe_move_tensor(emb_cpu[1], device, non_blocking = True)
            )
            self.alt_rope_embed[device] = emb
        return emb