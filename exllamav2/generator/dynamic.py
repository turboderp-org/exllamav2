from __future__ import annotations

from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer, SeqTensor, ExLlamaV2Lora
from exllamav2.generator import ExLlamaV2Sampler
from exllamav2.generator.filters import ExLlamaV2Filter
from exllamav2.generator.dynamic_embeddings import ExLlamaV2MMEmbedding
from exllamav2.cache import ExLlamaV2CacheBase, ExLlamaV2Cache_8bit
from exllamav2.attn import ExLlamaV2Attention, assert_paged_attn
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
from exllamav2.util import cuda_sync_active, timed
from exllamav2 import mrope
from concurrent.futures import ThreadPoolExecutor

from exllamav2.compat import pairwise
import torch
import random
import numpy as np
import time
import threading
import pprint
from collections import deque
import hashlib
import itertools
from dataclasses import dataclass
# import xxhash
# from line_profiler import profile

# TODO:
#  - ExLlamaV2StreamingGenerator wrapper
#  - Input embeddings
#  - Faster hash algorithm (Murmur?)

PAGED_PAGE_SIZE = 256

def _tensor_blake2b_checksum(tensor: torch.Tensor, prev_hash: bytes | None) -> bytes:
    hasher = hashlib.blake2b(digest_size = 16)
    if prev_hash is not None:
        hasher.update(prev_hash)
    hasher.update(tensor.numpy().tobytes())
    return hasher.digest()

# xxhasher = xxhash.xxh128()
# def _tensor_xxhash128_checksum(tensor: torch.Tensor, prev_hash: bytes | None) -> bytes:
#     global xxhasher
#     xxhasher.reset()
#     if prev_hash is not None:
#         xxhasher.update(prev_hash)
#     xxhasher.update(tensor.numpy().tobytes())
#     return xxhasher.digest()

_tensor_hash_checksum = _tensor_blake2b_checksum

_uniquehash = 0
def _randomhash():
    global _uniquehash
    _uniquehash += 1
    return _uniquehash.to_bytes(16, byteorder = 'big')

@dataclass
class CachePage:

    generator: ExLlamaV2DynamicGenerator
    page_index: int
    # Hash of this page if kv_position == PAGE_SIZE, else random hash. Also used to index (un)referenced_pages
    phash: bytes
    phash_revert: bytes
    # Hash of previous page in chain
    prev_hash: bytes | None
    prev_hash_revert: bytes | None
    # Number of active jobs referencing page
    ref_count: int
    # Last time this page was assigned to a job
    access_serial: int
    access_serial_revert: int
    # Number of tokens in page for which KV is valid assuming prev_hash
    kv_position: int
    kv_position_revert: int
    # Specific tokens for which KV is valid assuming prev_hash
    sequence: torch.Tensor
    can_revert: bool
    # Used by defragmenter
    new_page_index: int

    def __repr__(self):
        return (
            f"CachePage: idx = {self.page_index}, ref_count = {self.ref_count}, "
            f"phash: ..{str(self.phash)[8:24]}.., prev_hash: ..{str(self.prev_hash)[8:24]}.., "
            f"kvp {self.kv_position}"
        )

    def backup(self):
        self.phash_revert = self.phash
        self.prev_hash_revert = self.prev_hash
        self.access_serial_revert = self.access_serial
        self.kv_position_revert = self.kv_position
        self.can_revert = True

    def revert(self):
        assert self.can_revert
        self.phash = self.phash_revert
        self.prev_hash = self.prev_hash_revert
        self.access_serial = self.access_serial_revert
        self.kv_position = self.kv_position_revert
        self.can_revert = False

    def add_ref(self, serial):
        if self.ref_count == 0:
            del self.generator.unreferenced_pages[self.phash]
            assert self.phash not in self.generator.referenced_pages
            self.generator.referenced_pages[self.phash] = self
        self.ref_count += 1
        self.access_serial = max(serial, self.access_serial)
        self.can_revert = False

    def add_ref_clear(self, serial, newhash):
        assert self.ref_count == 0
        del self.generator.unreferenced_pages[self.phash]
        self.phash = newhash
        assert self.phash not in self.generator.referenced_pages
        self.generator.referenced_pages[self.phash] = self
        self.ref_count += 1
        self.access_serial = serial
        self.prev_hash = None
        self.can_revert = False
        self.kv_position = 0

    def add_ref_unique(self, serial):
        self.backup()
        assert self.ref_count == 0
        del self.generator.unreferenced_pages[self.phash]
        self.phash = _randomhash()
        assert self.phash not in self.generator.referenced_pages
        self.generator.referenced_pages[self.phash] = self
        self.ref_count += 1
        self.access_serial = serial
        self.prev_hash = None
        self.kv_position = 0

    def sub_ref(self):
        self.ref_count -= 1
        if self.ref_count == 0:
            del self.generator.referenced_pages[self.phash]
            if self.can_revert:
                self.revert()
            if self.phash in self.generator.referenced_pages or self.phash in self.generator.unreferenced_pages:
                self.phash = _randomhash()
                self.prev_hash = None
            assert self.phash not in self.generator.unreferenced_pages
            self.generator.unreferenced_pages[self.phash] = self

    def clear(self):
        assert self.ref_count == 0
        del self.generator.unreferenced_pages[self.phash]
        self.phash = _randomhash()
        self.prev_hash = None
        self.kv_position = 0
        self.can_revert = False
        self.sequence[:, :] = 0
        assert self.phash not in self.generator.unreferenced_pages
        self.generator.unreferenced_pages[self.phash] = self

    def update_hash(self, newhash):
        assert self.ref_count > 0
        assert self.kv_position == self.generator.page_size
        del self.generator.referenced_pages[self.phash]
        self.phash = newhash
        self.can_revert = False
        assert self.phash not in self.generator.referenced_pages
        self.generator.referenced_pages[self.phash] = self


class NGramTrie:

    token: int
    count: int
    children: dict[int, NGramTrie]
    winningest_child: NGramTrie | None

    def __init__(self, token: int = -1):
        self.token = token
        self.count = 0
        self.children = {}
        self.winningest_child = None


class ExLlamaV2DynamicGenerator:

    model: ExLlamaV2
    cache: ExLlamaV2CacheBase
    draft_model: ExLlamaV2
    draft_cache: ExLlamaV2CacheBase
    tokenizer: ExLlamaV2Tokenizer

    max_batch_size: int
    max_total_tokens: int
    max_seq_len: int
    max_chunk_size: int
    padded_vocab_size: int

    num_draft_tokens: int
    max_ngram: int
    use_ngram_draft: bool

    # Page table

    paged: bool
    max_pages: int
    referenced_pages: dict[bytes: CachePage]
    unreferenced_pages: dict[bytes: CachePage]
    all_pages: list[CachePage]
    access_serial: int
    job_serial: int

    last_defrag_serial: int
    defrag_buffers: dict[tuple, torch.Tensor]

    # Job queue

    pending_jobs: list[ExLlamaV2DynamicJob]
    active_jobs: list[ExLlamaV2DynamicJob]

    # Pinned buffer for receiving logits

    logits_pinned: torch.Tensor
    draft_input_ids_pinned: torch.Tensor
    draft_ids_pinned: torch.Tensor

    # LoRAs

    current_loras: list[ExLlamaV2Lora] | None

    # Sampling threads

    max_sampling_threads: int
    min_sampling_threads: int
    sampling_pool: ThreadPoolExecutor | None
    filter_pool: ThreadPoolExecutor | None
    filter_queue: list | None


    def __init__(
        self,
        model: ExLlamaV2,
        cache: ExLlamaV2CacheBase,
        tokenizer: ExLlamaV2Tokenizer,
        max_batch_size: int = None,
        max_seq_len: int | None = None,
        max_chunk_size: int | None = None,
        max_q_size: int = 8,
        draft_model: ExLlamaV2 | None = None,
        draft_cache: ExLlamaV2CacheBase | None = None,
        num_draft_tokens: int = 4,
        use_ngram_draft: bool = False,
        max_ngram: int = 4,
        max_sampling_threads: int = 16,
        min_sampling_threads: int = 3,
        paged: bool = True,
        filter_background_eval: bool = True,
        **kwargs
    ):
        """
        Initialize generator

        :param model:
            The model (loaded)

        :param cache:
            ExLlamaV2Cache allocated with batch size 1. The max_seq_len of the cache defines the total
            number of tokens that the generator can assign to a batch of jobs.

        :param tokenizer:
            ExLlamaV2Tokenizer

        :param max_batch_size:
            The maximum number of sequences to process in parallel. The generator will also limit this
            dynamically considering the available cache space. Specify None to calculate automatically

        :param max_seq_len:
            Maximum length of each individual sequence. Defaults to the model's max_seq_len.

        :param max_chunk_size:
            Maximum number of tokens to process in parallel during prefill (prompt ingestion). Should not
            exceed the model's max_input_len but can be lowered to trade off prompt speed for a shorter
            interruption to ongoing jobs when a new job is started.

        :param max_q_size:
            Maximum number of tokens to evaluate per sequence during generation. Leave this at the default
            (16) unless there's a good reason to increase it.

        :param draft_model:
            Draft model. Enables speculative decoding with draft, and must be specified along with
            draft_cache. Note that speculative decoding with many parallel jobs is likely not advantageous.

        :param draft_cache:
            ExLlamaV2Cache allocated for draft model. Must have batch_size 1 and same max_seq_len as the
            main cmodel cache.

        :param num_draft_tokens:
            Number of future tokens to draft.

        :param use_ngram_draft:
            Use n-gram speculative decoding. Uses a simple n-gram model created from the input sequence
            to predict future tokens.

        :param max_ngram:
            Longest n-gram to consider.

        :param max_sampling_threads:
            Maximum number of concurrent threads used by sampler.

        :param min_sampling_threads:
            Minimum number of threads to spawn at once. If the batch size for an iteration is lower than this
            number, use single-threaded sampling instead to eliminate multithreading overhead.

        :param paged:
            Enable paged mode, defaults to True. If this is False, the generator uses a fallback unpaged mode which
            does not require paged attention support, but in which the max supported batch size is 1. CFG also will
            not work in this mode.

        :param filter_background_eval:
            Try to overlap filter evaluation with model forward pass. This should generally have no downside since
            filters are evaluated by the CPU which will otherwise be busywaiting after CUDA workload is scheduled.

        :param kwargs:
        """

        # torch.set_num_threads(1)

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        cfg = self.model.config
        self.padded_vocab_size = ((cfg.vocab_size + 31) // 32) * 32

        self.draft_model = draft_model
        self.draft_cache = draft_cache

        if draft_model or use_ngram_draft:
            assert num_draft_tokens <= max_q_size, \
                "num_draft_tokens cannot be larger than max_q_size."
            self.num_draft_tokens = num_draft_tokens
        else:
            self.num_draft_tokens = 0

        if draft_model:
            assert draft_cache is not None, \
                "Must supply cache for draft model"
            assert draft_cache.max_seq_len == cache.max_seq_len and \
                draft_cache.batch_size == cache.batch_size, \
                "Cache and draft cache must be same dimensions"
            assert draft_model.config.max_seq_len >= model.config.max_seq_len, \
                "Draft model seq len must be >= model seq len"

        if paged:
            assert_paged_attn()
            assert not cfg.no_flash_attn, \
                "Paged mode requires flash-attn, but flash-attn is disabled in model config."

        assert not isinstance(cache, ExLlamaV2Cache_8bit), \
            "Dynamic generator does not currently work with 8-bit cache. Use either FP16 or Q4."

        if not max_batch_size:
            max_batch_size = cfg.max_input_len // max_q_size
            self.max_batch_size = max_batch_size
        else:
            model_max_q = cfg.max_batch_size * cfg.max_input_len
            req_max_q = max_q_size * max_batch_size
            assert req_max_q <= model_max_q, \
                f"Model has max_batch_size * max_input_len = {cfg.max_batch_size} * {cfg.max_input_len} tokens, " + \
                f"generator requires max_batch_size * max_q_size = {max_batch_size} * {max_q_size} tokens."
            self.max_batch_size = max_batch_size

        if max_seq_len is not None:
            assert max_seq_len <= model.config.max_seq_len, \
                f"Model initialized with max seq len {model.config.max_seq_len}, " + \
                f"requested seq len is {max_seq_len}"
            self.max_seq_len = max_seq_len
        else:
            self.max_seq_len = model.config.max_seq_len

        # Initialize cache/page table

        self.paged = paged
        self.page_size = PAGED_PAGE_SIZE if paged else self.cache.max_seq_len

        assert cache.batch_size == 1, \
            f"DynamicGenerator requires cache to have batch_size = 1"

        assert self.cache.max_seq_len % PAGED_PAGE_SIZE == 0, \
            f"cache.max_seq_len must be multiple of {PAGED_PAGE_SIZE}, received {cache.max_seq_len}"
        self.max_pages = max(cache.max_seq_len // self.page_size, 1)
        self.max_total_tokens = cache.max_seq_len

        self.reset_page_table()

        # Chunking

        if max_chunk_size is not None:
            assert max_chunk_size <= model.config.max_input_len, \
                f"max_chunk_size must be less than model max_input_len ({model.config.max_input_len}), " + \
                f"received {max_chunk_size}"
            self.max_chunk_size = max_chunk_size
        else:
            self.max_chunk_size = model.config.max_input_len

        if paged:
            assert self.max_chunk_size % self.page_size == 0, \
                f"max_chunk_size must be multiple of {self.page_size}, received {max_chunk_size}"

        # Jobs

        self.job_serial = 0
        self.pending_jobs = []
        self.active_jobs = []

        # Buffers

        self.logits_pinned = torch.empty(
            (max_batch_size, max_q_size, self.padded_vocab_size),
            dtype = torch.float,
            pin_memory = False
        )

        if draft_model:
            self.draft_input_ids_pinned = torch.empty(
                (max_batch_size, 1),
                dtype = torch.long,
                pin_memory = False
            )
            self.draft_ids_pinned = torch.empty(
                (max_batch_size, num_draft_tokens),
                dtype = torch.long,
                pin_memory = False
            )

        # Ngram

        if use_ngram_draft:
            assert draft_model is None, \
                "Cannot use both draft model and ngram draft"
        self.max_ngram = max_ngram
        self.use_ngram_draft = use_ngram_draft

        # LoRAs

        self.current_loras = None

        # Sampling threads

        self.max_sampling_threads = max_sampling_threads
        self.min_sampling_threads = min_sampling_threads
        if max_sampling_threads > 1:
            self.sampling_pool = ThreadPoolExecutor(max_workers = max_sampling_threads)

        # Filter threads

        if filter_background_eval:
            self.filter_pool = ThreadPoolExecutor(max_workers = 16)
            self.filter_queue = []
        else:
            self.filter_pool = None
            self.filter_queue = None

        # Temp buffers for defrag

        if self.paged:

            self.defrag_buffer = {}
            cache_tensors = self.cache.all_tensors()
            if self.draft_cache:
                cache_tensors += self.draft_cache.all_tensors()

            for c in cache_tensors:
                if c is None:
                    continue
                key = (c.device.index, c.dtype, c.shape[2], c.shape[3])
                if key not in self.defrag_buffer:
                    t = torch.empty((1, self.page_size, c.shape[2], c.shape[3]), dtype = c.dtype, device = c.device)
                    self.defrag_buffer[key] = t

        else:
            self.defrag_buffer = None


    def reset_page_table(self):
        """
        Reset the page table.
        """
        self.referenced_pages = {}
        self.unreferenced_pages = {}
        self.all_pages = []
        for idx in range(self.max_pages):
            h = _randomhash()
            cp = CachePage(
                generator = self,
                page_index = idx,
                phash = h,
                phash_revert = h,
                prev_hash = None,
                prev_hash_revert = None,
                sequence = torch.empty((1, self.page_size), dtype = torch.long),
                ref_count = 0,
                access_serial = idx,
                access_serial_revert = 0,
                kv_position = 0,
                kv_position_revert = 0,
                can_revert = False,
                new_page_index = 0
            )
            self.all_pages.append(cp)
            self.unreferenced_pages[h] = cp
        self.access_serial = self.max_pages
        self.last_defrag_serial = self.access_serial


    def warmup(self):
        """
        Warm up the generator by generating some text, making sure kernel autotune has time to complete.
        """
        self.generate("Once upon a time,", max_new_tokens = 32)
        self.reset_page_table()


    def set_loras(self, loras: list[ExLlamaV2Lora] | None):
        """
        Enable LoRAs. Queue must be empty when this is called.

        :param loras:
            List of LoRAs to enable, or None to disable all.
        """

        assert not self.num_remaining_jobs(), \
            "LoRAs cannot be updated while there are jobs in the generator queue."

        if loras is None:
            self.current_loras = []
        elif isinstance(loras, list):
            self.current_loras = loras
        else:
            self.current_loras = [loras]


    def generate(
        self,
        prompt: list[tuple] | list[str] | tuple | str,
        max_new_tokens: int,
        min_new_tokens: int = 0,
        seed: int or None = None,
        gen_settings: ExLlamaV2Sampler.Settings | list[ExLlamaV2Sampler.Settings] | None = None,
        token_healing: bool = False,
        encode_special_tokens: bool = False,
        decode_special_tokens: bool = False,
        stop_conditions: list[int | str] | None = None,
        add_bos: bool = False,
        abort_event: threading.Event | None = None,
        completion_only: bool = False,
        filters: list[list[ExLlamaV2Filter]] | list[ExLlamaV2Filter] | None = None,
        filter_prefer_eos: bool = False,
        return_last_results: bool = False,
        embeddings: list[ExLlamaV2MMEmbedding] | list[list[ExLlamaV2MMEmbedding]] | None = None,
        **kwargs
    ):
        """
        Generate one or more completions.

        :param prompt:
            If this argument is a list, its length determines the batch size, and the output will be a list of strings
            as well. Each prompt is either a string or a pair of prompts for CFG sampling. If CFG is used, sampler
            settings must contain cfg_scale.

        :param gen_settings:
            Sample settings for all prompts in batch or list of settings for each prompt.

        :param max_new_tokens:
            Max number of tokens to generate.

        :param min_new_tokens:
            Minimum number of tokens to generate before stop tokens become active. Until this number have been
            sampled, stop tokens are suppressed but stop strings will still end response.

        :param seed:
            Seed for the sampling RNG. Doesn't guarantee perfect determinism from the implementation.

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param encode_special_tokens:
            Encode special tokens (BOS etc.) represented as text in the input. If False, special tokens are
            interpreted as text by the tokenizer.

        :param decode_special_tokens:
            Decode special tokens output by the model. If False, tokens marked as special in the tokenizer
            are decoded as empty strings.

        :param stop_conditions:
            List of strings and/or token IDs that will end generation. The stop condition is not included
            in the output.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :param filters:
            (List of) list of ExLlamaV2Filters to apply during generation. Each prompt in a batch needs
            its own filter list, or a value of None to disable filters for individual prompts.

        :param filter_prefer_eos:
            If True, always sample the tokenizer's defined EOS token as soon as it's allowed by the filters

        :param return_last_results:
            If True, returns the last results dict for each job

        :param embeddings:
            Optional list of ExLlamaV2MMEmbeddings to use for, or list of lists for batched generation

        :return:
            Completion(s): (str or list[str] depending on the type of the input prompt argument)
            Optionally, last results: (dict or list[dict] depending on the type of the input prompt argument)
        """

        order = {}
        if isinstance(prompt, list):
            prompts = prompt
        else:
            prompts = [prompt]
            filters = [filters]
            embeddings = [embeddings]

        if not filters:
            filters = [None] * len(prompts)
        else:
            assert len(filters) == len(prompts) and \
                all((f is None or isinstance(f, list)) for f in filters), \
                "If using filters, must provide one filter list (or None-value) per prompt."

        if not embeddings:
            embeddings = [None] * len(prompts)
        else:
            assert len(embeddings) == len(prompts) and all((isinstance(f, list) or not f) for f in embeddings), \
                "Must provide one list of embeddings per prompt."

        prompts = prompt if isinstance(prompt, list) else [prompt]
        batch_size = len(prompts)
        for idx, p in enumerate(prompts):

            if isinstance(p, str):
                input_ids = self.tokenizer.encode(
                    p,
                    encode_special_tokens = encode_special_tokens,
                    add_bos = add_bos,
                    embeddings = embeddings[idx]
                )
            elif isinstance(p, tuple):
                input_ids = [self.tokenizer.encode(
                    p_,
                    encode_special_tokens = encode_special_tokens,
                    add_bos = add_bos,
                    embeddings = embeddings[idx]
                ) for p_ in p]
            else:
                assert False, "Unexpected type in prompt"

            if gen_settings is None:
                p_settings = None
            elif isinstance(gen_settings, ExLlamaV2Sampler.Settings):
                p_settings = gen_settings
            elif isinstance(gen_settings, list):
                assert len(gen_settings) == len(prompts)
                p_settings = gen_settings[idx]
            else:
                assert False, "Unexpected type in gen_settings"

            job = ExLlamaV2DynamicJob(
                input_ids = input_ids,
                max_new_tokens = max_new_tokens,
                min_new_tokens = min_new_tokens,
                seed = seed,
                stop_conditions = stop_conditions,
                gen_settings = p_settings,
                filters = filters[idx] or [],
                filter_prefer_eos = filter_prefer_eos,
                token_healing = token_healing,
                decode_special_tokens = decode_special_tokens,
                embeddings = embeddings[idx] or []
            )

            if seed is not None: seed += 1

            serial = self.enqueue(job)
            order[serial] = idx

        # Collect outputs until all jobs finish

        completions = [""] * batch_size
        last_results = [None] * batch_size

        while self.num_remaining_jobs():
            results = self.iterate()

            for r in results:
                idx = order[r["serial"]]
                if r["stage"] == "streaming":
                    text = r.get("text", "")
                    completions[idx] += text
                if r["eos"]:
                    last_results[idx] = r
            if abort_event is not None and abort_event.is_set():
                self.clear_queue()
                return None

        # Return results

        if not completion_only:
            completions = [(p if isinstance(p, str) else p[0]) + c for p, c in zip(prompts, completions)]

        if not isinstance(prompt, list):
            completions = completions[0]
            last_results = last_results[0]

        if return_last_results:
            return completions, last_results
        else:
            return completions


    def print_page_list(self, short: bool = True):
        for cp in self.all_pages:
            if cp.phash in self.referenced_pages:
                assert cp.ref_count > 0
                ref = str(cp.ref_count) if cp.ref_count < 10 else "+"
            elif cp.phash in self.unreferenced_pages:
                assert cp.ref_count == 0
                ref = "."
            else:
                ref = "#"
            if short: print(ref, end = "")
            else: print(str(cp) + f", ref {ref}")
        print()


    def validate_cache(self):
        pass

    def ___validate_cache(self):
        try:
            assert len(self.referenced_pages) + len(self.unreferenced_pages) == self.max_pages, "sum"
            ref_counts = [0] * self.max_pages
            for job in self.active_jobs:
                for seq in job.sequences:
                    for page in seq.allocated_pages:
                        ref_counts[page.page_index] += 1
            page_refs = set()
            for page in self.referenced_pages.values():
                assert page.page_index not in page_refs
                page_refs.add(page.page_index), "r dup " + str(page)
            for page in self.unreferenced_pages.values():
                assert page.page_index not in page_refs
                page_refs.add(page.page_index), "u dup " + str(page)
            assert len(page_refs) == self.max_pages
            for page in self.all_pages:
                assert page.ref_count >= 0, "ref_count < 0 " + str(page)
                if page.ref_count == 0:
                    assert page.phash in self.unreferenced_pages, "u not found" + str(page)
                else:
                    assert page.phash in self.referenced_pages, "r not found" + str(page)
                n = 0
                if page.phash in self.referenced_pages: n += 1
                if page.phash in self.unreferenced_pages: n += 1
                if n != 1:
                    print("-- Referenced:")
                    pprint.pprint(self.referenced_pages)
                    print("-- Unreferenced:")
                    pprint.pprint(self.unreferenced_pages)
                    assert False, f"n == {n} " + str(page)
            for (h, page) in self.unreferenced_pages.items():
                assert page.ref_count == 0, "u ref_count != 0 " + str(page)
                assert page.phash == h, "u hash " + str(page)
                assert page.ref_count == ref_counts[page.page_index], "u refc " + str(page)
                assert h not in self.referenced_pages, "u2r " + str(page)
            for (h, page) in self.referenced_pages.items():
                assert page.ref_count > 0, "r ref_count == 0 " + str(page)
                assert page.phash == h, "r hash " + str(page)
                assert page.ref_count == ref_counts[page.page_index], "r refc " + str(page)
                assert h not in self.unreferenced_pages, "r2u " + str(page)
            # for job in self.active_jobs:
            #     for seq in job.sequences:
            #         spos = 0
            #         prev_hash = None
            #         for page in seq.allocated_pages:
            #             spos2 = min(spos + self.page_size, seq.kv_position)
            #             ids = seq.sequence_ids.torch()[:, spos:spos2]
            #             assert page.kv_position >= ids.shape[-1]
            #             if ids.shape[-1] > 0:
            #                 assert page.prev_hash == prev_hash, "bad prev_hash " + str(job) + " -> " + str(page)
            #             if ids.shape[-1] == self.page_size:
            #                 phash = _tensor_hash_checksum(ids, prev_hash)
            #                 assert page.phash == phash, "bad phash " + str(job) + " -> " + str(page)
            #                 prev_hash = phash
            #             spos = spos2


        except Exception as ex:
            print(ex)
            raise ex


    def num_remaining_jobs(self):
        return len(self.pending_jobs) + len(self.active_jobs)


    def clear_queue(self):

        num_jobs = self.num_remaining_jobs()

        for job in self.active_jobs + self.pending_jobs:
            job.deallocate_pages()
        self.active_jobs.clear()
        self.pending_jobs.clear()

        if num_jobs and not self.num_remaining_jobs():
            self.defrag_cache()

    def enqueue(
        self,
        job: ExLlamaV2DynamicJob | list[ExLlamaV2DynamicJob]
    ) -> int | list[int]:
        """
        Adds a job or list of jobs to the queue.

        returns:
            int: (List of) unique serial number(s) for job(s)
        """

        if isinstance(job, list):
            serials = []
            for j in job:
                serials.append(self.enqueue(j))
            return serials

        job.prepare_for_queue(self, self.job_serial)
        self.job_serial += 1
        self.pending_jobs.append(job)
        job.time_enqueue = time.time()
        return job.serial_number


    def cancel(
        self,
        job: ExLlamaV2DynamicJob
    ):

        num_jobs = self.num_remaining_jobs()

        if job in self.pending_jobs:
            self.pending_jobs.remove(job)
        elif job in self.active_jobs:
            job.deallocate_pages()
            self.active_jobs.remove(job)

        if num_jobs and not self.num_remaining_jobs():
            self.defrag_cache()

        self.validate_cache()


    def get_paged_params(
        self,
        batch_size: int,
        block_index: torch.Tensor,
        cache_seqlens: torch.Tensor,
        q_len: int,
        alt_rope_embed: dict | None = None,
        rope_offsets: torch.Tensor | None = None
    ):

        assert batch_size == 1 or not alt_rope_embed, \
            "Can't use alt_rope_embed if fwd batch_size > 1"

        # assert all(
        #     cache_seqlens[i].item() + q_len <= block_index.shape[-1] * self.page_size
        #     for i in range(batch_size)
        # )

        if self.paged:
            return ExLlamaV2Attention.PagedParams(
                batch_size = batch_size,
                block_index = block_index,
                cache_seqlens = cache_seqlens,
                max_cache_seqlen = cache_seqlens.max().item(),
                page_size = self.page_size,
                q_len = q_len,
                alt_rope_embed = alt_rope_embed,
                rope_offsets = rope_offsets,
            )
        else:
            assert cache_seqlens.shape[0] == 1
            return ExLlamaV2Attention.Params(
                batch_size = 1,
                seq_len = q_len,
                past_len = cache_seqlens[0].item(),
                alt_rope_embed = alt_rope_embed,
                rope_offsets = rope_offsets,
            )

    @torch.inference_mode
    def iterate(self) -> list[dict]:
        """
        Performs inference on available jobs.

        :return:
            List of dicts:

            # Job has started
            {
                "job": ExLlamaV2DynamicJob  - reference to job
                "stage": "started"
                "identifier":  - optional identifier
                "serial": int  - job serial number
                "eos": bool  - always False at this stage
            }

            # Prefill is underway
            {
                "job": ExLlamaV2DynamicJob  - reference to job
                "stage": "prefill"
                "curr_progress": int  - prompt tokens ingested so far
                "max_progress": int  - total prompt tokens to ingest
                "identifier":  - optional identifier
                "serial": int   - job serial number
                "eos": bool  - always False at this stage
            }

            # Generation is underway
            {
                "job": ExLlamaV2DynamicJob  - reference to job
                "stage": "streaming"
                "identifier":  - optional identifier
                "serial": int   - job serial number
                "eos": bool  - True if stop condition has been met

                optional, if eos:
                    "eos_reason":  - one of:
                        "stop_token"
                        "stop_string"
                        "max_new_tokens"
                        "end_filter"
                    optional, if "eos_reason" == "stop_token":
                        "eos_triggering_token_id": int
                        "eos_triggering_token_str": str
                    optional, if "eos_reason" == "stop_string":
                        "eos_triggering_string": str
                    "full_completion": str  - full text completion
                    "new_tokens": int  - number of tokens generated
                    "time_enqueued": float  - time from job was enqueued until it started, in seconds
                    "time_prefill": float  - time to first token, in seconds
                    "time_generate": float  - time to last token, in seconds
                    optional, if SD enabled:
                        "accepted_draft_tokens": int
                        "rejected_draft_tokens": int

                "text": str  - streamed text output. Does not include prefix from healed token, or stop string
                "token_ids": torch.Tensor  - output tokens, shape (1, n)
                "token_probs": torch.Tensor  - last sampling probability of output tokens, shape (1, n)
                "top_k_tokens": torch.Tensor  - shape (1, n, k)
                "top_k_probs": torch.Tensor  - shape (1, n, k)
                "logits": torch.Tensor  - shape (1, n, vocab_size)
            }
        """

        results = []
        self.iterate_start_jobs(results)

        # Perform one round of prefill

        for job in self.active_jobs:
            job.prefill(results)

        # Generation with draft model

        if self.draft_model:
            draft_tokens = self.iterate_draftmodel_gen(results)
            self.iterate_gen(results, draft_tokens)

        # Generation with ngram draft

        elif self.use_ngram_draft:
            draft_tokens = self.iterate_ngram_gen(results)
            self.iterate_gen(results, draft_tokens)

        # Regular generation

        else:
            self.iterate_gen(results)

        # Finished iteration

        return results


    def iterate_ngram_gen(self, results: list):

        draft_ids_list = []

        for job in self.active_jobs:
            if not job.is_prefill_done(): continue

            # Update trie

            sequence = job.sequences[0].sequence_ids
            pos_a = job.ngram_position
            pos_b = len(sequence) - self.max_ngram
            if pos_b > pos_a:
                subs = sequence.torch()[0, pos_a : pos_b + self.max_ngram].tolist()
                job.ngram_position = pos_b

                for i in range(len(subs) - self.max_ngram):
                    node = job.ngrams
                    for j in range(i, i + self.max_ngram):
                        token = subs[j]
                        if token not in node.children:
                            node.children[token] = NGramTrie(token)
                        child = node.children[token]
                        child.count += 1
                        if not node.winningest_child or child.count > node.winningest_child.count:
                            node.winningest_child = child
                        node = child

            # Predict

            subs = sequence.torch()[0, -(self.max_ngram - 1):].tolist()
            ids = []

            for i in range(self.num_draft_tokens):
                w = subs[-1]
                for j in range(self.max_ngram - 1):
                    node = job.ngrams
                    for k in range(j, self.max_ngram - 1):
                        token = subs[k]
                        node = node.children.get(token)
                        if not node: break
                    if node:
                        w = node.winningest_child.token
                        break
                ids.append(w)
                subs = subs[1:] + [w]

            draft_ids_list.append(torch.tensor([ids], dtype = torch.long))

        return torch.cat(draft_ids_list, dim = 0) if len(draft_ids_list) > 0 else None


    def iterate_draftmodel_gen(self, results: list):

        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens + 1)
            batch_size += 1

        if batch_size == 0:
            return None

        max_pages_batch = (max_seq_len + self.page_size - 1) // self.page_size
        block_index = torch.zeros((batch_size, max_pages_batch), dtype = torch.int32)
        cache_seqlens = torch.zeros((batch_size,), dtype = torch.int32)

        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                seq_block_index = job.get_block_index(seq, len(seq.sequence_ids) + self.num_draft_tokens + 1)
                block_index[batch, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Collect input IDs

        input_ids_list = []
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            job_ids = job.get_input_ids_list()
            input_ids_list += job_ids

        batch_ids = self.draft_input_ids_pinned[:batch_size, :]
        batch_ids.copy_(torch.cat(input_ids_list, dim = 0))

        # Indexed embeddings not supported when drafting

        for job in self.active_jobs:
            assert not job.embeddings, \
                "Embeddings not supported while using draft model."

        # Greedy sample draft IDs

        for idx in range(self.num_draft_tokens):

            attn_params = self.get_paged_params(batch_size, block_index, cache_seqlens, 1)

            device_logits = self.draft_model.forward_chunk(
                input_ids = batch_ids,
                attn_params = attn_params,
                cache = self.draft_cache,
            )["logits"]

            new_ids = torch.argmax(device_logits, dim = -1)
            self.draft_ids_pinned[:batch_size, idx:idx+1].copy_(new_ids)
            batch_ids.copy_(new_ids)
            cache_seqlens += 1

        # TODO: Need keys/values for the last token, but only if it's accepted. This could be delayed until after
        #   sampling to skip one pass of the draft model sometimes.

        attn_params = self.get_paged_params(batch_size, block_index, cache_seqlens, 1)

        self.draft_model.forward_chunk(
            input_ids = batch_ids,
            attn_params = attn_params,
            cache = self.draft_cache,
            preprocess_only = True
        )

        return self.draft_ids_pinned


    # @profile
    def iterate_gen(self, results: list, draft_tokens: torch.Tensor | None = None):

        batch_size = 0
        max_seq_len = 0
        for job in self.active_jobs:
            if not job.is_prefill_done():
                continue
            max_seq_len = max(max_seq_len, job.get_max_seq_len() + self.num_draft_tokens)
            batch_size += len(job.sequences)

        if batch_size == 0:
            return  # Nothing more to do this iteration

        max_pages_batch = (max_seq_len + self.page_size - 1) // self.page_size
        block_index = torch.zeros((batch_size, max_pages_batch), dtype = torch.int32)
        cache_seqlens = torch.zeros((batch_size,), dtype = torch.int32)

        batch = 0
        for job in self.active_jobs:
            if not job.is_prefill_done(): continue
            for seq in job.sequences:
                t_len = len(seq.sequence_ids)
                if draft_tokens is not None:
                    t_len += draft_tokens.shape[-1]
                seq_block_index = job.get_block_index(seq, t_len)
                block_index[batch, :seq_block_index.shape[-1]].copy_(seq_block_index)
                cache_seqlens[batch] = seq.kv_position
                batch += 1

        # Collect input IDs and indexed embeddings

        input_ids_list = []
        active_embeddings = []
        logit_mapping = []
        rope_offsets_list = [] if self.model.config.arch.lm.mrope else None
        for job in self.active_jobs:
            logit_mapping.append(len(input_ids_list))
            if not job.is_prefill_done(): continue
            if job.time_first_token is None:
                cuda_sync_active()
                job.time_first_token = time.time()
            if draft_tokens is None:
                job_ids = job.get_input_ids_list(add_to_cache = True)
            else:
                job_ids = job.get_input_ids_list(draft_tokens, len(input_ids_list), add_to_cache = True)
            input_ids_list += job_ids
            active_embeddings += job.embeddings
            if rope_offsets_list is not None:
                rope_offsets_list += [job.alt_rope_offset] * len(job_ids)

        logit_mapping.append(len(input_ids_list))

        batch_ids = torch.cat(input_ids_list, dim = 0)

        # Get logit batch from model

        attn_params = self.get_paged_params(
            batch_size,
            block_index,
            cache_seqlens,
            batch_ids.shape[-1],
            rope_offsets = torch.tensor(rope_offsets_list, dtype = torch.int) if rope_offsets_list is not None else None
        )

        device_logits = self.model.forward_chunk(
            input_ids = batch_ids,
            attn_params = attn_params,
            cache = self.cache,
            loras = self.current_loras,
            indexed_embeddings = active_embeddings
        )["logits"]

        # GPU workload is scheduled here, so launch any sampling filters that can run while waiting for CUDA

        if self.filter_queue:
            for f, p in self.filter_queue:
                if p:
                    f.background_prepare_logit_mask(self.filter_pool)
                else:
                    f.background_next(self.filter_pool)
            # time.sleep(0)
            self.filter_queue.clear()

        # Pass logits to jobs for sampling

        batch_logits = self.logits_pinned[:device_logits.shape[0], :device_logits.shape[1], :]
        batch_logits.copy_(device_logits, non_blocking = False)
        # device_logits = device_logits.float().cpu()
        # ext_c.fast_copy_cpu(batch_logits, device_logits)
        # torch.cuda.synchronize()

        if self.max_sampling_threads > 1 and len(self.active_jobs) >= self.min_sampling_threads:
            mt_sample = True
            futures = deque()
            for job, a, b in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
                if a == b: continue
                job_logits = batch_logits[a:b, :1, :]
                futures.append(self.sampling_pool.submit(job.receive_logits, job_logits))
        else:
            mt_sample = False

        completed_jobs = []
        j = 0
        for job, a, b in zip(self.active_jobs, logit_mapping[:-1], logit_mapping[1:]):
            if a == b: continue
            for i in range(batch_logits.shape[1]):
                job_logits = batch_logits[a:b, i:i+1, :]
                if i == 0 and mt_sample:
                    next_token, next_k_tokens, next_k_probs, next_prob, filter_eos = \
                    futures.popleft().result()
                else:
                    next_token, next_k_tokens, next_k_probs, next_prob, filter_eos = \
                    job.receive_logits(job_logits)

                eos, sampled_token = job.receive_sample(
                    job_logits,
                    next_token,
                    next_k_tokens,
                    next_k_probs,
                    next_prob,
                    filter_eos,
                    results,
                    i == 0
                )

                if eos:
                    completed_jobs.append(job)
                    break
                if draft_tokens is not None and i < batch_logits.shape[1] - 1:
                    if draft_tokens[j, i].item() != sampled_token.item():
                        rejected = batch_logits.shape[1] - 1 - i
                        job.rejected_draft_tokens += rejected
                        for seq in job.sequences:
                            r = rejected
                            while r:
                                pos = seq.kv_position + r
                                page = seq.allocated_pages[(pos - 1) // self.page_size]
                                rp = min(page.kv_position, r)
                                page.kv_position -= rp
                                r -= rp
                        break
                    else:
                        job.accepted_draft_tokens += 1
            j += 1

        # Release pages for completed jobs

        num_jobs = self.num_remaining_jobs()

        for job in completed_jobs:
            job.deallocate_pages()
            self.active_jobs.remove(job)

        if num_jobs and not self.num_remaining_jobs():
            self.defrag_cache()


    def iterate_start_jobs(self, results: list):

        # Get current max batch

        current_max_batch = 0
        for job in self.active_jobs:
            current_max_batch += len(job.sequences)

        # Start new jobs if possible

        if (len(self.unreferenced_pages) and
            len(self.pending_jobs) and
            current_max_batch < self.max_batch_size):

            skipped_jobs = []
            for job in self.pending_jobs.copy():

                if (len(job.sequences) + current_max_batch > self.max_batch_size or
                        job.current_new_pages_required() > len(self.unreferenced_pages)):
                    skipped_jobs.append(job)
                    continue

                # Make sure the job we're about to add doesn't skip a job that's been skipped too many times

                for j in skipped_jobs:
                    if j.skips >= j.max_skips:
                        return
                for j in skipped_jobs:
                    j.skips += 1

                # Add job to active list

                self.pending_jobs.remove(job)
                self.active_jobs.append(job)

                # Allocate pages for job

                job.allocate_pages()
                current_max_batch += len(job.sequences)

                self.validate_cache()

                r = {
                    "job": job,
                    "stage": "started",
                    "eos": False,
                    "serial": job.serial_number,
                }
                if job.identifier is not None:
                    r.update({ "identifier": job.identifier })
                results.append(r)


    @torch.inference_mode
    def defrag_cache(self):

        if not self.paged:
            return

        if self.access_serial < self.last_defrag_serial + self.max_pages:
            return
        self.last_defrag_serial = self.access_serial

        assert not self.referenced_pages

        @dataclass
        class CacheNode:
            page: CachePage | None
            parent: CachePage | None = None
            children: set[CacheNode] = None
            left_page: int = len(self.all_pages)
            def __init__(self, page_):
                self.page = page_
                if self.page:
                    self.left_page = page_.access_serial
                self.children = set()
            def __hash__(self):
                return id(self)
            def __eq__(self, other):
                return self is other

        # Build a tree of the current cache

        root_node = CacheNode(None)
        node_index = {}

        for page in self.all_pages:
            assert page.phash is not None
            node_index[page.phash] = CacheNode(page)

        for node in node_index.values():
            parent = node_index.get(node.page.prev_hash, root_node)
            node.parent = parent
            parent.children.add(node)
            while node.parent:
                node.parent.age = min(node.parent.left_page, node.left_page)
                node = node.parent

        # Remove oldest branch until tree is empty

        new_page_index = 0
        while root_node.children:
            oldest = min(root_node.children, key = lambda x: x.left_page)
            node = oldest
            skipped_nodes = set()
            while True:
                node.page.new_page_index = new_page_index
                new_page_index += 1
                if not node.children: break
                next_node = min(node.children, key = lambda x: x.left_page)
                skipped_nodes |= set([n for n in node.children if n != next_node])
                node = next_node
            root_node.children.remove(oldest)
            root_node.children |= skipped_nodes

        # Order of operations

        defrag_map = {}
        for page in self.all_pages:
            if page.page_index != page.new_page_index:
                defrag_map[page.new_page_index] = page.page_index

        # Shuffle pages

        cache_tensors = self.cache.all_tensors()
        if self.draft_cache:
            cache_tensors += self.draft_cache.all_tensors()
        defrag_buffers = [self.defrag_buffer[c.device.index, c.dtype, c.shape[2], c.shape[3]] for c in cache_tensors]

        while defrag_map:
            assert len(defrag_map) >= 2
            target = next(iter(defrag_map))
            source = defrag_map[target]
            del defrag_map[target]

            rotation = [target]
            while source != rotation[0]:
                rotation.append(source)
                target = source
                source = defrag_map[target]
                del defrag_map[target]

            rotation = [r * self.page_size for r in rotation]
            for cache, buffer in zip(cache_tensors, defrag_buffers):
                buffer[:, :, :, :].copy_(cache[:, rotation[0] : rotation[0] + self.page_size, :, :])
                for a, b in pairwise(rotation):
                    cache[:, a : a + self.page_size, :, :].copy_(cache[:, b : b + self.page_size, :, :])
                cache[:, rotation[-1] : rotation[-1] + self.page_size, :, :].copy_(buffer[:, :, :, :])

        # Update page table

        for page in self.all_pages:
            page.page_index = page.new_page_index

        self.validate_cache()


# Convert list of strings to UTF32 format to pass by reference to partial matching function

def _strings_to_utf32(strings: list[str]) -> (np.array, list[int]):

    if not strings: return bytearray(), None

    encoded_strings = [s.encode("utf-32-le") for s in strings]
    encoded_lengths = [len(s) for s in encoded_strings]
    offsets = [0] + encoded_lengths
    for i in range(1, len(offsets)):
        offsets[i] += offsets[i - 1]
    total_length = offsets[-1]
    concat_strings = bytearray(total_length)
    for s, offset in zip(encoded_strings, offsets[:-1]):
        concat_strings[offset:offset + len(s)] = s

    concat_strings = np.frombuffer(concat_strings, dtype = np.uint8)
    offsets = np.frombuffer(np.array(offsets, dtype = np.int32), dtype = np.uint8)
    return concat_strings, offsets


# Count matching elements from left between two (1, n) tensors
#
# def _count_match(a: torch.Tensor, b: torch.Tensor):
#     m = min(a.shape[-1], b.shape[-1])
#     for i in range(m):
#         if a[0, i] != b[0, i]:
#             return i
#     return m


class ExLlamaV2DynamicJob:

    generator: ExLlamaV2DynamicGenerator | None
    serial_number: int | None

    class Sequence:
        input_ids: SeqTensor
        sequence_ids: SeqTensor
        kv_position: int
        page_hashes: list[bytes]
        new_unique_pages: int
        allocated_pages: list[CachePage] | None
        block_index_tensor: torch.Tensor | None
        prefill_complete: bool
        live: bool

    sequences: list[Sequence]
    all_unique_hashes: list[bytes]

    max_new_tokens: int
    min_new_tokens: int
    new_tokens: int
    gen_settings: ExLlamaV2Sampler.Settings
    rng: random.Random

    decode_special_tokens: bool
    return_top_tokens: int
    prefix_token: torch.Tensor | None

    skips: int
    max_skips: int | None

    # Measurement

    time_enqueue: float | None
    time_first_prefill: float | None
    time_first_token: float | None
    time_last_token: float | None
    accepted_draft_tokens: int
    rejected_draft_tokens: int
    cached_pages: int
    cached_tokens: int
    is_finished: bool
    non_sequential_pages: int
    total_pages: int

    # Output buffers

    held_text: str
    held_tokens: SeqTensor
    held_k_tokens: SeqTensor
    held_k_probs: SeqTensor
    held_probs: SeqTensor
    held_logits: SeqTensor

    full_completion: str

    # Ngrams

    ngrams: NGramTrie
    ngram_position: int

    # Filters

    filters: list[ExLlamaV2Filter] | None
    filter_prefer_eos: bool

    # Stop conditions

    stop_strings: set
    stop_strings_list: list
    stop_strings_utf32_buffer: np.array or None
    stop_strings_utf32_offsets: np.array or None
    stop_tokens: set
    stop_tokens_list: list

    # Banned strings

    banned_strings: list[str]
    banned_strings_utf32_buffer: np.array or None
    banned_strings_utf32_offsets: np.array or None
    checkpoint: dict | None

    # Hold reference to embeddings

    embeddings: list[ExLlamaV2MMEmbedding]
    alt_rope_embed: dict
    alt_rope_offset: int


    def __init__(
        self,
        input_ids: torch.Tensor | list[torch.Tensor],
        max_new_tokens: int,
        min_new_tokens: int = 0,
        max_skips: int | None = 4,
        gen_settings: ExLlamaV2Sampler.Settings | None = None,
        seed: int = None,
        stop_conditions: list | tuple | set = None,
        decode_special_tokens: bool = False,
        return_top_tokens: int = 0,
        return_logits: bool = False,
        return_probs: bool = False,
        filters: list[ExLlamaV2Filter] | None = None,
        filter_prefer_eos: bool = False,
        token_healing: bool = False,
        identifier: object | None = None,
        banned_strings: list[str] | None = None,
        embeddings: list[ExLlamaV2MMEmbedding] | None = None,
        **kwargs
    ):
        """
        Create new job.

        :param input_ids:
            Tokenized IDs of the input prompt, shape (1, n). Alternatively, list of tokenized IDs to inference on
            seperately but sample collectively (e.g. CFG prompt pair)

        :param max_new_tokens:
            Max no. output tokens to allow

        :param min_new_tokens:
            Minimum number of tokens to generate before stop tokens become active. Until this number have been
            sampled, stop tokens are suppressed but stop strings will still end response.

        :param max_skips:
            In the event that the job is too large to fit in the cache at any given moment but there are
            smaller jobs pending that would fit, those smaller jobs are started instead. This number
            specifies the maximum number of times a job can be skipped in favor of a smaller job before it
            stalls the queue. After this, the job is guaranteed to be the next job started.

        :param gen_settings:
            ExLlamaV2Sampler.Settings containing sampling parameters

        :param seed:
             RNG seed (determinism is not guaranteed)

        :param stop_conditions:
            List of strings and/or token IDs that will trigger the EOS condition. If a stop condition is
            encountered it is not emitted as output. If the beginning of a stop string is sampled, stream
            output will be held until the stop condition can be resolved.

        :param decode_special_tokens:
            If True, special tokens like <|im_start|> etc. will be decoded and included in the text output.
            If False, special tokens will still be respected as stop conditions.

        :param return_top_tokens:
            Number of top tokens to return, along with their final sampling probabilities. There is some
            performance penalty for enabling this.

        :param return_logits:
            Return pre-sampling logits along with output tokens.

        :param return_probs:
            Return final sampling probability for each chosen token.

        :param filters:
            List of ExLlamaV2Filters to apply during generation.

        :param filter_prefer_eos:
            If True, the sampler will prefer whatever token the filter presents as an EOS condition, e.g.
            the outer closing bracket in a JSON grammar, as soon as that (sub)token is legal under the
            grammar.

        :param token_healing:
            Resample the last token of the input with a prefix constraint. E.g. if the last token is
            "_Hel", it is removed from the input and the first token of the output will be constrained to
            one of "_Hello", "_Help", "_Helium", etc. Only the added part of the healed token is emitted as
            text, i.e. "lo", "p", "ium" etc.

        :param identifier:
            Object to return with every stream event relating to this job

        :param embeddings:
            Optional list of ExLlamaV2MMEmbeddings to use for, or list of lists for batched generation

        :param kwargs:
        """

        assert all(ids.device.type == "cpu" for ids in input_ids), \
                "input_ids must reside in system memory"

        self.generator = None
        self.serial_number = None
        self.identifier = identifier

        self.max_skips = max_skips
        self.allocated_pages = None

        if gen_settings is None:
            gen_settings = ExLlamaV2Sampler.Settings()

        # Prepare sequences

        if not isinstance(input_ids, list):
            input_ids = [input_ids]

        if token_healing and all(ids.shape[-1] > 1 for ids in input_ids):
            input_seq_ids = [ids[:, :-1] for ids in input_ids]
            self.prefix_token = torch.cat([ids[:, -1:] for ids in input_ids], dim = 0)
        else:
            input_seq_ids = input_ids
            self.prefix_token = None

        self.sequences = []
        for ids, seq_ids in zip(input_ids, input_seq_ids):
            assert ids.shape[-1] > 0, \
                "Input IDs cannot be empty."
            assert ids.shape[0] == 1, \
                "input_ids must be [1, seq_len] tensor or list of [1, seq_len] tensors"
            seq = ExLlamaV2DynamicJob.Sequence()
            seq.input_ids = SeqTensor.from_tensor(ids, seq_dim = -1)
            seq.sequence_ids = SeqTensor.from_tensor(seq_ids, seq_dim = -1)
            seq.kv_position = 0
            seq.page_hashes = None
            seq.new_unique_pages = 0
            seq.allocated_pages = None
            seq.block_index_tensor = None
            seq.live = True
            seq.prefill_complete = False
            self.sequences.append(seq)

        # Generation parameters

        self.max_new_tokens = max_new_tokens
        self.min_new_tokens = min_new_tokens
        self.new_tokens = 0 if self.prefix_token is None else -1
        self.gen_settings = gen_settings
        self.rng = random.Random() if seed is None else random.Random(seed)

        # Output options

        self.decode_special_tokens = decode_special_tokens
        self.return_top_tokens = return_top_tokens
        self.return_logits = return_logits
        self.return_probs = return_probs

        # Stop conditions

        self.stop_strings = set()
        self.stop_tokens = set()
        if stop_conditions is not None:
            for t in stop_conditions:
                if isinstance(t, int): self.stop_tokens.add(t)
                elif isinstance(t, str): self.stop_strings.add(t)
                else: raise ValueError("Unsupported type in stop_conditions")
            self.stop_strings_utf32_buffer, self.stop_strings_utf32_offsets = \
                _strings_to_utf32(list(self.stop_strings))
        else:
            self.stop_strings_utf32_buffer, self.stop_strings_utf32_offsets = None, None

        self.stop_tokens_list = list(self.stop_tokens)
        self.stop_strings_list = list(self.stop_strings)

        # Banned strings

        if banned_strings:
            assert filters is None or len(filters) == 0, \
                "Cannot combine banned strings with filters"
            self.banned_strings = [s.lower() for s in banned_strings]
            self.banned_strings_utf32_buffer, self.banned_strings_utf32_offsets = \
                _strings_to_utf32(self.banned_strings)
        else:
            self.banned_strings = []
            self.banned_strings_utf32_buffer, self.banned_strings_utf32_offsets = None, None

        self.checkpoint = None

        # Measurement

        self.time_enqueue = None
        self.time_first_prefill = None
        self.time_first_token = None
        self.time_last_token = None
        self.accepted_draft_tokens = 0
        self.rejected_draft_tokens = 0
        self.cached_pages = 0
        self.cached_tokens = 0
        self.is_finished = False
        self.non_sequential_pages = 0
        self.total_pages = 0

        # Ngram

        self.ngrams = NGramTrie()
        self.ngram_position = 0

        # Filters

        self.filters = filters if filters is not None else []
        self.filter_prefer_eos = filter_prefer_eos

        # Embeddings

        self.embeddings = embeddings or []
        self.alt_rope_embed = {}
        self.alt_rope_offset = 0


    def __repr__(self):
        if self.serial_number is None:
            return "ExLlamaV2DynamicJob (new)"
        else:
            return f"ExLlamaV2DynamicJob #{self.serial_number}"


    def is_prefill_done(self):
        return all(seq.kv_position == len(seq.sequence_ids) - 1 for seq in self.sequences)


    def get_max_seq_len(self):
        if not self.is_prefill_done():
            return 0
        max_seq_len = 0
        for seq in self.sequences:
            if seq.kv_position == len(seq.sequence_ids) - 1:
                max_seq_len = max(max_seq_len, len(seq.sequence_ids))
        return max_seq_len


    def get_input_ids_list(self, draft_tokens: torch.Tensor | None = None, idx: int = 0, add_to_cache: bool = False):
        input_ids_list = []
        for seq in self.sequences:
            ids = seq.sequence_ids.torch_slice(seq.kv_position, None)
            if draft_tokens is not None:
                ids = torch.cat((ids, draft_tokens[idx:idx+1, :]), dim = -1)
            input_ids_list.append(ids)
            if add_to_cache:
                tokens_to_add = ids.shape[-1]
                skvp = seq.kv_position
                while tokens_to_add:
                    page = seq.allocated_pages[skvp // self.generator.page_size]
                    assert page.ref_count == 1
                    tokens_page = min(tokens_to_add, self.generator.page_size - page.kv_position)
                    page.sequence[:, page.kv_position:page.kv_position + tokens_page] = ids[:, :tokens_page]
                    # if page.kv_position == 0:
                    #     page.prev_hash = None
                    page.kv_position += tokens_page
                    skvp += tokens_page
                    ids = ids[:, tokens_page:]
                    tokens_to_add -= tokens_page
                    page.can_revert = False
        return input_ids_list


    def receive_logits(
        self,
        logits: torch.Tensor,
    ):
        # Support single seq and CFG for now

        assert logits.shape[0] == len(self.sequences) == (2 if self.gen_settings.cfg_scale is not None else 1)
        assert self.is_prefill_done()
        assert all(seq.live for seq in self.sequences)

        # Start filters

        if self.new_tokens == 0:
            for f in self.filters:
                f.background_drop()
                f.begin("")

        # Sample

        if self.checkpoint and self.checkpoint["offset"] == 0:
            blocked_tokens = self.checkpoint["explored_tokens"]
        else:
            blocked_tokens = None

        if self.new_tokens < self.min_new_tokens:
            if blocked_tokens:
                blocked_tokens += self.stop_tokens_list
            else:
                blocked_tokens = self.stop_tokens_list

        next_token, next_k_tokens, next_k_probs, next_prob, filter_eos = \
        ExLlamaV2Sampler.sample(
            logits,
            self.gen_settings,
            self.sequences[0].sequence_ids.torch(),
            self.rng.random(),
            self.generator.tokenizer,
            self.prefix_token if self.new_tokens == -1 else None,
            self.return_top_tokens,
            blocked_tokens = blocked_tokens,
            filters = self.filters if self.new_tokens >= 0 else None,
            filter_prefer_eos = self.filter_prefer_eos,
            # sync = True
        )

        return next_token, next_k_tokens, next_k_probs, next_prob, filter_eos


    def receive_sample(
            self,
            logits: torch.Tensor | None,
            next_token: torch.Tensor | None,
            next_k_tokens: torch.Tensor | None,
            next_k_probs: torch.Tensor | None,
            next_prob: torch.Tensor | None,
            filter_eos: bool | None,
            results: list,
            first_sample_in_sd_batch: bool = True
    ):
        page_size = self.generator.page_size

        # Feed filters

        if self.new_tokens >= 0:
            all_mask = True
            for f in self.filters:
                f.feed(next_token)
                if not f.can_mask_logits() or not f.use_background_worker():
                    all_mask = False
            if first_sample_in_sd_batch and self.generator.filter_queue is not None:
                if all_mask:
                    # Using logit mask(s)
                    for f in self.filters:
                        self.generator.filter_queue.append((f, True))
                else:
                    # Using allowed token list(s)
                    for f in self.filters:
                        if f.use_background_worker():
                            self.generator.filter_queue.append((f, False))

        # Accept token

        self.new_tokens += 1

        for seq in self.sequences:

            # Accept new token

            seq.sequence_ids.append(next_token)

            page_before = seq.kv_position // page_size
            seq.kv_position += 1
            pos = seq.kv_position
            if self.checkpoint:
                pos -= self.checkpoint["offset"]
            page_after = pos // page_size

            # Hash completed page

            if page_after > page_before:
                assert page_after == page_before + 1

                page = seq.allocated_pages[page_before]

                if page_before > 0:
                    last_page = seq.allocated_pages[page_before - 1]
                    last_hash = last_page.phash
                else:
                    last_hash = None

                page_ids = seq.sequence_ids.torch_slice(page_before * page_size, page_after * page_size)
                # assert page.sequence.shape[-1] == self.generator.page_size
                # assert torch.all(page_ids == page.sequence)
                # assert page_ids.shape[-1] == self.generator.page_size
                new_hash = _tensor_hash_checksum(page_ids, last_hash)

                # If another referenced page has the same hash, switch to referencing that instead

                if new_hash in self.generator.referenced_pages:
                    new_serial = page.access_serial
                    page.sub_ref()
                    page = self.generator.referenced_pages[new_hash]
                    assert page.kv_position == page_size
                    seq.allocated_pages[page_before] = page
                    seq.block_index_tensor = None
                    page.add_ref(new_serial)

                else:

                    # If an unreferenced page has the same hash, clear that page

                    if new_hash in self.generator.unreferenced_pages:
                        up = self.generator.unreferenced_pages[new_hash]
                        up.clear()

                    # Update the hash

                    page.update_hash(new_hash)

                # if page_after >= len(seq.allocated_pages):
                #     pass

                page = seq.allocated_pages[page_after]
                page.prev_hash = new_hash
                page.can_revert = False

        # Stream output

        def emit(
            results: list,
            emit_eos: bool = False,
            eos_reason: str = None,
            emit_held = False,
            suppressed_text = None,
            suppressed_tokens = None,
            stop_token: int = None,
            stop_string: str = None,
            rem_held_text: str = None
        ):
            r = {
                "job": self,
                "stage": "streaming",
                "eos": emit_eos,
                "serial": self.serial_number,
            }

            if eos_reason is not None:
                r.update({ "eos_reason": eos_reason })
                if eos_reason == "stop_token":
                    id_to_piece = self.generator.tokenizer.get_id_to_piece_list(True)
                    r.update({
                        "eos_triggering_token_id": stop_token,
                        "eos_triggering_token_str": id_to_piece[stop_token]
                    })
                    pass
                if eos_reason == "stop_string":
                    r.update({ "eos_triggering_string": stop_string })

            if emit_held:
                if self.held_text != "":
                    self.full_completion += self.held_text
                    r.update({ "text": self.held_text })
                    self.held_text = ""
                if self.held_tokens:
                    r.update({ "token_ids": self.held_tokens.torch().clone() })
                    self.held_tokens.clear()
                if self.held_probs:
                    r.update({ "token_probs": self.held_probs.torch().clone() })
                    self.held_probs.clear()
                if self.held_k_tokens:
                    r.update({ "top_k_tokens": self.held_k_tokens.torch().clone() })
                    r.update({ "top_k_probs": self.held_k_probs.torch().clone() })
                    self.held_k_tokens.clear()
                    self.held_k_probs.clear()
                if self.held_logits:
                    r.update({ "logits": self.held_logits.torch().clone() })
                    self.held_logits.clear()

            if suppressed_text:
                r.update({ "suppressed_text": suppressed_text })
                r.update({ "suppressed_tokens": suppressed_tokens.torch() })

            if emit_eos:
                self.is_finished = True
                self.time_last_token = time.time()
                r.update({
                    "full_completion": self.full_completion,
                    "new_tokens": self.new_tokens,
                    "prompt_tokens": len(self.sequences[0].input_ids),
                    "time_enqueued": self.time_first_prefill - self.time_enqueue,
                    "time_prefill": self.time_first_token - self.time_first_prefill,
                    "time_generate": self.time_last_token - self.time_first_token,
                    "cached_pages": self.cached_pages // len(self.sequences),
                    "cached_tokens": (self.cached_pages * page_size + self.cached_tokens) // len(self.sequences),
                })
                if self.generator.draft_model or self.generator.use_ngram_draft:
                    r.update({
                        "accepted_draft_tokens": self.accepted_draft_tokens,
                        "rejected_draft_tokens": self.rejected_draft_tokens
                    })
                if eos_reason == "stop_string":
                    self.held_text = rem_held_text
                rh = {}
                if self.held_text:
                    rh.update({ "text": self.held_text })
                if self.held_tokens:
                    rh.update({ "token_ids": self.held_tokens.torch().clone() })
                if self.held_probs:
                    rh.update({ "token_probs": self.held_probs.torch().clone() })
                if self.held_k_tokens:
                    rh.update({ "top_k_tokens": self.held_k_tokens.torch().clone() })
                    rh.update({ "top_k_probs": self.held_k_probs.torch().clone() })
                if self.held_logits:
                    rh.update({ "logits": self.held_logits.torch().clone() })
                if rh:
                    r.update({ "held": rh })

            if self.identifier is not None:
                r.update({ "identifier": self.identifier })

            results.append(r)
            return emit_eos, next_token

        # Decode and buffer output

        id_to_piece = self.generator.tokenizer.get_id_to_piece_list(self.decode_special_tokens)
        new_text = id_to_piece[next_token.item()]

        if self.new_tokens == 0:
            unhealed = id_to_piece[self.prefix_token[0].item()]
            new_text = new_text[len(unhealed):]

        self.held_text += new_text
        self.held_tokens.append(next_token)
        if self.return_probs:
            self.held_probs.append(next_prob)
        if self.return_top_tokens > 0:
            self.held_k_tokens.append(next_k_tokens)
            self.held_k_probs.append(next_k_probs)
        if self.return_logits:
            self.held_logits.append(logits[:1, :, :])

        # End on stop tokens

        if next_token.item() in self.stop_tokens:
            return emit(results, emit_eos = True, eos_reason = "stop_token", stop_token = next_token.item())

        # Stop if we reach max_new_tokens
        # TODO: Auto-extend option

        if self.new_tokens >= self.max_new_tokens - self.generator.num_draft_tokens:
            return emit(results, emit_eos = True, emit_held = True, eos_reason = "max_new_tokens")

        # End now if newly added token ends a filter

        if filter_eos:
            return emit(results, emit_eos = True, emit_held = True, eos_reason = "end_filter")

        # Hold text if it contains an incomplete character

        if 1 <= self.held_text.count("�") < 5:
            test_decode = self.generator.tokenizer.decode(
                self.held_tokens.torch(),
                decode_special_tokens = self.decode_special_tokens
            )[0]
            if not "�" in test_decode:
                self.held_text = test_decode
            else:
                return emit(results)

        # Hold text as long as it contains part of a banned string

        def unset_checkpoint():
            self.checkpoint = None

        def set_checkpoint():
            if self.checkpoint is None:
                self.checkpoint = {
                    "offset": 1,
                    "held_text": self.held_text[:-len(new_text)],
                    "held_tokens": self.held_tokens.clone(1),
                    "held_probs": self.held_probs.clone(1),
                    "held_k_tokens": self.held_k_tokens.clone(1),
                    "held_k_probs": self.held_k_probs.clone(1),
                    "held_logits": self.held_logits.clone(1),
                    "explored_tokens": [next_token.item()],
                }
            else:
                self.checkpoint["offset"] += 1
                if self.checkpoint["offset"] == 1:
                    self.checkpoint["explored_tokens"].append(next_token.item())

        def rewind_checkpoint():
            assert self.checkpoint is not None
            offset = self.checkpoint["offset"]
            self.new_tokens -= offset
            for seq in self.sequences:
                p_page = seq.kv_position // self.generator.page_size
                seq.kv_position -= offset
                seq.sequence_ids.truncate(len(seq.sequence_ids) - offset)
                n_page = seq.kv_position // self.generator.page_size
                for pi in range(n_page, p_page + 1):
                    page = seq.allocated_pages[pi]
                    page.can_revert = False
                    if page.kv_position == self.generator.page_size:
                        page.update_hash(_randomhash())
                    if pi == n_page:
                        page.kv_position = seq.kv_position - pi * self.generator.page_size
                    else:
                        page.kv_position = 0
            off_tokens = self.held_tokens.slice(len(self.checkpoint["held_tokens"]), None)
            off_text = self.held_text[len(self.checkpoint["held_text"]):]
            self.held_text = self.checkpoint["held_text"]
            self.held_token = self.checkpoint["held_tokens"]
            self.held_probs = self.checkpoint["held_probs"]
            self.held_k_tokens = self.checkpoint["held_k_tokens"]
            self.held_k_probs = self.checkpoint["held_k_probs"]
            self.held_logits = self.checkpoint["held_logits"]
            self.checkpoint["offset"] = 0
            return off_tokens, off_text

        if self.banned_strings_utf32_offsets is not None and self.new_tokens > 0:
            match = ext_c.partial_strings_match(
                np.frombuffer(self.held_text.lower().encode("utf-32-le"), dtype = np.uint8),
                self.banned_strings_utf32_offsets,
                self.banned_strings_utf32_buffer
            )
            if match >= 0:
                set_checkpoint()
                offending_tokens, offending_text = rewind_checkpoint()
                return emit(results, emit_held = True, suppressed_text = offending_text, suppressed_tokens = offending_tokens)
            elif match == -2:
                set_checkpoint()
                return emit(results)
            else:
                unset_checkpoint()

        # End on stop strings

        if self.stop_strings_utf32_offsets is not None:
            match = ext_c.partial_strings_match(
                np.frombuffer(self.held_text.encode("utf-32-le"), dtype = np.uint8),
                self.stop_strings_utf32_offsets,
                self.stop_strings_utf32_buffer
            )
            if match >= 0:
                held = self.held_text[match:]
                self.held_text = self.held_text[:match]
                for s in self.stop_strings:
                    if held.startswith(s):
                        return emit(
                            results,
                            emit_eos = True,
                            emit_held = True,
                            eos_reason = "stop_string",
                            stop_string = s,
                            rem_held_text = held
                        )
                assert False, "Detected stop string but couldn't identify it (logic error)"
            if match == -2:
                return emit(results)

        # Stream output

        return emit(results, emit_held = True)


    def prepare_for_queue(self, generator, serial_number: int):

        self.serial_number = serial_number
        self.generator = generator
        self.skips = 0
        page_size = self.generator.page_size

        # Hash full pages of input IDs

        all_unique_hashes = set()
        all_unique_pages = 0

        for seq in self.sequences:

            seq.page_hashes = []

            max_len = len(seq.sequence_ids) + self.max_new_tokens
            if self.prefix_token:
                max_len += 1
            context_pages = (len(seq.sequence_ids) - 1) // page_size
            total_pages = (max_len + page_size - 1) // page_size

            r_hash = None
            for i in range(context_pages):
                page_ids = seq.sequence_ids.torch_slice(i * page_size, (i + 1) * page_size)
                assert page_ids.shape[-1] == self.generator.page_size
                r_hash = _tensor_hash_checksum(page_ids, r_hash)
                seq.page_hashes.append(r_hash)
                all_unique_hashes.add(r_hash)

            seq.new_unique_pages = total_pages - context_pages
            all_unique_pages += seq.new_unique_pages

            # seq.context_pages = context_pages
            # seq.total_pages = total_pages

        self.all_unique_hashes = list(all_unique_hashes)

        # Make sure the request can potentially fit

        total_pages = len(self.all_unique_hashes) + seq.new_unique_pages
        assert total_pages <= self.generator.max_pages, \
            f"Job requires {total_pages} pages (only {self.generator.max_pages} available) and cannot " + \
            f"be enqueued. Total cache allocated is {self.generator.max_pages} * {page_size} = " + \
            f"{self.generator.max_total_tokens} tokens"
        assert len(self.sequences) <= self.generator.max_batch_size, \
            f"Job requires a minimum batch size of {len(self.sequences)}. Max supported batch size in" + \
            f"generator is {self.generator.max_batch_size}."

        # Initial conditions

        self.held_text = ""
        self.held_tokens = SeqTensor((1, 0), dtype = torch.long, seq_dim = -1)
        self.held_k_tokens = SeqTensor((1, 0, self.return_top_tokens), dtype = torch.long, seq_dim = 1)
        self.held_k_probs = SeqTensor((1, 0, self.return_top_tokens), dtype = torch.float, seq_dim = 1)
        self.held_probs = SeqTensor((1, 0), dtype = torch.float, seq_dim = -1)
        self.held_logits = SeqTensor((1, 0, self.generator.padded_vocab_size), dtype = torch.float, seq_dim = 1)

        self.full_completion = ""

        # Prepare MRoPE embeddings

        if self.embeddings and generator.model.config.arch.lm.mrope:
            ids = self.sequences[0].sequence_ids.torch()
            e, offset = mrope.gen_mrope_embed(
                generator.model.config,
                ids,
                self.embeddings,
                ids.shape[-1],  # + self.max_new_tokens
            )
            self.alt_rope_embed = {"cpu": e}
            self.alt_rope_offset = offset - ids.shape[-1]
        else:
            self.alt_rope_embed = {}
            self.alt_rope_offset = 0

    def current_new_pages_required(self):
        new_pages = 0
        for h in self.all_unique_hashes:
            if h not in self.generator.referenced_pages:
                new_pages += 1
        for s in self.sequences:
            new_pages += s.new_unique_pages
        return new_pages


    def prefill(self, results: list):

        page_size = self.generator.page_size

        if self.time_first_prefill is None:
            self.time_first_prefill = time.time()

        progress = 0
        for seq in self.sequences:
            if seq.prefill_complete:
                continue

            if self.generator.paged:

                prefill_start = seq.kv_position
                prefill_end = seq.kv_position + self.generator.max_chunk_size
                if self.generator.paged:
                    prefill_end = (prefill_end // page_size) * page_size
                prefill_end = min(prefill_end, len(seq.sequence_ids) - 1)

                p0 = prefill_start // page_size
                p1 = (prefill_end + page_size - 1) // page_size
                for local_idx in range(p0, p1):
                    page = seq.allocated_pages[local_idx]
                    if page.kv_position == page_size:
                        prefill_start = (local_idx + 1) * page_size
                        seq.kv_position = prefill_start
                        self.cached_pages += 1
                        page.can_revert = False
                    else:
                        break

                p0 = prefill_start // page_size
                for local_idx in range(p0, p1):
                    page = seq.allocated_pages[local_idx]
                    if page.kv_position == page_size:
                        prefill_end = local_idx * page_size
                        break

                if prefill_end <= prefill_start:
                    continue

                prefill_ids = seq.sequence_ids.torch_slice(prefill_start, prefill_end)

                # Special case for partial last page, check if there's a page anywhere in the cache that
                # partially matches, then copy keys/values from there

                p0 = prefill_start // page_size
                p1 = prefill_end // page_size
                if prefill_start == p0 * page_size:
                    prev_hash = None if p0 == 0 else seq.allocated_pages[p0 - 1].phash
                    best_match = 0
                    best_match_page = None
                    for page in self.generator.all_pages:
                        if page.prev_hash != prev_hash or page == seq.allocated_pages[p0]:
                            continue
                        # match = _count_match(page.sequence[:, :page.kv_position], prefill_ids)
                        match = ext_c.count_match(page.sequence, prefill_ids, page.kv_position)
                        if match > best_match:
                            best_match = match
                            best_match_page = page
                    if best_match_page and best_match > 1:
                        page = seq.allocated_pages[p0]
                        # print([sap.page_index for sap in seq.allocated_pages])
                        for c in [self.generator.cache] if not self.generator.draft_model else \
                            [self.generator.cache, self.generator.draft_cache]:
                            c.copy_states(
                                c,
                                best_match_page.page_index * page_size, best_match,
                                page.page_index * page_size, best_match,
                                0, 1,
                                0, 1,
                            )
                        page.prev_hash = best_match_page.prev_hash
                        page.sequence[:, :best_match].copy_(prefill_ids[:, :best_match])
                        prefill_ids = prefill_ids[:, best_match:]
                        prefill_start += best_match
                        seq.kv_position += best_match
                        page.kv_position = best_match
                        page.can_revert = False
                        self.cached_tokens += best_match
                        progress += best_match

            # In unpaged mode there is only one page to compare against

            else:
                prefill_start = seq.kv_position
                progress = prefill_start
                page = self.generator.all_pages[0]

                prefill_ids = seq.sequence_ids.torch()[:, prefill_start:-1]
                if prefill_start == 0:
                    match = ext_c.count_match(
                        page.sequence[:, prefill_start:],
                        prefill_ids,
                        min(page.kv_position_revert, len(seq.sequence_ids) - 1)
                    )
                    prefill_ids = prefill_ids[:, match:]
                    seq.kv_position = prefill_start + match
                    page.kv_position = prefill_start + match
                    prefill_start += match
                    self.cached_tokens += match
                    progress += match

                prefill_end = min(seq.kv_position + self.generator.max_chunk_size, len(seq.sequence_ids) - 1)
                if prefill_end <= prefill_start:
                    seq.prefill_complete = True
                    continue

                assert page.can_revert

                prefill_ids = prefill_ids[:, :self.generator.max_chunk_size]
                p0 = p1 = 0

            # Inference

            if prefill_end > prefill_start:

                attn_params = self.generator.get_paged_params(
                    1,
                    self.get_block_index(seq, prefill_end).unsqueeze(0),
                    torch.tensor([prefill_start], dtype = torch.int32),
                    prefill_ids.shape[-1],
                    alt_rope_embed = self.alt_rope_embed,
                )

                if self.generator.draft_model:
                    self.generator.draft_model.forward_chunk(
                        input_ids = prefill_ids,
                        preprocess_only = True,
                        attn_params = attn_params,
                        cache = self.generator.draft_cache,
                    )

                if not self.generator.paged:
                    self.generator.cache.current_seq_len = prefill_start

                self.generator.model.forward_chunk(
                    input_ids = prefill_ids,
                    preprocess_only = True,
                    attn_params = attn_params,
                    cache = self.generator.cache,
                    loras = self.generator.current_loras,
                    indexed_embeddings = self.embeddings
                )

                seq.kv_position = prefill_end

                if self.generator.paged:
                    p2 = min(p1 + 1, len(seq.allocated_pages))
                    for local_idx in range(p0, p2):
                        page = seq.allocated_pages[local_idx]
                        page.kv_position = min(max(prefill_end - local_idx * page_size, 0), page_size)
                        if local_idx == 0:
                            page.prev_hash = None
                        else:
                            page.prev_hash = seq.allocated_pages[local_idx - 1].phash
                        pf_a = max(local_idx * page_size, prefill_start)
                        pf_b = min(local_idx * page_size + page_size, prefill_end)
                        pfp_a = pf_a - local_idx * page_size
                        pfp_b = pf_b - local_idx * page_size
                        page.sequence[:, pfp_a:pfp_b].copy_(seq.sequence_ids.torch_slice(pf_a, pf_b))
                        page.can_revert = False
                else:
                    page = seq.allocated_pages[0]
                    page.kv_position = prefill_end
                    page.prev_hash = None
                    page.sequence[:, prefill_start:prefill_end].copy_(seq.sequence_ids.torch_slice(prefill_start, prefill_end))

                progress += prefill_end - prefill_start
                if progress >= len(seq.sequence_ids) - 1:
                    seq.prefill_complete = True
                    if not self.generator.paged:
                        page = seq.allocated_pages[0]
                        page.can_revert = False

        if progress:
            r = {
                "job": self,
                "stage": "prefill",
                "eos": False,
                "curr_progress": sum(seq.kv_position for seq in self.sequences),
                "max_progress": sum(len(seq.sequence_ids) - 1 for seq in self.sequences),
                "serial": self.serial_number,
            }
            if self.identifier is not None:
                r.update({"identifier": self.identifier})
            results.append(r)

            self.generator.validate_cache()


    def get_block_index(self, seq: Sequence, max_len) -> torch.Tensor:

        page_size = self.generator.page_size

        if seq.block_index_tensor is None:
            block_index = [page.page_index for page in seq.allocated_pages]
            seq.block_index_tensor = torch.tensor(block_index, dtype = torch.int32)

        num_blocks = (max_len + page_size - 1) // page_size
        return seq.block_index_tensor[:num_blocks]


    def allocate_pages(self):

        page_size = self.generator.page_size

        for seq in self.sequences:

            seq.allocated_pages = []
            available_pages = None

            # Allocate whole pages

            for h in seq.page_hashes:

                self.generator.access_serial += 1

                # Find matching referenced page

                rp = self.generator.referenced_pages.get(h)
                if rp:
                    rp.add_ref(self.generator.access_serial)
                    seq.allocated_pages.append(rp)

                # If possible, reuse an unreferenced page with matching hash

                else:
                    up = self.generator.unreferenced_pages.get(h)
                    if up:
                        up.add_ref(self.generator.access_serial)
                        seq.allocated_pages.append(up)

                    # No matching pages

                    else:

                        # Get list of unreferenced pages in order of oldest to newest

                        if available_pages is None:
                            available_pages = list(self.generator.unreferenced_pages.values())
                            available_pages.sort(key = lambda x: x.access_serial)
                            available_pages = deque(available_pages)
                        else:
                            while available_pages[0].ref_count:
                                available_pages.popleft()

                        # assert all((p.phash in self.generator.unreferenced_pages) for p in available_pages)
                        # assert all((p.phash not in self.generator.referenced_pages) for p in available_pages)
                        # assert all(p.ref_count == 0 for p in available_pages)

                        # Allocate oldest unreferenced page

                        np = available_pages.popleft()
                        np.add_ref_clear(self.generator.access_serial, h)
                        seq.allocated_pages.append(np)

            # Allocate unique pages

            for npi in range(seq.new_unique_pages):

                self.generator.access_serial += 1

                # Get list of unreferenced pages in order of oldest to newest

                if available_pages is None:
                    available_pages = list(self.generator.unreferenced_pages.values())
                    available_pages.sort(key = lambda x: x.access_serial)
                    available_pages = deque(available_pages)
                else:
                    while available_pages[0].ref_count:
                        available_pages.popleft()

                # assert all((p.phash in self.generator.unreferenced_pages) for p in available_pages)
                # assert all((p.phash not in self.generator.referenced_pages) for p in available_pages)
                # assert all(p.ref_count == 0 for p in available_pages)

                np = available_pages.popleft()
                np.add_ref_unique(self.generator.access_serial)
                seq.allocated_pages.append(np)

            # Advance cache over prefilled pages

            for page in seq.allocated_pages:
                if page.kv_position == page_size:
                    seq.kv_position += page_size
                    self.cached_pages += 1
                else:
                    break

            # Metrics

            self.total_pages += len(seq.allocated_pages)
            for page_a, page_b in pairwise(seq.allocated_pages):
                if page_b.page_index != page_a.page_index + 1:
                    self.non_sequential_pages += 1

        self.generator.validate_cache()


    def deallocate_pages(self):

        if not self.generator.paged:
            self.generator.all_pages[0].backup()

        for seq in self.sequences:
            if seq.allocated_pages is not None:
                for page in seq.allocated_pages:
                    page.sub_ref()
                seq.allocated_pages = []

        self.generator.validate_cache()
