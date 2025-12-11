from __future__ import annotations

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2CacheBase,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2Sampler
)
from exllamav2.generator.filters import ExLlamaV2Filter
import torch
import random
import threading
from exllamav2.generator.hooks import ExLlamaV2PostSamplingHook, ExLlamaV2PostSamplingResult
from exllamav2.embedding import EMBEDDING_INDEX
from exllamav2.util import cuda_sync_active

class ExLlamaV2BaseGenerator:

    # Internal state

    model: ExLlamaV2
    cache: ExLlamaV2CacheBase
    tokenizer: ExLlamaV2Tokenizer

    sequence_ids: torch.Tensor | None

    abort_event: threading.Event | None


    def __init__(self,
                 model: ExLlamaV2,
                 cache: ExLlamaV2CacheBase,
                 tokenizer: ExLlamaV2Tokenizer):

        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer
        self.sequence_ids = None
        self.abort_event = None

    # For testing purposes, run a forward pass to make sure CUDA is fully initialized

    def warmup(self):

        input_ids = torch.zeros((1, 2), dtype = torch.long)
        self.model.forward(input_ids, cache = None, input_mask = None, preprocess_only = True)
        cuda_sync_active()


    def full(self):

        return self.sequence_ids.shape[-1] >= self.model.config.max_seq_len


    def generate_simple(
        self,
        prompt: str or list,
        gen_settings: ExLlamaV2Sampler.Settings,
        num_tokens: int,
        seed: int or None = None,
        token_healing: bool = False,
        encode_special_tokens: bool = False,
        decode_special_tokens: bool = False,
        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] | None = None,
        stop_token: int or None = -1,
        add_bos: bool = False,
        abort_event: threading.Event | None = None,
        input_embeddings: torch.Tensor | None = None,
        completion_only: bool = False,
        filters: list[ExLlamaV2Filter] | None = None,
        filter_prefer_eos: bool = False,
    ):

        """
        Generate one or more completions.

        :param prompt:
            String or list of strings. If this argument is a list, its length determinse the batch size, and
            the output will be list of strings as well.

        :param gen_settings:
            ExLlamaV2Sampler.Settings

        :param num_tokens:
            Max number of tokens to generate.

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

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param stop_token:
            ID of the stop token. If this argument is None, no stop token will be considered. The default
            value is -1, which is interpreted as whatever the EOS token is defined to be in the tokenizer
            model.

        :param add_bos:
            Prepend the tokenizer's specified BOS token to the input.

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param input_embeddings:
            Tensor of shape (batch_size, n, hidden_size) added to the beginning of the prompt. Batching
            is not supported when passing input embeddings unless all prompts are the same. Prompt must
            contain the string `{{EMBED_HERE}}` to indicate where embeddings are to be inserted.

        :param completion_only:
            Only return completion. If False, returned string will include the input prompt.

        :param filters:
            List of ExLlamaV2Filters to apply during generation.

        :param filter_prefer_eos:
            If True, always sample the tokenizer's defined EOS token as soon as it's allowed by the filters

        :return:
            Completion(s) (str or list[str] depending on the type of the input prompt argument)
        """


        self.abort_event = abort_event
        if self.abort_event: self.abort_event.clear()

        # Filters

        if filters is None: filters = []

        # Default stop token

        if stop_token == -1: stop_token = self.tokenizer.eos_token_id

        # Accept LoRA or list of LoRAs

        if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]

        # Apply seed

        if seed is not None: random.seed(seed)

        # Tokenize input and produce padding mask if needed, inserting embeddings if provided

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        prompts_identical = batch_size == 1 or all(s == prompt[0] for s in prompt)

        if input_embeddings is not None:

            embed_marker = "{{EMBED_HERE}}"
            prompt_split = prompt.split(embed_marker)
            assert len(prompt_split) == 2, \
                f"Prompt must contain one instance of {embed_marker} when embeddings are provided"

            if batch_size > 1: assert prompts_identical, \
                "Batched generation with input embeddings requires all prompts to be identical."

            assert input_embeddings.shape[0] == batch_size, \
                "Input embeddings tensor does not match batch size of prompt."

            pre_ids, _ = self.tokenizer.encode(prompt_split[0].rstrip(" \t"),
                                               encode_special_tokens = encode_special_tokens,
                                               return_offsets = True,
                                               add_bos = add_bos)
            post_ids, _ = self.tokenizer.encode(prompt_split[1].lstrip(" \t"),
                                               encode_special_tokens = encode_special_tokens,
                                               return_offsets = True,
                                               add_bos = False)

            num_emb_tokens = input_embeddings.shape[1]
            image_ids = torch.arange(EMBEDDING_INDEX, EMBEDDING_INDEX + num_emb_tokens, dtype = torch.long).unsqueeze(0)
            ids = torch.cat((pre_ids, image_ids, post_ids), dim = -1)

            position_offsets = None

        else:
            ids, position_offsets = self.tokenizer.encode(prompt,
                                                          encode_special_tokens = encode_special_tokens,
                                                          return_offsets = True,
                                                          add_bos = add_bos)

            pre_ids = torch.empty(*ids.shape[:-1], 0)

            if prompts_identical:
                position_offsets = None

        # Truncate prompt if generation would cause cache overflow

        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]
        else: overflow = 0

        mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None

        first_token = max(-overflow, 0)

        # Completion only

        if completion_only:
            first_token = ids.shape[-1]

        # Prepare for healing

        unhealed_token = None
        if ids.shape[-1] < 2: token_healing = False
        if token_healing:
            unhealed_token = ids[:, -1:]
            ids = ids[:, :-1]

        # Process prompt and begin gen

        self._gen_begin_base(ids,
                             mask,
                             loras,
                             position_offsets = position_offsets,
                             input_embeddings = input_embeddings)

        if self.abort_event and self.abort_event.is_set():
            if isinstance(prompt, str): return ""
            else: return [""] * len(prompt)

        # Remove indexed embeddings from generator's sequence

        if input_embeddings is not None:
            self.sequence_ids[self.sequence_ids >= EMBEDDING_INDEX] = self.tokenizer.pad_token_id

        # Begin filters

        healed_token = []
        id_to_piece = self.tokenizer.get_id_to_piece_list()
        if unhealed_token is not None:
            unhealed_token_list = unhealed_token.flatten().tolist()
            heal = [id_to_piece[x] for x in unhealed_token_list]
        else:
            heal = None

        for f in filters: f.begin(heal)

        # Generate tokens

        batch_eos = [False] * batch_size

        for i in range(num_tokens):

            if self.abort_event and self.abort_event.is_set():
                break

            logits = self.model.forward(self.sequence_ids[:, -1:],
                                        self.cache,
                                        input_mask = mask,
                                        loras = loras,
                                        position_offsets = position_offsets,
                                        indexed_embeddings = input_embeddings).float().cpu()

            token, ptokens, pprobs, prob, eos = \
            ExLlamaV2Sampler.sample(
                logits,
                gen_settings,
                pre_ids,
                self.sequence_ids,
                random.random(),
                self.tokenizer,
                prefix_token = unhealed_token,
                filters = filters,
                filter_prefer_eos = filter_prefer_eos
            )

            if unhealed_token is not None:
                unhealed_token_copy = unhealed_token
                healed_token = token

            if stop_token is not None:
                for b in range(batch_size):
                    if token[b, 0].item() == stop_token:
                        batch_eos[b] = True
                        if all(batch_eos): eos = True
                    if batch_eos[b]:
                        token[b, 0] = self.tokenizer.pad_token_id

            # Post sampling hook

            if gen_settings.post_sampling_hooks:
                p = ExLlamaV2PostSamplingResult(
                    sampled_token = token,
                    sampled_prob = prob,
                    logits = logits,
                    candidate_tokens = None if ptokens.is_meta else ptokens,
                    candidate_probs = None if pprobs.is_meta else pprobs
                )
                for h in gen_settings.post_sampling_hooks:
                    h(p)
                token = p.sampled_token
                if p.feed_filters:
                    for f in filters: f.feed(token)
            else:
                for f in filters: f.feed(token)

            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

            unhealed_token = None
            if eos: break

        # Decode

        decode_ids = self.sequence_ids[:, first_token:]
        if input_embeddings is not None:
            decode_ids = torch.stack([decode_ids[i][decode_ids[i] != self.tokenizer.pad_token_id] for i in range(batch_size)])

        if len(healed_token) and completion_only:
            decode_ids = torch.cat([healed_token, decode_ids], dim = -1)

        text = self.tokenizer.decode(decode_ids, decode_special_tokens = decode_special_tokens)

        if len(healed_token) and completion_only:
            pre_text = self.tokenizer.decode(unhealed_token_copy, decode_special_tokens = decode_special_tokens)
            text = [t[len(p):] for t, p in zip(text, pre_text)]

        if isinstance(prompt, str):
            return text[0]
        else:
            return text


    def _gen_begin_base(self,
                        input_ids: torch.Tensor,
                        mask: torch.Tensor | None = None,
                        loras: ExLlamaV2Lora or list[ExLlamaV2Lora] or None = None,
                        position_offsets: torch.Tensor | None = None,
                        input_embeddings: torch.Tensor | None = None):

        self.cache.current_seq_len = 0
        self.sequence_ids = input_ids

        self.model.forward(input_ids[:, :-1],
                           self.cache,
                           input_mask = mask,
                           preprocess_only = True,
                           loras = loras,
                           position_offsets = position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = input_embeddings)
        if self.abort_event and self.abort_event.is_set():
            self.sequence_ids = self.sequence_ids[:, :self.cache.current_seq_len + 1]
