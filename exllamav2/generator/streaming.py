from __future__ import annotations

from ast import Tuple
from typing import Union, Tuple
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2CacheBase,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora,
)
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2BaseGenerator
)
from exllamav2.generator.filters import ExLlamaV2Filter
from exllamav2.generator.ngram import NgramCache
import torch
import random
import threading
from exllamav2.generator.hooks import ExLlamaV2PostSamplingHook, ExLlamaV2PostSamplingResult
from exllamav2.embedding import EMBEDDING_INDEX
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
import numpy as np

class ExLlamaV2StreamingGenerator(ExLlamaV2BaseGenerator):

    # Constants

    tail_decode_tokens: int = 2
    max_fallback_tokens: int = 4

    no_tokens: torch.Tensor
    no_ptokens: torch.Tensor
    no_probs: torch.Tensor
    no_pprobs: torch.Tensor
    no_logits: torch.Tensor

    # Generation settings

    settings: ExLlamaV2Sampler.Settings

    return_probabilities: bool
    return_top_tokens: int
    return_logits: bool

    position_offsets: torch.Tensor | None
    input_mask: torch.Tensor | None
    indexed_embeddings: torch.Tensor | None

    # Stop conditions

    stop_strings: set
    stop_strings_utf32_buffer: np.array or None
    stop_strings_utf32_offsets: np.array or None
    stop_tokens: set
    remaining_tokens: int

    # Speculative decoding

    future_logits: torch.Tensor | None
    future_tokens: torch.Tensor | None

    total_draft_tokens: int
    total_tokens: int
    accepted_draft_tokens: int

    # Draft model

    draft_model: ExLlamaV2 | None
    draft_cache: ExLlamaV2CacheBase | None
    num_speculative_tokens: int
    speculative_prob_threshold: float

    # N-gram decoding

    speculative_ngram: bool
    speculative_ngram_min: int
    speculative_ngram_max: int
    speculative_ngram_max_inf: int
    speculative_ngram_threshold: int
    ngram: NgramCache | None
    ngram_preloaded: NgramCache | None

    # Output buffers

    held_text: str
    held_tokens: torch.Tensor | None
    held_ptokens: torch.Tensor | None
    held_probs: torch.Tensor | None
    held_pprobs: torch.Tensor | None
    held_logits: torch.Tensor | None

    # Token healing

    first_token: bool
    heal_next_token: bool

    # LoRAs

    active_loras: list[ExLlamaV2Lora]

    # Extra decoding options

    decode_special_tokens: bool

    # Banned strings

    banned_strings: list[str]
    banned_strings_utf32_buffer: np.array or None
    banned_strings_utf32_offsets: np.array or None
    ban_checkpoint: dict | None
    blocked_tokens: list[int]
    blocked_position: int
    current_blocked_tokens: list[int]
    reuse_logits: torch.Tensor | None

    # Filters

    filters: list[ExLlamaV2Filter] | None
    filter_prefer_eos: bool


    def __init__(self, model, cache, tokenizer, draft_model = None, draft_cache = None, num_speculative_tokens = 5):
        super().__init__(model, cache, tokenizer)

        # Stop conditions

        self.stop_strings = set()
        self.stop_strings_utf32_buffer = None
        self.stop_strings_utf32_offsets = None
        self.stop_tokens = {tokenizer.eos_token_id,}
        self.remaining_tokens = 0

        # Generation settings

        self.return_probabilities = False
        self.return_top_tokens = 0
        self.return_logits = False
        self.indexed_embeddings = None

        # Speculative decoding

        self.future_logits = None
        self.future_tokens = None

        self.total_draft_tokens = 0
        self.total_tokens = 0
        self.accepted_draft_tokens = 0

        # Draft model

        if draft_model:
            self.draft_model = draft_model
            self.num_speculative_tokens = num_speculative_tokens
            if draft_cache:
                self.draft_cache = draft_cache
            else:
                self.draft_cache = ExLlamaV2Cache(draft_model,
                                                  batch_size = cache.batch_size,
                                                  max_seq_len = cache.max_seq_len)
        else:
            self.draft_model = None
            self.draft_cache = None

        self.speculative_prob_threshold = 0.25

        # N-gram decoding

        self.speculative_ngram = False
        self.speculative_ngram_min = 1
        self.speculative_ngram_max = 5
        self.speculative_ngram_max_inf = 3
        self.speculative_ngram_threshold = 1
        self.ngram = None
        self.ngram_preloaded = None

        # Token healing

        self.active_loras = []

        # Banned strings

        self.banned_strings = []
        self.banned_strings_utf32_buffer = None
        self.banned_strings_utf32_offsets = None
        self.ban_checkpoint = None
        self.blocked_tokens = []
        self.blocked_position = 0
        self.current_blocked_tokens = []
        self.reuse_logits = None


    def set_stop_conditions(self,
                            stop_conditions: list | tuple | set):
        """
        :param stop_conditions:
            List of either token IDs or strings that will cause stream_ex to emit the EOS signal. String values do not
            have to match whole tokens and can span multiple tokens.

        Example:
            generator.set_stop_conditions(tokenizer.eos_token_id, "\nUser:", "###")
        """

        self.stop_strings = set()
        self.stop_tokens = set()
        for t in stop_conditions:
            if isinstance(t, int): self.stop_tokens.add(t)
            elif isinstance(t, str): self.stop_strings.add(t)
            else: raise ValueError("Unsupported type in stop_conditions")
        self.stop_strings_utf32_buffer, self.stop_strings_utf32_offsets = \
            self.strings_to_utf32(list(self.stop_strings))


    # Legacy function

    def begin_stream(self,
                     input_ids: torch.Tensor,
                     gen_settings: ExLlamaV2Sampler.Settings,
                     token_healing: bool = False,
                     loras: ExLlamaV2Lora | list[ExLlamaV2Lora] = None,
                     input_mask: torch.Tensor | None = None,
                     position_offsets: torch.Tensor | None = None,
                     abort_event: threading.Event = None,
                     **kwargs):
        """
        See ExLlamaV2StreamingGenerator.begin_stream_ex
        """

        self.begin_stream_ex(input_ids,
                             gen_settings,
                             token_healing,
                             loras,
                             input_mask,
                             position_offsets,
                             abort_event = abort_event)


    # Begin stream

    def begin_stream_ex(
        self,
        input_ids: torch.Tensor,
        gen_settings: ExLlamaV2Sampler.Settings,
        token_healing: bool = False,
        loras: ExLlamaV2Lora | list[ExLlamaV2Lora] = None,
        input_mask: torch.Tensor | None = None,
        position_offsets: torch.Tensor | None = None,
        return_probabilities: bool = False,
        return_top_tokens: int = 0,
        return_logits: bool = False,
        abort_event: threading.Event = None,
        input_embeddings: torch.Tensor | None = None,
        decode_special_tokens: bool = False,
        banned_strings: list[str] | None = None,
        filters: list[ExLlamaV2Filter] | None = None,
        filter_prefer_eos: bool = False,
        **kwargs
    ):
        """
        Resets the generator and starts a new completion of the supplied input_ids. Reuses the existing
        cache for any token IDs matching the previous sequence.

        :param input_ids:
            Input token ID sequence, as produced by ExLlamaV2Tokenizer. Streaming generator does not
            currently support batch size > 1 except when performing CFG, in which case batch size
            must be 2.

        :param gen_settings:
            Sampling settings

        :param token_healing:
            Apply token healing by regenerating the last token of the input sequence with prefix
            constraint.

        :param loras:
            (List of) ExLlamaV2Lora objects to apply during generation

        :param input_mask:
            Optional attention mask for the input

        :param position_offsets:
            Optional position offsets

        :param return_probabilities:
            Return tensor of post-sampler probabilities for each selected token

        :param return_top_tokens:
            Number of top candidate tokens to return for each selected token, along with their final
            probabilities.

        :param return_logits:
            Return the raw logits output by the model for each streamed token

        :param abort_event:
            Forwarded to the model during generation. Will abort prefill/context ingestion if triggered.

        :param input_embeddings:
            Tensor of shape (batch_size, n, hidden_size) added to the beginning of the prompt. Batching
            is not supported when passing input embeddings unless all prompts are the same. Prompt must
            contain the string `{{EMBED_HERE}}` to indicate where embeddings are to be inserted.

        :param decode_special_tokens:
            Also decode special tokens into output text stream

        :param filters:
            List of ExLlamaV2Filters to apply during generation.

        :param filter_prefer_eos:
            If True, always sample the tokenizer's defined EOS token as soon as it's allowed by the filters

        :param banned_strings:
            List of strings that the generator will refuse to output. As soon as a partial match happens,
            a checkpoint is saved that the generator can rewind to if need be. Subsequent tokens are then
            held until the full string is resolved (match or no match) and either emitted or discarded,
            accordingly. Strings are case-insensitive.
        """

        self.return_probabilities = return_probabilities
        self.return_top_tokens = return_top_tokens
        self.return_logits = return_logits

        self.abort_event = abort_event

        self.decode_special_tokens = decode_special_tokens

        assert input_ids.shape[0] <= 2, "Streaming generator does not support batch size > 1"
        if input_ids.shape[0] == 2:
            assert gen_settings.cfg_scale is not None, "No CFG scale set"
        self.input_ids = input_ids

        self.position_offsets = position_offsets
        self.input_mask = input_mask

        if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]
        self.active_loras = loras

        self.indexed_embeddings = input_embeddings

        # Decluttering

        self.no_logits = torch.empty((0, ((self.model.config.vocab_size + 31) // 32) * 32), dtype = torch.float)
        self.no_tokens = torch.empty((1, 0), dtype = torch.long)
        self.no_probs = torch.empty((1, 0), dtype = torch.float)
        self.no_ptokens = torch.empty((1, 0, self.return_top_tokens), dtype = torch.long)
        self.no_pprobs = torch.empty((1, 0, self.return_top_tokens), dtype = torch.float)

        # Initialize state

        self.held_text = ""

        self.held_tokens = self.no_tokens
        self.held_ptokens = self.no_ptokens
        self.held_probs = self.no_probs
        self.held_pprobs = self.no_pprobs
        self.held_logits = self.no_logits
        self.settings = gen_settings

        # Ingest prompt

        assert input_embeddings is None or self.draft_model is None, \
            "Can not use input embeddings with draft model"

        self._gen_begin_reuse(input_ids, gen_settings)
        self.heal_next_token = (token_healing and self.sequence_ids.shape[-1] >= 2)

        # Remove indexed embeddings from generator's sequence

        if input_embeddings is not None:
            self.sequence_ids[self.sequence_ids >= EMBEDDING_INDEX] = self.tokenizer.pad_token_id

        # Initialize n-gram cache

        if self.speculative_ngram:
            self.ngram = NgramCache(self.speculative_ngram_min,
                                    self.speculative_ngram_max,
                                    self.ngram_preloaded)
            self.ngram.update(self.sequence_ids[0].tolist())

        # Banned strings

        if banned_strings is None: banned_strings = []
        self.banned_strings = [s.lower() for s in banned_strings]
        self.banned_strings_utf32_buffer, self.banned_strings_utf32_offsets = \
            self.strings_to_utf32(self.banned_strings)

        self.ban_checkpoint = None
        self.blocked_tokens = []
        self.blocked_position = -1
        self.current_blocked_tokens = []
        self.reuse_logits = None

        # Filters

        self.filters = filters if filters is not None else []
        self.filter_prefer_eos = filter_prefer_eos


    # Convert list of strings to UTF32 format needed, to pass by reference to partial matching function

    def strings_to_utf32(self, strings: list[str]) -> (np.array, list[int]):

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


    # Get the next chunk of text in the stream

    def stream_ex(self, ban_tokens: list[int] | None = None, **kwargs):
        """
        Perform one streaming iteration, returning one chunk of text.

        :param ban_tokens:
            List of tokens to disallow for this iteration only.

        Returns a dict with the following entries:

            chunk:
                Decoded output text. May be an empty string if the generator holds text to resolve a
                stop condition.

            eos:
                Boolean EOS signal. True if one of the specified stop conditions has been met

            chunk_token_ids:
                Tensor of tokens corresponding to the output text, of shape (1, n). n may be zero if
                tokens are being held due to a partially met stop condition. Held tokens will be emitted
                on subsequent calls to stream_ex() when the stop condition is resolved.

                In the case of token healing, this does not include the last token of the input even
                though its value may have changed.

                If a string stop condition matches a partial token, the ID for the full token is included.

            probs: (if return_probabilities == True)
                Tensor of probabilities (1, n) corresponding to the chunk_token_ids

            top_tokens, top_probs: (if return_top_tokens > 0)
                Top-K token IDs with corresponding probabilities, shape (1, n, k). Sorted in descending
                order. Probabilities may sum to less than 1 if more than k tokens were candidates at the
                final sampling stage. If less than k tokens were candidates, the list will be padded with
                zero-probability tokens whose order is undefined.

            return_logits: (if return_logits == True)
                Raw output logits for the model, shape (1, n, vocab_size)
        """

        chunk, eos, chunk_token_ids, probs, ptokens, pprobs, logits, extra = self._stream(
            ban_tokens = ban_tokens
        )

        ret = { "chunk": chunk,
                "eos": eos,
                "chunk_token_ids": chunk_token_ids }

        if self.return_probabilities:
            ret["probs"] = probs

        if self.return_top_tokens > 0:
            ret["top_probs"] = pprobs
            ret["top_tokens"] = ptokens

        if self.return_logits:
            ret["logits"] = logits.unsqueeze(0)

        if extra:
            ret.update(extra)

        return ret


    # Legacy function

    def stream(self, **kwargs) -> Union[Tuple[str, bool, torch.Tensor],
                                  Tuple[str, bool, torch.Tensor, torch.Tensor],
                                  Tuple[str, bool, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Legacy functions that returns a tuple rather than a dict. See ExLlamaV2StreamingGenerator.stream_ex
        """

        assert self.return_top_tokens == 0, "Use stream_ex() to return top K probs"

        chunk, eos, chunk_token_ids, probs, _, _, logits, _ = self._stream()
        ret = [chunk, eos, chunk_token_ids]

        if self.return_probabilities:
            ret.append(probs)

        if self.return_logits:
            ret.append(logits)

        return tuple(ret)


    def _stream(self, ban_tokens: list[str] | None = None) -> (str, bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict | None):

        # Blocked/banned tokens

        self.current_blocked_tokens = [] if ban_tokens is None else ban_tokens
        if self.cache.current_seq_len == self.blocked_position:
            self.current_blocked_tokens += self.blocked_tokens

        # Token healing

        if self.heal_next_token:

            # Pop the last token

            old_tail = self.tokenizer.decode(self.sequence_ids[:, -self.tail_decode_tokens:],
                                             decode_special_tokens = self.decode_special_tokens)[0]
            last_token = self.sequence_ids[:, -1:]
            self.sequence_ids = self.sequence_ids[:, :-1]
            self.cache.current_seq_len -= 1

            # Start filters

            if self.first_token:

                for f in self.filters:
                    f.begin(self.tokenizer.get_id_to_piece_list(self.decode_special_tokens)[last_token])
                self.first_token = False

            # Regenerate the last token again, with prefix

            healed_token, _, _, _, eos, logits, dev_logits = self._gen_single_token(self.settings, prefix_token = last_token)
            new_tail = self.tokenizer.decode(self.sequence_ids[:, -self.tail_decode_tokens:],
                                             decode_special_tokens = self.decode_special_tokens)[0]
            self.held_text += new_tail[len(old_tail):]

            self.heal_next_token = False

            # In case we only needed the healed token

            if eos: return self.held_text, True, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits, None

        # Start filters when not healing

        else:

            if self.first_token:

                for f in self.filters: f.begin("")
                self.first_token = False

        # Generate a single token and append to the sequence

        next_token, next_ptokens, next_pprobs, next_prob, eos, next_logits, dev_logits = self._gen_single_token(self.settings)

        # End immediately if it was a stop token

        if next_token.item() in self.stop_tokens:
            return self.held_text, True, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits, None

        id_to_piece = self.tokenizer.get_id_to_piece_list(self.decode_special_tokens)
        new_text = id_to_piece[next_token]

        self.held_text += new_text
        self.held_tokens = torch.cat([self.held_tokens, next_token], dim = -1)
        if self.return_probabilities:
            self.held_probs = torch.cat([self.held_probs, next_prob], dim = 1)
        if self.return_top_tokens > 0:
            self.held_ptokens = torch.cat([self.held_ptokens, next_ptokens], dim = 1)
            self.held_pprobs = torch.cat([self.held_pprobs, next_pprobs], dim = 1)
        if self.return_logits:
            self.held_logits = torch.cat([self.held_logits, next_logits], dim = 0)

        # Return now if newly added token ends a filter

        if eos: return self.held_text, True, self.held_tokens, self.held_probs, self.held_ptokens, self.held_pprobs, self.held_logits, None

        # Hold text if it contains an incomplete character

        if 1 <= self.held_text.count("�") < 5:
            test_decode = self.tokenizer.decode(
                self.held_tokens,
                decode_special_tokens = self.decode_special_tokens
            )[0]
            if not "�" in test_decode:
                self.held_text = test_decode
            else:
                return "", False, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits, None

        # Hold text as long as it contains part of a banned string

        def set_checkpoint():
            self.ban_checkpoint = {
                "position": self.cache.current_seq_len - 1,
                "held_text": self.held_text[:-len(new_text)],
                "held_tokens": self.held_tokens[:, :-1],
                "held_probs": self.held_probs[:, :-1],
                "held_ptokens": self.held_ptokens[:, :-1, :],
                "held_pprobs": self.held_pprobs[:, :-1, :],
                "held_logits": self.held_logits[:-1, :],
                "offending_token": next_token,
                "next_logits": dev_logits
            }
            self.blocked_position = self.cache.current_seq_len - 1

        def rewind_checkpoint():
            cp = self.ban_checkpoint
            self.sequence_ids = self.sequence_ids[:, :cp["position"]+1]
            self.cache.current_seq_len = cp["position"]
            off_text = self.held_text[len(cp["held_text"]):]
            self.held_text = cp["held_text"]
            self.held_tokens = cp["held_tokens"]
            self.held_probs = cp["held_probs"]
            self.held_ptokens = cp["held_ptokens"]
            self.held_pprobs = cp["held_pprobs"]
            self.held_logits = cp["held_logits"]
            self.future_logits = None
            self.future_tokens = None
            self.ban_checkpoint = None
            self.reuse_logits = cp["next_logits"]
            return cp["offending_token"], off_text

        if self.banned_strings_utf32_offsets is not None:
            match = ext_c.partial_strings_match(
                np.frombuffer(self.held_text.lower().encode("utf-32-le"), dtype = np.uint8),
                self.banned_strings_utf32_offsets,
                self.banned_strings_utf32_buffer
            )
            if match >= 0:
                if self.ban_checkpoint is None: set_checkpoint()
                offending_token, offending_text = rewind_checkpoint()
                self.blocked_tokens.append(offending_token.item())
                extra_ret = { "suppressed": offending_text }
                return "", False, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits, extra_ret
            if match == -2:
                if self.ban_checkpoint is None: set_checkpoint()
                return "", False, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits, None

        # Check for stop strings and hold text as long as it contains part of a stop string

        if self.stop_strings_utf32_offsets is not None:
            match = ext_c.partial_strings_match(
                np.frombuffer(self.held_text.encode("utf-32-le"), dtype = np.uint8),
                self.stop_strings_utf32_offsets,
                self.stop_strings_utf32_buffer
            )
            if match >= 0:
                return self.held_text[:match], True, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits, None
            if match == -2:
                return "", False, self.no_tokens, self.no_probs, self.no_ptokens, self.no_pprobs, self.no_logits, None

        # No stop condition or banned string, so clear checkpoint and return whatever is being held

        self.ban_checkpoint = None
        self.blocked_tokens = []
        self.blocked_position = -1
        stream_text = self.held_text
        stream_tokens = self.held_tokens
        stream_probs = self.held_probs
        stream_ptokens = self.held_ptokens
        stream_pprobs = self.held_pprobs
        stream_logits = self.held_logits
        self.held_text = ""
        self.held_tokens = self.no_tokens
        self.held_probs = self.no_probs
        self.held_ptokens = self.no_ptokens
        self.held_pprobs = self.no_pprobs
        self.held_logits = self.no_logits
        return stream_text, False, stream_tokens, stream_probs, stream_ptokens, stream_pprobs, stream_logits, None


    # Helper for limiting the sequence length to the cache length. Necessary in case the cache prefill was
    # aborted since the incomplete cache might otherwise be reused

    def _truncate_seq_to_cache(self):
        cachelen = self.cache.current_seq_len
        if self.draft_cache: cachelen = min(cachelen, self.draft_cache.current_seq_len)
        self.sequence_ids = self.sequence_ids[:, :cachelen + 1]


    # Begin a generation (prefill/ingest)

    def _gen_begin(self, in_tokens, gen_settings):

        self.sequence_ids = in_tokens.clone()
        self.cache.current_seq_len = 0
        self.model.forward(self.sequence_ids[:, :-1],
                           self.cache,
                           preprocess_only = True,
                           loras = self.active_loras,
                           input_mask = self.input_mask,
                           position_offsets = self.position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = self.indexed_embeddings)
        if self.abort_event and self.abort_event.is_set():
            self._truncate_seq_to_cache()
            return

        if self.draft_model is not None:
            self.draft_cache.current_seq_len = 0
            self.draft_model.forward(self.sequence_ids[:1, :-1],
                                     self.draft_cache,
                                     input_mask = self.input_mask,
                                     position_offsets = self.position_offsets,
                                     preprocess_only = True,
                                     abort_event = self.abort_event)
            if self.abort_event and self.abort_event.is_set():
                self._truncate_seq_to_cache()
                return
            self.future_logits = None
            self.future_tokens = None

        if self.speculative_ngram:
            self.future_logits = None
            self.future_tokens = None

        self.first_token = True


    # Begin a generation (prefill/ingest) while reusing the K/V cache for any starting tokens that are
    # unchanged since the last generation

    def _gen_begin_reuse(self, in_tokens, gen_settings):

        if self.sequence_ids is None or self.cache.current_seq_len == 0:
            self._gen_begin(in_tokens, gen_settings)
            return

        reuse = 0
        while reuse < self.sequence_ids.shape[-1] and reuse < in_tokens.shape[-1] and self.sequence_ids[0, reuse] == in_tokens[0, reuse]:
            reuse += 1

        if reuse < 2:
            self._gen_begin(in_tokens, gen_settings)
            return

        self.cache.current_seq_len = reuse - 1
        if self.draft_model is not None:
            self.draft_cache.current_seq_len = reuse - 1
        self.sequence_ids = in_tokens[:, :reuse]

        if reuse < in_tokens.shape[-1]: self._gen_feed_tokens(in_tokens[:, reuse:], gen_settings)

        if self.speculative_ngram or self.draft_model is not None:
            self.future_logits = None
            self.future_tokens = None

        self.first_token = True


    # Ingest a number of tokens, appending to the current sequence

    def _gen_feed_tokens(self, in_tokens, gen_settings):

        if self.sequence_ids is None:
            self._gen_begin(in_tokens, gen_settings)
            return

        start = self.cache.current_seq_len
        self.sequence_ids = torch.cat((self.sequence_ids, in_tokens), dim = 1)

        self.model.forward(self.sequence_ids[:, start : -1],
                           self.cache,
                           preprocess_only = True,
                           loras = self.active_loras,
                           input_mask = self.input_mask,
                           position_offsets = self.position_offsets,
                           abort_event = self.abort_event,
                           indexed_embeddings = self.indexed_embeddings)
        if self.abort_event and self.abort_event.is_set():
            self._truncate_seq_to_cache()
            return

        if self.draft_model is not None:
            self.draft_model.forward(self.sequence_ids[:, start: -1],
                                     self.draft_cache,
                                     preprocess_only = True,
                                     input_mask = self.input_mask,
                                     position_offsets = self.position_offsets,
                                     abort_event = self.abort_event)
            if self.abort_event and self.abort_event.is_set():
                self._truncate_seq_to_cache()
                return
            self.future_logits = None
            self.future_tokens = None


    # Generate a single token and append to sequence

    def _gen_single_token(self, gen_settings, prefix_token = None):

        if self.speculative_ngram:

            token, ptokens, pprobs, prob, eos, logits = self._gen_single_token_ngram(gen_settings, prefix_token)
            dev_logits = None

        elif self.draft_model is None:

            if self.reuse_logits is not None:
                dev_logits = self.reuse_logits
                self.reuse_logits = None
                self.cache.current_seq_len += 1
                logits = dev_logits.float().cpu()
            else:
                dev_logits = self.model.forward(
                    self.sequence_ids[:, -1:],
                    self.cache,
                    loras = self.active_loras,
                    input_mask = self.input_mask,
                    position_offsets = self.position_offsets
                )
                logits = dev_logits.float().cpu()

            token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.sample(
                logits,
                gen_settings,
                self.input_ids,
                self.sequence_ids[:1, :],
                random.random(),
                self.tokenizer,
                prefix_token,
                self.return_top_tokens,
                blocked_tokens = self.current_blocked_tokens,
                filters = self.filters,
                filter_prefer_eos = self.filter_prefer_eos
            )

        else:

            token, ptokens, pprobs, prob, eos, logits = \
                self._gen_single_token_speculative(gen_settings, prefix_token)
            dev_logits = None

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
                for f in self.filters: f.feed(token)
        else:
            for f in self.filters: f.feed(token)

        # Accept token

        if self.sequence_ids.shape[0] > 1 and token.shape[0] == 1:
            self.sequence_ids = torch.cat([self.sequence_ids, token.repeat(self.sequence_ids.shape[0], 1)], dim = 1)
        else:
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

        return token, ptokens, pprobs, prob, eos, logits.flatten(1), dev_logits


    # Speculative decoding with draft model

    def _gen_single_token_speculative(self, gen_settings, prefix_token = None):

        if self.future_tokens is None:

            # Generate draft

            draft_gen_settings = gen_settings.greedy_clone()
            draft_sequence_ids = self.sequence_ids[:1, :]
            num_drafted_tokens = 0

            for k in range(self.num_speculative_tokens):

                logits = self.draft_model.forward(draft_sequence_ids[:, -1:],
                                                  self.draft_cache,
                                                  input_mask = self.input_mask,
                                                  position_offsets = self.position_offsets).float().cpu()
                token, _, _, prob, _ = ExLlamaV2Sampler.sample(
                    logits,
                    draft_gen_settings,
                    self.input_ids,
                    draft_sequence_ids,
                    random.random(),
                    self.tokenizer,
                    prefix_token if k == 0 else None
                )

                if prob < self.speculative_prob_threshold:
                    self.draft_cache.current_seq_len -= 1
                    break

                draft_sequence_ids = torch.cat((draft_sequence_ids, token), dim = 1)
                num_drafted_tokens += 1

            self.total_draft_tokens += num_drafted_tokens

            # Rewind draft cache

            self.draft_cache.current_seq_len -= num_drafted_tokens

            # Forward last sampled token plus draft through model

            if self.sequence_ids.shape[0] > 1:
                self.future_tokens = draft_sequence_ids[:, -1 - num_drafted_tokens:].repeat(self.sequence_ids.shape[0], 1)
            else:
                self.future_tokens = draft_sequence_ids[:, -1 - num_drafted_tokens:]
            self.future_logits = self.model.forward(self.future_tokens,
                                                    self.cache,
                                                    loras = self.active_loras,
                                                    input_mask = self.input_mask,
                                                    position_offsets = self.position_offsets).float().cpu()

            # Rewind model cache

            self.cache.current_seq_len -= num_drafted_tokens + 1

        # Sample the first future logits

        logits = self.future_logits[:, :1, :]
        token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.sample(
            logits,
            gen_settings,
            self.input_ids,
            self.sequence_ids[:1, :], random.random(),
            self.tokenizer,
            prefix_token,
            self.return_top_tokens,
            blocked_tokens = self.current_blocked_tokens,
            filters = self.filters,
            filter_prefer_eos = self.filter_prefer_eos
        )
        self.future_logits = self.future_logits[:, 1:, :]
        self.future_tokens = self.future_tokens[:, 1:]
        self.cache.current_seq_len += 1
        self.draft_cache.current_seq_len += 1

        # If sampled token doesn't match future token or no more future tokens

        if self.future_tokens.shape[-1] == 0 or self.future_tokens[0, 0] != token[0, 0]:
            self.future_tokens = None
            self.future_logits = None
        else:
            self.accepted_draft_tokens += 1
        self.total_tokens += 1

        return token, ptokens, pprobs, prob, eos, logits


    # Speculative decoding with n-gram predictions

    def _gen_single_token_ngram(self, gen_settings, prefix_token = None):

        if self.future_tokens is None:

            inf_ids = self.sequence_ids[0, -1:].tolist()
            pred_ids = self.sequence_ids[0, -self.speculative_ngram_max:].tolist()

            threshold = self.speculative_ngram_threshold
            while len(inf_ids) < self.speculative_ngram_max_inf:
                t = self.ngram.predict_next(pred_ids, threshold, self.ngram_preloaded)
                if t is None: break
                pred_ids = pred_ids[1:] + [t]
                inf_ids += [t]
                threshold += 1

            if len(inf_ids) + self.cache.current_seq_len > self.cache.max_seq_len:
                inf_ids = inf_ids[:-(len(inf_ids) + self.cache.current_seq_len - self.cache.max_seq_len)]

            self.future_tokens = torch.tensor([inf_ids], dtype = torch.long)
            self.future_logits = self.model.forward(self.future_tokens,
                                                    self.cache,
                                                    loras = self.active_loras,
                                                    input_mask = self.input_mask,
                                                    position_offsets = self.position_offsets)
            self.future_logits = self.future_logits.float().cpu()

            self.cache.current_seq_len -= len(inf_ids)
            self.total_draft_tokens += len(inf_ids) - 1

        # Sample the first future logits

        logits = self.future_logits[:, :1, :]
        token, ptokens, pprobs, prob, eos = ExLlamaV2Sampler.sample(
            logits,
            gen_settings,
            self.input_ids,
            self.sequence_ids[:1, :],
            random.random(),
            self.tokenizer,
            prefix_token,
            self.return_top_tokens,
            blocked_tokens = self.current_blocked_tokens,
            filters=self.filters,
            filter_prefer_eos=self.filter_prefer_eos
        )
        self.future_logits = self.future_logits[:, 1:, :]
        self.future_tokens = self.future_tokens[:, 1:]
        self.cache.current_seq_len += 1

        # Update predictor

        tail_ids = self.sequence_ids[0, -(self.speculative_ngram_max - 1):].tolist() + [token.item()]
        self.ngram.update_single(tail_ids)

        # If sampled token doesn't match future token or no more future tokens

        if self.future_tokens.shape[-1] == 0 or self.future_tokens[0, 0] != token[0, 0]:
            self.future_tokens = None
            self.future_logits = None
        else:
            self.accepted_draft_tokens += 1
        self.total_tokens += 1

        return token, ptokens, pprobs, prob, eos, logits


    # Some metrics

    def reset_sd_stats(self):

        self.total_tokens = 0
        self.total_draft_tokens = 0
        self.accepted_draft_tokens = 0


    def get_sd_stats(self):

        efficiency = self.accepted_draft_tokens / self.total_tokens if self.total_tokens else 0
        accuracy = self.accepted_draft_tokens / self.total_draft_tokens if self.total_draft_tokens else 0
        return efficiency, accuracy, self.total_tokens, self.total_draft_tokens, self.accepted_draft_tokens


    def ngram_preload(self,
                      input_ids: torch.Tensor):
        """
        Preload the n-gram cache with some tokenized example text. (Every call to begin_stream_ex creates
        a new dynamic cache based on the provided context and generated tokens.)
        """

        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids[0].tolist()

        self.ngram_preloaded = NgramCache(self.speculative_ngram_min, self.speculative_ngram_max, None)
        self.ngram_preloaded.update(input_ids)
