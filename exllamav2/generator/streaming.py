from ast import Tuple
from typing import Union, Tuple
from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
    ExLlamaV2Lora
)
from exllamav2.generator import (
    ExLlamaV2Sampler,
    ExLlamaV2BaseGenerator
)

import torch
import random

class ExLlamaV2StreamingGenerator(ExLlamaV2BaseGenerator):

    tail_decode_tokens: int = 2

    remaining_tokens: int = 0
    held_text: str = ""
    held_utf8_tokens: torch.tensor = None
    expect_utf8: int = 0
    held_tokens: torch.Tensor or None = None
    held_ptokens: torch.Tensor or None = None
    held_probs: torch.Tensor or None = None
    held_logits: torch.Tensor or None = None
    settings: ExLlamaV2Sampler.Settings = None
    stop_strings: set = set()
    stop_tokens: set = set()

    no_tokens: torch.Tensor = None
    no_ptokens: torch.Tensor = None
    no_probs: torch.Tensor = None
    no_logits: torch.Tensor = None

    heal_prefix_token = None
    heal_old_tail_len = None

    draft_model: ExLlamaV2 or None = None
    draft_cache: ExLlamaV2Cache or None = None

    future_logits: torch.tensor or None = None
    future_tokens: torch.tensor or None = None
    num_speculative_tokens: int
    speculative_prob_threshold: float = 0.25
    total_draft_tokens: int = 0
    total_tokens: int = 0
    accepted_draft_tokens: int = 0

    return_probabilities: bool = False      # Return final sampling probabilities, one per token
    return_probabilities_k: int = 1        # Number of probabilities to return per token
    return_logits: bool = False             # Return raw logits prior to softmax, per token

    active_loras = None
    position_offsets = None
    input_mask = None

    queued_logits = None


    def __init__(self, model, cache, tokenizer, draft_model = None, draft_cache = None, num_speculative_tokens = 5):
        super().__init__(model, cache, tokenizer)

        self.stop_strings = set()
        self.stop_tokens = {tokenizer.eos_token_id,}

        if draft_model:
            self.draft_model = draft_model
            self.num_speculative_tokens = num_speculative_tokens
            if draft_cache:
                self.draft_cache = draft_cache
            else:
                self.draft_cache = ExLlamaV2Cache(draft_model,
                                                  batch_size = cache.batch_size,
                                                  max_seq_len = cache.max_seq_len)


    def set_stop_conditions(self, stop_conditions):

        assert isinstance(stop_conditions, (list, tuple, set))

        self.stop_strings = set()
        self.stop_tokens = set()
        for t in stop_conditions:
            if isinstance(t, int): self.stop_tokens.add(t)
            elif isinstance(t, str): self.stop_strings.add(t)
            else: raise ValueError("Unsupported type in stop_conditions")


    def begin_stream(self, input_ids: torch.Tensor, gen_settings: ExLlamaV2Sampler.Settings, token_healing = False, loras = None, input_mask = None, position_offsets = None):

        self.no_logits = torch.empty((0, ((self.model.config.vocab_size + 31) // 32) * 32), dtype = torch.float)
        self.no_tokens = torch.empty((1, 0), dtype=torch.long)
        if self.return_probabilities_k == 1:
            self.no_ptokens = torch.empty((1, 0), dtype = torch.long)
            self.no_probs = torch.empty((1, 0), dtype = torch.float)
        else:
            self.no_ptokens = torch.empty((1, 0, self.return_probabilities_k), dtype = torch.long)
            self.no_probs = torch.empty((1, 0, self.return_probabilities_k), dtype = torch.float)

        assert input_ids.shape[0] <= 2, "Streaming generator does not support batch size > 1"
        if input_ids.shape[0] == 2:
            assert gen_settings.cfg_scale is not None, "No CFG scale set"

        self.position_offsets = position_offsets
        self.input_mask = input_mask

        # Accept LoRA or list of LoRAs
        if loras is not None and isinstance(loras, ExLlamaV2Lora): loras = [loras]
        self.active_loras = loras

        self.held_text = ""
        self.held_utf8_tokens = self.no_tokens
        self.expect_utf8 = 0
        self.held_tokens = self.no_tokens
        self.held_ptokens = self.no_ptokens
        self.held_probs = self.no_probs
        self.held_logits = self.no_logits
        self.settings = gen_settings
        self._gen_begin_reuse(input_ids, gen_settings)

        self.queued_logits = []

        # Initialize token healing
        if token_healing and self.sequence_ids.shape[-1] >= max(2, self.tail_decode_tokens + 1):

            # Pop the last token, remembering tail len for first stream decode

            self.heal_old_tail_len = len(self.tokenizer.decode(self.sequence_ids[:, -(self.tail_decode_tokens + 1):])[0])
            self.heal_prefix_token = self.sequence_ids[:, -1:]
            self.sequence_ids = self.sequence_ids[:, :-1]
            self.cache.current_seq_len -= 1

            # Start filters

            self.settings.begin_filters(self.tokenizer.get_id_to_piece_list()[self.heal_prefix_token])
        else:
            self.settings.begin_filters()


    def stream(self) -> Union[Tuple[str, bool, torch.Tensor],
                              Tuple[str, bool, torch.Tensor, torch.Tensor],
                              Tuple[str, bool, torch.Tensor, torch.Tensor, torch.Tensor],
                              Tuple[str, bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:

        chunk, eos, chunk_token_ids, probs, ptokens, logits = self._stream()
        ret = [chunk, eos, chunk_token_ids]

        if self.return_probabilities:
            ret.append(probs)
            if self.return_probabilities_k > 1:
                ret.append(ptokens)
        
        if self.return_logits:
            ret.append(logits)
        
        return tuple(ret)


    # Get the next chunk of text in the stream. Returns eos if stop condition has been met but does not count tokens

    def _stream(self) -> (str, bool, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        if self.heal_old_tail_len is not None:
            old_tail_len = self.heal_old_tail_len
            self.heal_old_tail_len = None
        else:
            old_tail_len = len(self.tokenizer.decode(self.sequence_ids[:1, -self.tail_decode_tokens:])[0])

        # Generate a single token and append to the sequence

        next_token, next_ptokens, next_prob, eos, next_logits = self._gen_single_token(self.settings, prefix_token = self.heal_prefix_token)
        self.heal_prefix_token = None

        # End immediately if it was a stop token

        if next_token.item() in self.stop_tokens:
            return self.held_text, True, self.no_tokens, self.no_probs, self.no_ptokens, self.no_logits

        # Decode the tail end of the sequence with the added token to get (actual) characters added

        new_tail = self.tokenizer.decode(self.sequence_ids[:1, -(self.tail_decode_tokens + 1):])[0]
        new_text = new_tail[old_tail_len:]

        next_token, new_text = self._catch_utf8(next_token, new_text)

        self.held_text += new_text
        self.held_tokens = torch.cat([self.held_tokens, next_token], dim = -1)
        if self.return_probabilities:
            self.held_probs = torch.cat([self.held_probs, next_prob], dim = 1)
            if self.return_probabilities_k > 1:
                self.held_ptokens = torch.cat([self.held_ptokens, next_ptokens], dim = 1)
        if self.return_logits:
            self.held_logits = torch.cat([self.held_logits, next_logits], dim = 0)

        # Return now if newly added token ends a filter

        if eos: return self.held_text, True, self.held_tokens, self.held_probs, self.held_ptokens, self.held_logits

        # Hold text as long as it contains part of a stop string

        partial_ss = False
        for ss in self.stop_strings:

            # Check if held_text fully contains stop string

            position = self.held_text.find(ss)
            if position != -1:
                return self.held_text[:position], True, self.no_tokens, self.no_probs, self.no_ptokens, self.no_logits

            # Check for overlap between end of held_text and start of stop string

            overlap = 0
            for j in range(1, min(len(self.held_text), len(ss)) + 1):
                if self.held_text[-j:] == ss[:j]: overlap = j
            if overlap > 0: partial_ss = True

        # If holding text because of a partial stop condition, return nothing but also EOS = False

        if partial_ss:
            return "", False, self.no_tokens, self.no_probs, self.no_ptokens, self.no_logits

        # No stop condition, so return whatever is being held

        stream_text = self.held_text
        stream_tokens = self.held_tokens
        stream_probs = self.held_probs
        stream_ptokens = self.held_ptokens
        stream_logits = self.held_logits
        self.held_text = ""
        self.held_tokens = self.no_tokens
        self.held_probs = self.no_probs
        self.held_ptokens = self.no_ptokens
        self.held_logits = self.no_logits
        return stream_text, False, stream_tokens, stream_probs, stream_ptokens, stream_logits


    def _decode_utf8(self):

        if self.held_utf8_tokens.shape[-1] == 0: return self.no_tokens, ""

        try:
            id_to_ord = self.tokenizer.get_id_to_ord_list()
            b = [id_to_ord[x] for x in self.held_utf8_tokens[0].tolist()]
            c = bytes(b).decode('utf-8')
        except ValueError:
            id_to_piece = self.tokenizer.get_id_to_piece_list()
            c = "".join(id_to_piece[x] for x in self.held_utf8_tokens[0].tolist())
        except UnicodeDecodeError:
            c = "�"

        pre_t = self.held_utf8_tokens
        self.held_utf_tokens = self.no_tokens
        return pre_t, c


    def _catch_utf8(self, next_token, new_text):

        if self.expect_utf8 == 0:

            if new_text != "�": return next_token, new_text

            id_to_ord = self.tokenizer.get_id_to_ord_list()
            t = next_token[0, 0].item()
            b = id_to_ord[t]

            if 0 < b < 256:
                if b & 0b1100000 == 0b1000000: self.expect_utf8 = 2
                if b & 0b1110000 == 0b1100000: self.expect_utf8 = 3
                if b & 0b1111000 == 0b1110000: self.expect_utf8 = 4
                if b & 0b1111100 == 0b1111000: self.expect_utf8 = 5
            self.held_utf8_tokens = self.no_tokens
            if self.expect_utf8 == 0: return next_token, new_text
            new_text = ""

        if self.expect_utf8:

            if len(new_text) > 1:

                pre_t, pre_c = self._decode_utf8()
                next_token = torch.cat((pre_t, next_token), dim = -1)
                new_text = pre_c + new_text
                return next_token, new_text

            self.held_utf8_tokens = torch.cat((self.held_utf8_tokens, next_token), dim = -1)
            self.expect_utf8 -= 1
            if self.expect_utf8 == 0: return self._decode_utf8()
            return self.no_tokens, ""


    def _gen_begin(self, in_tokens, gen_settings):

        self.sequence_ids = in_tokens.clone()
        self.cache.current_seq_len = 0
        self.model.forward(self.sequence_ids[:, :-1], self.cache, preprocess_only = True, loras = self.active_loras, input_mask = self.input_mask, position_offsets = self.position_offsets)

        if self.draft_model is not None:
            self.draft_cache.current_seq_len = 0
            self.draft_model.forward(self.sequence_ids[:1, :-1], self.draft_cache, preprocess_only = True)
            self.future_logits = None
            self.future_tokens = None


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

        if self.draft_model is not None:
            self.future_logits = None
            self.future_tokens = None


    def _gen_feed_tokens(self, in_tokens, gen_settings):

        if self.sequence_ids is None:
            self._gen_begin(in_tokens, gen_settings)
            return

        start = self.cache.current_seq_len
        self.sequence_ids = torch.cat((self.sequence_ids, in_tokens), dim = 1)

        self.model.forward(self.sequence_ids[:, start : -1], self.cache, preprocess_only = True, loras = self.active_loras, input_mask = self.input_mask, position_offsets = self.position_offsets)

        if self.draft_model is not None:
            self.draft_model.forward(self.sequence_ids[:, start: -1], self.draft_cache, preprocess_only = True)
            self.future_logits = None
            self.future_tokens = None


    def append_logits(self, logits):

        assert self.draft_model is None
        assert logits.shape[0] == self.sequence_ids.shape[0]

        self.queued_logits.append(logits)

    def _gen_single_token(self, gen_settings, prefix_token = None):

        if self.draft_model is None:

            if self.queued_logits:
                logits = self.queued_logits.pop()
            else:
                logits = self.model.forward(self.sequence_ids[:, -1:], self.cache, loras = self.active_loras, input_mask = self.input_mask, position_offsets = self.position_offsets).float().cpu()
            token, ptokens, prob, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids[:1, :], random.random(), self.tokenizer, prefix_token, self.return_probabilities_k)

        else:

            token, ptokens, prob, eos, logits = self._gen_single_token_speculative(gen_settings, prefix_token)

        if self.sequence_ids.shape[0] > 1 and token.shape[0] == 1:
            self.sequence_ids = torch.cat([self.sequence_ids, token.repeat(self.sequence_ids.shape[0], 1)], dim = 1)
        else:
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim = 1)

        gen_settings.feed_filters(token)
        return token, ptokens, prob, eos, logits.flatten(1)


    def _gen_single_token_speculative(self, gen_settings, prefix_token = None):

        if self.future_tokens is None:

            # Generate draft

            draft_gen_settings = gen_settings.greedy_clone()
            draft_sequence_ids = self.sequence_ids[:1, :]
            num_drafted_tokens = 0

            for k in range(self.num_speculative_tokens):

                logits = self.draft_model.forward(draft_sequence_ids[:, -1:], self.draft_cache).float().cpu()
                token, _, prob, _ = ExLlamaV2Sampler.sample(logits, draft_gen_settings, draft_sequence_ids, random.random(), self.tokenizer, prefix_token if k == 0 else None)

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
            self.future_logits = self.model.forward(self.future_tokens, self.cache, loras = self.active_loras, input_mask = self.input_mask, position_offsets = self.position_offsets).float().cpu()

            # Rewind model cache

            self.cache.current_seq_len -= num_drafted_tokens + 1

        # Sample the first future logits

        logits = self.future_logits[:, :1, :]
        token, ptokens, prob, eos = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids[:1, :], random.random(), self.tokenizer, prefix_token, self.return_probabilities_k)
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

        return token, ptokens, prob, eos, logits


    def reset_sd_stats(self):

        self.total_tokens = 0
        self.total_draft_tokens = 0
        self.accepted_draft_tokens = 0


    def get_sd_stats(self):

        efficiency = self.accepted_draft_tokens / self.total_tokens
        accuracy = self.accepted_draft_tokens / self.total_draft_tokens
        return efficiency, accuracy, self.total_tokens, self.total_draft_tokens, self.accepted_draft_tokens
