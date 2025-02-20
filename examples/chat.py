
import sys, os, time, math
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Cache_8bit,
    ExLlamaV2Cache_Q4,
    ExLlamaV2Cache_Q6,
    ExLlamaV2Cache_Q8,
    ExLlamaV2Cache_TP,
    ExLlamaV2Tokenizer,
    model_init,
)

import argparse
import torch

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

from chat_formatting import CodeBlockFormatter
from chat_prompts import prompt_formats
prompt_formats_list = list(prompt_formats.keys())

# Options
# (!!!) NOTE: These go on top of the engine arguments that can be found in `model_init.py` (!!!)

parser = argparse.ArgumentParser(description = "Simple Llama2 chat example for ExLlamaV2")
parser.add_argument("-dm", "--draft_model_dir", type = str, default = None, help = "Path to draft model directory")
parser.add_argument("-nds", "--no_draft_scale", action = "store_true", help = "If draft model has smaller context size than model, don't apply alpha (NTK) scaling to extend it")
parser.add_argument("-dn", "--draft_n_tokens", type = int, default = 5, help = "How many tokens to speculate ahead (defaults to 5)")

parser.add_argument("-modes", "--modes", action = "store_true", help = "List available modes and exit.")
parser.add_argument("-mode", "--mode", choices = prompt_formats_list, help = "Chat mode. Use llama for Llama 1/2 chat finetunes.")
parser.add_argument("-un", "--username", type = str, default = "User", help = "Username when using raw chat mode")
parser.add_argument("-bn", "--botname", type = str, default = "Chatbort", help = "Bot name when using raw chat mode")
parser.add_argument("-sp", "--system_prompt", type = str, help = "Use custom system prompt")
parser.add_argument("-nsp", "--no_system_prompt", action = "store_true", help = "Do not use any system prompt")

parser.add_argument("-temp", "--temperature", type = float, default = 0.95, help = "Sampler temperature, default = 0.95 (1 to disable)")
parser.add_argument("-smooth", "--smoothing_factor", type = float, default = 0.0, help = "Smoothing Factor, default = 0.0 (0 to disable")
parser.add_argument("-dyntemp", "--dynamic_temperature", type = str, help = "Dynamic temperature min,max,exponent, e.g. -dyntemp 0.2,1.5,1")

parser.add_argument("-topk", "--top_k", type = int, default = 50, help = "Sampler top-K, default = 50 (0 to disable)")
parser.add_argument("-topp", "--top_p", type = float, default = 0.8, help = "Sampler top-P, default = 0.8 (0 to disable)")
parser.add_argument("-topa", "--top_a", type = float, default = 0.0, help = "Sampler top-A, default = 0.0 (0 to disable)")
parser.add_argument("-skew", "--skew", type = float, default = 0.0, help = "Skew sampling, default = 0.0 (0 to disable)")
parser.add_argument("-typical", "--typical", type = float, default = 0.0, help = "Sampler typical threshold, default = 0.0 (0 to disable)")
parser.add_argument("-repp", "--repetition_penalty", type = float, default = 1.01, help = "Sampler repetition penalty, default = 1.01 (1 to disable)")
parser.add_argument("-freqpen", "--frequency_penalty", type = float, default = 0.0, help = "Sampler frequency penalty, default = 0.0 (0 to disable)")
parser.add_argument("-prespen", "--presence_penalty", type = float, default = 0.0, help = "Sampler presence penalty, default = 0.0 (0 to disable)")
parser.add_argument("-xtcp", "--xtc_probability", type = float, default = 0.0, help = "XTC sampling probability, default = 0.0 (0 to disable)")
parser.add_argument("-xtct", "--xtc_threshold", type = float, default = 0.1, help = "XTC sampling threshold, default = 0.1, ignored when xtc_probability is 0")
parser.add_argument("-drym", "--dry_multiplier", type = float, default = 0.0, help = "DRY multiplier, default = 0.0 (0 to disable)")
parser.add_argument("-drya", "--dry_allowed_length", type = int, default = 2, help = "DRY allowed length, default = 2, ignored when dry_multiplier is 0")
parser.add_argument("-dryb", "--dry_base", type = float, default = 1.75, help = "DRY base value, default = 1.75, ignored when dry_multiplier is 0")
parser.add_argument("-dryr", "--dry_range", type = int, default = 0, help = "DRY range, default = 0 (0 for unlimited range)")

parser.add_argument("-maxr", "--max_response_tokens", type = int, default = 1000, help = "Max tokens per response, default = 1000")
parser.add_argument("-resc", "--response_chunk", type = int, default = 250, help = "Space to reserve in context for reply, default = 250")
parser.add_argument("-ncf", "--no_code_formatting", action = "store_true", help = "Disable code formatting/syntax highlighting")

parser.add_argument("-c8", "--cache_8bit", action = "store_true", help = "Use 8-bit (FP8) cache")
parser.add_argument("-cq4", "--cache_q4", action = "store_true", help = "Use Q4 cache")
parser.add_argument("-cq6", "--cache_q6", action = "store_true", help = "Use Q6 cache")
parser.add_argument("-cq8", "--cache_q8", action = "store_true", help = "Use Q8 cache")

parser.add_argument("-ngram", "--ngram_decoding", action = "store_true", help = "Use n-gram speculative decoding")

parser.add_argument("-mli", "--mli", action = "store_true", help = "Enable multi line input")
parser.add_argument("-pt", "--print_timings", action = "store_true", help = "Output timings/stats after each prompt")
parser.add_argument("-amnesia", "--amnesia", action = "store_true", help = "Forget context after every response")

# Arrrgs

model_init.add_args(parser)
args = parser.parse_args()

# Prompt templates/modes

if args.modes:
    print(" -- Available formats:")
    for k, v in prompt_formats.items():
        print(f" --   {k:12} : {v().description}")
    sys.exit()

username = args.username
botname = args.botname
system_prompt = args.system_prompt
if args.no_system_prompt:
    system_prompt = ""

if args.mode is None:
    print(" ## Error: No mode specified.")
    sys.exit()

prompt_format = prompt_formats[args.mode]()
prompt_format.botname = botname
prompt_format.username = username
if system_prompt is None: system_prompt = prompt_format.default_system_prompt()

# Initialize model and tokenizer

model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(args, allow_auto_split = True, max_output_len = 16, skip_load = True)

# Initialize draft model if provided, assume it always fits on first device

draft_model = None
draft_cache = None

if args.draft_model_dir:

    print(f" -- Draft model: {args.draft_model_dir}")

    draft_config = ExLlamaV2Config()
    draft_config.model_dir = args.draft_model_dir
    draft_config.prepare()

    if draft_config.max_seq_len < model.config.max_seq_len:

        if args.no_draft_scale:
            print(f" !! Warning: Draft model native max sequence length is less than sequence length for model. Speed may decrease after {draft_config.max_seq_len} tokens.")
        else:
            ratio = model.config.max_seq_len / draft_config.max_seq_len
            alpha = -0.13436 + 0.80541 * ratio + 0.28833 * ratio ** 2
            draft_config.scale_alpha_value = alpha
            print(f" -- Applying draft model RoPE alpha = {alpha:.4f}")

    draft_config.max_seq_len = model.config.max_seq_len
    draft_config.no_flash_attn = args.no_flash_attn
    draft_config.scale_pos_emb = args.rope_scale

    print(" -- Loading draft model...")

    draft_model = ExLlamaV2(draft_config)
    draft_model.load()

    if args.cache_8bit:
        draft_cache = ExLlamaV2Cache_8bit(draft_model)
    elif args.cache_q4:
        draft_cache = ExLlamaV2Cache_Q4(draft_model)
    elif args.cache_q6:
        draft_cache = ExLlamaV2Cache_Q6(draft_model)
    elif args.cache_q8:
        draft_cache = ExLlamaV2Cache_Q8(draft_model)
    else:
        draft_cache = ExLlamaV2Cache(draft_model)

# Load model after draft model

print(" -- Loading model...")

model_init.post_init_load(model, args, allow_auto_split = True)

# Create cache

if args.cache_8bit:
    cache_type = ExLlamaV2Cache_8bit
elif args.cache_q4:
    cache_type = ExLlamaV2Cache_Q4
elif args.cache_q6:
    cache_type = ExLlamaV2Cache_Q6
elif args.cache_q8:
    cache_type = ExLlamaV2Cache_Q8
else:
    cache_type = ExLlamaV2Cache

if model.tp_context:
    cache = ExLlamaV2Cache_TP(model, base = cache_type)
else:
    cache = cache_type(model, lazy = not model.loaded)

# Load model now if auto split enabled

if not model.loaded:

    model.load_autosplit(cache)

# Chat context

def format_prompt(user_prompt, first):
    global system_prompt, prompt_format

    if first:
        return prompt_format.first_prompt(bool(system_prompt)) \
            .replace("<|system_prompt|>", system_prompt) \
            .replace("<|user_prompt|>", user_prompt)
    else:
        return prompt_format.subs_prompt() \
            .replace("<|user_prompt|>", user_prompt)

def encode_prompt(text):
    global tokenizer, prompt_format

    add_bos, add_eos, encode_special_tokens = prompt_format.encoding_options()
    return tokenizer.encode(text, add_bos = add_bos, add_eos = add_eos, encode_special_tokens = encode_special_tokens)

user_prompts = []
responses_ids = []

def get_tokenized_context(max_len):
    global user_prompts, responses_ids

    while True:

        context = torch.empty((1, 0), dtype=torch.long)

        for turn in range(len(user_prompts)):

            up_text = format_prompt(user_prompts[turn], context.shape[-1] == 0)
            up_ids = encode_prompt(up_text)
            context = torch.cat([context, up_ids], dim=-1)

            if turn < len(responses_ids):
                context = torch.cat([context, responses_ids[turn]], dim=-1)

        if context.shape[-1] < max_len: return context

        # If the context is too long, remove the first Q/A pair and try again. The system prompt will be moved to
        # the first entry in the truncated context

        user_prompts = user_prompts[1:]
        responses_ids = responses_ids[1:]


# Generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer, draft_model, draft_cache, num_speculative_tokens=args.draft_n_tokens)
generator.speculative_ngram = args.ngram_decoding

settings = ExLlamaV2Sampler.Settings(
    temperature = args.temperature,
    top_k = args.top_k,
    top_p = args.top_p,
    top_a = args.top_a,
    typical = args.typical,
    skew = args.skew,
    token_repetition_penalty = args.repetition_penalty,
    token_frequency_penalty = args.frequency_penalty,
    token_presence_penalty = args.presence_penalty,
    smoothing_factor = args.smoothing_factor,
    xtc_probability = args.xtc_probability,
    xtc_threshold = args.xtc_threshold,
    dry_allowed_length = args.dry_allowed_length,
    dry_base = args.dry_base,
    dry_multiplier = args.dry_multiplier,
    dry_range = args.dry_range,
)

if args.dynamic_temperature:
    dt_args = [float(alloc) for alloc in args.dynamic_temperature.split(",")]
    settings.min_temp = dt_args[0]
    settings.max_temp = dt_args[1]
    settings.temp_exponent = dt_args[2]

max_response_tokens = args.max_response_tokens
min_space_in_context = args.response_chunk

# Stop conditions

sc = prompt_format.stop_conditions(tokenizer)
sc = [x for x in sc if x is not None]
generator.set_stop_conditions(sc)

# ANSI color codes

col_default = "\u001b[0m"
col_user = "\u001b[33;1m"  # Yellow
col_bot = "\u001b[34;1m"  # Blue
col_error = "\u001b[31;1m"  # Magenta
col_sysprompt = "\u001b[37;1m"  # Grey

# Code block formatting

codeblock_formatter = None if args.no_code_formatting else CodeBlockFormatter()
in_code_block = False

delim_overflow = ""

# Other options

print_timings = args.print_timings
amnesia = args.amnesia

# Main loop

print(f" -- Prompt format: {args.mode}")
if system_prompt:
    print(f" -- System prompt:")
    print()
    print(col_sysprompt + system_prompt.strip() + col_default)

while True:

    # Get user prompt

    print()
    print(col_user + username + ": " + col_default, end='', flush=True)

    # multi-lin support
    if args.mli:
        content = sys.stdin.read().rstrip()
    else:
        content = input().strip()

    # clear context
    if content == "clear":
        user_prompts = []
        responses_ids = []
        print(col_user + "Context cleared." + col_default, end='', flush=True)
        continue

    up = username + ": " + content
    print()

    # Add to context

    user_prompts.append(up)

    # Send tokenized context to generator

    active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
    generator.begin_stream_ex(active_context, settings)

    # Stream response

    if prompt_format.print_bot_name():

        print(col_bot + botname + ": " + col_default, end = "")

    response_tokens = 0
    response_text = ""
    responses_ids.append(torch.empty((1, 0), dtype = torch.long))

    if print_timings:
        time_begin_stream = time.time()
        if args.ngram_decoding or draft_model is not None: generator.reset_sd_stats()

    while True:

        # Get response stream

        res = generator.stream_ex()
        chunk = res["chunk"]
        eos = res["eos"]
        tokens = res["chunk_token_ids"]

        if len(response_text) == 0: chunk = chunk.lstrip()
        response_text += chunk
        responses_ids[-1] = torch.cat([responses_ids[-1], tokens], dim = -1)

        # Check for code block delimiters
        # Let formatter suppress text as long as it may be part of delimiter
        chunk, codeblock_delimiter = (chunk, False) if codeblock_formatter is None else codeblock_formatter.process_delimiter(chunk)

        # Enter code block

        if not in_code_block:

            # Start of codeblock
            if codeblock_delimiter:
                codeblock_formatter.begin()
                print("\n")
                in_code_block = True
                codeblock_delimiter = False

        # Print

        if in_code_block:

            # Print unformatted
            codeblock_formatter.print_code_block(chunk)

        else:

            # Print formatted
            print(chunk, end = "")

        # Exit code block

        if in_code_block:

            # End of code block
            if codeblock_delimiter:

                # Edge case when we get EOS right after code block
                if eos: codeblock_formatter.print_code_block("\n")

                print("\033[0m")  # Reset block color to be certain
                in_code_block = False
                codeblock_delimiter = False

        sys.stdout.flush()
        # time.sleep(1)

        # If model has run out of space, rebuild the context and restart stream

        if generator.full():

            active_context = get_tokenized_context(model.config.max_seq_len - min_space_in_context)
            generator.begin_stream(active_context, settings)

        # If response is too long, cut it short, and append EOS if that was a stop condition

        response_tokens += 1
        if response_tokens == max_response_tokens:

            if tokenizer.eos_token_id in generator.stop_tokens:
                responses_ids[-1] = torch.cat([responses_ids[-1], tokenizer.single_token(tokenizer.eos_token_id)], dim = -1)

            print()
            print(col_error + f" !! Response exceeded {max_response_tokens} tokens and was cut short." + col_default)
            break

        # EOS signal returned

        if eos:

            if prompt_format.print_extra_newline():
                print()

            break

    # Prompt timings

    if print_timings:

        time_end_stream = time.time()
        speed = response_tokens / (time_end_stream - time_begin_stream)

        if draft_model is not None or args.ngram_decoding:
            eff, acc, _, _, _ = generator.get_sd_stats()
            sd_stats = f", SD eff. {eff*100:.2f}%, SD acc. {acc*100:.2f}%"
        else:
            sd_stats = ""

        ctx_tokens = active_context.shape[-1]
        print()
        print(col_sysprompt + f"(Context: {ctx_tokens} tokens, response: {response_tokens} tokens, {speed:.2f} tokens/second{sd_stats})" + col_default)

    # Optionally forget context after each response

    if amnesia:
        user_prompts = []
        responses_ids = []
