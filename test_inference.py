
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

from exllamav2.generator import (
    ExLlamaV2BaseGenerator,
    ExLlamaV2Sampler
)

from exllamav2.attn import ExLlamaV2Attention
from exllamav2.mlp import ExLlamaV2MLP
from exllamav2.moe_mlp import ExLlamaV2MoEMLP
from exllamav2.parallel_decoder import ExLlamaV2ParallelDecoder

import argparse, os, math, time
import torch
import torch.nn.functional as F
from exllamav2.conversion.tokenize import get_tokens
from exllamav2.conversion.quantize import list_live_tensors
import gc

# from exllamav2.mlp import set_catch

import sys
import json

torch.cuda._lazy_init()
torch.set_printoptions(precision = 5, sci_mode = False, linewidth = 150)

# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
# torch.set_float32_matmul_precision("medium")

# (!!!) NOTE: These go on top of the engine arguments that can be found in `model_init.py` (!!!)
parser = argparse.ArgumentParser(description = "Test inference on ExLlamaV2 model")
parser.add_argument("-ed", "--eval_dataset", type = str, help = "Perplexity evaluation dataset (.parquet file)")
parser.add_argument("-er", "--eval_rows", type = int, default = None, help = "Number of rows to apply from dataset (default 128)")
parser.add_argument("-el", "--eval_length", type = int, default = 2048, help = "Max no. tokens per sample")
parser.add_argument("-et", "--eval_token", action = "store_true", help = "Evaluate perplexity on token-by-token inference using cache")
parser.add_argument("-e8", "--eval_token_8bit", action = "store_true", help = "Evaluate perplexity on token-by-token inference using 8-bit (FP8) cache")
parser.add_argument("-eq4", "--eval_token_q4", action = "store_true", help = "Evaluate perplexity on token-by-token inference using Q4 cache")
parser.add_argument("-eq6", "--eval_token_q6", action = "store_true", help = "Evaluate perplexity on token-by-token inference using Q6 cache")
parser.add_argument("-eq8", "--eval_token_q8", action = "store_true", help = "Evaluate perplexity on token-by-token inference using Q8 cache")
parser.add_argument("-ecl", "--eval_context_lens", action = "store_true", help = "Evaluate perplexity at range of context lengths")
# parser.add_argument("-eb", "--eval_bos", action = "store_true", help = "Add BOS token to every row in perplexity test (required by Gemma and maybe other models.)")
parser.add_argument("-p", "--prompt", type = str, help = "Generate from prompt (basic sampling settings)")
parser.add_argument("-pnb", "--prompt_no_bos", action = "store_true", help = "Don't add BOS token to prompt")
parser.add_argument("-t", "--tokens", type = int, default = 128, help = "Max no. tokens")
parser.add_argument("-ps", "--prompt_speed", action = "store_true", help = "Test prompt processing (batch) speed over context length")
parser.add_argument("-s", "--speed", action = "store_true", help = "Test raw generation speed over context length")
parser.add_argument("-mix", "--mix_layers", type = str, help = "Load replacement layers from secondary model. Example: --mix_layers 1,6-7:/mnt/models/other_model")
parser.add_argument("-nwu", "--no_warmup", action = "store_true", help = "Skip warmup before testing model")
parser.add_argument("-sl", "--stream_layers", action = "store_true", help = "Load model layer by layer (perplexity evaluation only)")
parser.add_argument("-sp", "--standard_perplexity", choices = ["wiki2"], help = "Run standard (HF) perplexity test, stride 512 (experimental)")
parser.add_argument("-rr", "--rank_reduce", type = str, help = "Rank-reduction for MLP layers of model, in reverse order (for experimentation)")
parser.add_argument("-mol", "--max_output_len", type = int, help = "Set max output chunk size (incompatible with ppl tests)")

# Initialize model and tokenizer

model_init.add_args(parser)
args = parser.parse_args()

# Check conflicting settings

if args.stream_layers:
    if args.eval_token or args.eval_token_8bit or args.eval_token_q4 or args.eval_token_q6 or args.eval_token_q8:
        print(" ## Can't test token ppl while streaming layers")
        sys.exit()
    if args.prompt:
        print(" ## Can't generate while streaming layers")
        sys.exit()
    if args.speed or args.prompt_speed:
        print(" ## Can't test speed while streaming layers")
        sys.exit()
    if args.gpu_split:
        print(" ## Can only use one GPU when streaming layers")
        sys.exit()
    if args.eval_context_lens and args.stream_layers:
        print(" ## eval_context_lens not compatible with stream_layers")
        sys.exit()
    if args.eval_dataset:
        if args.length and args.eval_length != args.length:
            print(" !! Overriding model context length to match eval row length")
        args.length = args.eval_length

# Init

model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(
    args,
    allow_auto_split = True,
    skip_load = args.stream_layers,
    benchmark = True,
    max_output_len = args.max_output_len,
    progress = True
)
cache = None

# Auto split

if not model.loaded and not args.stream_layers:

    if args.mix_layers:
        print(" !! Warning, auto split does not account for VRAM requirement of replacement layers")

    print(" -- Loading model...")
    cache = ExLlamaV2Cache(model, lazy = True)
    t = time.time()
    model.load_autosplit(cache, progress = True)
    t = time.time() - t
    print(f" -- Loaded model in {t:.4f} seconds")

if args.stream_layers:

    stream_batch_size = 2
    model.config.max_batch_size = stream_batch_size
    model.load(lazy = True)

# Rank reduction

if args.rank_reduce:

    if args.stream_layers:
        print(" ## --rank_reduce can not be combined with --stream_layers")
        sys.exit()

    rr = args.rank_reduce.split(",")
    idx = len(model.modules) - 1
    for r in rr:
        k = float(r)

        while True:
            idx -= 1
            module = model.modules[idx]
            if isinstance(module, ExLlamaV2ParallelDecoder): break
            if isinstance(module, ExLlamaV2MLP): break
            if isinstance(module, ExLlamaV2MoEMLP): break
            if idx < 0:
                print(" ## Not enough layers")
                sys.exit()

        print(f" -- Reducing {module.key} ({module.name}) to {k * 100:.2f}%")
        module.rank_reduce(k)

# Replacement

if args.mix_layers:
    intervals_, extra_dir = args.mix_layers.split(":")

    print(f" -- Loading replacement layers from: {extra_dir}")

    extra_config = ExLlamaV2Config()
    extra_config.model_dir = extra_dir
    extra_config.prepare()
    intervals = intervals_.split(",")
    for interval in intervals:
        ab = interval.split("-")
        a, b = int(ab[0]), int(ab[-1])
        for idx in range(a, b + 1):
            print(f" --   Layer {idx}...")
            layerkey = "model.layers." + str(idx) + "."
            remove = [k for k in model.config.tensor_file_map.keys() if k.startswith(layerkey)]
            replace = [k for k in extra_config.tensor_file_map.keys() if k.startswith(layerkey)]
            # reload = [k for k in model.modules_dict.keys() if k.startswith(layerkey)]
            for k in remove: del model.config.tensor_file_map[k]
            for k in replace: model.config.tensor_file_map[k] = extra_config.tensor_file_map[k]
            # for k in reload:
            #     model.modules_dict[k].unload()
            #     model.modules_dict[k].load()
            if not args.stream_layers:
                model.modules[idx * 2 + 1].reload()
                model.modules[idx * 2 + 2].reload()

# Test generation

if args.prompt:

    with torch.inference_mode():

        if cache is None:
            cache = ExLlamaV2Cache(model) if not model.tp_context else ExLlamaV2Cache_TP(model)

        ids = tokenizer.encode(args.prompt)
        tokens_prompt = ids.shape[-1]

        print(f" -- Warmup...")

        generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)
        if not args.no_warmup: generator.warmup()

        print(f" -- Generating...")
        print()

        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = 1.0
        settings.top_k = 0
        settings.top_p = 0.8
        settings.token_repetition_penalty = 1.02
        settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

        time_begin = time.time()

        output = generator.generate_simple(args.prompt, settings, args.tokens, token_healing = True, add_bos = not args.prompt_no_bos)

        torch.cuda.synchronize()
        time_prompt = time.time()

        time_end = time.time()

    print(output)
    print()

    total_gen = time_end - time_begin
    print(f" -- Response generated in {total_gen:.2f} seconds, {args.tokens} tokens, {args.tokens / total_gen:.2f} tokens/second (includes prompt eval.)")


# Test perplexity

if args.eval_dataset or args.standard_perplexity:

    with torch.inference_mode():

        print(f" -- Running perplexity test")

        if args.standard_perplexity:

            eval_length = args.eval_length
            if args.eval_dataset:
                print(f" !! Note, overriding specified --eval_dataset with {args.standard_perplexity}")

            from datasets import load_dataset

            if args.standard_perplexity == "wiki2":
                ds = "wikitext"
                part = "wikitext-2-raw-v1"
                split = "test"
            # if args.standard_perplexity == "c4":
            #     ds = "allenai/c4"
            #     part = "allenai--c4"
            #     split = "train"

            print(f" -- Loading dataset {ds}, {part}, {split}...")
            test = load_dataset(ds, part, split = split)

            print(f" -- Tokenizing samples...")
            text = "\n\n".join(test["text"])
            eval_tokens = tokenizer.encode(text)

            stride = 512
            seqs = []
            eval_len = []
            a = 0
            while True:
                b = a + model.config.max_seq_len
                if b > eval_tokens.shape[-1]: break
                seqs.append(eval_tokens[:, a:b])
                eval_len.append(b if a == 0 else stride)
                a += stride
                if args.eval_rows and len(seqs) >= args.eval_rows:
                    break

            eval_tokens = torch.cat(seqs, dim = 0)

        else:

            eval_dataset = args.eval_dataset
            eval_rows = args.eval_rows or 128
            eval_length = args.eval_length

            print(f" -- Dataset: {eval_dataset}")
            print(f" -- Tokenizing eval data, {eval_rows} rows x {eval_length} tokens...")

            eval_tokens = get_tokens(eval_rows, eval_length, eval_dataset, tokenizer)
            eval_len = [eval_tokens.shape[1]] * eval_tokens.shape[0]

            # if args.eval_bos:
            if model.config.arch.lm.requires_bos:
                boss = torch.full((eval_tokens.shape[0], 1), tokenizer.bos_token_id, dtype = torch.long)
                eval_tokens = torch.cat((boss, eval_tokens[:, :-1]), dim = 1)

        if args.eval_context_lens:
            logprob_sum = []
            logprob_count = []
        else:
            logprob_sum = 0.0
            logprob_count = 0

        def ppl(input_ids__, logits__, lengths__, bins = False):

            logits_device = model.modules[-1].device() if not model.tp_context else \
                            torch.device(model.tp_context.device)

            if bins:
                num_bins = (max(lengths__) + 255) // 256
                logprob_sum_ = [0.0] * num_bins
                logprob_count_ = [0] * num_bins
            else:
                logprob_sum_ = 0.0
                logprob_count_ = 0

            assert logits__.shape[0] == input_ids__.shape[0]
            ll = logits__.shape[1]

            for bi in range(logits__.shape[0]):
                cl = max(ll - lengths__[bi], 0)
                logits_ = logits__[bi:bi+1, cl:, :]
                input_ids_ = input_ids__[bi:bi+1, cl:]

                if bins:
                    chunksize = 256
                else:
                    chunksize = logits_.shape[1] * 4000 // logits_.shape[2] + 1
                b_ = 0
                while b_ < logits_.shape[1]:
                    a_ = b_
                    b_ = min(b_ + chunksize, logits_.shape[1])

                    logits_f = logits_[:, a_:b_, :].to(logits_device).float() + 1e-10
                    target_ids = input_ids_[:, a_ + 1:b_ + 1].to(logits_f.device)

                    log_probs = F.log_softmax(logits_f, dim=-1)
                    token_log_probs = log_probs.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
                    if bins:
                        # for cbin in range(a_ // 256 + 1):
                        cbin = a_ // 256
                        logprob_sum_[cbin] += token_log_probs.sum().item()
                        logprob_count_[cbin] += target_ids.numel()
                    else:
                        logprob_sum_ += token_log_probs.sum().item()
                        logprob_count_ += target_ids.numel()

            return logprob_sum_, logprob_count_

        if args.stream_layers:

            print(f" -- Inference (streamed)", end = "")
            sys.stdout.flush()

            batch_size, seq_len = eval_tokens.shape
            attn_params = ExLlamaV2Attention.Params(stream_batch_size, seq_len, 0, None, None)
            # attn_mask = model.build_attn_mask(stream_batch_size, seq_len, 0, None, "cuda:0")

            for idx, module in enumerate(model.modules):
                module.set_device_idx(-1 if idx == 0 else 0)

            model.modules[0].load()
            hidden_state = model.modules[0].forward(eval_tokens)
            model.modules[0].unload()

            for idx, module in enumerate(model.modules):
                if idx == 0: continue

                print(".", end = "")
                sys.stdout.flush()
                module.load()

                b = 0
                while b < eval_tokens.shape[0]:
                    a = b
                    b = min(b + stream_batch_size, eval_tokens.shape[0])
                    x = hidden_state[a:b, :, :].to("cuda:0")
                    x = module.forward(x, cache = None, attn_params = attn_params, past_len = 0, loras = None)

                    if idx < len(model.modules) - 1:
                        hidden_state[a:b, :, :] = x.to("cpu")

                    else:
                        input_ids = eval_tokens[a:b, :]
                        logits = x[:, :-1, :]

                        # if model.config.logit_scale != 1:
                        #     logits.mul_(model.config.logit_scale)

                        logprob_sum__, logprob_count__ = ppl(input_ids, logits, eval_len[a:b])
                        logprob_sum += logprob_sum__
                        logprob_count += logprob_count__

                module.unload()

            print()

        else:

            print(f" -- Inference", end = "")
            sys.stdout.flush()

            if cache is None:
                if eval_length > model.config.max_input_len:
                    cache = ExLlamaV2Cache(model, max_seq_len = eval_length) if not model.tp_context else ExLlamaV2Cache_TP(model, max_seq_len = eval_length)
                else:
                    cache = None

            for i in range(eval_tokens.shape[0]):

                if i % 10 == 0: print(".", end = "")
                sys.stdout.flush()

                input_ids = eval_tokens[i:i+1, :]

                input_ids = input_ids[:, :]
                if cache is not None: cache.current_seq_len = 0
                logits = model.forward(input_ids, cache, cpu_logits = input_ids.numel() > 2048)
                logits = logits[:, :-1, :]

                logprob_sum__, logprob_count__ = ppl(input_ids, logits, eval_len[i:i+1], args.eval_context_lens)
                if args.eval_context_lens:
                    while len(logprob_sum) < len(logprob_sum__):
                        logprob_sum.append(0.0)
                        logprob_count.append(0)
                    for j in range(len(logprob_sum__)):
                        logprob_sum[j] += logprob_sum__[j]
                        logprob_count[j] += logprob_count__[j]
                else:
                    logprob_sum += logprob_sum__
                    logprob_count += logprob_count__

        if not args.eval_context_lens:
            print()
            mean_log_prob = logprob_sum / logprob_count
            perplexity = math.exp(-mean_log_prob)
            print(f" -- Evaluation perplexity: {perplexity:.4f}")
        else:
            print()
            for j in range(len(logprob_sum__)):
                mean_log_prob = logprob_sum[j] / logprob_count[j]
                perplexity = math.exp(-mean_log_prob)
                dl = min((j + 1) * 256, eval_length)
                print(f" -- Evaluation perplexity: {dl} {perplexity:.4f}")

        def test_ppl_token():
            global logprob_sum, logprob_count, i, input_ids
            global logits, target_ids, log_probs, token_log_probs
            global mean_log_prob, perplexity

            # set_catch("model.layers.3")

            logprob_sum = 0
            logprob_count = 0

            for i in range(eval_tokens.shape[0]):

                cache.current_seq_len = 0

                for j in range(eval_tokens.shape[1] - 1):
                    if j % 256 == 0: print(".", end = "")
                    sys.stdout.flush()

                    input_ids = eval_tokens[i:i + 1, j:j + 1]
                    logits = model.forward(input_ids, cache)
                    logits = logits.float() + 1e-10

                    log_probs = F.log_softmax(logits, dim = -1)
                    logprob_sum += log_probs[0, 0, eval_tokens[i, j+1]]
                    logprob_count += 1

                    # mean_log_prob = logprob_sum / logprob_count
                    # perplexity = math.exp(-mean_log_prob)
                    # print(f" -- Token {j}: {perplexity:.4f}")

            print()

            mean_log_prob = logprob_sum / logprob_count
            perplexity = math.exp(-mean_log_prob)
            print(f" -- Evaluation perplexity: {perplexity:.4f}")

        if args.eval_token:
            if args.standard_perplexity:
                print(f" !! Note, can't evalutate token perplexity on standard test")
            else:
                print(f" -- Inference (token)", end = "")
                sys.stdout.flush()
                cache = ExLlamaV2Cache(model, max_seq_len = eval_length) if not model.tp_context else \
                        ExLlamaV2Cache_TP(model, max_seq_len = eval_length)
                test_ppl_token()

        if args.eval_token_8bit:
            if args.standard_perplexity:
                print(f" !! Note, can't evalutate token perplexity on standard test")
            else:
                print(f" -- Inference (token, 8-bit cache)", end = "")
                sys.stdout.flush()
                cache = ExLlamaV2Cache_8bit(model, max_seq_len = eval_length) if not model.tp_context else \
                        ExLlamaV2Cache_TP(model, max_seq_len = eval_length, base = ExLlamaV2Cache_8bit)
                test_ppl_token()

        if args.eval_token_q4:
            if args.standard_perplexity:
                print(f" !! Note, can't evalutate token perplexity on standard test")
            else:
                print(f" -- Inference (token, Q4 cache)", end = "")
                sys.stdout.flush()
                cache = ExLlamaV2Cache_Q4(model, max_seq_len = eval_length) if not model.tp_context else \
                        ExLlamaV2Cache_TP(model, max_seq_len = eval_length, base = ExLlamaV2Cache_Q4)
                # cache.calibrate(tokenizer)
                test_ppl_token()

        if args.eval_token_q6:
            if args.standard_perplexity:
                print(f" !! Note, can't evalutate token perplexity on standard test")
            else:
                print(f" -- Inference (token, Q6 cache)", end = "")
                sys.stdout.flush()
                cache = ExLlamaV2Cache_Q6(model, max_seq_len = eval_length) if not model.tp_context else \
                        ExLlamaV2Cache_TP(model, max_seq_len = eval_length, base = ExLlamaV2Cache_Q6)
                # cache.calibrate(tokenizer)
                test_ppl_token()

        if args.eval_token_q8:
            if args.standard_perplexity:
                print(f" !! Note, can't evalutate token perplexity on standard test")
            else:
                print(f" -- Inference (token, Q8 cache)", end = "")
                sys.stdout.flush()
                cache = ExLlamaV2Cache_Q8(model, max_seq_len = eval_length) if not model.tp_context else \
                        ExLlamaV2Cache_TP(model, max_seq_len = eval_length, base = ExLlamaV2Cache_Q8)
                # cache.calibrate(tokenizer)
                test_ppl_token()


# Test prompt speed

if args.prompt_speed:

    with torch.inference_mode():

        if cache is None:
            cache = ExLlamaV2Cache(model) if not model.tp_context else ExLlamaV2Cache_TP(model)

        ids = torch.randint(0, model.config.vocab_size - 1, (1, model.config.max_seq_len))

        print(f" -- Warmup...")

        if not args.no_warmup:
            model.forward(ids[:, -1:])

        print(f" -- Measuring prompt speed...")

        torch.cuda.synchronize()

        current_len = 128
        step = 128
        prompt_iters = 3
        while True:

            total_time = 0
            for i in range(prompt_iters):

                torch.cuda.synchronize()
                time_begin = time.time()

                cache.current_seq_len = 0
                model.forward(ids[:, :current_len], cache, preprocess_only = True)

                torch.cuda.synchronize()
                time_end = time.time()
                total_time += time_end - time_begin

            tps = current_len / (total_time / prompt_iters)

            print(f" ** Length {current_len:>5} tokens: {tps:>11.4f} t/s")

            if current_len >= 1024: step = 1024
            if current_len >= 4096: step = 4096
            if current_len >= 16384: step = 8192

            current_len_ = current_len
            current_len = min(current_len + step, model.config.max_seq_len)
            if current_len == current_len_: break


# Test token speed

if args.speed:

    with torch.inference_mode():

        if cache is None:
            cache = ExLlamaV2Cache(model) if not model.tp_context else ExLlamaV2Cache_TP(model)
        cache.current_seq_len = 0

        print(f" -- Measuring token speed...")
        ids = tokenizer.encode("X")
        model.forward(ids[:, :])

        current_idx = ids.shape[-1]
        next_stop = 128

        while True:

            time_begin = time.time()

            tokens = next_stop - current_idx
            for i in range(tokens):

                logits = model.forward(ids[:, -1:], cache)
                sample = torch.argmax(logits[0, -1]).cpu().unsqueeze(0).unsqueeze(0)
                sample.clamp_(0, tokenizer.get_vocab_size() - 1)
                ids = torch.cat((ids, sample), dim=-1)

            time_end = time.time()
            tps = tokens / (time_end - time_begin)

            print(f" ** Position {current_idx:>5} + {tokens:>3} tokens: {tps:>9.4f} t/s")

            current_idx = next_stop
            next_stop = min(next_stop + 128, model.config.max_seq_len)
            if next_stop == current_idx: break

