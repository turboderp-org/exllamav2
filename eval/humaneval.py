from __future__ import annotations
import sys
import os
import argparse
import subprocess
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from human_eval.data import write_jsonl, read_problems
from exllamav2 import model_init
from exllamav2 import ExLlamaV2Cache, ExLlamaV2Cache_Q4, ExLlamaV2Cache_Q6, ExLlamaV2Cache_Q8
from exllamav2.generator import ExLlamaV2DynamicGenerator, ExLlamaV2DynamicJob, ExLlamaV2Sampler
import util

# Args

parser = argparse.ArgumentParser(description="Run HumanEval evaluation on EXL2 model")
parser.add_argument("-o", "--output", type=str, help="Output .jsonl filename", required=True)
parser.add_argument("-cs", "--cache_size", type=int, default=None)
parser.add_argument("-spt", "--samples_per_task", type=int, default=200)
parser.add_argument("-cq4", "--cache_q4", action="store_true", help="Use Q4 cache")
parser.add_argument("-cq6", "--cache_q6", action="store_true", help="Use Q6 cache")
parser.add_argument("-cq8", "--cache_q8", action="store_true", help="Use Q8 cache")
parser.add_argument("--max_tokens", type=int, default=768, help="Max number of tokens for each completion")
parser.add_argument("-pf", "--prompt_format", type=str,
                    help="Instruct format to apply. Default is raw completion (for base models)")
parser.add_argument("-v", "--verbose", action="store_true", help="Spam completions to console while generating")
parser.add_argument("-e", "--eval", action="store_true", help="Run evaluation script on output file after sampling")
parser.add_argument("-temp", "--temperature", type=float, default=0.6, help="Sampling temperature (0 for greedy)")
parser.add_argument("-bs", "--batch_size", type=int, default=50, help="Number of problems to process in each batch")
model_init.add_args(parser)
args = parser.parse_args()

# Validate args

directory = os.path.dirname(args.output)
if directory and not os.path.isdir(directory):
    print(f" ## Directory for output file {args.output} does not exist.")
    sys.exit()
if os.path.exists(args.output):
    print(f" !! Warning: Output file exists and will be overwritten.")

# Prompt formats

prompt_formats = {
    "raw": ("```python\n{{problem}}    ", "    "),
    "granite": (
        "Question:\nComplete the following Python function:\n\n{{problem}}\n\nAnswer:\n"
        "Sure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    ),
    "llama": (
        "[INST] <<SYS>>\nYou are a helpful AI coding assistant.\n<</SYS>>\n\n"
        "Complete the following Python function:\n\n{{problem}} [/INST] "
        "Sure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    ),
    "llama3": (
        "<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful AI coding assistant.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nComplete the following Python function:\n\n{{problem}}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nSure! Here is how you might implement the function:\n\n```python\n{{problem}}",
        "    "
    ),
    "gemma": (
        "<bos><start_of_turn>user\nComplete the following Python function:\n\n{{problem}}<|eot_id|>"
        "<start_of_turn>model\n```python\n{{problem}}",
        "    "
    )
}

if args.prompt_format is None:
    prompt_format, prefix = "{{problem}}", "    "
elif args.prompt_format in prompt_formats:
    prompt_format, prefix = prompt_formats[args.prompt_format]
else:
    print("Prompt format is not supported. Available formats:")
    print("\n".join(prompt_formats.keys()))
    sys.exit()

# Init model and cache

model_init.check_args(args)
model_init.print_options(args)
model, tokenizer = model_init.init(
    args,
    allow_auto_split=True,
    progress=True,
    max_output_len=4,
    max_input_len=2048
)

if args.cache_q4:
    cache_type = ExLlamaV2Cache_Q4
elif args.cache_q6:
    cache_type = ExLlamaV2Cache_Q6
elif args.cache_q8:
    cache_type = ExLlamaV2Cache_Q8
else:
    cache_type = ExLlamaV2Cache
cache = cache_type(
    model,
    lazy=not model.loaded,
    max_seq_len=args.cache_size or model.config.max_seq_len
)

if not model.loaded:
    model.load_autosplit(cache, progress=True)

# Generator

generator = ExLlamaV2DynamicGenerator(
    model=model,
    cache=cache,
    tokenizer=tokenizer,
    max_batch_size=256,
    max_q_size=4
)

gen_settings = ExLlamaV2Sampler.Settings(
    token_repetition_penalty=1.0,
    temperature=args.temperature,
    top_k=50,
    top_p=0.6
)


def process_batch(batch_problems, batch_size, progress, sample_task, generate_task):
    samples = []

    for problem_id, problem in batch_problems.items():
        b_problem = problem["prompt"]
        f_problem = prompt_format.replace("{{problem}}", b_problem)
        input_ids = tokenizer.encode(f_problem, encode_special_tokens=True, add_bos=True)

        for s in range(batch_size):
            job = ExLlamaV2DynamicJob(
                input_ids=input_ids,
                gen_settings=gen_settings,
                max_new_tokens=args.max_tokens,
                stop_conditions=[tokenizer.eos_token_id],
                token_healing=True,
                identifier=(problem_id, s),
                min_new_tokens=6
            )
            generator.enqueue(job)
            progress.update(sample_task, advance=1)

    while generator.num_remaining_jobs():
        results = generator.iterate()
        for result in results:
            job = result["job"]
            eos = False
            completion = job.full_completion
            last_newline_index = completion.rfind("\n")
            if last_newline_index >= 0:
                last_line = completion[last_newline_index + 1:]
                if last_line != "" and not last_line[0].isspace():
                    completion = completion[:last_newline_index]
                    eos = True
            eos = eos or result["eos"]

            if eos:
                identifier = result["identifier"]
                sample = batch_problems[identifier[0]]["prompt"] + prefix + completion.strip()
                if not result["eos"]:
                    generator.cancel(job)

                if args.verbose:
                    print("----------------------------------------------------------------------")
                    print(f" ** Problem {identifier[0]}, sample {identifier[1] + 1} / {batch_size}")
                    print("----------------------------------------------------------------------")
                    print(sample)
                    print()
                else:
                    progress.update(generate_task, advance=1)

                samples.append(dict(task_id=identifier[0], completion=prefix + completion.strip()))

    return samples


# Main execution
problems = read_problems()
all_samples = []
batch_size = args.batch_size
total_samples = len(problems) * args.samples_per_task

with util.get_progress() as progress:
    sample_task = progress.add_task("[red]Sample", total=total_samples, name="Creating sample jobs")
    generate_task = progress.add_task("[green]Sample", total=total_samples, name="Generating samples")

    for i in range(0, len(problems), batch_size):
        batch_problems = dict(list(problems.items())[i:i + batch_size])
        batch_samples = process_batch(batch_problems, args.samples_per_task, progress, sample_task, generate_task)
        all_samples.extend(batch_samples)

        # Optional: Clear CUDA cache to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# Save output
print(f" -- Saving: {args.output}")
write_jsonl(args.output, all_samples)

# Optionally launch eval script
if args.eval:
    subprocess.run(["evaluate_functional_correctness", args.output])