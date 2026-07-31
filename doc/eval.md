# ExLlamaV2 Evaluation scripts

Common arguments:

- **-m / --model *directory***: _(required)_ Path to model (EXL2, GPTQ or FP16)

- **-gs / --gpu_split *list***: List of memory allocations per GPU, in GB for model weights and static buffers 
(excluding cache). Example: `-gs 10.5,0,10.5` would allocate 10.5 GB on CUDA devices 0 and 2 while skipping
device 1. `-gs auto` will load the model in auto split mode, which fills available devices in order.

- **-l / --length *int***: Context length. The default is the model's native context length, which may be
excessive for most benchmarks.

- **-rs / --rope_scale *float***: RoPE scale factor (linear)

- **-ra / --rope_alpha *float***: RoPE scale factor (NTK)

- **-nfa / --no_flash_attn**: Don't use flash-attn.

- **-nxf / --no_xformers**: Don't use xformers.

- **-fst / --fast_safetensors**: Use alternative loading mode. On Linux, this mode uses direct I/O and pinned
buffers and can potentially load faster from very fast NVMe RAID arrays with a cold cache. On Windows, this
uses regular non-memorymapped I/O and is typically just slower. However, in either case this can fix situations
in which ExLlama runs out of system memory when loading large models.

## HumanEval

This is the standard [HumanEval](https://github.com/openai/human-eval) test implemented for ExLlamaV2 with
dynamic batching.

```sh
pip install human-eval
python eval/humaneval.py -m <model_dir> -o humaneval_output.json
evaluate-functional-correctness humaneval_output.json
```

Arguments:

- **-o / --output *file***: _(required)_ Output JSON file to receive generated samples

- **-spt / --samples_per_task *int***: Number of samples for each HumanEval task. A single sample per task is
sufficient to compute an approximate pass@1 score, but more samples give a more accurate score. At least 10 
samples is required for a pass@10 score, etc.

- **--max_tokens *int***: Maximum number of tokens to generate before cutting a sample short. The stop condition
for each generation, if this limit isn't reached first, is the first newline character not followed by 
whitespace, i.e. the first non-indented line after the function definition has been generated. Default is 768
which seems sufficient for most HumanEval tasks.

- **-pf / --prompt *str***: By default, the sample is a raw completion suitable for both base models and instruct
tuned models Supplying a prompt format turns each task into an instruct prompt asking for the completion with a
prefix for the response.

- **-v / --verbose**: Output completions as they're being generated (otherwise show a progress bar.)

- **-cs / --cache_size *int***: Total number of cache tokens. Set this as high as possible for best batching
performance.

- **-cq4 / --cache_q4**: Use Q4 cache
  
- **-cq6 / --cache_q6**: Use Q6 cache

- **-cq8 / --cache_q8**: Use Q8 cache

## MMLU

This is the standard [MMLU](https://github.com/hendrycks/test) test implemented for ExLlamaV2 with
dynamic batching.

```sh
pip install datasets
python eval/mmlu.py -m <model_dir>
```

Arguments:

- **-sub / --subjects *list***: Limit test to the listed subjects, otherwise test on all subjects. E.g.
`-sub anatomy,nutrition,professional_medicine`. See [the dataset](https://huggingface.co/datasets/cais/mmlu) for
the full list of subjects.

- **-fs / --fewshot_examples *int***: Number of fewshot examples before each question. Default is 5.

- **-shf / --shuffle**: Shuffle the answer choices to each question.

- **-cs / --cache_size *int***: Total number of cache tokens. Set this as high as possible for best batching
performance.

- **-cq4 / --cache_q4**: Use Q4 cache

- **-cq6 / --cache_q6**: Use Q6 cache

- **-cq8 / --cache_q8**: Use Q8 cache
