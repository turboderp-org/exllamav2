
import argparse, sys, os, glob, time
import torch

from exllamav2 import(
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Tokenizer
)

def add_args(parser):

    parser.add_argument("-m", "--model_dir", type = str, help = "Path to model directory")
    parser.add_argument("-gs", "--gpu_split", type = str, help = "\"auto\", or VRAM allocation per GPU in GB. \"auto\" is implied by default in tensor-parallel mode.")
    parser.add_argument("-tp", "--tensor_parallel", action = "store_true", help = "Load in tensor-parallel mode (not fully supported for all models)")
    parser.add_argument("-l", "--length", type = int, help = "Maximum sequence length")
    parser.add_argument("-rs", "--rope_scale", type = float, help = "RoPE scaling factor")
    parser.add_argument("-ra", "--rope_alpha", type = float, help = "RoPE alpha value (NTK)")
    parser.add_argument("-ry", "--rope_yarn", type = float, help = "Set RoPE YaRN factor (use default max_seq_len as original_max_position_embeddings if not configured)")
    parser.add_argument("-nfa", "--no_flash_attn", action = "store_true", help = "Disable Flash Attention")
    parser.add_argument("-nxf", "--no_xformers", action = "store_true", help = "Disable xformers, an alternative plan of flash attn for older devices")
    parser.add_argument("-nsdpa", "--no_sdpa", action = "store_true", help = "Disable Torch SDPA")
    parser.add_argument("-ng", "--no_graphs", action = "store_true", help = "Disable Graphs")
    parser.add_argument("-lm", "--low_mem", action = "store_true", help = "Enable VRAM optimizations, potentially trading off speed")
    parser.add_argument("-ept", "--experts_per_token", type = int, help = "Override MoE model's default number of experts per token")
    parser.add_argument("-lq4", "--load_q4", action = "store_true", help = "Load weights in Q4 mode")
    parser.add_argument("-fst", "--fast_safetensors", action = "store_true", help = "Deprecated (does nothing)")
    parser.add_argument("-ic", "--ignore_compatibility", action = "store_true", help = "Do not override model config options in case of compatibility issues")
    parser.add_argument("-chunk", "--chunk_size", type = int, help = "Chunk size ('input length')")


def print_options(args):

    print(f" -- Model: {args.model_dir}")

    print_opts = []
    if args.gpu_split is not None: print_opts += [f"gpu_split: {args.gpu_split}"]
    if args.tensor_parallel: print_opts += ["tensor_parallel"]
    if args.length is not None: print_opts += [f"length: {args.length}"]
    if args.rope_scale is not None: print_opts += [f"rope_scale: {args.rope_scale}"]
    if args.rope_alpha is not None: print_opts += [f"rope_alpha: {args.rope_alpha}"]
    if args.rope_yarn is not None: print_opts += [f"rope_yarn: {args.rope_yarn}"]
    if args.no_flash_attn: print_opts += ["no_flash_attn"]
    if args.no_xformers: print_opts += ["no_xformers"]
    if args.no_sdpa: print_opts += ["no_sdpa"]
    if args.no_graphs: print_opts += ["no_graphs"]
    if args.low_mem: print_opts += ["low_mem"]
    if args.experts_per_token is not None: print_opts += [f"experts_per_token: {args.experts_per_token}"]
    if args.load_q4: print_opts += ["load_q4"]
    if args.ignore_compatibility: print_opts += ["ignore_compatibility"]
    if args.chunk_size is not None: print_opts += [f"chunk_size: {args.chunk_size}"]
    print(f" -- Options: {print_opts}")


def check_args(args):

    if not args.model_dir:
        print(" ## Error: No model directory specified")
        sys.exit()

    if not os.path.exists(args.model_dir):
        print(f" ## Error: Can't find model directory: {args.model_dir}")
        sys.exit()

    required_files = ["config.json",
                      ["tokenizer.model", "tokenizer.json"],
                      "*.safetensors"]

    for filename in required_files:
        if isinstance(filename, str):
            filename = [filename]
        all_matches = []
        for file in filename:
            path = os.path.join(args.model_dir, file)
            matches = glob.glob(path)
            all_matches += matches
        if len(all_matches) == 0:
            print(f" ## Error: Cannot find {filename} in {args.model_dir}")
            sys.exit()


def init(
    args,
    quiet: bool = False,
    allow_auto_split: bool = False,
    skip_load: bool = False,
    benchmark: bool = False,
    max_batch_size: int = None,
    max_input_len: int = None,
    max_output_len: int = None,
    progress: bool = False
):

    # Create config

    config = ExLlamaV2Config()
    config.model_dir = args.model_dir
    config.prepare()

    # Set config options

    if args.rope_yarn:
        if config.alt_rope_method != "yarn":
            config.yarn_rope_original_max_position_embeddings = config.max_seq_len
        config.alt_rope_method = "yarn"
        config.yarn_rope_factor = args.rope_yarn

    if args.length: config.max_seq_len = args.length
    if args.rope_scale: config.scale_pos_emb = args.rope_scale
    if args.rope_alpha: config.scale_alpha_value = args.rope_alpha
    if args.no_flash_attn: config.no_flash_attn = True
    if args.no_xformers: config.no_xformers = True
    if args.no_sdpa: config.no_sdpa = True
    if args.no_graphs: config.no_graphs = True
    if args.experts_per_token: config.num_experts_per_token = args.experts_per_token

    if max_batch_size: config.max_batch_size = max_batch_size
    config.max_output_len = max_output_len
    if max_input_len: config.max_input_len = max_input_len

    # Set low-mem options

    if args.low_mem: config.set_low_mem()
    if args.load_q4: config.load_in_q4 = True

    if args.chunk_size is not None:
        config.max_input_len = args.chunk_size
        config.max_attention_size = args.chunk_size ** 2

    # Compatibility warnings

    config.arch_compat_overrides(warn_only = args.ignore_compatibility)

    # Load model
    # If --gpu_split auto, return unloaded model. Model must be loaded with model.load_autosplit() supplying cache
    # created in lazy mode

    model = ExLlamaV2(config)

    if not skip_load:
        post_init_load(
            model,
            args,
            quiet,
            allow_auto_split,
            benchmark,
            progress,
        )

    # Load tokenizer

    if not quiet: print(" -- Loading tokenizer...")

    tokenizer = ExLlamaV2Tokenizer(config)

    return model, tokenizer


def post_init_load(
    model: ExLlamaV2,
    args,
    quiet: bool = False,
    allow_auto_split: bool = False,
    benchmark: bool = False,
    progress: bool = False,
):

    split = None
    if args.gpu_split and args.gpu_split != "auto":
        split = [float(alloc) for alloc in args.gpu_split.split(",")]
        if len(split) > torch.cuda.device_count():
            print(f" ## Error: Too many entries in gpu_split. {torch.cuda.device_count()} CUDA devices are available.")
            sys.exit()

    if args.tensor_parallel:
        if args.gpu_split == "auto": split = None
        model.load_tp(split, progress = progress)

    elif args.gpu_split != "auto":
        if not quiet and not progress: print(" -- Loading model...")
        t = time.time()
        model.load(split, progress = progress)
        t = time.time() - t
        if benchmark and not quiet:
            print(f" -- Loaded model in {t:.4f} seconds")

    else:
        assert allow_auto_split, "Auto split not allowed."
