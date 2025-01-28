from __future__ import annotations

import torch
import math
from exllamav2.stloader import STFile, cleanup_stfiles
from exllamav2.architecture import ExLlamaV2ArchParams
import os, glob, json
from typing import Any, Dict, List, TypeVar, Union, cast

T = TypeVar('T')
no_default = object()

def read(
    input_dict: dict[str, Any],
    expected_type: type | list[type],
    keys: str | list[str],
    default = no_default,
    opt_subkey: str | None = None
) -> T:

    expected_types = expected_type if isinstance(expected_type, list) else [expected_type]

    if isinstance(keys, str): keys = [keys]

    if opt_subkey is not None:
        keys = keys + [opt_subkey + "->" + k for k in keys]

    for key in keys:
        input_dict_s = input_dict

        key_split = key.split("->")
        for subk in key_split[:-1]:
            input_dict_s = input_dict_s.get(subk, None)
            if not input_dict_s:
                key = None
                break
        if key is None: continue
        key = key_split[-1]

        x = input_dict_s.get(key, None)
        if x is not None:

            if expected_type == float and isinstance(x, int):
                x = float(x)
            if expected_type == int and isinstance(x, float) and x == int(x):
                x = int(x)

            for t in expected_types:
                if isinstance(x, t):
                    return cast(T, x)
            raise TypeError(f"Value for {key} is not of expected type {expected_type}")

    if default != no_default: return default
    raise ValueError(f"Missing any of the following keys: {keys}")


class ExLlamaV2Config:

    model_dir: str | None                       # Directory containing model files

    max_seq_len: int                            # Maximum sequence length. Sequences longer than this will throw an exception
    max_batch_size: int                         # Maximum size of batches to process
    max_input_len: int                          # Maximum length of input IDs in a single forward pass. Sequences longer than this will be processed in multiple steps
    max_attention_size: int                     # Sequences will be processed in chunks to keep the size of the attention weights matrix <= this
    max_output_len: int | None                  # Maximum number of output tokens per forward pass

    scale_pos_emb: float                        # Factor by which to scale positional embeddings, e.g. for 4096-token sequence use a scaling factor of 2.0, requires finetuned model or LoRA
    scale_alpha_value: float                    # Alpha value for NTK RoPE scaling. Similar to compress_pos_emb but works without finetuned model

    no_flash_attn: bool                         # Implementation will automatically use flash-attn-2 when available, set True to override
    no_xformers: bool                           # Implementation will automatically use xformers for sm<80 when available, unless flash-attn-2 is available, set True to override
    no_sdpa: bool                               # Do not use Torch SDPA even if causal_lower_right bias is available (seems to be unreliable on ROCm (?))
    load_in_q4: bool                            # Load float linear layers in Q4 format (for test/dev purposes, not performant)
    no_graphs: bool                             # Do not use CUDA graphs

    max_dq_size: int                            # Max number of elements to dequantize at once

    # Loaded/set by .prepare():

    architecture: str
    arch: ExLlamaV2ArchParams

    model_config: str
    tensor_file_map: dict
    tensor_files: list

    tokenizer_path: str

    bos_token_id: int
    eos_token_id: int
    pad_token_id: int

    hidden_size: int
    initializer_range: float
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    num_key_value_groups: int
    num_hidden_layers: int
    norm_eps: float | None
    vocab_size: int
    rotary_embedding_base: float
    scale_long_factor: list[float] | None
    scale_short_factor: list[float] | None
    alt_rope_method: str | None
    original_max_seq_len: int
    head_dim: int
    num_experts: int | None
    num_experts_per_token: int | None
    logit_scale: float
    scale_depth: float
    scale_emb: float
    use_qk_norm: bool
    query_pre_attn_scalar: float | None
    final_logit_softcapping: float | None
    attn_logit_softcapping: float | None
    sliding_window: int
    sliding_window_pattern: int
    norm_head: int | None
    l3_rope_factor: float | None
    l3_rope_low_freq_factor: float | None
    l3_rope_high_freq_factor: float | None
    l3_rope_original_max_position_embeddings: int | None
    yarn_rope_factor: float | None
    yarn_rope_original_max_position_embeddings: int | None
    checkpoint_fused_mlp: bool
    checkpoint_offset_qzeros: bool
    mrope_section: list | None
    attention_multiplier: float | None

    vision_model_type: str | None
    vision_head_dim: int | None
    vision_num_attention_heads: int | None
    vision_num_key_value_heads: int | None
    vision_num_key_value_groups: int | None
    vision_hidden_size: int | None
    vision_intermediate_size: int | None
    vision_hidden_act: str | None
    vision_rope_theta: float | None
    vision_feature_layer: int | None
    vision_patch_size: dict | None
    vision_image_mean: list | None
    vision_image_std: list | None
    vision_resample: int | None
    vision_rescale_factor: float | None
    vision_size: dict | None
    vision_num_channels: int | None
    vision_num_layers: int | None
    vision_spatial_merge_size: int | None
    vision_spatial_patch_size: int | None
    vision_min_pixels: int | None
    vision_max_pixels: int | None
    vision_temporal_patch_size: int | None
    vision_max_size: int | None

    # Deprecated fields, kept for compatibiltiy

    fasttensors: bool                           # Fasttensors loader removed in v0.2.3


    def __init__(self,
                 model_dir: str | None = None):
        """
        :param model_dir:
            If specified, initialize ExLlamaV2Config with values read from model config.
        """

        self.max_batch_size = 1
        self.max_input_len = 2048
        self.max_attention_size = 2048**2
        self.max_output_len = None
        self.scale_pos_emb = 1.0
        self.scale_alpha_value = 1.0
        self.scale_long_factor = None
        self.scale_short_factor = None
        self.alt_rope_method = None

        self.no_flash_attn = 'EXLLAMA_NO_FLASH_ATTN' in os.environ
        self.no_xformers = 'EXLLAMA_NO_XFORMERS' in os.environ
        self.no_sdpa = 'EXLLAMA_NO_SDPA' in os.environ
        self.load_in_q4 = False
        self.no_graphs = 'EXLLAMA_NO_GRAPHS' in os.environ

        if model_dir is not None:
            self.model_dir = model_dir
            self.prepare()
        else:
            self.model_dir = None

        self.max_dq_size = 512*(1024**2)


    # Set low-mem options

    def set_low_mem(self):

        self.max_input_len = 1024
        self.max_attention_size = 1024 ** 2
        self.max_output_len = None if self.max_output_len is None else min(self.max_output_len, 1024)


    # Populate config with required files from model_dir

    def prepare(self, no_tensors: bool = False):

        assert self.model_dir is not None, "No model_dir specified in ExLlamaV2Config"
        assert os.path.exists(self.model_dir), "Can't find " + self.model_dir

        # Load config.json

        self.model_config = os.path.join(self.model_dir, "config.json")
        assert os.path.exists(self.model_config), "Can't find " + self.model_config

        with open(self.model_config, encoding = "utf8") as f:
            read_config = json.load(f)

        # Load generation_config.json

        generation_config_path = os.path.join(self.model_dir, "generation_config.json")
        if os.path.exists(generation_config_path):
            with open(generation_config_path, encoding = "utf8") as f:
                gen_config = json.load(f)
                self.generation_config = {}
                try:
                    self.generation_config['eos_token_id'] = read(gen_config, list, "eos_token_id", None)
                except (ValueError, TypeError):
                    eos_token_id_as_int = read(gen_config, int, "eos_token_id", None)
                    if eos_token_id_as_int is not None:
                        self.generation_config['eos_token_id'] = [eos_token_id_as_int]
                    else:
                        self.generation_config['eos_token_id'] = None

        # Model architecture

        assert len(read_config["architectures"]) == 1, "Multiple architectures defined in config.json"
        self.architecture = read_config["architectures"][0]
        self.arch = ExLlamaV2ArchParams(self.architecture, read_config)

        # Vocab params

        self.bos_token_id = read(read_config, int, "bos_token_id", None)  # 1
        self.eos_token_id = read(read_config, [int, list], "eos_token_id", None)  # 2
        self.pad_token_id = read(read_config, int, "pad_token_id", None)  # 0
        self.vocab_size = read(read_config, int, "vocab_size", opt_subkey = "text_config")

        if isinstance(self.eos_token_id, list):
            self.eos_token_id = self.eos_token_id[0]  # TODO: Figure out a way to maybe use all the EOS tokens somehow

        # Standard params

        self.initializer_range = read(read_config, float, ["initializer_range"], 0.02)
        self.num_hidden_layers = read(read_config, int, ["num_hidden_layers", "n_layers", "n_layer"], opt_subkey = "text_config")

        # Norm params

        if self.arch.lm.keys["norm_eps"]:
            self.norm_eps = read(read_config, float, self.arch.lm.keys["norm_eps"], opt_subkey = "text_config")
        else:
            self.norm_eps = 1e-5  # Torch default

        # Model dimensions

        self.hidden_size = read(read_config, int, ["hidden_size", "d_model", "n_embd"], opt_subkey = "text_config")

        # Attn params

        self.num_attention_heads = read(read_config, int, ["num_attention_heads", "n_heads", "n_head"], 0, opt_subkey = "text_config")
        self.head_dim = read(
            read_config,
            int,
            "head_dim",
            (self.hidden_size // self.num_attention_heads) if self.num_attention_heads else no_default,
            opt_subkey = "text_config"
        )

        if not self.num_attention_heads:
            self.num_attention_heads = self.hidden_size // self.head_dim

        if self.arch.lm.mqa:
            self.num_key_value_heads = 1
        else:
            self.num_key_value_heads = read(
                read_config,
                int,
                ["num_key_value_heads", "attn_config->kv_n_heads"],
                self.num_attention_heads,
                opt_subkey = "text_config",
            )
            self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads

        self.use_qk_norm = read(read_config, bool, ["use_qk_norm"], False)

        self.query_pre_attn_scalar = read(read_config, float, "query_pre_attn_scalar", None)
        self.attention_multiplier = read(read_config, float, "attention_multiplier", None)

        # MLP params

        if self.arch.lm.default_inner_dim_mult is not None:
            default_intermediate_size = self.arch.lm.default_inner_dim_mult * self.hidden_size
        else:
            default_intermediate_size = no_default

# Deci overrides num_key_value_heads, num_key_value_groups and intermediate size
        if self.architecture == "DeciLMForCausalLM":
            if "block_configs" in read_config: # # Llama-3_1-Nemotron-51B
                _block_configs: list[dict[str,Any]] = read_config["block_configs"]
                assert self.num_hidden_layers == len(_block_configs)
                self.num_key_value_heads = list()
                self.num_key_value_groups = list()
                self.intermediate_size = list()
                self.arch.lm.layer_keys = list()
                for il in range(len(_block_configs)):
                    if _block_configs[il]["attention"]["n_heads_in_group"] is None:
                        if _block_configs[il]["attention"]["replace_with_linear"] is True:
                            self.num_key_value_heads.append(0)
                            self.arch.lm.layer_keys.append([["input_layernorm"],["post_attention_layernorm"],["self_attn.linear_attn"],["mlp.down_proj"],["mlp.gate_proj"],["mlp.up_proj"]])
                        else:
                            self.num_key_value_heads.append(0)
                            self.arch.lm.layer_keys.append([["mlp.down_proj"],["mlp.gate_proj"],["mlp.up_proj"]])
                    else:
                        self.num_key_value_heads.append(self.num_attention_heads // _block_configs[il]["attention"]["n_heads_in_group"])
                        self.arch.lm.layer_keys.append([["input_layernorm"],["post_attention_layernorm"],["self_attn.q_proj"], ["self_attn.k_proj"],["self_attn.v_proj"],["self_attn.o_proj"],["mlp.down_proj"],["mlp.gate_proj"],["mlp.up_proj"]])
                    if self.num_key_value_heads[il] == 0:
                        self.num_key_value_groups.append(0)
                    else:
                        self.num_key_value_groups.append(self.num_attention_heads // self.num_key_value_heads[il])
                    ffn_mult = _block_configs[il]["ffn"]["ffn_mult"]
                    intm_size = int(2 * ffn_mult * self.hidden_size / 3)
                    if intm_size % 256 != 0:
                        intm_size = intm_size + 256 - (intm_size % 256)
                    self.intermediate_size.append(intm_size)
            else: # Deci-7B, no need to override intermediate_size
                self.num_key_value_heads: list[int] = read_config["num_key_value_heads_per_layer"]
                self.num_key_value_groups = list()
                for il in range(len(self.num_key_value_heads)):
                    self.num_key_value_groups.append(self.num_attention_heads // self.num_key_value_heads[il])
                self.intermediate_size = read(
                    read_config,
                    int,
                    ["intermediate_size", "ffn_config->ffn_hidden_size", "n_inner"],
                    default_intermediate_size,
                    opt_subkey = "text_config",
                )
        else:
            self.intermediate_size = read(
                read_config,
                int,
                ["intermediate_size", "ffn_config->ffn_hidden_size", "n_inner"],
                default_intermediate_size,
                opt_subkey = "text_config",
            )
        self.num_experts = read(read_config, int, ["num_local_experts", "ffn_config->moe_num_experts"], None)
        self.num_experts_per_token = read(read_config, int,["num_experts_per_tok", "ffn_config->moe_top_k"], None)

        # Logit/embedding/residual scale

        self.logit_scale = read(read_config, float, "logit_scale", 1)
        if self.arch.lm.logit_scale_basedim:
            dim_model_base = read(read_config, int, "dim_model_base", self.hidden_size)
            self.logit_scale /= (self.hidden_size / dim_model_base)

        logit_scaling = read(read_config, float, "logits_scaling", None)  # Granite is backwards
        if logit_scaling:
            self.logit_scale = 1.0 / logit_scaling

        self.scale_emb = read(read_config, float, ["scale_emb", "embedding_multiplier"], 1)
        residual_multiplier = read(read_config, float, "residual_multiplier", None)
        scale_depth = read(read_config, float, "scale_depth", None)
        self.scale_depth = 1
        if residual_multiplier:
            self.scale_depth = residual_multiplier
        elif scale_depth:
            self.scale_depth = scale_depth / math.sqrt(self.num_hidden_layers)

        self.attn_logit_softcapping = read(read_config, float, "attn_logit_softcapping", None)
        self.final_logit_softcapping = read(read_config, float, "final_logit_softcapping", None)

        # Normalize weights in head layer

        self.norm_head = read(read_config, int, "norm_head", None)

        # Positional embeddings

        self.rotary_embedding_base = read(
            read_config,
            float,
            ["rope_theta", "attn_config->rope_theta"],
            10000.0,
            opt_subkey = "text_config",
        )

        self.max_seq_len = read(
            read_config,
            int,
            ["max_sequence_length", "model_max_length", "max_position_embeddings", "max_seq_len", "n_positions"],
            2048,
            opt_subkey = "text_config"
        )
        self.original_max_seq_len = self.max_seq_len

        self.sliding_window = read(read_config, int, ["sliding_window", "sliding_window_size"], 0, opt_subkey = "text_config")
        self.sliding_window_pattern = read(read_config, int, ["sliding_window_pattern"], 1)

        rs = read(read_config, dict, "rope_scaling", None)
        if rs:
            scaling_type = rs.get("type", None)
            rope_type = rs.get("rope_type", None)
            assert not (scaling_type and rope_type), "rope_scaling key has both `type` and `rope_type` subkeys"
            if scaling_type == "linear":
                assert "factor" in rs, "'factor' missing from 'rope_scaling' config"
                self.scale_pos_emb = rs.get("factor", 1.0)
            if scaling_type == "su" or scaling_type == "longrope":
                assert "long_factor" in rs, "'long_factor' missing from 'rope_scaling' config ('su' mode)"
                assert "short_factor" in rs, "'short_factor' missing from 'rope_scaling' config ('su' mode)"
                assert "original_max_position_embeddings" in read_config, \
                    "'original_max_position_embeddings' required for 'su' scaling"
                self.scale_long_factor = rs["long_factor"]
                self.scale_short_factor = rs["short_factor"]
                self.original_max_seq_len = read_config["original_max_position_embeddings"]
                self.alt_rope_method = "su"
            if scaling_type == "yarn":
                self.alt_rope_method = "yarn"
                self.yarn_rope_factor = rs["factor"]
                self.yarn_rope_original_max_position_embeddings = rs["original_max_position_embeddings"]
            if rope_type == "llama3":
                self.alt_rope_method = "llama3"
                self.l3_rope_factor = rs["factor"]
                self.l3_rope_low_freq_factor = rs["low_freq_factor"]
                self.l3_rope_high_freq_factor = rs["high_freq_factor"]
                self.l3_rope_original_max_position_embeddings = rs["original_max_position_embeddings"]
            if scaling_type == "mrope":
                self.mrope_section = rs["mrope_section"]


        # Checkpoint format (for GPTQ models)

        checkpoint_format = read(read_config, str, ["quantization_config->checkpoint_format"], None)
        self.checkpoint_offset_qzeros = (checkpoint_format == "gptq_v2")

        # Create map of model tensors

        if no_tensors: return

        self.tensor_file_map = {}

        st_pattern = os.path.join(self.model_dir, "*.safetensors")
        self.tensor_files = glob.glob(st_pattern)

        if len(self.tensor_files) == 0:
            raise ValueError(f" ## No .safetensors files found in {self.model_dir}")

        for st_file in self.tensor_files:
            f = STFile.open(st_file, keymap = self.arch.keymap)
            for key in f.get_dict():
                self.tensor_file_map[key] = st_file

        # For loading checkpoints with fused MLP layers

        if "model.layers.0.mlp.down_proj.weight" not in self.tensor_file_map and \
            "model.layers.0.mlp.swiglu.w12.weight" in self.tensor_file_map:
            self.checkpoint_fused_mlp = True
            self.arch.make_fused_mlp()
        else:
            self.checkpoint_fused_mlp = False

        # Make sure we found all the layers we need

        def check_keys(archparams, prefix):

            expect_keys = archparams.expect_keys.copy()

            if not self.num_experts or self.num_experts == 1:
                per_layer_keys = archparams.layer_keys
            else:
                per_layer_keys = set()
                for expert_idx in range(self.num_experts):
                    for k in archparams.layer_keys:
                        skt = [sk.replace(".*.", f".{expert_idx}.") for sk in k]
                        per_layer_keys.add(tuple(skt))
                per_layer_keys = list(per_layer_keys)

            for layer_idx in range(self.num_hidden_layers):
                for ks in per_layer_keys:
                    prefixes = [f"model.layers.{layer_idx}.{k}" for k in ks]
                    expect_keys.append(prefixes)

            if self.arch.lm_prefix:
                expect_keys = [
                    [prefix + k for k in k2]
                    for k2 in expect_keys
                ]

            all_keys = set(self.tensor_file_map.keys())
            suffixes = [".q_weight", ".qweight", ".weight", ""]

#            for k in all_keys:
#                print(k)
#            print("****End of all_keys****")

            for prefixes in expect_keys:
#                print(prefixes)
                match = False
                for prefix in prefixes:
                    for suffix in suffixes:
                        if (prefix + suffix) in all_keys:
                            match = True
                            break
                        if match: break
                    if match: break
                if not match:
                    raise ValueError(f" ## Could not find {prefix}.* in model")

        def check_deci_keys(archparams, prefix):
            expect_keys = archparams.expect_keys.copy()

            per_layer_keys = archparams.layer_keys

            for layer_idx in range(self.num_hidden_layers):
                for ks in per_layer_keys[layer_idx]:
                    prefixes = [f"model.layers.{layer_idx}.{k}" for k in ks]
                    expect_keys.append(prefixes)

            if self.arch.lm_prefix:
                expect_keys = [
                    [prefix + k for k in k2]
                    for k2 in expect_keys
                ]

            all_keys = set(self.tensor_file_map.keys())
            suffixes = [".q_weight", ".qweight", ".weight", ""]

#            for k in all_keys:
#                print(k)
#            print("****End of all_keys****")

            for prefixes in expect_keys:
                match = False
                for prefix in prefixes:
                    for suffix in suffixes:
                        if (prefix + suffix) in all_keys:
                            match = True
                            break
                        if match: break
                    if match: break
                if not match:
                    raise ValueError(f" ## Could not find {prefix}.* in model")

        if self.architecture == "DeciLMForCausalLM" and "block_configs" in read_config: # # Llama-3_1-Nemotron-51B
            check_deci_keys(self.arch.lm, self.arch.lm_prefix)
        else:
            check_keys(self.arch.lm, self.arch.lm_prefix)
        check_keys(self.arch.mmp, self.arch.mmp_prefix)
        check_keys(self.arch.vt, self.arch.vt_prefix)

        # Vision models

        self.vision_model_type = read(read_config, str, "vision_config->model_type", None)

        if self.vision_model_type:
            self.model_config = os.path.join(self.model_dir, "preprocessor_config.json")
            assert os.path.exists(self.model_config), "Can't find " + self.model_config
            with open(self.model_config, encoding = "utf8") as f:
                read_prep_config = json.load(f)

        # TODO: Cleanup & refactor

        if self.vision_model_type is None:
            pass

        elif self.vision_model_type == "pixtral":
            self.vision_head_dim = read(read_config, int, ["vision_config->head_dim"], no_default)
            self.vision_num_attention_heads = read(read_config, int, ["vision_config->num_attention_heads"], no_default)
            self.vision_num_key_value_heads = read(read_config, int, ["vision_config->num_key_value_heads"], self.vision_num_attention_heads)
            self.vision_num_key_value_groups = self.vision_num_attention_heads // self.vision_num_key_value_heads
            self.multimodal_projector_bias = read(read_config, bool, ["multimodal_projector_bias"], True)

            self.vision_hidden_act = read(read_config, str, ["vision_config->hidden_act"], no_default)
            self.vision_hidden_size = read(read_config, int, ["vision_config->hidden_size"], 1024)
            patch_size = read(read_config, int, ["vision_config->patch_size"], no_default)
            self.vision_rope_theta = read(read_config, int, ["vision_config->rope_theta"], no_default)
            self.vision_feature_layer = read(read_config, int, ["vision_feature_layer"], no_default)
            self.vision_num_layers = read(read_config, int, ["vision_config->num_hidden_layers"], 24)
            self.vision_intermediate_size = read(read_config, int, ["vision_config->intermediate_size"], self.hidden_size)

            image_processor_type = read(read_prep_config, str, ["image_processor_type"], no_default)
            assert image_processor_type == "PixtralImageProcessor", \
                f"Wrong image processor type: {image_processor_type}"
            self.vision_image_mean = read(read_prep_config, list, ["image_mean"], no_default)
            self.vision_image_std = read(read_prep_config, list, ["image_std"], no_default)
            self.vision_patch_size = read(read_prep_config, dict, ["patch_size"], no_default)
            assert all(self.vision_patch_size.get(x) == patch_size for x in ["width", "height"]), \
                "Patch size inconsistency between config.json and preprocessor_config.json"
            self.vision_resample = read(read_prep_config, int, ["resample"], no_default)
            self.vision_rescale_factor = read(read_prep_config, float, ["rescale_factor"], no_default)
            self.vision_size = read(read_prep_config, dict, ["size"], no_default)
            self.vision_num_channels = 3
            self.vision_spatial_merge_size = 1
            self.vision_max_size = 16384

        elif self.vision_model_type == "qwen2":
            self.vision_num_attention_heads = read(read_config, int, ["vision_config->num_heads"], no_default)
            self.vision_num_key_value_heads = self.vision_num_attention_heads
            self.vision_hidden_size = read(read_config, int, ["vision_config->embed_dim"], no_default)
            self.vision_head_dim = self.vision_hidden_size // self.vision_num_attention_heads
            self.vision_num_key_value_groups = 1
            self.vision_hidden_act = "quickgelu"
            self.vision_spatial_merge_size = read(read_config, int, ["vision_config->spatial_merge_size"], no_default)
            self.vision_spatial_patch_size = read(read_config, int, ["vision_config->spatial_patch_size"], no_default)
            patch_size = read(read_config, int, ["vision_config->patch_size"], no_default)
            self.vision_rope_theta = read(read_config, int, ["vision_config->rope_theta"], 10000.0)
            self.vision_num_layers = read(read_config, int, ["vision_config->depth"], no_default)
            mlp_ratio = read(read_config, int, ["vision_config->mlp_ratio"], no_default)
            self.vision_intermediate_size = self.vision_hidden_size * mlp_ratio

            image_processor_type = read(read_prep_config, str, ["image_processor_type"], no_default)
            assert image_processor_type == "Qwen2VLImageProcessor", \
                f"Wrong image processor type: {image_processor_type}"
            self.vision_image_mean = read(read_prep_config, list, ["image_mean"], no_default)
            self.vision_image_std = read(read_prep_config, list, ["image_std"], no_default)
            assert read(read_prep_config, int, ["patch_size"], no_default) == patch_size, \
                "Incorrect patch size in preprocessor_config.json"
            self.vision_patch_size = {"height": patch_size, "width": patch_size}
            self.vision_temporal_patch_size = read(read_prep_config, int, ["temporal_patch_size"], no_default)
            self.vision_resample = 3
            self.vision_rescale_factor = None
            assert read(read_prep_config, int, ["merge_size"], no_default) == self.vision_spatial_merge_size, \
                "Incorrect merge size in preprocessor_config.json"
            self.vision_rescale_factor = read(read_prep_config, float, ["rescale_factor"], 0.00392156862745098)
            self.vision_num_channels = 3
            self.vision_min_pixels = read(read_prep_config, int, ["min_pixels"], no_default)
            self.vision_max_pixels = read(read_prep_config, int, ["max_pixels"], no_default)
            self.vision_max_size = 16384

        else:
            raise ValueError(f"Unsupported vision model type: {self.vision_model_type}")

        # Cleanup

        cleanup_stfiles()


    def arch_compat_overrides(self, quiet: bool = False, warn_only = False):

        from exllamav2.attn import (
            has_flash_attn,
            has_flash_attn_with_window,
            has_flash_attn_with_softcap,
            has_xformers
        )

        warnings = []

        if self.arch.lm.eager_attn_only:
            warnings.append(" !! Warning: Architecture currently supports only eager attention")
            if not warn_only:
                warnings.append(" !! Warning: flash-attn, xformers and SDPA are disabled")
                self.no_flash_attn = True
                self.no_xformers = True
                self.no_sdpa = True
            else:
                warnings.append(" !! Warning: flash-attn, xformers and SDPA should be disabled for correct inference")

        if has_flash_attn and not self.no_flash_attn:
            disable = False
            if self.attn_logit_softcapping and not has_flash_attn_with_softcap:
                warnings.append(" !! Warning: model requires softcap, not supported in installed version of flash-attn")
                disable = True
            if (self.arch.lm.swa or self.arch.lm.alternating_swa) and not has_flash_attn_with_window:
                warnings.append(" !! Warning: model requires SWA, not supported in installed version of flash-attn")
                disable = True
            if disable and not warn_only:
                warnings.append(" !! Warning: disabling flash-attn")
                self.no_flash_attn = True

        if has_xformers and not self.no_xformers:
            disable = False
            if self.attn_logit_softcapping:
                warnings.append(" !! Warning: model requires softcap, not supported in xformers")
                disable = True
            if self.arch.lm.swa or self.arch.lm.alternating_swa:
                warnings.append(" !! Warning: model requires SWA, not supported in xformers")
                disable = True
            if disable and not warn_only:
                warnings.append(" !! Warning: disabling xformers")
                self.no_xformers = True

        if not quiet:
            for w in warnings:
                print(w)
