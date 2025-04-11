from dataclasses import dataclass, field
from enum import IntEnum

# Common keys

layer_keys_llama_norms = [["input_layernorm"],
                          ["post_attention_layernorm"]]
layer_keys_cohere_norms = [["input_layernorm"]]
layer_keys_gpt2_norms = [["ln_1"],
                         ["ln_2"]]
layer_keys_yi_norms = [["ln1", "input_layernorm"],
                       ["ln2", "post_attention_layernorm"]]
layer_keys_gemma2_norms = [["input_layernorm"],
                           ["post_attention_layernorm"],
                           ["pre_feedforward_layernorm"],
                           ["post_feedforward_layernorm"]]
layer_keys_internlm2_norms = [["attention_norm"],
                              ["ffn_norm"]]
layer_keys_llama_attn = [["self_attn.q_proj"],
                         ["self_attn.k_proj"],
                         ["self_attn.v_proj"],
                         ["self_attn.o_proj"]]
layer_keys_gpt2_attn = [["self_attn.c_attn", "self_attn.q_proj"],
                        ["self_attn.c_attn", "self_attn.k_proj"],
                        ["self_attn.c_attn", "self_attn.v_proj"],
                        ["self_attn.o_proj"]]
layer_keys_internlm2_attn = [["self_attn.wqkv", "self_attn.q_proj"],
                             ["self_attn.wqkv", "self_attn.k_proj"],
                             ["self_attn.wqkv", "self_attn.v_proj"],
                             ["self_attn.o_proj"]]
layer_keys_dbrx_attn = [["self_attn.Wqkv", "self_attn.q_proj"],
                        ["self_attn.Wqkv", "self_attn.k_proj"],
                        ["self_attn.Wqkv", "self_attn.v_proj"],
                        ["self_attn.o_proj"]]
layer_keys_phi3_attn = [["self_attn.qkv_proj", "self_attn.q_proj"],
                        ["self_attn.qkv_proj", "self_attn.k_proj"],
                        ["self_attn.qkv_proj", "self_attn.v_proj"],
                        ["self_attn.o_proj"]]
layer_keys_llama_mlp = [["mlp.down_proj"],
                        ["mlp.gate_proj"],
                        ["mlp.up_proj"]]
layer_keys_internlm2_mlp = [["feed_forward.w1"],
                            ["feed_forward.w2"],
                            ["feed_forward.w3"]]
layer_keys_phi3_mlp = [["mlp.down_proj"],
                       ["mlp.gate_up_proj", "mlp.gate_proj"],
                       ["mlp.gate_up_proj", "mlp.up_proj"]]
layer_keys_mixtral_mlp = [["block_sparse_moe.experts.*.w1"],
                          ["block_sparse_moe.experts.*.w2"],
                          ["block_sparse_moe.experts.*.w3"],
                          ["block_sparse_moe.gate"]]
layer_keys_dbrx_mlp = [["block_sparse_moe.experts.*.v1", "block_sparse_moe.experts.v1"],
                       ["block_sparse_moe.experts.*.w1", "block_sparse_moe.experts.w1"],
                       ["block_sparse_moe.experts.*.w2", "block_sparse_moe.experts.w2"],
                       ["block_sparse_moe.gate"]]
layer_keys_llama_mlp_swiglu = [["mlp.swiglu.w12"],
                               ["mlp.swiglu.w3"]]
layer_keys_starcoder2_mlp = [["mlp.c_fc"],
                             ["mlp.c_proj"]]
layer_keys_gpt2_mlp = [["mlp.c_fc"],
                       ["mlp.c_proj"]]
expect_keys_llama = [["lm_head"],
                     ["model.norm"],
                     ["model.embed_tokens"]]
expect_keys_gemma = [["model.norm"],
                     ["model.embed_tokens"]]
expect_keys_starcoder2 = [["model.norm"],
                          ["model.embed_tokens"]]
expect_keys_gpt2 = [["model.embed_tokens"]]

dbrx_keymap = [("transformer.", "model."),
               (".blocks.", ".layers."),
               (".ffn.experts.mlp.", ".block_sparse_moe.experts."),
               (".ffn.router.layer.", ".block_sparse_moe.gate."),
               (".norm_attn_norm.norm_1.", ".input_layernorm."),
               (".norm_attn_norm.norm_2.", ".post_attention_layernorm."),
               (".norm_attn_norm.attn.", ".self_attn."),
               (".out_proj.", ".o_proj."),
               (".norm_f.", ".norm."),
               (".wte.", ".embed_tokens.")]
bigcode_keymap = [("transformer.ln_f", "model.norm"),
                  ("transformer.", "model."),
                  (".attn.c_proj.", ".self_attn.o_proj."),
                  (".attn.", ".self_attn."),
                  (".h.", ".layers."),
                  (".wte.", ".embed_tokens.")]
gpt2_keymap = [("$ln_f.", "model.norm."),
               (".attn.c_proj.", ".self_attn.o_proj."),
               (".attn.", ".self_attn."),
               ("$h.", "model.layers."),
               ("$wte.", "model.embed_tokens."),
               ("$wpe.", "model.wpe.")]
internlm2_keymap = [("$output.", "lm_head."),
                    ("$model.tok_embeddings.", "model.embed_tokens."),
                    (".attention.", ".self_attn."),
                    (".wo.", ".o_proj.")]

class RopeStyle(IntEnum):
    NONE = 0
    GPTJ = 1
    NEOX = 2

class ExLlamaV2ArchParams:

    def __init__(self, arch_string: str, read_config: dict):
        """
        Get architecture definition from model config. If the architecture isn't recognized, defaults to Llama
        architecture.

        :param arch_string:
            Architecture string from config.json

        :param read_config:
            config.json as Python dict
        """

        self.arch_string = arch_string
        arch_recognized = False

        self.keymap = None

        @dataclass
        class Params:
            keys: dict = field(default_factory = lambda: {
                "norm_eps": "rms_norm_eps",
                "norm_1": ".input_layernorm",
                "norm_1_post": None,
                "fused_qkv": None,
                "mlp_gate": ".mlp.gate_proj",
                "mlp_up": ".mlp.up_proj",
                "mlp_down": ".mlp.down_proj",
                "lm_head": "lm_head",
                "norm_2": ".post_attention_layernorm",
                "norm_2_post": None,
                "fused_mlp_12": None,
                "fused_mlp_3": None,
                "learned_pos_emb": None,
                "attn_q": ".self_attn.q_proj",
                "attn_k": ".self_attn.k_proj",
                "attn_v": ".self_attn.v_proj",
                "attn_o": ".self_attn.o_proj",
                "linear_attn": ".self_attn.linear_attn",
                "layers": "layers",
                "patch_conv": "patch_conv",
            })

            # Compute logit scale from `dim_model_base` key in config.json (MiniCPM quirk)
            logit_scale_basedim = False

            # Clamp hidden states to FP16 range
            clamp_hidden_states = False

            # Upcast hidden state to FP32 before adding to residual stream
            residual_stream_fp32 = False

            # Normalize embeddings (Gemma quirk)
            normalize_embeddings = False

            # Constant bias for layernorm (Gemma quirk)
            norm_constant_bias = 0

            # Alternate packing scheme for fused QKV tensor (InternLM2 quirk)
            fused_qkv_altpack = False

            # SWA required by architecture
            swa = False
            alternating_swa = False

            # Model only works with eager attention
            eager_attn_only = False

            # Expect bias for linear layers
            attention_bias_qkv = False
            attention_bias_o = False
            mlp_bias = False

            # Default multiplier for MLP inner dim (GPT2 quirk)
            default_inner_dim_mult = None

            # Use gated MLP
            mlp_gate = True

            # Use block-sparse MLP
            is_moe = False

            # Use parallel decoder blocks (Cohere quirk)
            parallel_decoder_blocks = False

            # Use MQA, effectively num_key_value_heads = 1 (GPTBigCode quirk)
            mqa = False

            # Model is incoherent without BOS at the start of the context
            requires_bos = False

            # Scale attn weights (GPT2 quirk, not important for inference)
            scale_attn_weights = False

            # Model implementation works in tensor-parallel mode
            supports_tp = False

            # Activation function
            mlp_act_func = "silu"

            # Layer norm type
            norm = "rmsnorm"

            # RoPE style
            rope_style = RopeStyle.NEOX

            # Expected keys
            expect_keys: list[str] = field(default_factory = lambda: [])
            layer_keys: list[str] = field(default_factory = lambda: [])

            # Vision stuff
            patch_conv_bias: bool = False
            is_vision: bool = False
            vision_input_norm: bool = True
            vision_conv3d: bool = False
            mrope: bool = False
            rope_freq_half: bool = False

        # Component models
        self.lm_prefix = ""
        self.vt_prefix = ""
        self.mmp_prefix = ""
        self.lm = Params()
        self.mmp = Params()
        self.vt = Params()

        self.mmp.keys.update({
            "norm_1": None,
            "norm_1_post": None,
            "norm_2": None,
            "norm_2_post": None,
            "fused_mlp_12": None,
            "fused_mlp_3": None,
        })
        self.mmp.rope_style = RopeStyle.NONE

        self.vt.is_vision = True

        # Tensors are transposed in original model weights
        self.orig_weights_transposed = False

        # Add noise rows to calibration while quantizing
        self.standard_calib_noise = None

        # Mistral

        if arch_string == "MistralForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.supports_tp = True

        # Mixtral

        if arch_string == "MixtralForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_mixtral_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.keys.update({
                "mlp_gate": ".block_sparse_moe.experts.*.w1",
                "mlp_up": ".block_sparse_moe.experts.*.w3",
                "mlp_down": ".block_sparse_moe.experts.*.w2",
                "mlp_expert_gate": ".block_sparse_moe.gate"
            })
            self.lm.is_moe = True

        # Pixtral

        if (
            arch_string == "LlavaForConditionalGeneration" and
            "vision_config" in read_config and
            read_config["vision_config"].get("model_type") == "pixtral"
        ):
            arch_recognized = True
            self.lm_prefix = "language_model."
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama

            self.vt_prefix = "vision_tower."
            self.vt.keys.update({
                "attn_q": ".attention.q_proj",
                "attn_k": ".attention.k_proj",
                "attn_v": ".attention.v_proj",
                "attn_o": ".attention.o_proj",
                "mlp_gate": ".feed_forward.gate_proj",
                "mlp_up": ".feed_forward.up_proj",
                "mlp_down": ".feed_forward.down_proj",
                "norm_1": ".attention_norm",
                "norm_2": ".ffn_norm",
                "layers": "transformer.layers",
            })

            self.mmp_prefix = "multi_modal_projector."
            self.mmp.keys.update({
                "mlp_gate": None,
                "mlp_up": "linear_1",
                "mlp_down": "linear_2",
            })
            self.mmp.mlp_gate = False
            self.mmp.mlp_act_func = "gelu"
            self.mmp.mlp_bias = bool(read_config.get("multimodal_projector_bias", True))

        # Yi

        if arch_string == "YiForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_yi_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.keys.update({
                "norm_1": ".ln1",
                "norm_2": ".ln2",
            })

        # Orion

        if arch_string == "OrionForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.norm = "layernorm"

        # Qwen2 (1.5, 2, 2.5)

        if arch_string == "Qwen2ForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.attention_bias_qkv = True
            self.lm.supports_tp = True

        # Qwen2-VL (2, 2.5)

        if arch_string in ["Qwen2VLForConditionalGeneration", "Qwen2_5_VLForConditionalGeneration"]:
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.attention_bias_qkv = True
            self.lm.mrope = True
            self.lm.rope_freq_half = True

            self.vt_prefix = "visual."
            if arch_string == "Qwen2VLForConditionalGeneration":
                read_config["vision_config"].update({"model_type": "qwen2"})
                self.vt.keys.update({
                    "fused_qkv": ".attn.qkv",
                    "attn_o": ".attn.proj",
                    "mlp_gate": None,
                    "mlp_up": ".mlp.fc1",
                    "mlp_down": ".mlp.fc2",
                    "norm_1": ".norm1",
                    "norm_2": ".norm2",
                    "layers": "blocks",
                    "patch_conv": "patch_embed.proj",
                })
                self.vt.mlp_gate = False
                self.vt.mlp_act_func = "quickgelu"
                self.vt.norm = "layernorm"
            elif arch_string == "Qwen2_5_VLForConditionalGeneration":
                read_config["vision_config"].update({"model_type": "qwen2.5"})
                self.vt.keys.update({
                    "fused_qkv": ".attn.qkv",
                    "attn_o": ".attn.proj",
                    "mlp_gate": ".mlp.gate_proj",
                    "mlp_up": ".mlp.up_proj",
                    "mlp_down": ".mlp.down_proj",
                    "norm_1": ".norm1",
                    "norm_2": ".norm2",
                    "layers": "blocks",
                    "patch_conv": "patch_embed.proj",
                })
                self.vt.mlp_gate = True
                self.vt.mlp_act_func = "silu"
                self.vt.norm = "rmsnorm"
            self.vt.mlp_bias = True
            self.vt.attention_bias_qkv = True
            self.vt.attention_bias_o = True
            self.vt.vision_input_norm = False
            self.vt.vision_conv3d = True

            self.mmp_prefix = "visual.merger."
            self.mmp.keys.update({
                "mlp_gate": None,
                "mlp_up": "mlp.0",
                "mlp_down": "mlp.2",
                "norm_2": "ln_q",
            })
            self.mmp.mlp_gate = False
            self.mmp.mlp_act_func = "gelu"
            self.mmp.mlp_bias = True
            self.mmp.norm = "layernorm"

            self.standard_calib_noise = (5, 30)

        # Gemma

        if arch_string == "GemmaForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_gemma
            self.lm.keys.update({
                "lm_head": "model.embed_tokens",
            })
            self.lm.mlp_act_func = "gelu"
            self.lm.normalize_embeddings = True
            self.lm.norm_constant_bias = 1
            self.lm.requires_bos = True

        # Gemma2

        if arch_string == "Gemma2ForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_gemma2_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_gemma
            self.lm.keys.update({
                "lm_head": "model.embed_tokens",
                "norm_1": ".input_layernorm",
                "norm_1_post": ".post_attention_layernorm",
                "norm_2": ".pre_feedforward_layernorm",
                "norm_2_post": ".post_feedforward_layernorm",
            })
            self.lm.mlp_act_func = "gelu"
            self.lm.normalize_embeddings = True
            self.lm.norm_constant_bias = 1
            self.lm.requires_bos = True
            self.lm.alternating_swa = True
            self.lm.residual_stream_fp32 = True

        # StarCoder2

        if arch_string == "Starcoder2ForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_starcoder2_mlp
            self.lm.expect_keys += \
                expect_keys_starcoder2
            self.lm.keys.update({
                "mlp_gate": None,
                "mlp_up": ".mlp.c_fc",
                "mlp_down": ".mlp.c_proj",
                "lm_head": "model.embed_tokens",
                "norm_eps": "layer_norm_epsilon",
            })
            self.lm.mlp_act_func = "gelu"
            self.lm.norm = "layernorm"
            self.lm.attention_bias_qkv = True
            self.lm.attention_bias_o = True
            self.lm.mlp_bias = True
            self.lm.mlp_gate = False

        # GemMoE

        if arch_string == "GemmoeForCausalLM":
            arch_recognized = True
            print(f" !! Warning, Gemmoe support is experimental and has not been fully tested")
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_mixtral_mlp
            self.lm.expect_keys += \
                expect_keys_gemma
            self.lm.keys.update({
                "mlp_gate": ".block_sparse_moe.experts.*.w1",
                "mlp_up": ".block_sparse_moe.experts.*.w3",
                "mlp_down": ".block_sparse_moe.experts.*.w2",
                "mlp_expert_gate": ".block_sparse_moe.gate",
                "lm_head": "model.embed_tokens",
            })
            self.lm.mlp_act_func = "gelu"
            self.lm.normalize_embeddings = True
            self.lm.norm_constant_bias = 1
            self.lm.is_moe = True
            self.lm.requires_bos = True

        # Cohere

        if arch_string == "CohereForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_cohere_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_gemma
            self.lm.keys.update({
                "norm_eps": "layer_norm_eps",
                "lm_head": "model.embed_tokens",
                "norm_1": ".input_layernorm",
                "norm_2": None,
            })
            self.lm.norm = "layernorm"
            self.lm.rope_style = RopeStyle.GPTJ
            self.lm.parallel_decoder_blocks = True
            self.lm.requires_bos = True

        # Cohere 2

        if arch_string == "Cohere2ForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_cohere_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_gemma
            self.lm.keys.update({
                "norm_eps": "layer_norm_eps",
                "lm_head": "model.embed_tokens",
                "norm_1": ".input_layernorm",
                "norm_2": None,
            })
            self.lm.norm = "layernorm"
            self.lm.rope_style = RopeStyle.GPTJ
            self.lm.parallel_decoder_blocks = True
            self.lm.requires_bos = True
            self.lm.alternating_swa = True

        # DBRX

        if arch_string == "DbrxForCausalLM":
            arch_recognized = True
            self.keymap = dbrx_keymap
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_dbrx_attn + \
                layer_keys_dbrx_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.keys.update({
                "norm_eps": None,
                "mlp_gate": ".block_sparse_moe.experts.*.w1",
                "mlp_up": ".block_sparse_moe.experts.*.v1",
                "mlp_down": ".block_sparse_moe.experts.*.w2",
                "mlp_expert_gate": ".block_sparse_moe.gate",
                "fused_qkv": ".self_attn.Wqkv",
            })
            self.lm.norm = "layernorm"
            self.lm.is_moe = True

        # Phi3

        if arch_string == "Phi3ForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_phi3_attn + \
                layer_keys_phi3_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.keys.update({
                "fused_qkv": ".self_attn.qkv_proj",
                "fused_mlp_12": "gate_up_proj",
            })

        # GPTBigCode

        if arch_string == "GPTBigCodeForCausalLM":
            arch_recognized = True
            self.keymap = bigcode_keymap
            self.lm.layer_keys += \
                layer_keys_gpt2_norms + \
                layer_keys_gpt2_attn + \
                layer_keys_gpt2_mlp
            self.lm.expect_keys += \
                expect_keys_gpt2
            self.lm.keys.update({
                "norm_eps": "layer_norm_epsilon",
                "mlp_gate": None,
                "mlp_up": ".mlp.c_fc",
                "mlp_down": ".mlp.c_proj",
                "lm_head": "model.embed_tokens",
                "norm_1": ".ln_1",
                "norm_2": ".ln_2",
                "fused_qkv": ".self_attn.c_attn",
                "learned_pos_emb": "model.wpe",
            })
            self.lm.mlp_act_func = "gelu"
            self.lm.norm = "layernorm"
            self.lm.rope_style = RopeStyle.NONE
            self.lm.mqa = True
            self.lm.attention_bias_qkv = True
            self.lm.attention_bias_o = True
            self.lm.mlp_bias = True
            self.lm.mlp_gate = False

        # GPT2

        if arch_string == "GPT2LMHeadModel":
            arch_recognized = True
            self.keymap = gpt2_keymap
            self.lm.layer_keys += \
                layer_keys_gpt2_norms + \
                layer_keys_gpt2_attn + \
                layer_keys_gpt2_mlp
            self.lm.expect_keys += \
                expect_keys_gpt2
            self.lm.keys.update({
                "norm_eps": "layer_norm_epsilon",
                "mlp_gate": None,
                "mlp_up": ".mlp.c_fc",
                "mlp_down": ".mlp.c_proj",
                "lm_head": "model.embed_tokens",
                "norm_1": ".ln_1",
                "norm_2": ".ln_2",
                "fused_qkv": ".self_attn.c_attn",
                "learned_pos_emb": "model.wpe",
            })
            self.lm.mlp_act_func = "gelu"
            self.lm.norm = "layernorm"
            self.lm.rope_style = RopeStyle.NONE
            self.lm.default_inner_dim_mult = 4
            self.lm.attention_bias_qkv = True
            self.lm.attention_bias_o = True
            self.lm.mlp_bias = True
            self.lm.mlp_gate = False
            self.orig_weights_transposed = True

        # MiniCPM

        if arch_string == "MiniCPMForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.logit_scale_basedim = True

        # InternLM2

        if arch_string == "InternLM2ForCausalLM":
            arch_recognized = True
            self.keymap = internlm2_keymap
            self.lm.layer_keys += \
                layer_keys_internlm2_norms + \
                layer_keys_internlm2_attn + \
                layer_keys_internlm2_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.keys.update({
                "mlp_gate": ".feed_forward.w1",
                "mlp_up": ".feed_forward.w3",
                "mlp_down": ".feed_forward.w2",
                "norm_1": ".attention_norm",
                "norm_2": ".ffn_norm",
                "fused_qkv": ".self_attn.wqkv",
            })
            self.lm.fused_qkv_altpack = True

        # Index

        if arch_string == "IndexForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama

        # Granite (v3)

        if arch_string == "GraniteForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama

        # Deci

        if arch_string == "DeciLMForCausalLM":
            arch_recognized = True
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama
#            self.lm.keys.update({
#                "attn_o": ".self_attn.linear_attn",
#            })
            self.lm.supports_tp = True

        # Llama (default + fallback)

        if arch_string != "LlamaForCausalLM" and not arch_recognized:
            print(f" !! Warning, unknown architecture: {arch_string}")
            print(f" !! Loading as LlamaForCausalLM")
            self.arch_string = "LlamaForCausalLM"
        if not arch_recognized:
            self.lm.layer_keys += \
                layer_keys_llama_norms + \
                layer_keys_llama_attn + \
                layer_keys_llama_mlp
            self.lm.expect_keys += \
                expect_keys_llama
            self.lm.supports_tp = True

        # Arch overrides

        if read_config.get("attention_bias", False):
            self.lm.attention_bias_qkv = True
            self.lm.attention_bias_o = True

        if read_config.get("mlp_bias", False):
            self.lm.mlp_bias = True

        if read_config.get("tie_word_embeddings", False):
            if ["lm_head"] in self.lm.expect_keys:
                self.lm.expect_keys.remove(["lm_head"])
                self.lm.keys.update({
                    "lm_head": "model.embed_tokens",
                })

        # Sanity checks

        if self.lm.residual_stream_fp32:
            assert self.lm.keys["norm_1_post"] and self.lm.keys["norm_2_post"], \
                "FP32 residual stream only implemented for arch with post layernorms"

    def make_fused_mlp(self):

        for x in layer_keys_llama_mlp: self.lm.layer_keys.remove(x)
        self.lm.layer_keys += layer_keys_llama_mlp_swiglu
        self.lm.keys.update({
            "fused_mlp_12": layer_keys_llama_mlp_swiglu[0][0],
            "fused_mlp_3": layer_keys_llama_mlp_swiglu[1][0],
        })
