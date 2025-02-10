from exllamav2.conversion.qparams import QParams
from exllamav2.ext import exllamav2_ext as ext_c, none_tensor
import math
import itertools
import time
from exllamav2.conversion.bot_status import print_stage

def optimize(job, save_fn, model):

    cfg = model.config
    km = cfg.arch.lm.keys

    has_gate = cfg.arch.lm.mlp_gate
    if has_gate: mlp_key_gate = km["mlp_gate"]
    mlp_key_up = km["mlp_up"]
    mlp_key_down = km["mlp_down"]

    norm_interval = (1.5, 3.5)
    norm_2ndstage = 0.15
    anneal_temp_max = 2
    anneal_temp_min = 0.0001
    anneal_cooling_factor = 0.995
    anneal_iter = 1000
    anneal_samples = 80
    anneal_stages = 3

    first_q_layer = 0
    while not model.modules[first_q_layer].key.startswith(cfg.arch.lm_prefix + "model.layers"):
        first_q_layer += 1

    # max_step_size = 2
    # first_layer_bias = 4
    # bias_layers = 2
    # bias_iter = 0

    key = cfg.arch.lm_prefix + "model.layers.0"
    key_q = key + km["attn_q"]
    key_k = key + km["attn_k"]
    key_v = key + km["attn_v"]
    key_o = key + km["attn_o"]

    if not cfg.arch.lm.is_moe:
        if has_gate: key_g = key + mlp_key_gate
        key_u = key + mlp_key_up
        key_d = key + mlp_key_down
        mlp_mode = "mlp"
    else:
        if has_gate: key_g = key + mlp_key_gate.replace("*", "0")
        key_u = key + mlp_key_up.replace("*", "0")
        key_d = key + mlp_key_down.replace("*", "0")
        mlp_mode = "block_sparse_moe"

    num_experts = cfg.num_experts if cfg.num_experts is not None else 1
    shape_q = model.modules_dict[key_q].matrix_shape()
    shape_k = model.modules_dict[key_k].matrix_shape()
    shape_v = model.modules_dict[key_v].matrix_shape()
    shape_o = model.modules_dict[key_o].matrix_shape()
    shape_g = model.modules_dict[key_g].matrix_shape() if has_gate else None
    shape_u = model.modules_dict[key_u].matrix_shape()
    shape_d = model.modules_dict[key_d].matrix_shape()
    numel_q = shape_q[0] * shape_q[1]
    numel_k = shape_k[0] * shape_k[1]
    numel_v = shape_v[0] * shape_v[1]
    numel_o = shape_o[0] * shape_o[1]
    numel_g = shape_g[0] * shape_g[1] * num_experts if has_gate else 0
    numel_u = shape_u[0] * shape_u[1] * num_experts
    numel_d = shape_d[0] * shape_d[1] * num_experts
    numel_attn = numel_q + numel_k + numel_v + numel_o
    numel_mlp = numel_g + numel_u + numel_d

    # Combined size of hidden layers

    num_layers = cfg.num_hidden_layers
    num_modules = num_layers * 2
    numel = sum(m.numel() for m in model.modules[first_q_layer : num_modules + first_q_layer])

    target_bpw = job["bits"]
    weight_budget = int(numel * target_bpw)

    # Compile options

    measurement = job["measurement"]
    slots = []
    params = []

    for i in range(num_layers):
        if cfg.arch.lm.parallel_decoder_blocks:
            m1 = measurement[cfg.arch.lm_prefix + "model.layers." + str(i) + ".parallel_decoder"]["attn"]
            m2 = measurement[cfg.arch.lm_prefix + "model.layers." + str(i) + ".parallel_decoder"]["mlp"]
        else:
            m1 = measurement[cfg.arch.lm_prefix + "model.layers." + str(i) + ".self_attn"]
            m2 = measurement[cfg.arch.lm_prefix + "model.layers." + str(i) + "." + mlp_mode]
        for m in [m1, m2]:
            slot = []
            param = []
            for opt in m:
                o = (int(opt["total_bits"]), 1 - opt["accuracy"])
                slot.append(o)
                param.append(opt)
            slots.append(slot)
            params.append(param)

    # Find some solutions

    last_update = 0
    m = float("inf")
    p = float("inf")
    c = weight_budget
    for i in range(anneal_stages * anneal_samples):
        if time.time() - last_update > 1 or i == anneal_samples - 1:
            print(f" -- Optimizing: {i + 1:4}/{anneal_stages * anneal_samples:4}")
            print_stage(job, "Optimizing", i + 1, anneal_stages * anneal_samples)
            last_update = time.time()

        if i < anneal_samples:
            t = i / (anneal_samples - 1)
            norm = (1 - t) * norm_interval[0] + t * norm_interval[1]

        elif i < anneal_samples * 2:
            if i == anneal_samples:
                norm_a = bestnorm - norm_2ndstage / 2
                norm_b = bestnorm + norm_2ndstage / 2
            t = i / (anneal_samples - 1) - 1
            norm = (1 - t) * norm_a + t * norm_b

        else:
            norm = bestnorm

        s_, si_, p_, c_, m_ = ext_c.sim_anneal(
            slots,
            weight_budget,
            anneal_temp_max,
            anneal_cooling_factor,
            anneal_temp_min,
            anneal_iter,
            norm
        )

        if i < anneal_samples * 2:
            if m_ < m:
                m = m_
                bestnorm = norm
        else:
            if p_ < p:
                s, si, p, c, m = s_, si_, p_, c_, m_

    # Tweak solution in case there is some budget left over

    while True:
        repeat = False
        for i in range(len(si)):
            if si[i] < len(slots[i]) - 1:
                delta_c = slots[i][si[i] + 1][0] - slots[i][si[i]][0]
                if c + delta_c <= weight_budget:
                    c += delta_c
                    si[i] = si[i] + 1
                    repeat = True
        if not repeat: break

    solution_idx = si
    print(f" -- max(err): {m:.6f}")
    print(f" -- error_norm: {bestnorm:.6f}")

    # Save strategy

    print(" -- Quantization strategy:")

    logerr = 0
    maxerr = 0
    job["strategy"] = {}
    for layer_ in range(num_layers):

        k1 = cfg.arch.lm_prefix + "model.layers." + str(layer_) + ".self_attn"
        k2 = cfg.arch.lm_prefix + "model.layers." + str(layer_) + "." + mlp_mode
        p1 = params[layer_ * 2][solution_idx[layer_ * 2]]
        p2 = params[layer_ * 2 + 1][solution_idx[layer_ * 2 + 1]]

        for (k, p, n) in zip((k1, k2), (p1, p2), (numel_attn, numel_mlp)):
            job["strategy"][k] = p
            bpw = p["total_bits"] / n
            err = 1 - p["accuracy"]
            print(f" --   {k:50} {bpw:1.4f} bpw - exp. error: {err:1.8f}")
            logerr += math.log(err)
            maxerr = max(err, maxerr)

    print(f" -- sum(log(err)): {logerr:.6f}")
    print(f" -- max(err): {maxerr:.6f}")

    xx = 0