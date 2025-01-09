#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "config.h"
#include "ext_quant.h"

#include "cuda/pack_tensor.cuh"
#include "cuda/quantize.cuh"

#include "cpp/util.h"

// Packing functions

void pack_rows_4
(
    torch::Tensor input,
    torch::Tensor output
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(input, kShort);
    TORCH_CHECK_DTYPE(output, kInt);
    TORCH_CHECK_SHAPES(input, 0, output, 0, 1);
    TORCH_CHECK_SHAPES(input, 1, output, 1, 8);

    int rows = input.size(0);
    int columns = input.size(1);

    pack_rows_4_cuda
    (
        stream,
        (uint16_t*) input.data_ptr(),
        (uint32_t*) output.data_ptr(),
        rows,
        columns
    );
}

void pack_columns
(
    torch::Tensor input,
    torch::Tensor output,
    int bits
)
{
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK_DTYPE(input, kShort);
    TORCH_CHECK_DTYPE(output, kInt);
    TORCH_CHECK_SHAPES(input, 1, output, 1, 1);

    int in_rows = input.size(0);
    int columns = input.size(1);
    int out_rows = output.size(0);
    int exp_out_rows = in_rows * bits / 32;
    TORCH_CHECK(out_rows == exp_out_rows, "Wrong output shape for input and bitrate")

    pack_columns_cuda
    (
        stream,
        (uint16_t*) input.data_ptr(),
        (uint32_t*) output.data_ptr(),
        in_rows,
        out_rows,
        columns,
        bits
    );
}


// Quantization functions

void quantize_err
(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    float qzero,
    float maxq,
    float err_norm,
    float min_p,
    float max_p,
    int p_grid
)
{
    TORCH_CHECK_DTYPE(input, kFloat);
    TORCH_CHECK_DTYPE(output, kFloat);
    // TORCH_CHECK_SHAPES(input, 0, output, 0, 1);
    // TORCH_CHECK_SHAPES(input, 1, output, 1, 1);
    TORCH_CHECK_SHAPES(input, 1, scale, 0, 1);
    TORCH_CHECK(output.size(0) == p_grid + 1, "Output vector shape doesn't match grid")

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int rows = input.size(0);
    int columns = input.size(1);

    quantize_err_cuda
    (
        stream,
        (float*) input.data_ptr(),
        (float*) output.data_ptr(),
        (float*) scale.data_ptr(),
        rows,
        columns,
        qzero,
        maxq,
        err_norm,
        min_p,
        max_p,
        p_grid
    );
}

void quantize
(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scale,
    torch::Tensor out_q,
    float qzero,
    float maxq
)
{
    TORCH_CHECK_DTYPE(input, kFloat);
    TORCH_CHECK_DTYPE(output, kFloat);
    TORCH_CHECK_SHAPES(input, 0, output, 0, 1);
    TORCH_CHECK_SHAPES(input, 1, output, 1, 1);
    TORCH_CHECK_SHAPES(input, 1, scale, 0, 1);

    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    int rows = input.size(0);
    int columns = input.size(1);

    quantize_cuda
    (
        stream,
        (float*) input.data_ptr(),
        (float*) output.data_ptr(),
        (float*) scale.data_ptr(),
        out_q.device().is_meta() ? NULL : (uint16_t*) out_q.data_ptr(),
        rows,
        columns,
        qzero,
        maxq
    );
}

std::tuple<std::vector<std::tuple<uint64_t, float>>, std::vector<int>, float, uint64_t, float> sim_anneal
(
    const std::vector<std::vector<std::tuple<uint64_t, float>>>& slots,
    uint64_t max_cost,
    float initial_temp,
    float cooling_factor,
    float min_temp,
    int iterations,
    float norm
)
{
    // --- Internal Parameters ---
    const int redistribution_iterations = 50; // Increased
    const float bpw_penalty_scale = 0.5f;  // Further increased BPW penalty
    const float min_bpw_base = 3.5f; // Increased baseline
    const int opportunistic_iterations = 20000;
    const float initial_opportunistic_temp = 0.05f;
    const float low_error_threshold = 0.001f;
    const float targeted_redistribution_bpw_threshold = 3.5f; // Increased threshold
    const float targeted_redistribution_max_err_increase_initial = 1.1f;
    const float targeted_redistribution_max_err_increase_final = 1.01f;
    const float high_bpw_donor_threshold = 5.0f;
    const int num_options_to_explore_per_layer = 8;  // Explore more options

    // --- Dynamic Minimum BPW ---
    auto calculate_dynamic_min_bpw = [&](float target_bpw, float temp_ratio) {
        float scaled_min_bpw = min_bpw_base + 0.5f * (target_bpw - min_bpw_base);
        return min_bpw_base + temp_ratio * (scaled_min_bpw - min_bpw_base);
    };

    // --- Calculate BPW ---
    auto calculate_bpw = [&](const std::tuple<uint64_t, float>& option) {
        return 8.0f * std::get<0>(option) / 1024.0f;
    };

    // --- Calculate BPW stats ---
    auto calculate_bpw_stats = [&](const std::vector<std::tuple<uint64_t, float>>& sol) {
        int num_slots = sol.size();
        std::vector<float> current_bpws(num_slots);
        for (int i = 0; i < num_slots; ++i) {
            current_bpws[i] = calculate_bpw(sol[i]);
        }
        float bpw_mean = std::accumulate(current_bpws.begin(), current_bpws.end(), 0.0f) / num_slots;
        float bpw_sq_sum = std::inner_product(current_bpws.begin(), current_bpws.end(), current_bpws.begin(), 0.0f);
        float bpw_variance = bpw_sq_sum / num_slots - bpw_mean * bpw_mean;
        return std::make_pair(bpw_mean, std::sqrt(std::max(0.0f, bpw_variance)));
    };

    // --- Original Simulated Annealing ---
    int num_slots = slots.size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::tuple<uint64_t, float>> solution(num_slots);
    std::vector<int> solution_idx(num_slots);

    uint64_t current_cost = 0;
    float current_max_exp_error = 0;

    float temp = initial_temp * 2;  // Higher initial temperature
    int iterations_outer = static_cast<int>(std::log(min_temp / temp) / std::log(cooling_factor));
    float target_bpw = max_cost * 8.0f / 1024.0f / num_slots;

    for (int i = 0; i < num_slots; ++i)
    {
        solution[i] = slots[i][0];
        current_cost += std::get<0>(slots[i][0]);
        current_max_exp_error = std::max(current_max_exp_error, std::get<1>(slots[i][0]));
    }

    for (int j = 0; j < iterations_outer; ++j)
    {
        float temp_ratio = temp / (initial_temp * 2);
        float min_bpw_limit = calculate_dynamic_min_bpw(target_bpw, temp_ratio);

        for (int k = 0; k < iterations; ++k)
        {
            int i = std::uniform_int_distribution<>(0, num_slots - 1)(gen);
            int n = std::uniform_int_distribution<>(0, slots[i].size() - 1)(gen);
            auto new_option = slots[i][n];
            auto old_option = solution[i];
            uint64_t delta_cost = std::get<0>(new_option) - std::get<0>(old_option);
            float delta_e = std::get<1>(new_option) - std::get<1>(old_option);

            float new_max_exp_error = current_max_exp_error;
            if (std::get<1>(old_option) == current_max_exp_error) {
                new_max_exp_error = std::get<1>(new_option);
                for (int slot_idx = 0; slot_idx < num_slots; slot_idx++) {
                    if (slot_idx == i) continue;
                    new_max_exp_error = std::max(new_max_exp_error, std::get<1>(solution[slot_idx]));
                }
            } else {
                new_max_exp_error = std::max(current_max_exp_error, std::get<1>(new_option));
            }

            // BPW Penalty (Dynamic, Temperature-Dependent, and Non-Linear)
            float bpw_new = calculate_bpw(new_option);
            float bpw_penalty = 0.0f;
            if (bpw_new < min_bpw_limit) {
                bpw_penalty = powf((min_bpw_limit - bpw_new), 2.0f) * bpw_penalty_scale * (1 + temp_ratio);
            }

            if (current_cost + delta_cost <= max_cost || (delta_cost < 0 && current_cost > max_cost))
            {
                if (delta_e + bpw_penalty < 0 ||
                    std::uniform_real_distribution<>(0, 1)(gen) < std::exp(-(delta_e + bpw_penalty) / temp))
                {
                    solution[i] = new_option;
                    solution_idx[i] = n;
                    current_cost += delta_cost;
                    current_max_exp_error = new_max_exp_error;
                }
            }
        }
        temp *= cooling_factor;
    }

    // --- Post-processing: Bit Redistribution ---

    for (int r = 0; r < redistribution_iterations; ++r) {
        float temp_ratio = temp / (initial_temp * 2);
        float min_bpw_limit = calculate_dynamic_min_bpw(target_bpw, temp_ratio);

        // Calculate BPW statistics and dynamic bpw_threshold
        auto [bpw_mean, bpw_stddev] = calculate_bpw_stats(solution);
        float bpw_threshold = std::max(min_bpw_limit, bpw_mean - 0.5f * bpw_stddev);

        std::vector<int> low_bpw_indices;
        std::vector<int> high_bpw_indices;

        for (int i = 0; i < num_slots; ++i) {
            float bpw = calculate_bpw(solution[i]);
            if (bpw < bpw_threshold) {
                low_bpw_indices.push_back(i);
            } else {
                high_bpw_indices.push_back(i);
            }
        }

        bool improved = false;
        for (int low_idx : low_bpw_indices) {
            if (high_bpw_indices.empty()) break;

            // Error-weighted selection of high_idx
            std::vector<float> high_bpw_errors;
            for (int high_idx : high_bpw_indices) {
                high_bpw_errors.push_back(std::get<1>(solution[high_idx]));
            }
            std::discrete_distribution<int> high_idx_dist(high_bpw_errors.begin(), high_bpw_errors.end());
            int high_idx = high_bpw_indices[high_idx_dist(gen)];

            // Find a higher BPW option for the low-BPW slot, with bias towards lower error
            int best_low_new_idx = -1;
            float best_low_new_error = 1e10f;
            for (int n = 0; n < slots[low_idx].size(); ++n) {
                if (calculate_bpw(slots[low_idx][n]) > calculate_bpw(solution[low_idx])) {
                    float error_factor = 1.0f + std::get<1>(slots[low_idx][n]);
                    if (error_factor * std::get<1>(slots[low_idx][n]) < best_low_new_error) {
                        best_low_new_error = error_factor * std::get<1>(slots[low_idx][n]);
                        best_low_new_idx = n;
                    }
                }
            }

            // Find a lower BPW option for the high-BPW slot, with bias towards lower error
            int best_high_new_idx = -1;
            float best_high_new_error = 1e10f;
            for (int n = 0; n < slots[high_idx].size(); ++n) {
                if (calculate_bpw(slots[high_idx][n]) < calculate_bpw(solution[high_idx])) {
                    float error_factor = 1.0f + std::get<1>(slots[high_idx][n]);
                    if (error_factor * std::get<1>(slots[high_idx][n]) < best_high_new_error) {
                        best_high_new_error = error_factor * std::get<1>(slots[high_idx][n]);
                        best_high_new_idx = n;
                    }
                }
            }

            if (best_low_new_idx != -1 && best_high_new_idx != -1) {
                auto new_low_option = slots[low_idx][best_low_new_idx];
                auto new_high_option = slots[high_idx][best_high_new_idx];

                uint64_t new_cost = current_cost - std::get<0>(solution[low_idx]) - std::get<0>(solution[high_idx]) + std::get<0>(new_low_option) + std::get<0>(new_high_option);

                if (new_cost <= max_cost) {
                    float new_max_exp_error = std::get<1>(new_low_option);
                    for (int i = 0; i < num_slots; i++) {
                        if (i == low_idx) continue;
                        if (i == high_idx) {
                            new_max_exp_error = std::max(new_max_exp_error, std::get<1>(new_high_option));
                        } else {
                            new_max_exp_error = std::max(new_max_exp_error, std::get<1>(solution[i]));
                        }
                    }

                    auto [current_bpw_mean, current_bpw_stddev] = calculate_bpw_stats(solution);
                    auto [new_bpw_mean, new_bpw_stddev] = calculate_bpw_stats({new_low_option, new_high_option});
                    float bpw_change_penalty = bpw_penalty_scale * std::max(0.0f, new_bpw_stddev - current_bpw_stddev); // Penalize increased variance

                    if (new_max_exp_error + bpw_change_penalty < current_max_exp_error) {
                        solution[low_idx] = new_low_option;
                        solution_idx[low_idx] = best_low_new_idx;
                        solution[high_idx] = new_high_option;
                        solution_idx[high_idx] = best_high_new_idx;
                        current_cost = new_cost;
                        current_max_exp_error = new_max_exp_error;
                        improved = true;
                    }
                }
            }
        }
    }

    // --- Opportunistic Optimization with Simulated Annealing ---
    float current_sum_log_err = 0;
    for (int i = 0; i < num_slots; ++i) {
        current_sum_log_err += log(std::get<1>(solution[i]));
    }

    float best_sum_log_err = current_sum_log_err;
    std::vector<std::tuple<uint64_t, float>> best_solution = solution;
    std::vector<int> best_solution_idx = solution_idx;
    float local_temp = initial_opportunistic_temp;

    for (int i = 0; i < opportunistic_iterations; ++i) {
        float temp_ratio = temp / (initial_temp * 2);
        float min_bpw_limit = calculate_dynamic_min_bpw(target_bpw, temp_ratio);

        // Select a slot to adjust
        int target_slot = std::uniform_int_distribution<>(0, num_slots - 1)(gen);

        // Calculate the global average BPW
        float global_bpw_sum = 0;
        for (int j = 0; j < num_slots; ++j) {
            global_bpw_sum += calculate_bpw(solution[j]);
        }
        float global_bpw_avg = global_bpw_sum / num_slots;

        // Adjust BPW of the target slot
        std::vector<std::tuple<uint64_t, float>> new_solution = solution;
        std::vector<int> new_solution_idx = solution_idx;
        float new_sum_log_err = current_sum_log_err;
        uint64_t new_cost = current_cost;

        float current_bpw = calculate_bpw(solution[target_slot]);

        // Adjust BPW towards the global average, weighted by error
        float avg_error = 0;
        for (int k = 0; k < num_slots; ++k) {
            avg_error += std::get<1>(solution[k]);
        }
        avg_error /= num_slots;
        float error_ratio = std::get<1>(solution[target_slot]) / avg_error;

        float adjustment = 0.25f + 0.25f * error_ratio;

        // Adjust BPW towards the target, weighted by error, with a bias towards higher BPW
        if (current_bpw < global_bpw_avg + adjustment) {
            // Search for a higher BPW option
            for (int n = 0; n < slots[target_slot].size(); ++n) {
                auto new_option = slots[target_slot][n];
                if (calculate_bpw(new_option) > current_bpw && calculate_bpw(new_option) <= current_bpw + adjustment) {
                    if (new_cost - std::get<0>(solution[target_slot]) + std::get<0>(new_option) <= max_cost) {
                        new_cost = new_cost - std::get<0>(solution[target_slot]) + std::get<0>(new_option);
                        new_sum_log_err = new_sum_log_err - log(std::get<1>(solution[target_slot])) + log(std::get<1>(new_option));
                        new_solution[target_slot] = new_option;
                        new_solution_idx[target_slot] = n;
                        break;
                    }
                }
            }
        } else if (current_bpw > global_bpw_avg) {
            // Search for a lower BPW option
            for (int n = slots[target_slot].size() - 1; n >= 0; --n) {
                auto new_option = slots[target_slot][n];
                if (calculate_bpw(new_option) < current_bpw && calculate_bpw(new_option) >= current_bpw - adjustment) {
                    if (new_cost - std::get<0>(solution[target_slot]) + std::get<0>(new_option) <= max_cost) {
                        new_cost = new_cost - std::get<0>(solution[target_slot]) + std::get<0>(new_option);
                        new_sum_log_err = new_sum_log_err - log(std::get<1>(solution[target_slot])) + log(std::get<1>(new_option));
                        new_solution[target_slot] = new_option;
                        new_solution_idx[target_slot] = n;
                        break;
                    }
                }
            }
        }

        // Calculate new max exp error
        float new_max_exp_error = 0;
        for (int j = 0; j < num_slots; ++j) {
            new_max_exp_error = std::max(new_max_exp_error, std::get<1>(new_solution[j]));
        }

        // Acceptance criterion with a small probability of accepting worse solutions
        bool accept = false;
        float delta_sum_log_err = new_sum_log_err - current_sum_log_err;

        // Dampen penalty for low errors
        float error_factor = 1.0f;
        if (current_max_exp_error < low_error_threshold) {
            error_factor = 0.1f;
        }

        // Introduce a small probability of accepting a worse solution (simulated annealing-like)
        if (new_cost <= max_cost) {
            float bpw_penalty_opp = 0.0f;
            if (calculate_bpw(new_solution[target_slot]) < min_bpw_limit) {
                bpw_penalty_opp = powf((min_bpw_limit - calculate_bpw(new_solution[target_slot])), 2.0f) * bpw_penalty_scale * (1 + temp_ratio);
            }
            if (delta_sum_log_err * error_factor + bpw_penalty_opp < 0 || std::uniform_real_distribution<>(0, 1)(gen) < std::exp(-(delta_sum_log_err * error_factor + bpw_penalty_opp) / local_temp)) {
                accept = true;
            }
        }

        if (accept) {
            solution = new_solution;
            solution_idx = new_solution_idx;
            current_sum_log_err = new_sum_log_err;
            current_cost = new_cost;
            current_max_exp_error = new_max_exp_error;

            if (current_sum_log_err < best_sum_log_err) {
                best_sum_log_err = current_sum_log_err;
                best_solution = solution;
                best_solution_idx = solution_idx;
            }
        }

        local_temp *= 0.95f;
    }

    // Use the best solution found during opportunistic optimization
    solution = best_solution;
    solution_idx = best_solution_idx;
    current_sum_log_err = best_sum_log_err;

    // --- BPW Smoothing (Post-processing) ---
    for (int i = 1; i < num_slots - 1; ++i) {
        float current_bpw = calculate_bpw(solution[i]);
        float prev_bpw = calculate_bpw(solution[i - 1]);
        float next_bpw = calculate_bpw(solution[i + 1]);
        float avg_neighbor_bpw = (prev_bpw + next_bpw) / 2.0f;

        if (current_bpw < avg_neighbor_bpw - 0.5f) {  // Significant difference
            // Find a higher BPW option for the current slot
            for (int n = 0; n < slots[i].size(); ++n) {
                auto new_option = slots[i][n];
                if (calculate_bpw(new_option) > current_bpw && calculate_bpw(new_option) <= avg_neighbor_bpw) {
                    if (current_cost - std::get<0>(solution[i]) + std::get<0>(new_option) <= max_cost) {
                        // Check if the new option doesn't significantly increase max_err
                        float new_max_err = 0;
                        for (int j = 0; j < num_slots; ++j) {
                            if (j == i) {
                                new_max_err = std::max(new_max_err, std::get<1>(new_option));
                            } else {
                                new_max_err = std::max(new_max_err, std::get<1>(solution[j]));
                            }
                        }

                        if (new_max_err < current_max_exp_error * 1.1f) { // Allow a small increase in max_err
                            current_cost = current_cost - std::get<0>(solution[i]) + std::get<0>(new_option);
                            solution[i] = new_option;
                            solution_idx[i] = n;
                            current_max_exp_error = new_max_err;
                            break;
                        }
                    }
                }
            }
        }
    }

    // --- Targeted Bit Redistribution (Post-processing) ---
    for (int iter = 0; iter < num_slots * 3; ++iter) {
        // Create a global pool of donor indices
        std::vector<int> donor_indices;
        for (int j = 0; j < num_slots; ++j) {
            if (calculate_bpw(solution[j]) > high_bpw_donor_threshold) {
                donor_indices.push_back(j);
            }
        }

        if (donor_indices.empty()) continue; // Skip if no donors

        for (int i = 0; i < num_slots; ++i) {
            float current_bpw = calculate_bpw(solution[i]);
            if (current_bpw < targeted_redistribution_bpw_threshold) {
                // Randomly select a donor from the global pool
                int donor_idx = donor_indices[std::uniform_int_distribution<>(0, donor_indices.size() - 1)(gen)];

                // Explore multiple higher BPW options for the current slot
                std::vector<int> higher_bpw_options;
                for (int n = 0; n < slots[i].size(); ++n) {
                    if (calculate_bpw(slots[i][n]) > current_bpw && calculate_bpw(slots[i][n]) >= targeted_redistribution_bpw_threshold) {
                        higher_bpw_options.push_back(n);
                    }
                }

                std::shuffle(higher_bpw_options.begin(), higher_bpw_options.end(), gen);
                int options_to_explore = std::min((int)higher_bpw_options.size(), num_options_to_explore_per_layer);

                for (int option_idx = 0; option_idx < options_to_explore; ++option_idx) {
                    int best_new_idx = higher_bpw_options[option_idx];
                    auto new_option = slots[i][best_new_idx];

                    // Find a lower BPW option for the donor slot
                    int best_donor_new_idx = -1;
                    float best_donor_new_error = 1e10f;
                    for (int n = 0; n < slots[donor_idx].size(); ++n) {
                        if (calculate_bpw(slots[donor_idx][n]) < calculate_bpw(solution[donor_idx])) {
                            float error_factor = 1.0f + std::get<1>(slots[donor_idx][n]);
                            if (error_factor * std::get<1>(slots[donor_idx][n]) < best_donor_new_error) {
                                best_donor_new_error = error_factor * std::get<1>(slots[donor_idx][n]);
                                best_donor_new_idx = n;
                            }
                        }
                    }

                    if (best_donor_new_idx != -1) {
                        auto donor_new_option = slots[donor_idx][best_donor_new_idx];
                        uint64_t new_cost = current_cost - std::get<0>(solution[i]) - std::get<0>(solution[donor_idx])
                                            + std::get<0>(new_option) + std::get<0>(donor_new_option);

                        if (new_cost <= max_cost) {
                            float new_max_err = std::get<1>(new_option);
                            for (int j = 0; j < num_slots; ++j) {
                                if (j == i) continue;
                                if (j == donor_idx) {
                                    new_max_err = std::max(new_max_err, std::get<1>(donor_new_option));
                                } else {
                                    new_max_err = std::max(new_max_err, std::get<1>(solution[j]));
                                }
                            }

                            // Adaptive max_err_increase
                            float max_err_increase = targeted_redistribution_max_err_increase_initial -
                                                    (targeted_redistribution_max_err_increase_initial - targeted_redistribution_max_err_increase_final) *
                                                    (static_cast<float>(iter) / (num_slots * 3));

                            if (new_max_err < current_max_exp_error * max_err_increase) {
                                current_cost = new_cost;
                                solution[i] = new_option;
                                solution_idx[i] = best_new_idx;
                                solution[donor_idx] = donor_new_option;
                                solution_idx[donor_idx] = best_donor_new_idx;
                                current_max_exp_error = new_max_err;
                                break; // Move to the next low-bpw layer after a successful redistribution
                            }
                        }
                    }
                }
            }
        }
    }

    // --- Final Cost Check and Rollback (if necessary) ---
    if (current_cost > max_cost) {
        std::vector<std::tuple<float, float, int>> bpw_error_indices(num_slots);
        for (int i = 0; i < num_slots; ++i) {
            float bpw = calculate_bpw(solution[i]);
            float error = std::get<1>(solution[i]);
            float penalty = (bpw < targeted_redistribution_bpw_threshold) ? 1000.0f : 0.0f; // High penalty for low BPW
            bpw_error_indices[i] = {error + penalty, bpw, i};
        }
        std::sort(bpw_error_indices.begin(), bpw_error_indices.end(), std::greater<std::tuple<float, float, int>>());

        for (const auto& tuple : bpw_error_indices) {
            int i = std::get<2>(tuple);
            for (int n = slots[i].size() - 1; n >= 0; --n) {
                if (calculate_bpw(slots[i][n]) < calculate_bpw(solution[i])) {
                    if (current_cost - std::get<0>(solution[i]) + std::get<0>(slots[i][n]) <= max_cost) {
                        uint64_t delta_cost = std::get<0>(slots[i][n]) - std::get<0>(solution[i]);
                        current_cost += delta_cost;
                        solution[i] = slots[i][n];
                        solution_idx[i] = n;
                        break;
                    }
                }
            }
            if (current_cost <= max_cost) break;
        }
    }

    // Calculate final max error and sum of log errors
    float max_err = 0.0f;
    float sum_log_err = 0.0;
    for (int i = 0; i < num_slots; ++i) {
        max_err = std::max(max_err, std::get<1>(solution[i]));
        sum_log_err += log(std::get<1>(solution[i]));
    }

    return { solution, solution_idx, sum_log_err, current_cost, max_err };
}

