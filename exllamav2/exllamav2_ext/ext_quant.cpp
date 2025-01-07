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
    const int redistribution_iterations = 25;
    const float bpw_penalty_scale = 0.01f;
    const float min_bpw_limit = 2.0f;
    const int opportunistic_iterations = 10000;
    const float initial_opportunistic_temp = 0.01f;
    const float low_error_threshold = 0.0009f;

    // --- Original Simulated Annealing ---
    int num_slots = slots.size();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<std::tuple<uint64_t, float>> solution(num_slots);
    std::vector<int> solution_idx(num_slots);

    uint64_t current_cost = 0;
    float current_max_exp_error = 0;

    float temp = initial_temp;
    int iterations_outer = static_cast<int>(std::log(min_temp / temp) / std::log(cooling_factor));

    for (int i = 0; i < num_slots; ++i)
    {
        solution[i] = slots[i][0];
        current_cost += std::get<0>(slots[i][0]);
        current_max_exp_error = std::max(current_max_exp_error, std::get<1>(slots[i][0]));
    }

    for (int j = 0; j < iterations_outer; ++j)
    {
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

            if (current_cost + delta_cost <= max_cost || (delta_cost < 0 && current_cost > max_cost))
            {
                if (delta_e < 0 ||
                    std::uniform_real_distribution<>(0, 1)(gen) < std::exp(-delta_e / temp))
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
    auto calculate_bpw = [&](const std::tuple<uint64_t, float>& option) {
        return 8.0f * std::get<0>(option) / 1024.0f;
    };

    auto calculate_bpw_stats = [&](const std::vector<std::tuple<uint64_t, float>>& sol) {
        std::vector<float> current_bpws(num_slots);
        for (int i = 0; i < num_slots; ++i) {
            current_bpws[i] = calculate_bpw(sol[i]);
        }
        float bpw_mean = std::accumulate(current_bpws.begin(), current_bpws.end(), 0.0f) / num_slots;
        float bpw_sq_sum = std::inner_product(current_bpws.begin(), current_bpws.end(), current_bpws.begin(), 0.0f);
        float bpw_variance = bpw_sq_sum / num_slots - bpw_mean * bpw_mean;
        return std::make_pair(bpw_mean, std::sqrt(std::max(0.0f, bpw_variance)));
    };

    for (int r = 0; r < redistribution_iterations; ++r) {
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
                    float bpw_penalty = bpw_penalty_scale * (new_bpw_stddev - current_bpw_stddev);

                    if (new_max_exp_error + bpw_penalty < current_max_exp_error) {
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
        // Select a neighborhood of slots
        int center_slot = std::uniform_int_distribution<>(0, num_slots - 1)(gen);
        int neighborhood_size = std::min(5, num_slots); // Example neighborhood size
        int start_slot = std::max(0, center_slot - neighborhood_size / 2);
        int end_slot = std::min(num_slots - 1, center_slot + neighborhood_size / 2);

        // Calculate average BPW in the neighborhood
        float neighborhood_bpw_sum = 0;
        for (int j = start_slot; j <= end_slot; ++j) {
            neighborhood_bpw_sum += calculate_bpw(solution[j]);
        }
        float neighborhood_bpw_avg = neighborhood_bpw_sum / (end_slot - start_slot + 1);

        // Adjust BPWs within the neighborhood
        std::vector<std::tuple<uint64_t, float>> new_solution = solution;
        std::vector<int> new_solution_idx = solution_idx;
        float new_sum_log_err = current_sum_log_err;
        uint64_t new_cost = current_cost;

        for (int j = start_slot; j <= end_slot; ++j) {
            float current_bpw = calculate_bpw(solution[j]);
            float target_bpw = neighborhood_bpw_avg;

            // Error-weighted adjustment
            float avg_error = 0;
            for (int k = start_slot; k <= end_slot; ++k) {
                avg_error += std::get<1>(solution[k]);
            }
            avg_error /= (end_slot - start_slot + 1);
            float error_ratio = std::get<1>(solution[j]) / avg_error;

            float adjustment = 0.125f;

            // Adjust BPW towards the target, weighted by error
            if (current_bpw < target_bpw) {
                // Search for a higher BPW option
                for (int n = 0; n < slots[j].size(); ++n) {
                    auto new_option = slots[j][n];
                    if (calculate_bpw(new_option) > current_bpw && calculate_bpw(new_option) <= current_bpw + adjustment) {
                        if (new_cost - std::get<0>(solution[j]) + std::get<0>(new_option) <= max_cost)
                         {
                            new_cost = new_cost - std::get<0>(solution[j]) + std::get<0>(new_option);
                            new_sum_log_err = new_sum_log_err - log(std::get<1>(solution[j])) + log(std::get<1>(new_option));
                            new_solution[j] = new_option;
                            new_solution_idx[j] = n;
                            break;
                        }
                    }
                }
            } else if (current_bpw > target_bpw) {
                // Search for a lower BPW option
                for (int n = 0; n < slots[j].size(); ++n) {
                    auto new_option = slots[j][n];
                    if (calculate_bpw(new_option) < current_bpw && calculate_bpw(new_option) >= current_bpw - adjustment) {
                        if (new_cost - std::get<0>(solution[j]) + std::get<0>(new_option) <= max_cost)
                         {
                            new_cost = new_cost - std::get<0>(solution[j]) + std::get<0>(new_option);
                            new_sum_log_err = new_sum_log_err - log(std::get<1>(solution[j])) + log(std::get<1>(new_option));
                            new_solution[j] = new_option;
                            new_solution_idx[j] = n;
                            break;
                        }
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
            error_factor = 0.1f; // Reduce the weight of sum_log_err
        }

        if (new_cost <= max_cost && calculate_bpw(new_solution[i]) >= min_bpw_limit) {
            if (delta_sum_log_err * error_factor < 0 || std::uniform_real_distribution<>(0, 1)(gen) < std::exp(-delta_sum_log_err * error_factor / local_temp)) {
                accept = true;
            }
            // Give high priority to the solution that has high minimum BPW
            if (calculate_bpw(new_solution[i]) < min_bpw_limit) {
                accept = false;
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

    // --- Final Cost Check and Rollback (if necessary) ---
    if (current_cost > max_cost) {
        std::vector<std::pair<float, int>> error_indices(num_slots);
        for (int i = 0; i < num_slots; ++i) {
            error_indices[i] = {std::get<1>(solution[i]), i};
        }
        std::sort(error_indices.begin(), error_indices.end());

        for (const auto& pair : error_indices) {
            int i = pair.second;
            for (int n = slots[i].size() - 1; n >= 0; --n) {
                if (calculate_bpw(slots[i][n]) < calculate_bpw(solution[i])) {
                    if (current_cost - std::get<0>(solution[i]) + std::get<0>(slots[i][n]) <= max_cost)
                     {
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

